# GeoLens Studio v2 Implementation Plan

> **For agentic workers:** 本计划由主会话内联执行（executing-plans 模式），逐任务勾选。步骤用 checkbox 跟踪。

**Goal:** 把 GeoLens 从"问答 Demo"升级为作品集级分析服务：滑窗大图推理 + GeoJSON/报告导出 + FastAPI 服务 + Gradio 界面 + int4 量化 + Docker + 双端基准。

**Architecture:** 三层——`engine/`（纯函数推理库，pytest 单测）→ `service/`（FastAPI 编排：任务队列/校验/产物管理）→ `ui/`（Gradio Blocks 只调 HTTP）。产物按任务 ID 隔离落盘。

**Tech Stack:** PyTorch/transformers（Qwen2.5-VL + weighted adapter）、rasterio（GeoTIFF，装不上则降级）、FastAPI、Gradio、AutoAWQ（备选 bitsandbytes NF4）、Docker。

设计文档：`docs/specs/2026-10-11-geolens-studio-design.md`

---

### Task 1: engine/vocab.py —— 共享类别词表与解析

**Files:** Create `engine/__init__.py`（空）、`engine/vocab.py`；Test `tests/test_vocab.py`

接口：`CLASS_ZH`（45 类 en→(zh,desc)）、`ZH2EN`、`SYNONYMS`、`parse_class(answer:str) -> str|None`（先标准名正则、后同义词、含 word-boundary 处理）。

- [ ] 失败测试：
```python
from engine.vocab import parse_class
def test_parse_direct(): assert parse_class("影像中的主要地物是山脉。") == "mountain"
def test_parse_synonym(): assert parse_class("这是飞机场") == "airport"
def test_parse_none(): assert parse_class("看不太出来") is None
def test_parse_bridge_substring(): assert parse_class("画面里有一座桥") == "bridge"
```
- [ ] 跑 `pytest tests/test_vocab.py -v` 确认 FAIL → 实现（把 `scripts/20_build_instructions.py` 与 `eval/30_zeroshot_eval.py` 的重复逻辑抽到这里，两处后续改引用）→ PASS → commit

### Task 2: engine/tiler.py —— 滑窗切片

**Files:** Create `engine/tiler.py`；Test `tests/test_tiler.py`

接口：`compute_tiles(w, h, window=448, overlap=0.25) -> list[tuple[x,y,ww,hh]]`（边缘钳制、最小窗口过滤≥64px、完整覆盖无越界）、`crop_tiles(img:PIL.Image, tiles) -> list[PIL.Image]`。

- [ ] 失败测试：
```python
from engine.tiler import compute_tiles
def test_exact_fit(): assert compute_tiles(448, 448) == [(0, 0, 448, 448)]
def test_cover_edges():
    ts = compute_tiles(1000, 700)
    assert ts[-1][0] + ts[-1][2] == 1000 and ts[-1][1] + ts[-1][3] == 700  # 边缘钳制
def test_no_overlap_min(): assert len(compute_tiles(900, 900, window=448, overlap=0.25)) >= 4
def test_small_image(): assert compute_tiles(300, 200) == [(0, 0, 300, 200)]  # 小图不切
```
- [ ] FAIL → 实现 → PASS → commit

### Task 3: engine/aggregator.py —— 全图聚合

**Files:** Create `engine/aggregator.py`；Test `tests/test_aggregator.py`

接口：`aggregate(tiles: list[dict]) -> dict`，输入每窗 `{x,y,w,h,top1,conf}`；输出 `{class_hist, class_ratio, dominant, n_tiles, low_conf_tiles, low_conf_ratio}`（低置信阈值默认 0.6，可配）。

- [ ] 失败测试：
```python
from engine.aggregator import aggregate
TILES = [
    {"x":0,"y":0,"w":448,"h":448,"top1":"forest","conf":0.9},
    {"x":336,"y":0,"w":448,"h":448,"top1":"forest","conf":0.8},
    {"x":0,"y":336,"w":448,"h":448,"top1":"river","conf":0.3},
]
def test_hist(): assert aggregate(TILES)["class_hist"] == {"forest":2,"river":1}
def test_dominant(): assert aggregate(TILES)["dominant"] == "forest"
def test_low_conf(): assert aggregate(TILES)["low_conf_ratio"] == 1/3
```
- [ ] FAIL → 实现 → PASS → commit

### Task 4: engine/classifier.py —— 窗口分类器（真实模型）

**Files:** Create `engine/classifier.py`

接口：`TileClassifier(model_path, adapter=None, load4bit=False, batch_size=8)`、`.classify_tiles(images:list[PIL]) -> list[dict{top1, conf, answer}]`。用中文问答 + `vocab.parse_class` 解析；conf 取答案中类别词可信度（无外部概率时置 1.0/0.0，后续由低置信解析失败体现）。冒烟验证（本地模型或服务器）：2 张已知类图正确。不写单测（模型重），冒烟脚本 `engine/smoke.py`。

- [ ] 本地预置模型：hf-mirror 下载 Qwen2.5-VL-3B 到本地 HF 缓存 + 从服务器 scp weighted adapter → commit（脚本入库）

### Task 5: engine/exporter.py —— 产物导出

**Files:** Create `engine/exporter.py`；Test `tests/test_exporter.py`

接口：`render_overlay(img, tiles, class_hist, out_path)`（半透明色块+类别标签）、`to_geojson(tiles, crs_info=None) -> dict`（像素坐标或 GeoTIFF 真实坐标）、`build_report(stats, out_path)`（Markdown：统计表/众数类/低置信建议复核）。

- [ ] 测试：geojson 结构合法（type=FeatureCollection、geometry=Polygon、properties 含 class/conf）；overlay 生成文件尺寸与原图一致 → 实现 → commit

### Task 6: 真实数据样例

**Files:** `docs/data-sources.md`、`samples/`（说明+下载脚本）

免登录 Sentinel-2 样例（ESA 公开样例页/公开 bucket）；受阻则：支持任意 GeoTIFF 的文档指引 + RESISC45 拼接大图演示（`scripts/30_make_mosaic.py`，标注"演示拼接"）→ commit

### Task 7: service/api.py —— FastAPI 服务

**Files:** Create `service/api.py`、`service/__init__.py`；Test `tests/test_api.py`（mock 引擎）

接口：`POST /analyze`（multipart 图像 → task_id）、`GET /result/{id}`（状态+产物路径列表）、`GET /result/{id}/download/{file}`、`GET /health`。内存任务表+ThreadPoolExecutor(2)；校验格式/大小≤500MB；产物到 `outputs/tasks/{id}/`。

- [ ] 测试：TestClient mock TileClassifier → 上传→轮询→产物存在 → commit

### Task 8: ui/app_studio.py —— Gradio 工具界面

**Files:** Create `ui/app_studio.py`

标签页：①大图分析（上传→调 /analyze 轮询→展示 overlay/统计条形图/报告+GeoJSON 下载）②服务信息（/health、基准表）。httpx 调 API。本地手测 + 服务器部署手测 → commit

### Task 9: 量化与基准

**Files:** `bench/quantize.py`（AutoAWQ，失败转 bitsandbytes NF4 记 D13）、`bench/benchmark.py`

P50/P99/单图全流程耗时/QPS/显存，bf16@5090 与 int4@3060 双行；int4 精度用 450 张测试集复测。结果入 README → commit

### Task 10: Docker + 收尾

**Files:** `Dockerfile`、`docker-compose.yml`、README v2（架构图、GIF/截图、基准表、简历条目文案）

→ commit → push → 邮件通知用户 v2 完成

---

**自检记录**：spec 覆盖（滑窗✓T2 GeoJSON✓T5 真实数据✓T6 服务✓T7 界面✓T8 量化✓T9 Docker✓T10 数据闭环钩子✓T5 report）；无占位符；接口命名一致（parse_class/compute_tiles/classify_tiles/aggregate/to_geojson）。
