# GeoLens Studio v2 设计文档

日期：2026-10-11 ｜ 状态：已批准（用户同日确认"可以，开工吧"）

## 1. 定位与用户故事

**GeoLens Studio：把任意遥感影像变成结构化地物报告的分析服务。**

使用者（遥感/GIS 从业者、需要分析大幅影像的人）上传大幅卫星影像（JPG/PNG/GeoTIFF），服务自动完成：滑窗地物识别 → 置信度评估 → 生成分类叠加图、地物统计、GeoJSON 矢量导出、Markdown 分析报告；低置信区域标记"建议人工复核"（数据闭环钩子，呼应项目自动标注主线）。

## 2. 架构

```
Gradio Blocks 工具界面（拖图上传 / 结果浏览 / 报告下载 / 性能页）
        │ HTTP
FastAPI 服务层  POST /analyze → 任务ID；GET /result/{id} → 产物；GET /health
        │
推理引擎（纯函数库，可独立测试，不依赖 Web）
  ├─ Tiler       滑窗切片（默认 448×448，重叠 25%，边缘窗口钳制）
  ├─ Classifier  微调模型批量推理（每窗 top-k + 置信度，中文问答→类别解析）
  ├─ Aggregator  全图聚合：类别占比、众数类、低置信窗口统计
  └─ Exporter    叠加图渲染、GeoJSON 生成（GeoTIFF 带真实地理坐标，
                 普通图用像素坐标）、Markdown 报告
        │
Docker 一键部署 + 基准测试（P50/P99 延迟、QPS、显存、int4 精度损失）
```

工程边界：`engine/` 纯库（pytest 单测）；`service/` 只做编排（任务队列、校验、产物管理）；`ui/` 只调 HTTP API。

## 3. 关键技术决策

| 项 | 决定 | 备选 |
|---|---|---|
| 形态 | FastAPI 引擎 + Gradio 界面 + Docker | 自写前端（更产品化但前端耗时不划算）；纯 Gradio（缺服务层） |
| 滑窗 | 448×448，重叠 25% | 参数可配 |
| 量化 | AutoAWQ int4（D4） | bitsandbytes NF4（已验证可行） |
| 真实数据 | 免登录 Sentinel-2 公开样例优先 | 支持任意 GeoTIFF + README 指引；大幅拼接演示图（标注） |
| 基准 | bf16@5090 与 int4@3060 双行对比，450 张测试集复测精度差 | — |

## 4. 数据流与错误处理

上传 → 校验（格式 jpg/png/tif，≤500MB）→ 内存任务队列（线程池，并发 1–2）→ Tiler → 批量推理 → Aggregator → 产物（overlay.png / stats.json / polygons.geojson / report.md）→ 结果页展示 + 下载 → 可选导出低置信窗口复核 JSON。

错误处理：不支持的格式/超大文件返回 4xx 带明确信息；推理异常任务标记 failed 带原因；产物目录按任务 ID 隔离。

## 5. 四天实施节奏

- D1 推理引擎（Tiler/Classifier/Aggregator）+ 单元测试
- D2 导出层（叠加图/GeoJSON/报告）+ 真实样例接入
- D3 FastAPI 服务 + Gradio 界面
- D4 量化 + 基准 + Docker + 架构图/README + 简历条目

纪律：每天 commit、DECISIONS.md 更新、完成发邮件。

## 6. 风险与备选

- AutoAWQ 对 Qwen2.5-VL 支持不确定 → bitsandbytes NF4
- Sentinel-2 免登录源不确定 → 退化方案见上表
- Windows 装 rasterio 受阻 → 退化为 PNG/JPG + 像素坐标 GeoJSON
