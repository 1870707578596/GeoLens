# GeoLens：遥感垂直域多模态助手

用"自动数据构建 + 置信度利用 + LoRA 微调"把通用小型多模态模型适配到遥感影像理解域。

**状态**：🔬 Phase 4 实验完成（v1.5 软加权变体进行中）。计划详见 [PLAN.md](./PLAN.md)，技术决策见 [DECISIONS.md](./DECISIONS.md)。

## 它解决什么问题

通用视觉-语言模型对遥感俯拍影像存在严重域差距（本项目实测 zero-shot 仅 24.7%）。GeoLens 用一套零人工标注的数据流水线——**CLIP 零样本打标（精度 58.4%）→ 置信度过滤（τ=0.9 时标签精度 97.4%）→ 中文指令生成**——全自动构建 6 万条遥感训练数据并微调小模型，同时用严格对照实验回答一个关键问题：**置信度信息应该用来"硬过滤"还是"软加权"？**

## 核心结果

遥感地物分类准确率（RESISC45 test split，45 类 × 10 张 = 450 张，中文问答解析评测）：

| 模型 | 准确率 | 训练数据 |
|---|---|---|
| Qwen2.5-VL-3B zero-shot | 24.7% | — |
| + LoRA（clean 硬过滤） | 52.2% | 3019 张高置信图（标签精度 97.4%） |
| + LoRA（raw-matched 等量对照） | 56.2% | 3019 张随机抽样 |
| + LoRA（raw-full 全量） | **60.4%** | 25200 张全量伪标签 |
| + LoRA（weighted 软加权 v1.5） | 实验中 | 全量 + 高置信样本 2 倍权重 |

**消融发现（诚实记录）**：朴素置信度硬过滤（clean）**劣于**等量未过滤对照（52.2% vs 56.2%）。逐类分析显示 clean 模型有 6 个类准确率为 0（CLIP 零样本盲区的长尾类被硬过滤完全剔除），harbor 0.10 vs 0.70、mountain 0.30 vs 0.80——**覆盖损失大于纯度收益**。这与弱监督学习里"软加权优于硬丢弃"的经典结论一致，v1.5 的 weighted 变体（保留全量数据、高置信样本加倍）即为验证该方向而设计。

## 流水线

```
RESISC45 31500 张图 ──► CLIP ViT-B-16 零样本打标（58.4%）
        │                     │
        │        max_prob ≥ 0.9 → 3738 张（标签精度 97.4%）= clean
        │                     └─ 其余 88% = raw（含 42% 正确标签）
        ▼
45 类中文知识库 + 四类指令模板 → 60483 条（raw）/ 7309 条（clean）指令
        ▼
LLaMA-Factory LoRA 微调 Qwen2.5-VL-3B（r=16，RTX 5090）
        ▼
统一评测（zero-shot 与微调用同一把尺子）→ 上表
```

## Demo（待 Phase 5）

上传遥感影像 → 中文问答 / 地物清单。

## 复现

```bash
# 1) 数据流水线（本地，3060 6G 即可）
python scripts/05_extract_parquet.py --parquet-dir data/resisc45/data --out-dir data/images --manifest data/manifest.csv
python scripts/10_clip_label.py --image-dir data/images --output data/labels.jsonl
python scripts/15_filter_analysis.py --labels data/labels.jsonl --decisions-out data/filter_decisions.jsonl --report-out data/filter_report.json
python scripts/20_build_instructions.py --labels data/labels.jsonl --decisions data/filter_decisions.jsonl \
    --out-clean data/instructions_clean.json --out-raw data/instructions_raw.json --out-raw-matched data/instructions_raw_matched.json

# 2) 训练与评测（服务器，见 training/setup_server.sh 初始化）
bash training/run_all.sh   # zero-shot 基线 + 三组训练 + 评测 + 汇总
```

数据来源：[timm/resisc45](https://huggingface.co/datasets/timm/resisc45)（CC BY 4.0）。
