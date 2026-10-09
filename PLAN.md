# GeoLens —— 遥感垂直域多模态助手（简历项目计划）

> 定位：数据构建 → 清洗 → LoRA 微调 → 评测 → Demo 部署的全流程多模态项目。
> 一句话简历版：用 CLIP 零样本 + 置信度过滤自动构建遥感指令数据（迁移论文 CFLR 方法论），LoRA 微调小参数多模态模型，量化"数据清洗"收益，Gradio 部署遥感影像问答助手。

## 为什么选遥感垂直域

1. **产业真实有岗**：商业航天/遥感解译是大模型落地的真实方向（航天宏图、长光卫星、商汤/阿里遥感线都有 VLM 岗），学生里做的人少，区分度高。
2. **数学背景加成**：遥感与几何、优化天然相关，和"计算数学 + CV"人设一致。
3. **痛点真实可量化**：通用 VLM 对遥感俯拍影像幻觉严重（域差距），zero-shot 基线差 → 微调提升明显，故事完整。
4. **数据全公开**：RESISC45（45 类地物、3.1 万图）、RSCC（约 1.4 万条中文 caption）等，无需爬虫，无版权/内容风险。
5. **Demo 视觉冲击力强**：上传卫星图 → 中文问答/地物识别，面试演示效果远好于"又一个通用 chatbot"。

## 技术方案

```
原始数据（RESISC45 图片 + RSCC 中文描述）
        │
        ▼  CLIP/SigLIP 零样本打标签
多标签候选 + 置信度/熵分数 ──► 置信度过滤（熵加权，复用论文 CFLR 思想）
        │                              │
   保留（清洗版）                  过滤掉（噪声）
        │
        ▼  模板 + 原生描述 → 指令数据（多轮 QA 对）
两份数据集：clean / raw（对照组）
        │
        ▼  LLaMA-Factory + LoRA（Qwen2-VL-2B 级小模型）
两个 checkpoint：clean 微调 vs raw 微调
        │
        ▼  评测：分类准确率 / captioning BLEU·ROUGE / 人工抽检
消融结论：数据清洗带来 X 点提升  ←—— 简历的核心量化结果
        │
        ▼  Gradio Demo：传图 → 中文问答 + 地物高亮
```

## 算力分工

| 环节 | 机器 | 说明 |
|---|---|---|
| 数据下载/打标/清洗 | 本地 3060 Laptop 6G（或 CPU） | CLIP ViT-B/16 fp16 推理约 1G 显存，无压力 |
| LoRA 微调 | 实验室 A40（校园网/VPN） | 2B 模型 LoRA 绰绰有余，LLaMA-Factory + tmux；6G 本地跑不了 QLoRA，训练锁定 A40 |
| 评测 | A40 或本地 | 批量推理在 A40，抽检在本地 |
| Gradio Demo | 本地 | Qwen2-VL-2B int4 量化（AWQ/GPTQ）约 2-3G 显存，6G 可跑；或直接在 A40 起 demo |

## 阶段拆解（总计约 2.5 周，每天 2–4 小时）

**Phase 0｜环境（0.5 天）**
- A40：conda 环境 + LLaMA-Factory + modelscope 下载 Qwen2-VL-2B（国内源快）
- 本地：transformers / open_clip / gradio / pandas 环境
- 产出：两端环境就绪，模型能加载推理

**Phase 1｜数据引擎（2–3 天）★ 项目灵魂**
- 下载 RESISC45、RSCC（HF/百度盘镜像）
- CLIP 零样本 45 类打标 → 每图得标签 + 置信度
- 熵加权 + 阈值过滤（阈值扫一遍看分布，画 P-R 曲线放 README）
- 模板生成 QA 对：识别/计数/对比/描述四类指令
- 产出：`data/clean.json`、`data/raw.json`（各约 1–2 万条）+ 数据构建脚本

**Phase 2｜基线（1 天）**
- Qwen2-VL-2B zero-shot 在测试集上：分类准确率 + 描述质量
- 产出：基线数字（讲"域差距"痛点的证据）

**Phase 3｜微调（3–5 天）**
- LLaMA-Factory LoRA：clean / raw 两组各训一轮
- swanlab/tensorboard 记录，超参写入 README
- 产出：两个 checkpoint + 训练曲线截图

**Phase 4｜评测与消融（1–2 天）**
- 测试集：clean vs raw vs zero-shot 三线对比表
- 人工抽检 50 例记录 badcase
- 产出：结果表（README 首屏）+ 结论

**Phase 5｜Demo 与收尾（1–2 天）**
- Gradio：传图 → 中文问答、地物列表、（可选）CLIP 注意力热图
- GitHub README（问题-方法-结果-复现命令）+ 简历条目定稿
- 产出：可投递状态

## 简历产出目标（写法预演）

**GeoLens：遥感垂直域多模态助手（个人项目）**
- 设计"零样本打标 + 置信度过滤"的自动数据构建流水线：CLIP 对 3.1 万张遥感影像打标，以熵加权过滤（迁移自科研中的标签重建方法）生成 X 万条中文指令数据，人工抽检准确率 X%；
- 基于 LLaMA-Factory 对 Qwen2-VL-2B 进行 LoRA 微调，消融显示清洗数据较原始数据提升 X 点，较 zero-shot 提升 X 点；
- Gradio 部署遥感影像中文问答 Demo，GitHub 开源全流程代码。

## 风险与备选

- 数据集下载受阻 → 备选 AID / NWPU-RESISC45 镜像 / RSCD 系；RSCC 难获取则用 RSICD 英文版
- A40 排期冲突 → 训练顺延（本地 6G 无法兜底 2B 训练）；极端情况换 SmolVLM-500M 级超小模型在本地 QLoRA 验证流水线
- 本地显存不足（3060 Laptop 6G）→ Demo 一律走 int4 量化；打标阶段 fp16 CLIP 仅约 1G 占用，安全

## 目录结构（骨架已建）

```
GeoLens/
├── README.md          # 项目门面（Phase 5 完善）
├── PLAN.md            # 本文件
├── data/              # 数据集与中间产物
├── scripts/           # 打标/清洗/生成指令数据
├── training/          # LLaMA-Factory 配置与启动脚本
├── eval/              # 评测脚本与结果
└── demo/              # Gradio 界面
```
