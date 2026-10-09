# GeoLens 技术决策记录

> 规则：每个技术决策一条记录，包含日期、背景、备选方案、决定、状态（采纳 / 试验中 / 废弃）。
> 决策变化时不删除历史，只更新状态并在备注里追加原因。新决策按编号往下排。

## D1 项目路线：多模态微调 + 数据质量（融合自动标注）

- 日期：2026-10-09
- 背景：需要一个与 RLR-VSI 科研互补的简历项目；调研发现多模态岗 JD 高频要求 LLaMA-Factory 等工具链与"数据构建处理经验"，腾讯面经直接考察"图文数据质量参差的处理"
- 备选：纯自动标注工具（数据闭环方向）/ AIGC 垂直域生成应用 / 模型部署与推理优化
- 决定：以多模态大模型 LoRA 微调为主线，把自动标注作为数据构建环节内嵌（CLIP 零样本打标 + 置信度过滤生成指令数据），用"清洗 vs 未清洗"消融量化数据质量收益
- 状态：采纳

## D2 垂直领域：遥感影像理解

- 日期：2026-10-09
- 背景：微调项目需要一个具体垂直域以形成区分度和面试记忆点
- 备选：中国美食识别 / 梗图理解 / 动漫角色 / 文档图表理解
- 决定：遥感。理由：岗位真实（商业航天/遥感解译）、与计算数学背景契合、痛点可量化（通用 VLM 对俯拍影像幻觉严重）、数据全公开零风险、Demo（卫星图中文问答）演示效果好
- 状态：采纳

## D3 底座模型：Qwen2-VL-2B 级小参数多模态模型

- 日期：2026-10-09
- 背景：需要本地可推理、A40 可 LoRA 的开源小模型
- 备选：LLaVA 系 / InternVL2-2B / SmolVLM（超小，兜底用）
- 决定：Qwen2-VL-2B（中文能力强、LLaMA-Factory 原生支持、社区教程多）；具体版本号在 Phase 0 于 A40 上验证后锁定
- 状态：采纳（版本待锁定）

## D4 训练框架：LLaMA-Factory + LoRA

- 日期：2026-10-09
- 背景：JD 点名要求 LLaMA-Factory/Transformers 熟练度，且不想重复造训练轮子
- 备选：手写 transformers Trainer / ms-swift / unsloth
- 决定：LLaMA-Factory 做 LoRA 微调；unsloth 仅作为省显存备选手段
- 状态：采纳

## D5 数据源：timm/resisc45 + BiliSakura/RSCC（经 hf-mirror 下载）

- 日期：2026-10-09
- 背景：需要地物分类图片源和中文 caption 源；本机直连 HF 受限，hf-mirror 已验证可达
- 备选：blanchon/RESISC45、jonathan-roberts1/NWPU-RESISC45（图片源备胎）；zhao-2025/RSCC、arampacha/rsicd（caption 备胎，RSICD 为英文）
- 决定：RESISC45 用 timm 版（下载量最高、最标准），中文描述用 BiliSakura/RSCC
- 状态：试验中（下载进行中，落盘结构确认后转采纳）

## D6 数据清洗策略：熵加权 + 置信度阈值（迁移论文 CFLR 思想）

- 日期：2026-10-09
- 背景：项目核心卖点是数据质量方法论；论文 RLR-VSI 的置信度过滤标签重建（熵权重 + 阈值 τ）可直接迁移
- 备选：仅阈值过滤 / 仅熵加权 / 不过滤（作为对照组保留）
- 决定：打标输出每图 max_prob 与熵，Phase 1 扫阈值画分布与 P-R 曲线后定参；raw（未清洗）数据保留作对照
- 状态：试验中（阈值待定）

## D7 本地环境：新建 geolens conda 环境（Python 3.11）

- 日期：2026-10-09
- 背景：本机已有环境均不可用——base 为 Python 3.12 无 torch；CLIP 环境为 Python 3.8 + torch 1.7.1（2020 年版本，open_clip 要求更高）；SARLP 环境在服务器不在本机
- 备选：复用 CLIP 环境 / 在 base 上装
- 决定：新建 geolens 环境，torch cu121 + open_clip_torch + pandas + gradio + huggingface_hub
- 状态：采纳

## D8 算力分工：本地 3060 Laptop 6G 做数据与 Demo，A40 做训练

- 日期：2026-10-09
- 背景：实测本地为 3060 Laptop 6GB 显存
- 备选：本地 QLoRA 兜底训练（已废弃——6GB 跑不动 2B QLoRA）；极端情况可用 SmolVLM-500M 级在本地验证流水线
- 决定：打标/清洗/Demo 在本地（CLIP fp16 约 1G 显存；Demo 用 int4 量化约 2-3G），LoRA 微调锁定实验室 A40（校园网连接）
- 状态：采纳
