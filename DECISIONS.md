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

## D3 底座模型：Qwen2.5-VL-3B-Instruct

- 日期：2026-10-09（2026-10-10 锁定版本）
- 背景：需要本地可推理（3060 Laptop 6G，int4）、服务器可 LoRA（32G）的开源小模型
- 备选：Qwen2-VL-2B（上一代，更小）/ Qwen2.5-VL-7B（本地 Demo 吃紧）/ LLaVA 系 / InternVL / SmolVLM
- 决定：**Qwen2.5-VL-3B-Instruct**——hf-mirror 已验证可得；较 Qwen2-VL 生成更新、中文与 OCR 更强；3.7B 参数 LoRA 在 32G 显存轻松训练，int4 量化约 2.5-3G 可进本地 6G 显存
- 状态：采纳

## D4 训练框架：LLaMA-Factory + LoRA

- 日期：2026-10-09
- 背景：JD 点名要求 LLaMA-Factory/Transformers 熟练度，且不想重复造训练轮子
- 备选：手写 transformers Trainer / ms-swift / unsloth
- 决定：LLaMA-Factory 做 LoRA 微调；unsloth 仅作为省显存备选手段
- 状态：采纳

## D5 数据源：timm/resisc45（采纳）；BiliSakura/RSCC 判定为同名不同物（废弃）

- 日期：2026-10-09（2026-10-09 修订）
- 背景：需要地物分类图片源和中文 caption 源；本机直连 HF 受限，hf-mirror 已验证可达
- 备选：blanchon/RESISC45、jonathan-roberts1/NWPU-RESISC45（图片源备胎）；zhao-2025/RSCC、arampacha/rsicd（caption 备胎，RSICD 为英文）
- 决定：RESISC45 用 timm 版——已下载（parquet 格式，31500 张），经 `05_extract_parquet.py` 解包为 imagefolder 并生成 manifest，**转采纳**
- 修订：下载后发现 BiliSakura/RSCC（及 zhao-2025/RSCC 镜像）实为 2025 年 xBD 灾前/灾后**变化描述**数据集（英文 change captioning，license 限制仅限科研），并非预期的遥感中文图注数据集，**废弃**作为图注源；中文描述类指令改由标签模板生成（流水线自洽），英文图注备选 arampacha/rsicd；已下载的 14GB 数据暂时保留在本地（data/rscc，未入 git），未来如做 v2"灾害变化描述"方向可复用
- 状态：图片源采纳 / RSCC(变化描述版) 废弃 / 图注源由模板替代

## D6 数据清洗策略：熵加权 + 置信度阈值（迁移论文 CFLR 思想）

- 日期：2026-10-09（同日定参）
- 背景：项目核心卖点是数据质量方法论；论文 RLR-VSI 的置信度过滤标签重建（熵权重 + 阈值 τ）可直接迁移
- 备选：仅阈值过滤 / 仅熵加权 / 不过滤（作为对照组保留）
- 决定：打标输出每图 max_prob 与熵，Phase 1 扫阈值画分布与 P-R 曲线后定参；raw（未清洗）数据保留作对照
- 定参结果（全量 31500 张）：CLIP 零样本基线精度 **58.4%**；**max_prob ≥ 0.90** 时保留 11.9%（3738 张）、精度 **97.4%**；熵分位 q=0.1 达 98.0% 精度但保留率同为 ~10%，与 max_prob 高度冗余，主指标取 max_prob、熵作辅助特征
- 状态：采纳（τ=0.90）

## D7 本地环境：新建 geolens conda 环境（Python 3.11）

- 日期：2026-10-09
- 背景：本机已有环境均不可用——base 为 Python 3.12 无 torch；CLIP 环境为 Python 3.8 + torch 1.7.1（2020 年版本，open_clip 要求更高）；SARLP 环境在服务器不在本机
- 备选：复用 CLIP 环境 / 在 base 上装
- 决定：新建 geolens 环境，torch cu121 + open_clip_torch + pandas + gradio + huggingface_hub
- 状态：采纳

## D8 算力分工：本地 3060 Laptop 6G 做数据与 Demo，实验室服务器做训练

- 日期：2026-10-09（2026-10-10 修订：实际 GPU 为 RTX 5090 32G）
- 背景：实测本地为 3060 Laptop 6GB 显存；2026-10-10 接入实验室服务器 lab-1（SSH 别名 `lab-1`，私钥免密，无需校园内网），实际为 **RTX 5090 32GB（Blackwell，sm_120）**，32 核 62G 内存，GPU 空闲
- 备选：本地 QLoRA 兜底训练（已废弃——6GB 跑不动）；极端情况可用 SmolVLM-500M 级在本地验证流水线
- 决定：打标/清洗/Demo 在本地（CLIP fp16 约 1G 显存；Demo 用 int4 量化约 2-3G），LoRA 微调上 lab-1；**5090 必须用 cu128 版 PyTorch**（cu121/cu124 不支持 sm_120）；服务器文件一律放 `~/xiaopeng/` 总文件夹下，自带独立 miniconda（不碰共享 `~/miniconda3`）
- 状态：采纳

## D9 CLIP 打标配置：ViT-B-16-quickgelu + "a centered satellite photo of {}."

- 日期：2026-10-09
- 背景：smoke 测试发现 open_clip 3.x 下 `ViT-B-16`+openai 权重存在 QuickGELU 激活不匹配警告，且 prompt 带冠词错误；在完整 validation 集（6300 张、45 类）上做了两组正式对照
- 备选：plain ViT-B-16 + "a satellite photo of a {}."（57.0%）；多模板 prompt ensemble（未实施，留作改进项）
- 决定：采用 quickgelu 变体 + CLIP 官方 RESISC45 模板，零样本精度 **58.3%**（此即过滤分析的原始标签精度基线）
- 状态：采纳

## D10 消融实验设计：三组对照（clean / raw-matched / raw-full）

- 日期：2026-10-09
- 背景：clean（3019 张训练图、7309 条指令）与 raw（25200 张、60483 条）数据量悬殊，直接对比会把"数据质量"与"数据数量"两个变量混在一起；另发现 CLIP 对 chaparral（0.4%）、terrace（0.7%）、sparse_residential（2.1%）等长尾类几乎无法识别，这些类永远进不了 clean 集
- 备选：只做 clean vs raw 两对比 / 按预测类做类自适应阈值保覆盖（未实施，列为改进项）
- 决定：三组对照——**clean**（置信度过滤）vs **raw-matched**（等量随机抽样，隔离数量变量，主对比）vs **raw-full**（全量）；README 中诚实讨论"置信度过滤牺牲长尾类覆盖"的代价及缓解思路
- 状态：采纳

## D11 消融结论：硬过滤劣于不过滤；v1.5 转向软加权

- 日期：2026-10-10
- 背景：Phase 4 四组评测完成：zero-shot 24.7% / clean 52.2% / raw-matched 56.2% / raw-full 60.4%
- 结果分析：clean **输给**等量对照 4 个点。逐类归因：clean 模型 6 个类准确率为 0（CLIP 零样本盲区长尾类被硬过滤完全剔除），harbor 0.10 vs 0.70、mountain 0.30 vs 0.80——**覆盖损失 > 纯度收益**，与 D10 预判一致
- 决定：v1.5 增加 **weighted** 变体：`instructions_weighted = raw_full + clean 副本`（LLaMA-Factory 0.9.5 无按样本权重列支持，用样本复制等效实现 2× 加权），保留全量覆盖、对高置信样本加权——对齐论文 RLR-VSI 中 CFLR"熵加权软标签"思想。若 weighted ≥ raw-full 则验证"置信度应加权而非丢弃"
- 状态：试验中（weighted 训练评测进行中）
