# 数据下载（Phase 1）

网络环境：国内，走 hf-mirror 镜像（已验证可达）。在 `geolens` 环境中执行。

## 1. RESISC45（45 类地物，约 3.15 万张，~4GB）

来源：`timm/resisc45`（HF 上最标准的版本，3262 次下载）
备选：`blanchon/RESISC45`、`jonathan-roberts1/NWPU-RESISC45`

```bash
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download timm/resisc45 --repo-type dataset --local-dir data/resisc45
```

注意：timm 版是 train/val 划分过的 imagefolder；打标时全量打（gt 取父文件夹名），后续再按 split 评估。

## 2. RSCC（遥感中文 caption，~1.4 万对）

来源：`BiliSakura/RSCC`
备选：`zhao-2025/RSCC`；英文备胎 `arampacha/rsicd`（RSICD，万余对英文描述）

```bash
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BiliSakura/RSCC --repo-type dataset --local-dir data/rscc
```

## 下载后检查

- RESISC45：`data/resisc45` 下应有 45 个类文件夹（可能在 train/val 子目录里），共约 31500 张
- RSCC：应为图片 + json/csv 格式的中文描述对，确认字段结构后写入 `scripts/20_build_instructions.py`
