#!/bin/bash
# GeoLens 服务器一键流程：zero-shot 基线 + 三组 LoRA 微调 + 三组评测（顺序执行）
# 前置：setup.sh 已完成（SETUP_DONE）、数据已解压到 ~/xiaopeng/geolens/GeoLens
# 用法: nohup bash training/run_all.sh > ~/xiaopeng/logs/run_all.log 2>&1 &
set -e
set -o pipefail
source ~/xiaopeng/env.sh
source ~/xiaopeng/miniconda3/etc/profile.d/conda.sh
conda activate geolens
cd ~/xiaopeng/geolens/GeoLens

echo "=== [0/4] 环境自检 ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
test -f data/instructions_clean.json || { echo "data missing"; exit 1; }

echo "=== [1/4] zero-shot 基线 ==="
python eval/30_zeroshot_eval.py --n 450 --out eval/results_zeroshot.json --tag zeroshot

echo "=== [2/4] LoRA 微调 x3 ==="
for v in clean raw_matched raw_full; do
  echo "--- training $v ---"
  bash training/train_lora.sh $v 2>&1 | tee ~/xiaopeng/logs/train_$v.log
done

echo "=== [3/4] 评测微调模型 x3 ==="
for v in clean raw_matched raw_full; do
  python eval/30_zeroshot_eval.py --n 450 \
    --adapter outputs/lora_$v \
    --out eval/results_lora_$v.json --tag lora_$v
done

echo "=== [4/4] 汇总 ==="
python - <<'EOF'
import json
rows = []
for tag, f in [("zero-shot", "eval/results_zeroshot.json"),
               ("clean", "eval/results_lora_clean.json"),
               ("raw-matched", "eval/results_lora_raw_matched.json"),
               ("raw-full", "eval/results_lora_raw_full.json")]:
    try:
        rows.append((tag, json.load(open(f, encoding="utf-8"))["summary"]["accuracy"]))
    except FileNotFoundError:
        rows.append((tag, "N/A"))
print(f"{'variant':<12} accuracy")
for t, a in rows:
    print(f"{t:<12} {a}")
EOF
echo RUN_ALL_DONE
