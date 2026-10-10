#!/bin/bash
# Phase 5 服务器侧：合并 weighted LoRA -> 准备示例图 -> 启动 Gradio Demo（公网分享链接）
# 用法: nohup bash demo/run_demo_server.sh > ~/xiaopeng/logs/demo.log 2>&1 &
set -e
source ~/xiaopeng/env.sh
source ~/xiaopeng/miniconda3/etc/profile.d/conda.sh
conda activate geolens
cd ~/xiaopeng/geolens/GeoLens

# 1) 合并 LoRA（若未合并）
if [ ! -f outputs/merged_weighted/config.json ]; then
  echo "=== merging LoRA (weighted) ==="
  llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter_name_or_path outputs/lora_weighted \
    --template qwen2_vl \
    --export_dir outputs/merged_weighted \
    --export_size 2
fi

# 2) gradio 与示例图
pip install -q gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p demo/examples
for c in harbor mountain circular_farmland airplane; do
  f=$(ls data/images/test/$c/*.jpg 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" demo/examples/$c.jpg
done

# 3) 启动 Demo
echo "=== launching demo ==="
python demo/app.py --model outputs/merged_weighted --share
