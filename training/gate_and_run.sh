#!/bin/bash
# 闸门：等待模型下载完成并校验后，发起 run_all 训练流水线
# 用法: nohup bash training/gate_and_run.sh > ~/xiaopeng/logs/gate.log 2>&1 &
source ~/xiaopeng/env.sh

echo "waiting for model download process to finish..."
while pgrep -f "hf download Qwen/Qwen2.5-VL-3B" >/dev/null; do sleep 30; done

SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/ 2>/dev/null | head -1)
NSAFE=$(ls $SNAP*.safetensors 2>/dev/null | wc -l)
if [ -z "$SNAP" ] || [ "$NSAFE" -lt 2 ] || [ ! -f "$SNAP/config.json" ]; then
  echo "MODEL INCOMPLETE (snapshot=$SNAP safetensors=$NSAFE), abort"; exit 1
fi
echo "model ready: $SNAP ($NSAFE shards)"

cd ~/xiaopeng/geolens/GeoLens
test -f data/instructions_clean.json || { echo "data missing"; exit 1; }
bash training/run_all.sh > ~/xiaopeng/logs/run_all.log 2>&1
echo GATE_DONE
