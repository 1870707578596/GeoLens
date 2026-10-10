#!/bin/bash
# GeoLens LoRA 微调（在 lab-1 的 geolens 环境中运行）
# 用法: bash train_lora.sh <clean|raw_matched|raw_full>
# 注意: 必须在仓库根目录 ~/xiaopeng/geolens/GeoLens 下执行（数据里是相对图片路径）
set -e
source ~/xiaopeng/env.sh
source ~/xiaopeng/miniconda3/etc/profile.d/conda.sh
conda activate geolens
cd ~/xiaopeng/geolens/GeoLens

VARIANT=$1
case $VARIANT in
  clean)       EPOCHS=3.0 ;;
  raw_matched) EPOCHS=3.0 ;;
  raw_full)    EPOCHS=1.0 ;;
  *) echo "usage: bash train_lora.sh <clean|raw_matched|raw_full>"; exit 1 ;;
esac

llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --stage sft \
  --do_train \
  --dataset geolens_${VARIANT} \
  --dataset_dir training \
  --template qwen2_vl \
  --finetuning_type lora \
  --lora_target all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --output_dir outputs/lora_${VARIANT} \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs ${EPOCHS} \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --bf16 \
  --cutoff_len 2048 \
  --logging_steps 20 \
  --save_steps 500 \
  --save_total_limit 2 \
  --plot_loss \
  --report_to none
