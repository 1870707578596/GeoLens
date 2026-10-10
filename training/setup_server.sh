#!/bin/bash
# GeoLens 服务器初始化脚本（lab-1，RTX 5090 32G，Blackwell 需 cu128）
# 用法：nohup bash ~/xiaopeng/downloads/setup.sh > ~/xiaopeng/logs/setup.log 2>&1 &
set -x
BASE=$HOME/xiaopeng
source $BASE/env.sh

# 1. 自有 miniconda（不碰共享的 ~/miniconda3）
if [ ! -x $BASE/miniconda3/bin/conda ]; then
  wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $BASE/downloads/miniconda.sh
  bash $BASE/downloads/miniconda.sh -b -p $BASE/miniconda3
fi
source $BASE/miniconda3/etc/profile.d/conda.sh

# 2. 独立环境
conda env list | grep -q "^geolens " || conda create -n geolens python=3.11 -y
conda activate geolens

# 3. cu128 版 torch（5090/Blackwell 必需）+ LLaMA-Factory
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install llamafactory -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import torch; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import llamafactory; print('LLAMAF OK', llamafactory.__version__)"

# 4. 底座模型下载（约 7GB，走 hf-mirror）
hf download Qwen/Qwen2.5-VL-3B-Instruct

echo SETUP_DONE
