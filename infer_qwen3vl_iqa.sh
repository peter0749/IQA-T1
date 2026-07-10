#!/bin/bash
# ===========================================
# IQA-T1 单图推理脚本
# ===========================================

set -e

# ---- 配置----
# 待评估图片路径
IMAGE_PATH=${IMAGE_PATH:-"/nas/wujinjian103/ThinkVis-IQA/IQA-T1/dataset/KONIQ/koniq/826373.jpg"}
# 模型路径
MODEL_PATH=${MODEL_PATH:-saves/qwen3-vl-4B-Instruct-all-SFT}
# GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
# ---------------------------

# 切换到脚本所在目录（IQA-T1 根）
cd "$(dirname "$0")"

echo "=============================================="
echo "IQA-T1 Inference"
echo "Model       : $MODEL_PATH"
echo "Image       : $IMAGE_PATH"
echo "=============================================="

python inference/infer.py \
    --model_path "$MODEL_PATH" \
    --image_path "$IMAGE_PATH" \
    "$@"
