#!/bin/bash
# ===========================================
# Qwen-VL-4B 全参数 SFT 微调训练脚本
# ===========================================

set -e

# 配置
LLAMAFACTORY_DIR="./train/sft/LlamaFactory_IQA-T1"
CONFIG_FILE="configs/qwen3vl_4b_iqa_full_sft.yaml"

# GPU 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=4
MASTER_PORT=29500

# 切换到 LLaMA-Factory 目录
cd "$LLAMAFACTORY_DIR"

# 设置 PYTHONPATH，让 Python 能找到 src/ 下的 llamafactory 包
export PYTHONPATH="src:$PYTHONPATH"

echo "=============================================="
echo "开始训练 Qwen-VL-4B"
echo "配置文件: $CONFIG_FILE"
echo "GPU 数量: $NUM_GPUS"
echo "=============================================="

FORCE_TORCHRUN=1 python -m llamafactory.cli train "$CONFIG_FILE"

echo "=============================================="
echo "训练完成！"
echo "=============================================="
