#!/bin/bash
# ===========================================
# Qwen3-VL-2B IQA GRPO Training Script
# Image Quality Assessment with GRPO
# ===========================================

set -x

# GPU configuration
export CUDA_VISIBLE_DEVICES=3,5,6,7

# Model path - the SFT-trained model
MODEL_PATH=YOUR_SFT_CHECKPOINT_PATH

# Working directory
cd ./train/grpo/EasyR1_IQA-T1

# Run GRPO training
python3 -m verl.trainer.main \
    config=examples/qwen3_vl_2b_iqa_grpo.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_2b_iqa_grpo \
    trainer.n_gpus_per_node=4
