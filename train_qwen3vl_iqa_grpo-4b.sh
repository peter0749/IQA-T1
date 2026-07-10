#!/bin/bash
set -x

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# 禁用 torch.compile/dynamo，避免 Triton 编译时找不到 libcuda 的问题
export TORCH_COMPILE_DISABLE=1

# ===========================================
# 调试选项：保存训练过程中的生成结果
# ===========================================
# 设为 1 启用保存，0 或不设置则不保存
export IQA_SAVE_GENERATIONS=${IQA_SAVE_GENERATIONS:-0}
# 保存路径（JSONL格式，每行一条记录）
export IQA_GENERATIONS_SAVE_PATH=${IQA_GENERATIONS_SAVE_PATH:-./generations.jsonl}


# Model path - the 4B SFT-trained model
MODEL_PATH=${MODEL_PATH:-../../../saves/qwen3-vl-4B-Instruct-all-SFT}

# Working directory
cd ./train/grpo/EasyR1_IQA-T1

# Run Agent GRPO (multi-turn IQA with tools)
python3 -m verl.trainer.main \
    config=examples/qwen3_vl_4b_iqa_agent_grpo-v3.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_4b_iqa_agent_grpo-v3\
    trainer.n_gpus_per_node=4 \
    "$@"
