<div align="center">

# [ECCV 2026] IQA-T1: Tool-based Visual Evidence Reasoning for Image Quality Assessment

This is the official repository for IQA-T1.

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/your-paper-id)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 📰 News

- **[2026-06-18]** 🚀 Our paper is accepted by **ECCV 2026**.

---

## 📁 Repository Structure

```
IQA-T1/
├── train/
│   ├── sft/
│   │   └── LlamaFactory_IQA-T1/         # Modified LLaMA-Factory for SFT (tool-augmented IQA)
│   └── grpo/
│       └── EasyR1_IQA-T1/               # Modified EasyR1 for GRPO (multi-turn tool-calling RL)
├── dataset/                             # Training dataset (see Step 1)
│   └── KONIQ/
│       ├── koniq/                        # Original KonIQ-10k images
│       ├── tools/                        # Pre-computed tool visualization images
│       └── metas/                        # Dataset metadata / annotation files
├── inference/
│   └── infer.py                         # Inference script
├── scripts/                             # Shared utility modules
│   ├── tools.py                         # Visual quality analysis tool functions
│   ├── iqa.py                           # IQA core module
│   └── iqa_mm_placeholders.py           # Multimodal placeholder utilities
├── train_qwen3vl_iqa_sft-4b.sh          # SFT training launch script
├── train_qwen3vl_iqa_grpo-4b.sh         # GRPO training launch script
└── README.md
```

---

## 🛠️ Environment Setup

```bash
git clone https://github.com/your-org/IQA-T1.git
cd IQA-T1

conda create -n iqa_t1 python=3.10 -y
conda activate iqa_t1

# Install training frameworks (see details below)
pip install torch torchvision  # match your CUDA version
```

---

## 🎓 Stage 1: Supervised Fine-Tuning (SFT)

### Step 1: Prepare the KonIQ-10k Dataset

Download and place under `dataset/KONIQ/` with the following structure:

```
dataset/KONIQ/
├── koniq/                    # 10,375 original images (*.jpg), e.g. KonIQ-10k dataset
├── tools/                    # Pre-computed tool maps (one folder per image)
│   ├── 10004473376/
│   │   ├── GradientMagnitudeMap.png
│   │   ├── ExtremeLuminanceMap.png
│   │   └── ...               # 15 tool types per image
│   └── ...
└── metas/
    └── koniq_training-multimodal-all.json   # Training data in ShareGPT format
```

The KonIQ-10k images can be obtained from the [official KonIQ-10k website](http://database.mmsp-kn.de/koniq-10k.html).

The pre-computed tool maps (`tools/`) and training data JSON (`metas/`) should be placed under `dataset/KONIQ/` as shown above.

### Step 2: Download the Base Model

Download [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and place it at the repository root:

```
IQA-T1/
└── Qwen3-VL-4B-Instruct/     # Download from HuggingFace
```

Alternatively, you can skip this step and let the training script auto-download the model from HuggingFace using the model ID `Qwen/Qwen3-VL-4B-Instruct`.

### Step 3: Configure the Model Path

Edit `train/sft/LlamaFactory_IQA-T1/configs/qwen3vl_4b_iqa_full_sft.yaml`:

```yaml
# Option A: Use local model (recommended)
model_name_or_path: ../../Qwen3-VL-4B-Instruct

# Option B: Auto-download from HuggingFace
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct

# Option C: Use absolute path
model_name_or_path: /absolute/path/to/IQA-T1/Qwen3-VL-4B-Instruct
```

> **Note**: If using a relative path with `../../`, newer versions of `huggingface_hub` may reject it. In that case, use an absolute path or the HuggingFace model ID directly.

### Step 4: Install SFT Framework Dependencies

```bash
cd train/sft/LlamaFactory_IQA-T1

# LLaMA-Factory requires its src/ on PYTHONPATH. Install in editable mode:
pip install setuptools  # ensure setuptools is available first
cd src
pip install -e .
cd ..

# Install additional requirements
pip install -r requirements/requirements_iqa.txt  # if exists
# Otherwise install common deps:
pip install deepspeed datasets accelerate peft
```

Alternatively, set `PYTHONPATH` manually (no pip install needed):

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

### Step 5: Run SFT Training

From the IQA-T1 root directory:

```bash
cd /path/to/IQA-T1
bash train_qwen3vl_iqa_sft-4b.sh
```

The script will:

- `cd` into `train/sft/LlamaFactory_IQA-T1`
- Launch `llamafactory-cli train` with `configs/qwen3vl_4b_iqa_full_sft.yaml`
- Use DeepSpeed ZeRO-3 across 3 GPUs (configurable via `CUDA_VISIBLE_DEVICES`)
- Save checkpoints to `saves/qwen3-vl-4B-Instruct-all-SFT/`

### Key Config Parameters

| Parameter                     | Value                                          | Notes                                       |
| ----------------------------- | ---------------------------------------------- | ------------------------------------------- |
| `model_name_or_path`          | `Qwen/Qwen3-VL-4B-Instruct`                    | Change to local path if needed              |
| `media_dir`                   | `../../../dataset/KONIQ`                       | Relative to LlamaFactory_IQA-T1 cwd         |
| `output_dir`                  | `../../../saves/...`                           | Checkpoint save location                    |
| `finetuning_type`             | `full`                                         | Full-parameter fine-tuning                  |
| `freeze_vision_tower`         | `true`                                         | Freeze vision encoder to save memory        |
| `deepspeed`                   | `examples/deepspeed/ds_z3_offload_config.json` | ZeRO-3 with offload                         |
| `per_device_train_batch_size` | `1`                                            | Adjust based on GPU memory                  |
| `gradient_accumulation_steps` | `16`                                           | Effective batch size = 1 × 16 × 3 GPUs = 48 |
| `num_train_epochs`            | `2.0`                                          | Number of training epochs                   |
| `cutoff_len`                  | `2560`                                         | Max token length (reduce if OOM)            |

---

## 🎓 Stage 2: GRPO Reinforcement Learning

We employ a modified version of [EasyR1](https://github.com/hiyouga/EasyR1) for GRPO training with multi-turn tool-calling.

1. Set `YOUR_SFT_CHECKPOINT_PATH` and `YOUR_KONIQ_DATA_DIR` in the GRPO config (e.g., `train/grpo/EasyR1_IQA-T1/examples/qwen3_vl_4b_iqa_agent_grpo-v3.yaml`):
   
   ```yaml
   data:
     train_files: ./data/train_koniq_7k_rl.json
     image_dir: YOUR_KONIQ_DATA_DIR
   worker:
     actor:
       model:
         model_path: YOUR_SFT_CHECKPOINT_PATH
   ```

2. Run GRPO training from the IQA-T1 root directory:
   
   ```bash
   bash train_qwen3vl_iqa_grpo-4b.sh
   ```

---

## 📊 Inference

From the IQA-T1 root directory:

```bash
python inference/infer.py --image_path /path/to/your/image.jpg --model_path /path/to/your/model
```

Arguments:

- `--image_path`: Path to the input image
- `--model_path`: Path to the trained model checkpoint (HuggingFace format)
- `--device`: Device to run inference on (default: `cuda` if available)

---

## ⭐ Citation

If you find IQA-T1 useful for your research and applications, please cite using this BibTeX:

```latex
@inproceedings{your2026iqat1,
  title={IQA-T1: Tool-based Visual Evidence Reasoning for Image Quality Assessment},
  author={...},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## 🤝 Acknowledgements

We thank the authors of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [EasyR1](https://github.com/hiyouga/EasyR1), and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for their contributions. Our modified versions (`LlamaFactory_IQA-T1` and `EasyR1_IQA-T1`) extend these frameworks with tool-based multi-turn reasoning capabilities for image quality assessment.
