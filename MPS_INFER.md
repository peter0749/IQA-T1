# IQA-T1 MPS inference (Apple Silicon)

This fork of [zibuyu-02/IQA-T1](https://github.com/zibuyu-02/IQA-T1) adds a minimal MPS path:

- `attn_implementation="sdpa"` when `device != cuda`
- `model.to(device)` when `device_map` is unset
- drop `fix_mistral_regex` for transformers 5.x

Proven smoke on M2 Ultra: **Score 4.22** on `demo/examples/826373.jpg` with `MAX_TOOL_CALLS=6`.

## Setup

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-mps.txt
huggingface-cli download zibuyu-02/IQA-T1 --local-dir ../scratch/IQA-T1-checkpoint
# or: --local-dir ./checkpoints/IQA-T1
```

## Run

```bash
./run_mps.sh
# MODEL_PATH=/path/to/checkpoint IMAGE_PATH=/path/to.jpg ./run_mps.sh
```

Training / dataset / Q-Tool are out of scope for this fork path.
