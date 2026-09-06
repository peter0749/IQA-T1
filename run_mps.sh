#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
CKPT="${MODEL_PATH:-$ROOT/../scratch/IQA-T1-checkpoint}"
# also allow checkpoint under ./checkpoints for clones without parent scratch/
if [[ ! -e "$CKPT" && -e "$ROOT/checkpoints/IQA-T1" ]]; then
  CKPT="$ROOT/checkpoints/IQA-T1"
fi
IMG="${IMAGE_PATH:-$ROOT/demo/examples/826373.jpg}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-6}"
TOOL_SAVE_DIR="${TOOL_SAVE_DIR:-$ROOT/scratch/mps_infer_out}"

if [[ ! -x "$PY" ]]; then
  echo "error: missing venv python at $PY" >&2
  echo "Create it with:" >&2
  echo "  /opt/homebrew/bin/python3 -m venv \"$ROOT/.venv\"" >&2
  echo "  \"$ROOT/.venv/bin/pip\" install -r \"$ROOT/requirements-mps.txt\"" >&2
  exit 1
fi

if [[ ! -e "$CKPT" ]]; then
  echo "error: MODEL_PATH not found: $CKPT" >&2
  echo "Download public weights: huggingface-cli download zibuyu-02/IQA-T1 --local-dir <dir>" >&2
  echo "Then: MODEL_PATH=<dir> ./run_mps.sh" >&2
  exit 1
fi

if [[ ! -f "$IMG" ]]; then
  echo "error: IMAGE_PATH not found: $IMG" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$TOOL_SAVE_DIR"
exec "$PY" inference/infer.py \
  --model_path "$CKPT" \
  --image_path "$IMG" \
  --device mps \
  --max_tool_calls "$MAX_TOOL_CALLS" \
  --tool_save_dir "$TOOL_SAVE_DIR" \
  "$@"
