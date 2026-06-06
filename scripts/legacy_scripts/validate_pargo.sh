#!/usr/bin/env bash
# scripts/run_validate_pargo.sh
# Run patched eval_pargo.py to load DCFormer + ParGo + QWEN from safetensors shards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"

PY_SCRIPT="${ROOT_DIR}/src/eval/eval_pargo.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
  PY_SCRIPT="${ROOT_DIR}/src/eval/inspect_pargo.py"
fi

# Defaults (edit if needed)
CHECKPOINT_DIR="${1:-${ROOT_DIR}/output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180}"
SAFETENSORS_INDEX="${2:-${CHECKPOINT_DIR}/model.safetensors.index.json}"
CONFIG_PATH="${3:-${CHECKPOINT_DIR}/config.json}"
DEVICE="${4:-cuda:0}"

echo "=== run_validate_pargo.sh ==="
echo "ROOT_DIR        = $ROOT_DIR"
echo "PY_SCRIPT       = $PY_SCRIPT"
echo "CHECKPOINT_DIR  = $CHECKPOINT_DIR"
echo "CONFIG_PATH     = $CONFIG_PATH"
echo "SAFETENSORS_IDX = $SAFETENSORS_INDEX"
echo "DEVICE          = $DEVICE"
echo

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] python script not found at: $PY_SCRIPT" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[ERROR] checkpoint dir not found: $CHECKPOINT_DIR" >&2
  exit 3
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config.json not found at: $CONFIG_PATH" >&2
  exit 4
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/run_validate_pargo_$(date +%Y%m%d-%H%M%S).log"
echo "Logfile: $LOGFILE"
echo "Started at $(date)" > "$LOGFILE"

# choose python runner (conda preferred)
if command -v conda >/dev/null 2>&1; then
  PY_CMD=(conda run -n Med3DVLM python)
else
  echo "[WARN] conda not found; ensure correct env is active" | tee -a "$LOGFILE"
  PY_CMD=(python)
fi

# Build command: pass checkpoint-dir, config, device, and pass --qwen-pretrained to force local HF load
CMD=( "${PY_CMD[@]}" "$PY_SCRIPT" --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG_PATH" --device "$DEVICE" )

# pass safetensors-index if file present and script accepts the flag
if [[ -f "$SAFETENSORS_INDEX" ]]; then
  if grep -qE -- '--safetensors-index' "$PY_SCRIPT"; then
    CMD+=( --safetensors-index "$SAFETENSORS_INDEX" )
  else
    echo "[INFO] target script does not accept --safetensors-index; skipping"
  fi
fi

# pass qwen pretrained as the checkpoint dir so from_pretrained tries local files
if grep -qE -- '--qwen-pretrained' "$PY_SCRIPT"; then
  CMD+=( --qwen-pretrained "$CHECKPOINT_DIR" )
elif grep -qE -- '--llm-hf' "$PY_SCRIPT"; then
  CMD+=( --llm-hf "$CHECKPOINT_DIR" )
else
  echo "[WARN] script does not accept --qwen-pretrained/--llm-hf; LLM may not be instantiated automatically"
fi

echo "Running command (logging to $LOGFILE):" | tee -a "$LOGFILE"
printf '%q ' "${CMD[@]}" >> "$LOGFILE"
echo >> "$LOGFILE"
echo "---- Output ----" >> "$LOGFILE"

# Run and tee
"${CMD[@]}" 2>&1 | tee -a "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]:-0}

echo "Finished at $(date) with exit code $EXIT_CODE" | tee -a "$LOGFILE"
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] runner exited non-zero. Inspect $LOGFILE"
else
  echo "[OK] Completed. Check diagnostics: output/pargo_full_load_diagnostics.json and $LOGFILE"
fi

exit $EXIT_CODE
