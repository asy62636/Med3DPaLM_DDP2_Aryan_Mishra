#!/usr/bin/env bash
# scripts/eval/run_eval_caption_pargo_with_dataroot.sh
# Wrapper that runs src/eval/eval_caption_pargo.py while forcing --data_root via sys.argv
# Use when the target script does not accept --data_root but downstream code expects args.data_root.

set -euo pipefail

ROOT_DIR="$(pwd)"   # run from repo root
SCRIPT_PATH="src/eval/eval_caption_pargo.py"

# Inputs (defaults)
CHECKPOINT_DIR="${1:-${ROOT_DIR}/output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180}"
QWEN_PRETRAINED="${2:-Qwen/Qwen2.5-7B-Instruct}"   # HF id or local tokenizer folder
CAP_DATA_PATH="${3:-${ROOT_DIR}/data/M3D_Cap_npy/M3D_Cap_subset.json}"
OUT_DIR="${4:-${ROOT_DIR}/output/eval_caption/cpu_diag_run2}"
DEVICE="${5:-cpu}"
TEST_SIZE="${6:-10}"
BATCH_SIZE="${7:-1}"
MAX_NEW_TOKENS="${8:-64}"
DATA_ROOT="${9:-${ROOT_DIR}/data}"

SAFETENSORS_INDEX="${CHECKPOINT_DIR}/model.safetensors.index.json"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR" "$OUT_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOGFILE="${LOG_DIR}/eval_caption_pargo_dataroot_${TIMESTAMP}.log"

echo "=== run_eval_caption_pargo_with_dataroot.sh ==="
echo "ROOT_DIR        = $ROOT_DIR"
echo "SCRIPT_PATH     = $SCRIPT_PATH"
echo "CHECKPOINT_DIR  = $CHECKPOINT_DIR"
echo "QWEN_PRETRAINED = $QWEN_PRETRAINED"
echo "CAP_DATA_PATH   = $CAP_DATA_PATH"
echo "OUT_DIR         = $OUT_DIR"
echo "DEVICE          = $DEVICE"
echo "TEST_SIZE       = $TEST_SIZE"
echo "BATCH_SIZE      = $BATCH_SIZE"
echo "MAX_NEW_TOKENS  = $MAX_NEW_TOKENS"
echo "DATA_ROOT       = $DATA_ROOT"
echo "SAFETENSORS_IDX = $SAFETENSORS_INDEX"
echo "LOGFILE         = $LOGFILE"
echo

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] script not found at: $SCRIPT_PATH (run this from repo root)" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[ERROR] checkpoint dir not found: $CHECKPOINT_DIR" >&2
  exit 3
fi

# choose python runner (prefer conda env)
MED3DVLM_ENV="${MED3DVLM_ENV:-Med3DVLM}"
if command -v conda >/dev/null 2>&1 && conda info --envs | awk '{print $1}' | grep -qx "$MED3DVLM_ENV"; then
  PY_BIN=(conda run -n "$MED3DVLM_ENV" python)
  echo "[INFO] using conda run -n $MED3DVLM_ENV python"
else
  PY_BIN=(python3)
  echo "[INFO] using system python3 (or active env)"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Build the argv list to hand to the script (explicit, no accidental empty args)
# We'll pass all known flags accepted by your script, then append --data_root
ARGS=(
  "--checkpoint-dir" "$CHECKPOINT_DIR"
  "--device" "$DEVICE"
  "--cap_data_path" "$CAP_DATA_PATH"
  "--output_dir" "$OUT_DIR"
  "--test_size" "$TEST_SIZE"
  "--batch_size" "$BATCH_SIZE"
  "--max_new_tokens" "$MAX_NEW_TOKENS"
  "--proj_out_num" "304"
)

if [[ -n "${QWEN_PRETRAINED}" ]]; then
  ARGS+=( "--qwen-pretrained" "$QWEN_PRETRAINED" )
fi

if [[ -f "$SAFETENSORS_INDEX" ]]; then
  ARGS+=( "--safetensors-index" "$SAFETENSORS_INDEX" )
fi

# Append the data_root in a way the script will receive it (we will set sys.argv to include this)
ARGS+=( "--data_root" "$DATA_ROOT" )

# Build Python invoker that sets sys.argv and runs the script via runpy.run_path
PYCODE=$(cat <<PYCODE
import sys, runpy, pathlib
# Construct argv: first entry should be the script path (as called)
script = "${SCRIPT_PATH}"
argv = [script] + ${ARGS!s}
print("[py-run] sys.argv will be:", argv)
sys.argv = argv
# Ensure working dir is repo root
pathlib.Path(".").resolve()
runpy.run_path(script, run_name="__main__")
PYCODE
)

# Run and log
echo "Started at $(date)" | tee "$LOGFILE"
echo "Running:" >> "$LOGFILE"
printf '%q ' "${PY_BIN[@]}" >> "$LOGFILE"
echo " -c 'runpy script wrapper'" >> "$LOGFILE"
echo "---- Output ----" >> "$LOGFILE"

# execute
# Using - <<'PY' so the inner code is passed exactly; pipe through tee to the logfile
"${PY_BIN[@]}" - <<PY 2>&1 | tee -a "$LOGFILE"
$PYCODE
PY

EXIT_CODE=${PIPESTATUS[0]:-0}
echo "Finished at $(date) with exit code $EXIT_CODE" | tee -a "$LOGFILE"

echo
echo "------ Log tail (last 200 lines) ------"
tail -n 200 "$LOGFILE" || true

DIAG="${OUT_DIR}/pargo_full_load_diagnostics.json"
if [[ -f "$DIAG" ]]; then
  echo
  echo "Diagnostics saved to: $DIAG"
  echo "Preview (first 200 lines):"
  sed -n '1,200p' "$DIAG" || true
fi

exit $EXIT_CODE
