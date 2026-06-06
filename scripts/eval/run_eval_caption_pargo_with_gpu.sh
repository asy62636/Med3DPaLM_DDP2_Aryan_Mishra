#!/usr/bin/env bash
# scripts/eval/run_eval_caption_pargo_safe.sh
# Safe wrapper for src/eval/eval_caption_pargo.py
# - ensures PYTHONPATH includes repo root so `import src.*` works
# - detects whether --data_root is supported and only passes it if allowed
# - streams logs to console and saves them to logs/
# Usage:
#   ./scripts/eval/run_eval_caption_pargo_safe.sh [CHECKPOINT_DIR] [QWEN_PRETRAINED] [CAP_DATA_PATH] [OUT_DIR] [CUDA_DEVICES] [TEST_SIZE] [BATCH_SIZE] [MAX_NEW_TOKENS] [DATA_ROOT]
#
# Example (from repo root):
#   ./scripts/eval/run_eval_caption_pargo_safe.sh \
#     ./output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180 \
#     Qwen/Qwen2.5-7B-Instruct \
#     data/M3D_Cap_npy/M3D_Cap_subset.json \
#     output/eval_caption/one_shot \
#     "0" 1 1 64 data

set -euo pipefail

# ---------- CONFIG / DEFAULTS ----------
REPO_ROOT="$(cd "$(dirname "$0")/../.."; pwd)"   # resolves scripts/eval -> repo root
SCRIPT_PATH="${REPO_ROOT}/src/eval/eval_caption_pargo.py"

CHECKPOINT_DIR="${1:-${REPO_ROOT}/output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180}"
QWEN_PRETRAINED="${2:-Qwen/Qwen2.5-7B-Instruct}"
CAP_DATA_PATH="${3:-${REPO_ROOT}/data/M3D_Cap_npy/M3D_Cap_subset.json}"
OUT_DIR="${4:-${REPO_ROOT}/output/eval_caption/safe_run}"
CUDA_DEVICES="${5:-0}"               # e.g. "0" or "0,1"
TEST_SIZE="${6:-1}"                  # default 1 for a quick end-to-end check
BATCH_SIZE="${7:-1}"
MAX_NEW_TOKENS="${8:-64}"
DATA_ROOT="${9:-${REPO_ROOT}/data}"  # optional; passed only if the script accepts it

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR" "$OUT_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOGFILE="${LOG_DIR}/eval_caption_pargo_safe_${TIMESTAMP}.log"

# ---------- sanity checks ----------
echo "=== run_eval_caption_pargo_safe.sh ==="
echo "REPO_ROOT       = $REPO_ROOT"
echo "SCRIPT_PATH     = $SCRIPT_PATH"
echo "CHECKPOINT_DIR  = $CHECKPOINT_DIR"
echo "QWEN_PRETRAINED = $QWEN_PRETRAINED"
echo "CAP_DATA_PATH   = $CAP_DATA_PATH"
echo "OUT_DIR         = $OUT_DIR"
echo "CUDA_DEVICES    = $CUDA_DEVICES"
echo "TEST_SIZE       = $TEST_SIZE"
echo "BATCH_SIZE      = $BATCH_SIZE"
echo "MAX_NEW_TOKENS  = $MAX_NEW_TOKENS"
echo "DATA_ROOT       = $DATA_ROOT"
echo "LOGFILE         = $LOGFILE"
echo

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] Python script not found at: $SCRIPT_PATH" >&2
  exit 2
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[ERROR] checkpoint dir not found: $CHECKPOINT_DIR" >&2
  exit 3
fi

# ---------- choose python runner (prefer conda env) ----------
MED3DVLM_ENV="${MED3DVLM_ENV:-Med3DVLM}"
if command -v conda >/dev/null 2>&1 && conda info --envs | awk '{print $1}' | grep -qx "$MED3DVLM_ENV"; then
  PY_BIN=(conda run -n "$MED3DVLM_ENV" python)
  echo "[INFO] using conda run -n $MED3DVLM_ENV python"
else
  PY_BIN=(python3)
  echo "[INFO] using system python3 (or active venv)"
fi

# ---------- ensure imports from repo root work ----------
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
echo "[INFO] PYTHONPATH set to: $PYTHONPATH"

# ---------- set GPUs ----------
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------- build base args ----------
ARGS=(
  "--checkpoint-dir" "$CHECKPOINT_DIR"
  "--qwen-pretrained" "$QWEN_PRETRAINED"
  "--device" "cuda"
  "--cap_data_path" "$CAP_DATA_PATH"
  "--output_dir" "$OUT_DIR"
  "--test_size" "$TEST_SIZE"
  "--batch_size" "$BATCH_SIZE"
  "--max_new_tokens" "$MAX_NEW_TOKENS"
  "--proj_out_num" "304"
)

# ---------- detect whether the script accepts --data_root ----------
# Use --help output to check if --data_root is a valid option; tolerant to failures.
ALLOW_DATA_ROOT="no"
HELP_OUT="$("${PY_BIN[@]}" "$SCRIPT_PATH" --help 2>&1 || true)"
if echo "$HELP_OUT" | grep -q -- "--data_root"; then
  ALLOW_DATA_ROOT="yes"
  echo "[INFO] eval script accepts --data_root -> will pass it"
else
  echo "[INFO] eval script does NOT accept --data_root -> not passing it"
fi

if [[ "$ALLOW_DATA_ROOT" == "yes" ]]; then
  ARGS+=( "--data_root" "$DATA_ROOT" )
fi

# if safetensors index exists, pass it (safe)
SAFETENSORS_INDEX="${CHECKPOINT_DIR}/model.safetensors.index.json"
if [[ -f "$SAFETENSORS_INDEX" ]]; then
  ARGS+=( "--safetensors-index" "$SAFETENSORS_INDEX" )
fi

# ---------- print final command for debugging ----------
echo "[CMD] ${PY_BIN[*]} $SCRIPT_PATH ${ARGS[*]}"
echo

# ---------- run and stream output to logfile and console ----------
echo "Started at $(date)" | tee "$LOGFILE"
echo "Running:" >> "$LOGFILE"
printf '%q ' "${PY_BIN[@]}" >> "$LOGFILE"
echo " $SCRIPT_PATH ${ARGS[*]}" >> "$LOGFILE"
echo "---- Output ----" >> "$LOGFILE"

# Run the script (stdout+stderr both tee'd to logfile and console)
# Use exec so that signals propagate to child process properly
"${PY_BIN[@]}" "$SCRIPT_PATH" "${ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]:-0}

echo "Finished at $(date) with exit code $EXIT_CODE" | tee -a "$LOGFILE"

# ---------- show brief diagnostics if produced ----------
DIAG="${OUT_DIR}/pargo_full_load_diagnostics.json"
if [[ -f "$DIAG" ]]; then
  echo
  echo "Diagnostics saved to: $DIAG"
  echo "Preview (first 200 lines):"
  sed -n '1,200p' "$DIAG" || true
else
  echo
  echo "No diagnostics file found at $DIAG"
fi

echo
echo "------ Log tail (last 200 lines) ------"
tail -n 200 "$LOGFILE" || true

exit $EXIT_CODE
