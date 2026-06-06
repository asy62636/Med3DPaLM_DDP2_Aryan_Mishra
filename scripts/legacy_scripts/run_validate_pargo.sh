# #!/usr/bin/env bash
# # scripts/run_validate_pargo.sh
# #
# # Robust runner for eval_pargo.py / inspect_pargo.py.
# # - Passes --config (required by eval_pargo.py) by default from checkpoint_dir/config.json
# # - Only passes --safetensors-index if the target script accepts that flag
# # - Auto-runs zero_to_fp32.py to create a .pt if index is missing and zero_to_fp32 is present
# # - Uses `conda run -n Med3DVLM python` if `conda` is available, else falls back to `python`
# # - Writes a timestamped log under logs/
# #
# # Usage:
# #   ./scripts/run_validate_pargo.sh [CHECKPOINT_DIR] [SAFETENSORS_INDEX] [CONFIG_PATH] [DEVICE] [QWEN_PRETRAINED] [PT_CHECKPOINT] [SKIP_LLM_FROM_SAFETENSORS] [SAVE_SMALL] [SAVE_SMALL_PATH]
# #
# # All args are optional when sensible defaults exist.

# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"     # repository root (parent of scripts/)
# # Prefer the patched eval_pargo.py, fallback to inspect_pargo.py
# INSPECT_SCRIPT="${ROOT_DIR}/src/eval/eval_pargo.py"
# if [[ ! -f "$INSPECT_SCRIPT" ]]; then
#   INSPECT_SCRIPT="${ROOT_DIR}/src/eval/inspect_pargo.py"
# fi

# # --- Inputs (defaults)
# CHECKPOINT_DIR="${1:-${ROOT_DIR}/output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180}"
# SAFETENSORS_INDEX="${2:-${CHECKPOINT_DIR}/model.safetensors.index.json}"
# # default config path: config.json inside the checkpoint dir
# CONFIG_PATH="${3:-${CHECKPOINT_DIR}/config.json}"
# DEVICE="${4:-cuda:0}"
# QWEN_PRETRAINED="${5:-}"    # HF id or local HF folder (optional)
# PT_CHECKPOINT="${6:-}"      # optional combined .pt (if you prefer)
# SKIP_LLM_FROM_SAFETENSORS="${7:-false}"  # 'true' or 'false'
# SAVE_SMALL="${8:-false}"    # 'true' or 'false'
# SAVE_SMALL_PATH="${9:-}"    # path to save combined checkpoint if SAVE_SMALL=true

# # Print summary
# echo "=== run_validate_pargo.sh ==="
# echo "ROOT_DIR                      = $ROOT_DIR"
# echo "INSPECT_SCRIPT                = $INSPECT_SCRIPT"
# echo "CHECKPOINT_DIR                = $CHECKPOINT_DIR"
# echo "CONFIG_PATH                   = $CONFIG_PATH"
# echo "SAFETENSORS_INDEX             = $SAFETENSORS_INDEX"
# echo "DEVICE                        = $DEVICE"
# echo "QWEN_PRETRAINED               = ${QWEN_PRETRAINED:-<none>}"
# echo "PT_CHECKPOINT                 = ${PT_CHECKPOINT:-<none>}"
# echo "SKIP_LLM_FROM_SAFETENSORS     = $SKIP_LLM_FROM_SAFETENSORS"
# echo "SAVE_SMALL                    = $SAVE_SMALL"
# echo "SAVE_SMALL_PATH               = ${SAVE_SMALL_PATH:-<none>}"
# echo

# # ensure repo root is on PYTHONPATH for src.* imports
# export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# # basic checks
# if [[ ! -d "$CHECKPOINT_DIR" ]]; then
#   echo "[ERROR] checkpoint dir not found: $CHECKPOINT_DIR" >&2
#   exit 2
# fi
# if [[ ! -f "$INSPECT_SCRIPT" ]]; then
#   echo "[ERROR] eval/inspect script not found at: $INSPECT_SCRIPT" >&2
#   exit 3
# fi

# # ensure config exists (eval_pargo requires --config)
# if [[ ! -f "$CONFIG_PATH" ]]; then
#   echo "[ERROR] config not found at: $CONFIG_PATH" >&2
#   echo "Provide config.json as 3rd arg or place config.json inside the checkpoint dir." >&2
#   exit 4
# fi

# # Logging
# TS="$(date +%Y%m%d-%H%M%S)"
# LOG_DIR="${ROOT_DIR}/logs"
# mkdir -p "$LOG_DIR"
# LOGFILE="${LOG_DIR}/eval_pargo_${TS}.log"
# echo "Logging to: $LOGFILE"
# echo "Started at $(date)" > "$LOGFILE"

# # choose python execution method
# if command -v conda >/dev/null 2>&1; then
#   PY_CMD=(conda run -n Med3DVLM python)
# else
#   echo "[WARN] 'conda' not found; using system python. Ensure the correct env is active." | tee -a "$LOGFILE"
#   PY_CMD=(python)
# fi

# # If safetensors index is missing and zero_to_fp32.py exists, optionally create a .pt
# AUTO_PT_CREATED=""
# if [[ ! -f "$SAFETENSORS_INDEX" && -z "${PT_CHECKPOINT:-}" ]]; then
#   echo "[INFO] safetensors index not found at: $SAFETENSORS_INDEX" | tee -a "$LOGFILE"
#   # try checkpoint root then any global_step* subdir
#   Z2F="${CHECKPOINT_DIR}/zero_to_fp32.py"
#   if [[ ! -f "$Z2F" ]]; then
#     GSDIR="$(ls -d "${CHECKPOINT_DIR}"/global_step* 2>/dev/null | head -n1 || true)"
#     if [[ -n "$GSDIR" && -f "${GSDIR}/zero_to_fp32.py" ]]; then
#       Z2F="${GSDIR}/zero_to_fp32.py"
#     fi
#   fi

#   if [[ -f "$Z2F" ]]; then
#     echo "[INFO] Found zero_to_fp32.py at: $Z2F ; attempting to produce a stitched .pt" | tee -a "$LOGFILE"
#     # choose input dir for zero_to_fp32
#     INPUT_DIR="${GSDIR:-$CHECKPOINT_DIR}"
#     OUT_PT="${CHECKPOINT_DIR}/model_fp32_from_zero_to_fp32.pt"
#     echo "[INFO] Running zero_to_fp32.py --input_dir $INPUT_DIR --output_file $OUT_PT" | tee -a "$LOGFILE"
#     "${PY_CMD[@]}" "$Z2F" --input_dir "$INPUT_DIR" --output_file "$OUT_PT" 2>&1 | tee -a "$LOGFILE" || {
#       echo "[WARN] zero_to_fp32.py failed; continuing without pt" | tee -a "$LOGFILE"
#     }
#     if [[ -f "$OUT_PT" ]]; then
#       PT_CHECKPOINT="$OUT_PT"
#       AUTO_PT_CREATED="$OUT_PT"
#       echo "[INFO] Created .pt at: $OUT_PT" | tee -a "$LOGFILE"
#     fi
#   else
#     echo "[INFO] zero_to_fp32.py not found; cannot auto-create .pt" | tee -a "$LOGFILE"
#   fi
# fi

# # Build command robustly: only pass flags the target script supports.
# CMD=( "${PY_CMD[@]}" "$INSPECT_SCRIPT" --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG_PATH" --device "$DEVICE" )

# # only pass --safetensors-index if the target script defines it (grep the script for the arg name)
# if grep -qE -- '--safetensors-index' "$INSPECT_SCRIPT"; then
#   if [[ -f "$SAFETENSORS_INDEX" ]]; then
#     CMD+=( --safetensors-index "$SAFETENSORS_INDEX" )
#   fi
# else
#   echo "[INFO] Target script does not accept --safetensors-index; skipping that argument" | tee -a "$LOGFILE"
# fi

# # pass --pt-checkpoint only if script supports it
# if grep -qE -- '--pt-checkpoint' "$INSPECT_SCRIPT"; then
#   if [[ -n "${PT_CHECKPOINT:-}" ]]; then
#     CMD+=( --pt-checkpoint "$PT_CHECKPOINT" )
#   fi
# else
#   # if script doesn't accept pt-checkpoint, skip it
#   echo "[INFO] Target script does not accept --pt-checkpoint; skipping that argument" | tee -a "$LOGFILE"
# fi

# # qwen-pretrained flag (if supported)
# if grep -qE -- '--qwen-pretrained' "$INSPECT_SCRIPT"; then
#   if [[ -n "${QWEN_PRETRAINED:-}" ]]; then
#     CMD+=( --qwen-pretrained "$QWEN_PRETRAINED" )
#   fi
# else
#   echo "[INFO] Target script does not accept --qwen-pretrained; skipping that argument" | tee -a "$LOGFILE"
# fi

# # skip-llm-from-safetensors flag (if supported)
# if grep -qE -- '--skip-llm-from-safetensors' "$INSPECT_SCRIPT"; then
#   if [[ "${SKIP_LLM_FROM_SAFETENSORS:-false}" == "true" ]]; then
#     CMD+=( --skip-llm-from-safetensors )
#   fi
# fi

# # save-small flags (if supported)
# if grep -qE -- '--save-small' "$INSPECT_SCRIPT"; then
#   if [[ "${SAVE_SMALL:-false}" == "true" ]]; then
#     CMD+=( --save-small )
#     if [[ -n "${SAVE_SMALL_PATH:-}" ]]; then
#       CMD+=( --save-small-path "$SAVE_SMALL_PATH" )
#     fi
#   fi
# fi

# # Show & run
# echo "Running command (logged to $LOGFILE):" | tee -a "$LOGFILE"
# printf '%q ' "${CMD[@]}" >> "$LOGFILE"
# echo >> "$LOGFILE"
# echo "---- Output ----" >> "$LOGFILE"

# # Execute and tee output to logfile
# "${CMD[@]}" 2>&1 | tee -a "$LOGFILE"
# EXIT_CODE=${PIPESTATUS[0]:-0}

# echo "Finished at $(date) with exit code $EXIT_CODE" | tee -a "$LOGFILE"
# if [[ -n "$AUTO_PT_CREATED" ]]; then
#   echo "[NOTE] Auto-created PT at: $AUTO_PT_CREATED" | tee -a "$LOGFILE"
# fi

# exit $EXIT_CODE

#!/usr/bin/env bash
# scripts/run_validate_pargo.sh
# Robust runner that forces loading DCformer + ParGo + QWEN from safetensors (or local checkpoint).
# It detects which CLI flags the python runner supports and passes the checkpoint dir as the LLM source
# so the script will attempt to instantiate & load Qwen into an `llm` object.
#
# Usage: ./scripts/run_validate_pargo.sh
# (You can override paths by editing the top variables.)

set -euo pipefail

# ------------------ Edit only if you moved files ------------------
ROOT_DIR="/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM"
CHECKPOINT_DIR="/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180"
SAFETENSORS_INDEX="${CHECKPOINT_DIR}/model.safetensors.index.json"
CONFIG_PATH="${CHECKPOINT_DIR}/config.json"
DEVICE="${DEVICE:-cuda:0}"
# -----------------------------------------------------------------

# Prefer eval_pargo.py if present, else fallback to inspect_pargo.py
PY_SCRIPT="${ROOT_DIR}/src/eval/eval_pargo.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
  PY_SCRIPT="${ROOT_DIR}/src/eval/inspect_pargo.py"
fi

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/run_validate_pargo_$(date +%Y%m%d-%H%M%S).log"
echo "Logfile: $LOGFILE"
echo "Started at $(date)" > "$LOGFILE"

echo "SUMMARY:" | tee -a "$LOGFILE"
echo " ROOT_DIR        = $ROOT_DIR" | tee -a "$LOGFILE"
echo " PY_SCRIPT       = $PY_SCRIPT" | tee -a "$LOGFILE"
echo " CHECKPOINT_DIR  = $CHECKPOINT_DIR" | tee -a "$LOGFILE"
echo " CONFIG_PATH     = $CONFIG_PATH" | tee -a "$LOGFILE"
echo " SAFETENSORS_IDX = $SAFETENSORS_INDEX" | tee -a "$LOGFILE"
echo " DEVICE          = $DEVICE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# basic existence checks for files we will definitely use
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[ERROR] checkpoint dir not found: $CHECKPOINT_DIR" | tee -a "$LOGFILE"
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] required config.json not found at: $CONFIG_PATH" | tee -a "$LOGFILE"
  exit 3
fi
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] runner script not found: $PY_SCRIPT" | tee -a "$LOGFILE"
  exit 4
fi

# Prepare env
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Choose python runner
if command -v conda >/dev/null 2>&1; then
  PY_CMD=(conda run -n Med3DVLM python)
else
  echo "[WARN] conda not found; ensure correct python env is active" | tee -a "$LOGFILE"
  PY_CMD=(python)
fi

# Build base command (always include --checkpoint-dir and --config and --device)
CMD=( "${PY_CMD[@]}" "$PY_SCRIPT" --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG_PATH" --device "$DEVICE" )

# Helper: add an arg only if PY_SCRIPT supports it (grep the script)
add_if_supported() {
  local flag="$1"
  shift
  if grep -qE -- "$flag" "$PY_SCRIPT"; then
    CMD+=( "$flag" "$@" )
    return 0
  else
    echo "[INFO] target script does not accept $flag; skipping" | tee -a "$LOGFILE"
    return 1
  fi
}

# Try to add safetensors-index if supported
if [[ -f "$SAFETENSORS_INDEX" ]]; then
  grep -qE -- '--safetensors-index' "$PY_SCRIPT" && CMD+=( --safetensors-index "$SAFETENSORS_INDEX" ) || echo "[INFO] --safetensors-index not supported by script" | tee -a "$LOGFILE"
fi

# Try to add pt-checkpoint if supported and present (not required here)
# (we don't have a .pt by default, leave this optional)

# Now ensure the Python runner will attempt to instantiate an LLM:
# common arg names we've seen: --qwen-pretrained, --qwen-pretrained, --llm-hf, --llm_hf
# We'll detect which one the Python script accepts and pass CHECKPOINT_DIR as its value so from_pretrained() will try local files.
LLM_FLAG_CHOSEN=""
if grep -qE -- '--qwen-pretrained' "$PY_SCRIPT"; then
  CMD+=( --qwen-pretrained "$CHECKPOINT_DIR" )
  LLM_FLAG_CHOSEN="--qwen-pretrained"
  echo "[INFO] passing --qwen-pretrained $CHECKPOINT_DIR to force LLM instantiation" | tee -a "$LOGFILE"
elif grep -qE -- '--llm-hf' "$PY_SCRIPT"; then
  CMD+=( --llm-hf "$CHECKPOINT_DIR" )
  LLM_FLAG_CHOSEN="--llm-hf"
  echo "[INFO] passing --llm-hf $CHECKPOINT_DIR to force LLM instantiation" | tee -a "$LOGFILE"
elif grep -qE -- '--llm_hf' "$PY_SCRIPT"; then
  CMD+=( --llm_hf "$CHECKPOINT_DIR" )
  LLM_FLAG_CHOSEN="--llm_hf"
  echo "[INFO] passing --llm_hf $CHECKPOINT_DIR to force LLM instantiation" | tee -a "$LOGFILE"
else
  echo "[WARN] target script does not support a known LLM flag (--qwen-pretrained/--llm-hf). If it can't auto-instantiate LLM from config, LLM weights will be skipped." | tee -a "$LOGFILE"
fi

# If script supports skip-llm-from-safetensors and you want to ensure it DOES attempt LLM load, don't pass skip.
# If script supports save-small, we skip by default (user can edit to enable)

# Show final command in log
echo "Final command (logging):" >> "$LOGFILE"
printf '%q ' "${CMD[@]}" >> "$LOGFILE"
echo >> "$LOGFILE"
echo "---- Begin output ----" >> "$LOGFILE"

# Run and tee
"${CMD[@]}" 2>&1 | tee -a "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]:-0}

echo "---- End output ----" >> "$LOGFILE"
echo "Finished at $(date) with exit code $EXIT_CODE" | tee -a "$LOGFILE"

if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] runner exited non-zero. Inspect $LOGFILE" | tee -a "$LOGFILE"
else
  echo "[OK] Completed. Check diagnostics: output/pargo_full_load_diagnostics.json and $LOGFILE" | tee -a "$LOGFILE"
fi

exit $EXIT_CODE
