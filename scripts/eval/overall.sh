#!/usr/bin/env bash
# run_smoke_cpu.sh
# Quick smoke test: runs the eval_caption_pargo runner with 1-sample CPU test.
set -euo pipefail

REPO_DIR="/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM"
RUNNER="${REPO_DIR}/scripts/eval/eval_caption_pargo.sh"
CHECKPOINT_DIR="${REPO_DIR}/output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180"
QWEN_PRETRAINED=""   # empty on purpose (no local HF folder passed)
CAP_DATA_JSON="${REPO_DIR}/data/M3D_Cap_npy/M3D_Cap_subset.json"
OUT_DIR="./output/eval_caption/smoke_cpu"
DEVICE="cpu"
TEST_SIZE=1
BATCH_SIZE=1

echo "Changing to repo directory: $REPO_DIR"
cd "$REPO_DIR"

# sanity checks
if [[ ! -f "$RUNNER" ]]; then
  echo "[ERROR] runner script not found: $RUNNER" >&2
  exit 2
fi
if [[ ! -d "$(dirname "$CHECKPOINT_DIR")" ]]; then
  echo "[WARN] checkpoint parent directory does not exist: $(dirname "$CHECKPOINT_DIR")" >&2
fi
if [[ ! -f "$CAP_DATA_JSON" ]]; then
  echo "[WARN] cap data json not found at: $CAP_DATA_JSON" >&2
fi

echo "Running smoke test..."
echo

# Run the exact command you requested
./scripts/eval/eval_caption_pargo.sh \
  "$CHECKPOINT_DIR" \
  "$QWEN_PRETRAINED" \
  "$CAP_DATA_JSON" \
  "$OUT_DIR" \
  "$DEVICE" $TEST_SIZE $BATCH_SIZE

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] smoke run exited with code $EXIT_CODE"
else
  echo "[OK] smoke run completed successfully."
fi

exit $EXIT_CODE
