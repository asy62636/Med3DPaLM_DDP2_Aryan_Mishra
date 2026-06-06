#!/usr/bin/env bash
# scripts/eval/patch_eval_loader.sh
# Patch eval_caption_pargo.py to add fallback for 1-D bias/decoder mismatch handling

set -euo pipefail

REPO_ROOT="/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM"
TARGET_FILE="${REPO_ROOT}/src/eval/eval_caption_pargo.py"
BACKUP_FILE="${TARGET_FILE}.bak_$(date +%Y%m%d-%H%M%S)"

cd "$REPO_ROOT"

if [[ ! -f "$TARGET_FILE" ]]; then
  echo "[ERROR] Target file not found: $TARGET_FILE" >&2
  exit 1
fi

echo "[STEP] Backing up $TARGET_FILE -> $BACKUP_FILE"
cp -v "$TARGET_FILE" "$BACKUP_FILE"

echo "[STEP] Applying fallback patch..."
python3 - <<'PY'
from pathlib import Path
p = Path("src/eval/eval_caption_pargo.py")
s = p.read_text()
old = 'res = module.load_state_dict(diag["new_state"], strict=False)'
if old not in s:
    print("ERROR: expected pattern not found. Aborting. Please paste the file header here if it changed.")
else:
    new = r'''
try:
    res = module.load_state_dict(diag["new_state"], strict=False)
except Exception as _e:
    # Fallback: attempt to fix 1-D bias/decoder mismatches by trimming/padding to expected shape and retry.
    import torch
    fixed_state = {}
    model_state = module.state_dict()
    for k, v in diag.get("new_state", {}).items():
        if isinstance(v, torch.Tensor) and k in model_state:
            exp_shape = tuple(model_state[k].shape)
            # handle 1-D biases explicitly
            if len(exp_shape) == 1:
                cur_n = v.numel()
                exp_n = exp_shape[0]
                if cur_n == exp_n:
                    fixed_state[k] = v.to(model_state[k].dtype)
                elif cur_n > exp_n:
                    fixed_state[k] = v[:exp_n].to(model_state[k].dtype).clone()
                else:
                    # pad with zeros
                    extra = torch.zeros((exp_n - cur_n,), dtype=model_state[k].dtype)
                    fixed_state[k] = torch.cat([v.to(model_state[k].dtype), extra], dim=0)
            else:
                # non-1D tensors: keep as-is (loader earlier handles many cases)
                fixed_state[k] = v
        else:
            fixed_state[k] = v
    try:
        res = module.load_state_dict(fixed_state, strict=False)
    except Exception as e2:
        # If still failing, re-raise original error to preserve trace
        raise _e
'''
    s = s.replace(old, new)
    p.write_text(s)
    print(f"[OK] Inserted fallback loader try/except into {p}")
PY

echo "[DONE] Patch applied to $TARGET_FILE"
echo "[INFO] Original file backed up at $BACKUP_FILE"
