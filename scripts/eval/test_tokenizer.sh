#!/usr/bin/env bash
# make_qwen_local_tokenizer_precise.sh
# Deterministically extend Qwen tokenizer vocabulary one token at a time
# until its vocab (len(tok.get_vocab())) equals TARGET_VOCAB.
#
# Safe but slower than adding in bulk; suitable for one-off setup.

set -euo pipefail

HF_ID="Qwen/Qwen2.5-7B-Instruct"
TARGET_VOCAB=151666
OUT_DIR="./qwen_local_tokenizer_151666_precise"
MAX_ADDITION_ATTEMPTS=2000   # safety cap; will abort if we need too many iterations

echo "[INFO] Will load tokenizer: $HF_ID"
echo "[INFO] Target vocab size: $TARGET_VOCAB"
echo "[INFO] Output folder: $OUT_DIR"
echo

python3 - <<PY
from transformers import AutoTokenizer
import os, sys, time

hf_id = "${HF_ID}"
target = ${TARGET_VOCAB}
out_dir = "${OUT_DIR}"
max_attempts = ${MAX_ADDITION_ATTEMPTS}

print("Loading tokenizer:", hf_id)
tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)

# Use len(tok.get_vocab()) as canonical current vocab size
try:
    cur = len(tok.get_vocab())
except Exception:
    # Fallback
    try:
        cur = tok.vocab_size
    except Exception:
        cur = len(tok)

print("Current vocab size (len(get_vocab())):", cur)
if cur == target:
    print("Already at target. Saving to", out_dir)
    tok.save_pretrained(out_dir)
    sys.exit(0)
elif cur > target:
    print(f"Current vocab ({cur}) is larger than target ({target}). Aborting.")
    sys.exit(2)

need = target - cur
print("Need to add", need, "tokens (will add one-by-one).")

# create out dir
os.makedirs(out_dir, exist_ok=True)

added = 0
attempt = 0
# Generate deterministic dummy token names using added index
while cur < target and attempt < max_attempts:
    attempt += 1
    # Use a token name unlikely to collide
    token_name = f"<_added_tok_{cur+added:06d}>"
    # add single token
    n_added = tok.add_tokens([token_name])
    # n_added is number of tokens actually added (usually 1 or 0 if collision)
    if n_added > 0:
        added += n_added
    # recompute canonical vocab size
    try:
        new_cur = len(tok.get_vocab())
    except Exception:
        try:
            new_cur = tok.vocab_size
        except Exception:
            new_cur = len(tok)
    print(f"[iter {attempt}] attempted token '{token_name}' -> n_added={n_added}; vocab {cur} -> {new_cur}")
    cur = new_cur
    # small sleep to avoid hogging (not strictly necessary)
    time.sleep(0.01)

if cur == target:
    print("Reached target vocab size:", cur)
    tok.save_pretrained(out_dir)
    print("Saved extended tokenizer to:", out_dir)
    sys.exit(0)
else:
    print("Failed to reach target within attempts limit.", "current:", cur, "target:", target)
    print("You can re-run this script or increase MAX_ADDITION_ATTEMPTS.")
    sys.exit(3)
PY
