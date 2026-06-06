"""
Merge LoRA + Save Complete Model — Experiment 2: Single-Scale ParGo
====================================================================
Stage 3 training saves checkpoints in PEFT format:
  - adapter_model.safetensors  → LoRA delta weights only
  - global_step*/              → ZeRO-sharded non-LoRA trainable weights
                                 (projector + vision tower)

This script:
  [1] Runs zero_to_fp32.py to reconstruct all trainable weights from ZeRO shards
  [2] Loads base Qwen2.5-7B-Instruct (frozen LLM backbone)
  [3] Initializes SingleScaleParGo projector + DCFormer vision tower
  [4] Wraps model in PEFT with same LoRA config as training
  [5] Loads the zero_to_fp32 weights into the PEFT model
  [6] Merges LoRA into the LLM backbone and unloads adapters
  [7] Saves encoder + projector + merged LLM as a single loadable model

Output: ~30 GB in safetensor shards at exp2-single-pargo-stage3-complete/
"""

import os
import sys
import glob
import subprocess
import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers import AutoTokenizer
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel

from src.model.llm.qwen import VLMQwenForCausalLM


# ==============================================================================
# CONFIGURE PATHS HERE
# ==============================================================================
STAGE3_CHECKPOINT = (
    "./output/exp2-single-pargo/stage3/checkpoint-150985"
)
OUTPUT_DIR = "./output/exp2-single-pargo-stage3-complete"
# ==============================================================================

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_BIAS    = "none"


def find_all_linear_names(model):
    """Identical to the function used during training — must match exactly."""
    ignore_keywords = ["vision_tower", "mm_projector", "embed_tokens", "lm_head"]
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(kw in name for kw in ignore_keywords):
            continue
        if isinstance(module, nn.Linear):
            lora_module_names.add(name)
    return list(lora_module_names)


def run_zero_to_fp32(checkpoint_dir, output_dir):
    """
    zero_to_fp32.py writes pytorch_model.bin INSIDE the output directory.
    Second argument is a directory, not a file path.
    """
    script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"zero_to_fp32.py not found in {checkpoint_dir}")

    output_file = os.path.join(output_dir, "pytorch_model.bin")
    if os.path.isfile(output_file):
        print(f"  ✓ zero_to_fp32 output already exists, skipping")
        return output_file

    os.makedirs(output_dir, exist_ok=True)
    print(f"  Running zero_to_fp32.py ...")
    result = subprocess.run(
        [sys.executable, script, checkpoint_dir, output_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-2000:])
        print("  STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"zero_to_fp32.py failed with code {result.returncode}")
    print(f"  ✓ zero_to_fp32 complete → {output_file}")
    return output_file


def main():
    print("=" * 80)
    print("MERGE LORA + SAVE COMPLETE MODEL — Experiment 2: Single-Scale ParGo")
    print("=" * 80)

    # ------------------------------------------------------------------
    # [1] Validate checkpoint
    # ------------------------------------------------------------------
    print(f"\n[1/8] Validating checkpoint...")
    if not os.path.isdir(STAGE3_CHECKPOINT):
        print(f"❌ ERROR: Checkpoint not found: {STAGE3_CHECKPOINT}")
        return

    adapter_file = os.path.join(STAGE3_CHECKPOINT, "adapter_model.safetensors")
    adapter_cfg  = os.path.join(STAGE3_CHECKPOINT, "adapter_config.json")
    zero_script  = os.path.join(STAGE3_CHECKPOINT, "zero_to_fp32.py")
    for f in [adapter_file, adapter_cfg, zero_script]:
        if not os.path.exists(f):
            print(f"❌ ERROR: Required file missing: {f}")
            return
        print(f"  ✓ {os.path.basename(f)}")

    global_step_dirs = glob.glob(os.path.join(STAGE3_CHECKPOINT, "global_step*"))
    if not global_step_dirs:
        print(f"❌ ERROR: No global_step* ZeRO shard directories found")
        return
    print(f"  ✓ ZeRO shard dir: {os.path.basename(global_step_dirs[0])}")

    # ------------------------------------------------------------------
    # [2] Run zero_to_fp32.py → get all trainable weights (LoRA + projector + VT)
    # ------------------------------------------------------------------
    print(f"\n[2/8] Extracting trainable weights from ZeRO shards...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # zero_fp32_path = os.path.join(OUTPUT_DIR, "zero_trainable_weights.pt")
    # run_zero_to_fp32(STAGE3_CHECKPOINT, zero_fp32_path)

    # print(f"  Loading extracted weights...")
    # zero_state_dict = torch.load(zero_fp32_path, map_location="cpu")

    zero_out_dir = os.path.join(OUTPUT_DIR, "zero_extracted")
    zero_fp32_path = run_zero_to_fp32(STAGE3_CHECKPOINT, zero_out_dir)

    print(f"  Loading extracted weights...")
    zero_state_dict = torch.load(zero_fp32_path, map_location="cpu")
    print(f"  ✓ Loaded {len(zero_state_dict)} parameters from zero_to_fp32")

    # Show key prefixes for diagnostics
    prefixes = {}
    for k in zero_state_dict:
        parts = k.split(".")
        p = ".".join(parts[:3]) if len(parts) >= 3 else k
        prefixes[p] = prefixes.get(p, 0) + 1
    print(f"  Top key prefixes:")
    for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:10]:
        print(f"    {p}: {c}")

    # ------------------------------------------------------------------
    # [3] Tokenizer
    # ------------------------------------------------------------------
    print(f"\n[3/8] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        padding_side="right",
        use_fast=False,
    )
    if "<im_patch>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>"]})
    print(f"  ✓ Tokenizer loaded | vocab: {len(tokenizer)} | <im_patch>: {tokenizer.convert_tokens_to_ids('<im_patch>')}")

    # ------------------------------------------------------------------
    # [4] Load base Qwen + initialize vision modules
    # ------------------------------------------------------------------
    print(f"\n[4/8] Loading base Qwen2.5-7B-Instruct...")
    model = VLMQwenForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    print(f"  ✓ Base Qwen loaded")

    model_args = SimpleNamespace(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        vision_tower="dcformer",
        vision_select_layer=-1,          # Exp 2 uses final layer only
        vision_select_feature="cls_patch",
        pretrain_vision_model=None,
        pretrain_clip_model=None,
        freeze_vision_tower=False,
        mm_projector_type="single_scale_pargo",
        proj_out_num=32,
        low_input_size=384,
        high_input_size=768,
        mm_hidden_size=768,
        num_global_queries=8,
        num_partial_queries=24,
        pargo_num_layers=2,
        use_pretrained_bert=True,
        pargo_dropout=0.0,
        bert_type="bert-base-uncased",
        num_query_tokens=32,
        img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
        vocab_size=len(tokenizer),
        dim=768, depth=12,
        input_size=(256, 256, 128),
        patch_size=(16, 16, 16),
        num_new_tokens=1,
        mm_mlp_depth=2,
        proj_layer_type="mlp", proj_layer_num=2,
        proj_pooling_type="spatial", proj_pooling_size=2,
        proj_residual=False,
        low_output_size=[192, 128],
        high_output_size=[64, 128],
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
        use_positional_embedding=False,
    )

    # Pre-set config so build_mm_projector reads correct values
    model.config.mm_projector_type   = "single_scale_pargo"
    model.config.low_input_size      = 384
    model.config.high_input_size     = 768
    model.config.mm_hidden_size      = 768
    model.config.mm_mlp_depth        = 2
    model.config.proj_out_num        = 32
    model.config.num_global_queries  = 8
    model.config.num_partial_queries = 24
    model.config.pargo_num_layers    = 2
    model.config.use_pretrained_bert = True
    model.config.pargo_dropout       = 0.0
    model.config.low_output_size     = [192, 128]
    model.config.high_output_size    = [64, 128]

    print(f"\n  Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args)
    model.initialize_vision_tokenizer(model_args, tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print(f"  ✓ Vision modules initialized")

    # ------------------------------------------------------------------
    # [5] Wrap with PEFT (same LoRA config as training)
    # ------------------------------------------------------------------
    print(f"\n[5/8] Applying LoRA adapters (matching training config)...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=find_all_linear_names(model),
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  ✓ LoRA applied")

    # ------------------------------------------------------------------
    # [6] Load zero_to_fp32 weights into PEFT model
    # ------------------------------------------------------------------
    print(f"\n[6/8] Loading trainable weights (LoRA + projector + vision tower)...")
    model_sd = model.state_dict()

    loaded      = 0
    skipped     = 0
    mismatched  = []

    for key, val in zero_state_dict.items():
        if key in model_sd:
            if model_sd[key].shape == val.shape:
                model_sd[key] = val
                loaded += 1
            else:
                mismatched.append(key)
                print(f"  ⚠ Shape mismatch: {key} model={model_sd[key].shape} ckpt={val.shape}")
        else:
            skipped += 1
            if skipped <= 5:
                print(f"  ⚠ Key not in model: {key}")

    if skipped > 5:
        print(f"  ... and {skipped - 5} more keys not in model")

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print(f"\n  Weight loading summary:")
    print(f"    ✓ Loaded     : {loaded}")
    print(f"    ⚠ Mismatched : {len(mismatched)}")
    print(f"    ⚠ Skipped    : {skipped}")
    print(f"    Missing (not in zero ckpt): {len(missing)}")
    if missing:
        # Missing = frozen LLM weights, which is expected since zero_to_fp32
        # only contains trainable params. These come from base Qwen already loaded.
        lora_missing = [k for k in missing if "lora" in k]
        non_lora_missing = [k for k in missing if "lora" not in k]
        print(f"      LoRA missing (unexpected): {len(lora_missing)}")
        if lora_missing:
            print(f"      First few: {lora_missing[:3]}")
        print(f"      Non-LoRA missing (frozen LLM, expected): {len(non_lora_missing)}")

    # ------------------------------------------------------------------
    # [7] Merge LoRA into LLM backbone
    # ------------------------------------------------------------------
    print(f"\n[7/8] Merging LoRA weights into LLM backbone...")
    model = model.merge_and_unload()
    print(f"  ✓ LoRA merged and adapters unloaded")

    # Fix BERT shared tensors (SingleScaleParGo uses BERT)
    projector = model.model.mm_projector
    if hasattr(projector, "bert"):
        bert = projector.bert
        if hasattr(bert, "cls") and hasattr(bert.cls, "predictions"):
            bert.cls.predictions.decoder.weight = torch.nn.Parameter(
                bert.cls.predictions.decoder.weight.clone()
            )
            bert.cls.predictions.decoder.bias = torch.nn.Parameter(
                bert.cls.predictions.decoder.bias.clone()
            )
        if hasattr(bert, "embeddings"):
            bert.embeddings.word_embeddings.weight = torch.nn.Parameter(
                bert.embeddings.word_embeddings.weight.clone()
            )
        print(f"  ✓ Fixed BERT shared tensors")

    # ------------------------------------------------------------------
    # [8] Save complete model
    # ------------------------------------------------------------------
    print(f"\n[8/8] Saving complete model to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model.config.architectures        = ["VLMQwenForCausalLM"]
    model.config.mm_projector_type    = "single_scale_pargo"
    model.config.vision_tower         = "dcformer"
    model.config.vision_select_layer  = -1
    model.config.vision_select_feature = "cls_patch"
    model.config.bert_type            = "bert-base-uncased"
    model.config.num_query_tokens     = 32
    model.config.proj_out_num         = 32
    model.config.num_global_queries   = 8
    model.config.num_partial_queries  = 24
    model.config.pargo_num_layers     = 2
    model.config.use_pretrained_bert  = True
    model.config.pargo_dropout        = 0.0
    model.config.low_input_size       = 384
    model.config.high_input_size      = 768
    model.config.low_output_size      = [192, 128]
    model.config.high_output_size     = [64, 128]
    model.config.use_positional_embedding = False
    model.config.save_pretrained(OUTPUT_DIR)
    print(f"  ✓ Config saved")

    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="5GB")
    print(f"  ✓ Model weights saved")

    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✓ Tokenizer saved")

    # Verification
    sf_files   = glob.glob(os.path.join(OUTPUT_DIR, "*.safetensors"))
    total_size = sum(os.path.getsize(f) for f in sf_files)
    print(f"\n{'=' * 80}")
    print(f"FINAL VERIFICATION:")
    print(f"  ✓ Safetensor shards : {len(sf_files)}")
    for sf in sorted(sf_files):
        print(f"    - {os.path.basename(sf)}: {os.path.getsize(sf)/1e9:.2f} GB")
    print(f"  ✓ Total size        : {total_size/1e9:.2f} GB")
    print(f"  ✓ config.json       : {os.path.exists(os.path.join(OUTPUT_DIR, 'config.json'))}")
    print(f"  ✓ tokenizer_config  : {os.path.exists(os.path.join(OUTPUT_DIR, 'tokenizer_config.json'))}")
    print(f"\n✓✓✓ EXP 2 STAGE 3 COMPLETE MODEL SAVED! ✓✓✓")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()