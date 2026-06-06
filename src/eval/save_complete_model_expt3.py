"""
Save Complete Model — Experiment 3: Dual-Scale ParGo
=====================================================
Consolidates the stage 2 checkpoint (projector + vision tower weights only)
with the full Qwen2.5-7B-Instruct LLM weights into a single loadable model.

Key differences from Exp 2 (Single-Scale ParGo):
  - mm_projector_type = "dual_scale_pargo"
  - proj_out_num = 288 (256 low-scale + 32 high-scale tokens)
  - No BERT component -> no shared tensor fix needed
  - Uses both low-level (penultimate) and high-level (final) DCFormer features
"""

import os
import torch
import glob
from types import SimpleNamespace

from transformers import AutoTokenizer
from safetensors.torch import load_file

from src.model.llm.qwen import VLMQwenForCausalLM


def main():
    print("=" * 80)
    print("SAVE COMPLETE MODEL — Experiment 3: Dual-Scale ParGo")
    print("=" * 80)

    # ======================================================================
    # CONFIGURE THESE PATHS
    # ======================================================================
    base_checkpoint = (
        "/home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/"
        "exp3-dual-pargo/stage2/checkpoint-130590"
    )
    output_dir = "./output/exp3-dual-pargo-complete"
    # ======================================================================

    # ------------------------------------------------------------------
    # [1/9] Validate paths
    # ------------------------------------------------------------------
    print(f"\n[1/9] Configuration")
    print(f"  Base checkpoint : {base_checkpoint}")
    print(f"  Output directory: {output_dir}")

    if not os.path.exists(base_checkpoint):
        print(f"\n❌ ERROR: Checkpoint directory does not exist!")
        return
    print(f"  ✓ Checkpoint directory exists")

    # Check required files
    required_files = ["config.json"]
    missing_files = []
    for req_file in required_files:
        fp = os.path.join(base_checkpoint, req_file)
        if os.path.exists(fp):
            print(f"    ✓ {req_file}")
        else:
            print(f"    ❌ {req_file}")
            missing_files.append(req_file)

    if missing_files:
        print(f"\n❌ ERROR: Missing required files: {missing_files}")
        return

    shard_files = glob.glob(os.path.join(base_checkpoint, "model-*.safetensors"))
    if not shard_files:
        print(f"\n❌ ERROR: No safetensor shards found in checkpoint!")
        return
    print(f"  ✓ Found {len(shard_files)} safetensor shard(s)")

    # ------------------------------------------------------------------
    # [2/9] Tokenizer
    # ------------------------------------------------------------------
    print(f"\n[2/9] Loading tokenizer...")
    tokenizer_path = (
        base_checkpoint
        if os.path.exists(os.path.join(base_checkpoint, "tokenizer_config.json"))
        else "Qwen/Qwen2.5-7B-Instruct"
    )
    print(f"  Loading from: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        use_fast=False,
    )

    if "<im_patch>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>"]})
        print(f"  Added <im_patch> special token")

    print(f"  ✓ Tokenizer loaded")
    print(f"  Vocabulary size       : {len(tokenizer)}")
    print(f"  <im_patch> token ID   : {tokenizer.convert_tokens_to_ids('<im_patch>')}")

    # ------------------------------------------------------------------
    # [3/9] Model arguments — Experiment 3: dual_scale_pargo
    # ------------------------------------------------------------------
    print(f"\n[3/9] Setting up model arguments...")
    model_args = SimpleNamespace(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        vision_tower="dcformer",
        vision_select_layer=-2,
        vision_select_feature="cls_patch",
        pretrain_vision_model=None,
        pretrain_clip_model=None,
        freeze_vision_tower=False,

        # ---- Experiment 3: dual_scale_pargo ----
        mm_projector_type="dual_scale_pargo",
        proj_out_num=288,           # 256 low-scale + 32 high-scale tokens

        # ---- DCFormer dual-stream feature sizes ----
        low_input_size=384,         # penultimate layer channels
        high_input_size=768,        # final layer channels
        mm_hidden_size=768,

        # ---- ParGo-specific (mirrored from training script defaults) ----
        num_global_queries=8,
        num_partial_queries=280,    # 288 total - 8 global = 280 partial
        pargo_num_layers=2,
        use_pretrained_bert=False,  # dual-scale ParGo does NOT use BERT
        pargo_dropout=0.0,

        # ---- Standard config ----
        bert_type="bert-base-uncased",
        num_query_tokens=288,
        img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
        vocab_size=len(tokenizer),
        dim=768,
        depth=12,
        input_size=(256, 256, 128),
        patch_size=(16, 16, 16),
        num_new_tokens=1,
        mm_mlp_depth=2,
        proj_layer_type="mlp",
        proj_layer_num=2,
        proj_pooling_type="spatial",
        proj_pooling_size=2,
        proj_residual=False,
        low_output_size=[192, 128],
        high_output_size=[64, 128],
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=None,
        use_positional_embedding=False,
    )
    print(f"  ✓ Model arguments configured")
    print(f"  Projector type  : {model_args.mm_projector_type}")
    print(f"  proj_out_num    : {model_args.proj_out_num}")
    print(f"  low_input_size  : {model_args.low_input_size}")
    print(f"  high_input_size : {model_args.high_input_size}")

    # ------------------------------------------------------------------
    # [4/9] Load base Qwen model
    # ------------------------------------------------------------------
    print(f"\n[4/9] Loading base Qwen2.5-7B-Instruct from HuggingFace...")
    model = VLMQwenForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        ignore_mismatched_sizes=True,
    )
    print(f"  ✓ Base Qwen model loaded")

    # Pre-set all config attributes that initialize_vision_modules needs
    print(f"\n  Pre-setting config attributes...")
    model.config.mm_projector_type   = "dual_scale_pargo"
    model.config.low_input_size      = 384
    model.config.high_input_size     = 768
    model.config.mm_hidden_size      = 768
    model.config.mm_mlp_depth        = 2
    model.config.proj_out_num        = 288
    model.config.num_global_queries  = 8
    model.config.num_partial_queries = 280
    model.config.pargo_num_layers    = 2
    model.config.use_pretrained_bert = False
    model.config.pargo_dropout       = 0.0
    model.config.low_output_size     = [192, 128]
    model.config.high_output_size    = [64, 128]
    print(f"  ✓ Config attributes set")

    # ------------------------------------------------------------------
    # [5/9] Initialize vision modules
    # ------------------------------------------------------------------
    print(f"\n[5/9] Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args)
    print(f"  ✓ Vision modules initialized")

    model.initialize_vision_tokenizer(model_args, tokenizer)
    print(f"  ✓ Vision tokenizer initialized")

    original_emb_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_emb_size = model.get_input_embeddings().weight.shape[0]
    print(f"  ✓ Token embeddings resized: {original_emb_size} → {new_emb_size}")

    # ------------------------------------------------------------------
    # [6/9] Load checkpoint safetensors
    # ------------------------------------------------------------------
    print(f"\n[6/9] Loading checkpoint weights from safetensors...")
    consolidated_state_dict = {}
    shard_files = sorted(
        glob.glob(os.path.join(base_checkpoint, "model-*.safetensors"))
    )

    print(f"  Found {len(shard_files)} shard(s):")
    for idx, shard_file in enumerate(shard_files, 1):
        print(f"    [{idx}/{len(shard_files)}] {os.path.basename(shard_file)} ...", end=" ")
        shard_dict = load_file(shard_file)
        consolidated_state_dict.update(shard_dict)
        print(f"({len(shard_dict)} params)")

    print(f"  ✓ Total parameters loaded from checkpoint: {len(consolidated_state_dict)}")

    # Diagnostic: show key prefixes
    print(f"\n  Sample parameter keys (first 5):")
    for key in list(consolidated_state_dict.keys())[:5]:
        print(f"    - {key}: {consolidated_state_dict[key].shape}")

    print(f"\n  Top parameter-key prefixes in checkpoint:")
    prefixes = {}
    for key in consolidated_state_dict.keys():
        parts = key.split(".")
        prefix = ".".join(parts[:3]) if len(parts) >= 3 else key
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
        print(f"    {p}: {c} params")

    # ------------------------------------------------------------------
    # [7/9] Merge checkpoint weights into model
    # ------------------------------------------------------------------
    print(f"\n[7/9] Merging checkpoint weights into model...")
    model_dict = model.state_dict()
    loaded_count        = 0
    skipped_count       = 0
    shape_mismatch_keys = []

    print(f"  Model total params      : {len(model_dict)}")
    print(f"  Checkpoint params       : {len(consolidated_state_dict)}")

    for key, value in consolidated_state_dict.items():
        if key in model_dict:
            if model_dict[key].shape == value.shape:
                model_dict[key] = value
                loaded_count += 1
            else:
                shape_mismatch_keys.append(key)
                print(
                    f"  ⚠ Shape mismatch for {key}: "
                    f"model={model_dict[key].shape}, ckpt={value.shape}"
                )
        else:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"  ⚠ Key not in model: {key}")

    if skipped_count > 5:
        print(f"  ... and {skipped_count - 5} more keys not in model")

    missing, unexpected = model.load_state_dict(model_dict, strict=False)

    print(f"\n{'=' * 60}")
    print(f"WEIGHT LOADING SUMMARY:")
    print(f"  ✓ Successfully loaded    : {loaded_count}")
    print(f"  ⚠ Shape mismatches      : {len(shape_mismatch_keys)}")
    print(f"  ⚠ Not in model          : {skipped_count}")
    if missing:
        print(f"  Missing (not in ckpt)   : {len(missing)}")
        print(f"  First few missing       : {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys         : {len(unexpected)}")
        print(f"  First few unexpected    : {unexpected[:5]}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # [8/9] No BERT shared-tensor fix needed for dual_scale_pargo
    # ------------------------------------------------------------------
    print(f"[8/9] Checking for shared tensors...")
    projector = model.model.mm_projector
    if hasattr(projector, "bert"):
        # Shouldn't be present for dual_scale_pargo, but guard just in case
        bert = projector.bert
        if hasattr(bert, "cls") and hasattr(bert.cls, "predictions"):
            bert.cls.predictions.decoder.weight = torch.nn.Parameter(
                bert.cls.predictions.decoder.weight.clone()
            )
            bert.cls.predictions.decoder.bias = torch.nn.Parameter(
                bert.cls.predictions.decoder.bias.clone()
            )
            print(f"  ✓ Fixed unexpected BERT cls.predictions shared tensors")
        if hasattr(bert, "embeddings") and hasattr(bert.embeddings, "word_embeddings"):
            bert.embeddings.word_embeddings.weight = torch.nn.Parameter(
                bert.embeddings.word_embeddings.weight.clone()
            )
            print(f"  ✓ Cloned BERT word embeddings")
    else:
        print(f"  ✓ No BERT in projector (expected for dual_scale_pargo) — nothing to fix")

    # ------------------------------------------------------------------
    # [9/9] Save complete model
    # ------------------------------------------------------------------
    print(f"\n[9/9] Saving complete model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Write all relevant config fields before saving
    model.config.architectures       = ["VLMQwenForCausalLM"]
    model.config.mm_projector_type   = "dual_scale_pargo"
    model.config.vision_tower        = "dcformer"
    model.config.bert_type           = "bert-base-uncased"
    model.config.num_query_tokens    = 288
    model.config.proj_out_num        = 288
    model.config.num_global_queries  = 8
    model.config.num_partial_queries = 280
    model.config.pargo_num_layers    = 2
    model.config.vision_select_layer = -2
    model.config.vision_select_feature = "cls_patch"
    model.config.use_positional_embedding = False
    model.config.low_input_size      = 384
    model.config.high_input_size     = 768
    model.config.low_output_size     = [192, 128]
    model.config.high_output_size    = [64, 128]
    model.config.save_pretrained(output_dir)
    print(f"  ✓ Config saved")

    print(f"  Saving model weights (safetensors, max shard 5 GB) ...")
    print(f"  (This may take a few minutes...)")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    print(f"  ✓ Model weights saved")

    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Tokenizer saved")

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------
    print(f"\n  Verifying saved files...")
    safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"\n{'=' * 80}")
    print(f"FINAL VERIFICATION:")
    print(f"  ✓ Safetensor shards : {len(safetensor_files)}")
    for sf in sorted(safetensor_files):
        size_mb = os.path.getsize(sf) / (1024 ** 2)
        print(f"    - {os.path.basename(sf)}: {size_mb:.2f} MB")
    print(f"  ✓ Total model size  : {total_size / 1e9:.2f} GB")
    print(f"  ✓ config.json       : {os.path.exists(os.path.join(output_dir, 'config.json'))}")
    print(f"  ✓ tokenizer_config  : {os.path.exists(os.path.join(output_dir, 'tokenizer_config.json'))}")
    print(f"\n✓✓✓ COMPLETE MODEL SAVED SUCCESSFULLY! ✓✓✓")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()