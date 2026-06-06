"""
Save Complete Model — Experiment 1: Baseline (low_high_mlp)
============================================================
Consolidates the stage 2 checkpoint (projector + vision tower weights only)
with the full Qwen2.5-7B-Instruct LLM weights into a single loadable model.
"""

import os
import torch
from transformers import AutoTokenizer
from src.model.llm.qwen import VLMQwenForCausalLM
from types import SimpleNamespace
import glob
from safetensors.torch import load_file


def main():
    print("=" * 80)
    print("SAVE COMPLETE MODEL — Experiment 1: Baseline (low_high_mlp)")
    print("=" * 80)

    # ======================================================================
    # CONFIGURE THESE PATHS
    # ======================================================================
    base_checkpoint = "/home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/exp1-baseline-stage2/checkpoint-202479"
    output_dir = "./output/exp1-baseline-complete"
    # ======================================================================

    print(f"\n[1/9] Configuration")
    print(f"  Base checkpoint: {base_checkpoint}")
    print(f"  Output directory: {output_dir}")

    if not os.path.exists(base_checkpoint):
        print(f"\n❌ ERROR: Checkpoint directory does not exist!")
        return

    print(f"  ✓ Checkpoint directory exists")

    required_files = ["config.json"]
    missing_files = []
    for req_file in required_files:
        file_path = os.path.join(base_checkpoint, req_file)
        if os.path.exists(file_path):
            print(f"    ✓ {req_file}")
        else:
            print(f"    ❌ {req_file}")
            missing_files.append(req_file)

    if missing_files:
        print(f"\n❌ ERROR: Missing required files: {missing_files}")
        return

    shard_files = glob.glob(os.path.join(base_checkpoint, "model-*.safetensors"))
    if not shard_files:
        print(f"\n❌ ERROR: No safetensor files found in checkpoint!")
        return
    print(f"  ✓ Found {len(shard_files)} safetensor files")

    # Load tokenizer — try checkpoint first, fall back to base model
    print(f"\n[2/9] Loading tokenizer...")
    tokenizer_path = base_checkpoint if os.path.exists(
        os.path.join(base_checkpoint, "tokenizer_config.json")
    ) else "Qwen/Qwen2.5-7B-Instruct"
    print(f"  Loading from: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        use_fast=False,
    )

    # Ensure <im_patch> token exists
    if "<im_patch>" not in tokenizer.get_vocab():
        special_token = {"additional_special_tokens": ["<im_patch>"]}
        tokenizer.add_special_tokens(special_token)
        print(f"  Added <im_patch> special token")

    print(f"  ✓ Tokenizer loaded successfully")
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Special token <im_patch> ID: {tokenizer.convert_tokens_to_ids('<im_patch>')}")

    # Model arguments — Experiment 1: low_high_mlp baseline
    print(f"\n[3/9] Setting up model arguments...")
    model_args = SimpleNamespace(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        vision_tower="dcformer",
        vision_select_layer=-2,
        vision_select_feature="cls_patch",
        pretrain_vision_model=None,
        pretrain_clip_model=None,
        freeze_vision_tower=False,
        # ---- Experiment 1: low_high_mlp ----
        mm_projector_type="low_high_mlp",
        proj_out_num=288,
        # ---- DCFormer-derived sizes (needed by build_mm_projector) ----
        # Penultimate layer (stage 3): (B, 256, 384) -> low_input_size = 384
        # Final layer (stage 4):       (B, 32, 768)  -> high_input_size = 768
        low_input_size=384,
        high_input_size=768,
        mm_hidden_size=768,       # vision encoder final output dim
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
    print(f"  Projector type: {model_args.mm_projector_type}")
    print(f"  proj_out_num: {model_args.proj_out_num}")
    print(f"  low_input_size: {model_args.low_input_size}")
    print(f"  high_input_size: {model_args.high_input_size}")

    print(f"\n[4/9] Loading base Qwen model from HuggingFace...")
    model = VLMQwenForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        ignore_mismatched_sizes=True,
    )
    print(f"  ✓ Base Qwen model loaded")

    # ---- Pre-set config attributes that build_mm_projector needs ----
    # initialize_vision_modules normally copies these from the vision tower,
    # but when we don't load the full vision tower, they may be missing.
    # Set them on the model config directly so the builder can find them.
    print(f"\n  Pre-setting config attributes for projector builder...")
    model.config.mm_projector_type = "low_high_mlp"
    model.config.low_input_size = 384       # DCFormer penultimate layer dim
    model.config.high_input_size = 768      # DCFormer final layer dim
    model.config.mm_hidden_size = 768       # vision encoder output dim
    model.config.mm_mlp_depth = 2
    model.config.proj_out_num = 288
    model.config.low_output_size = [192, 128]
    model.config.high_output_size = [64, 128]
    print(f"  ✓ Config attributes set")

    print(f"\n[5/9] Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args)
    print(f"  ✓ Vision modules initialized")
    model.initialize_vision_tokenizer(model_args, tokenizer)
    print(f"  ✓ Vision tokenizer initialized")

    original_embedding_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"  ✓ Token embeddings resized: {original_embedding_size} → {new_embedding_size}")

    print(f"\n[6/9] Loading checkpoint from safetensors...")
    consolidated_state_dict = {}
    shard_files = sorted(glob.glob(os.path.join(base_checkpoint, "model-*.safetensors")))

    print(f"  Found {len(shard_files)} safetensor shards:")
    for idx, shard_file in enumerate(shard_files, 1):
        print(f"    [{idx}/{len(shard_files)}] Loading {os.path.basename(shard_file)}...")
        shard_dict = load_file(shard_file)
        print(f"         → Loaded {len(shard_dict)} parameters from this shard")
        consolidated_state_dict.update(shard_dict)

    print(f"  ✓ Total parameters loaded from checkpoint: {len(consolidated_state_dict)}")

    print(f"\n  Sample parameter keys:")
    for i, key in enumerate(list(consolidated_state_dict.keys())[:5]):
        print(f"    - {key}: shape {consolidated_state_dict[key].shape}")

    print(f"\n[7/9] Loading checkpoint weights into model...")
    model_dict = model.state_dict()
    loaded_count = 0
    skipped_count = 0
    shape_mismatch_keys = []

    print(f"  Model has {len(model_dict)} parameters")
    print(f"  Attempting to load {len(consolidated_state_dict)} parameters from checkpoint...")

    for key, value in consolidated_state_dict.items():
        if key in model_dict:
            if model_dict[key].shape == value.shape:
                model_dict[key] = value
                loaded_count += 1
            else:
                shape_mismatch_keys.append(key)
                print(f"  ⚠ Shape mismatch for {key}: model={model_dict[key].shape}, checkpoint={value.shape}")
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"  Loading state dict into model...")
    missing, unexpected = model.load_state_dict(model_dict, strict=False)

    print(f"\n{'=' * 60}")
    print(f"WEIGHT LOADING SUMMARY:")
    print(f"  ✓ Successfully loaded: {loaded_count} parameters")
    print(f"  ⚠ Skipped: {skipped_count} parameters")
    if shape_mismatch_keys:
        print(f"  Shape mismatches: {len(shape_mismatch_keys)}")
    if missing:
        print(f"  Missing keys in checkpoint: {len(missing)}")
        print(f"  First few missing: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys in checkpoint: {len(unexpected)}")
        print(f"  First few unexpected: {unexpected[:5]}")
    print(f"{'=' * 60}\n")

    # low_high_mlp doesn't have BERT
    print(f"[8/9] Checking for shared tensor issues...")
    print(f"  ℹ low_high_mlp uses MLP-Mixer, no BERT shared tensor issues expected")

    print(f"\n[9/9] Saving complete model...")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.config.architectures = ["VLMQwenForCausalLM"]
    model.config.mm_projector_type = "low_high_mlp"
    model.config.vision_tower = "dcformer"
    model.config.proj_out_num = 288
    model.config.low_input_size = 384
    model.config.high_input_size = 768
    model.config.mm_hidden_size = 768
    model.config.mm_mlp_depth = 2
    model.config.vision_select_layer = -2
    model.config.vision_select_feature = "cls_patch"
    model.config.use_positional_embedding = False
    model.config.save_pretrained(output_dir)
    print(f"  ✓ Config saved")

    print(f"  Saving model weights in safetensors format...")
    print(f"  (This may take a few minutes...)")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    print(f"  ✓ Model weights saved")

    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Tokenizer saved")

    # Verify
    print(f"\n  Verifying saved files...")
    safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"\n{'=' * 80}")
    print(f"FINAL VERIFICATION:")
    print(f"  ✓ Created {len(safetensor_files)} safetensor files")
    for sf in safetensor_files:
        size_mb = os.path.getsize(sf) / (1024 ** 2)
        print(f"    - {os.path.basename(sf)}: {size_mb:.2f} MB")
    print(f"  ✓ Total model size: {total_size / 1e9:.2f} GB")
    print(f"  ✓ Config: {os.path.exists(os.path.join(output_dir, 'config.json'))}")
    print(f"  ✓ Tokenizer: {os.path.exists(os.path.join(output_dir, 'tokenizer_config.json'))}")
    print(f"\n✓✓✓ COMPLETE MODEL SAVED SUCCESSFULLY! ✓✓✓")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()