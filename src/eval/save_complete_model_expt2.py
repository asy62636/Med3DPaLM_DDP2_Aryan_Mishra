"""
Save Complete Model — Experiment 2: Single-Scale ParGo
=======================================================
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
    print("SAVE COMPLETE MODEL — Experiment 2: Single-Scale ParGo")
    print("=" * 80)

    # ======================================================================
    # CONFIGURE THESE PATHS
    # ======================================================================
    base_checkpoint = "/home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/exp2-single-pargo/stage2/checkpoint-97944"
    output_dir = "./output/exp2-single-pargo-complete"
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

    # Load tokenizer
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

    if "<im_patch>" not in tokenizer.get_vocab():
        special_token = {"additional_special_tokens": ["<im_patch>"]}
        tokenizer.add_special_tokens(special_token)
        print(f"  Added <im_patch> special token")

    print(f"  ✓ Tokenizer loaded successfully")
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Special token <im_patch> ID: {tokenizer.convert_tokens_to_ids('<im_patch>')}")

    # Model arguments — Experiment 2: single_scale_pargo
    print(f"\n[3/9] Setting up model arguments...")
    model_args = SimpleNamespace(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        vision_tower="dcformer",
        vision_select_layer=-2,
        vision_select_feature="cls_patch",
        pretrain_vision_model=None,
        pretrain_clip_model=None,
        freeze_vision_tower=False,
        # ---- Experiment 2: single_scale_pargo ----
        mm_projector_type="single_scale_pargo",
        proj_out_num=32,
        # ---- DCFormer-derived sizes ----
        # These are needed by initialize_vision_modules even for single-scale,
        # because the vision tower setup reads them.
        low_input_size=384,
        high_input_size=768,
        mm_hidden_size=768,
        # ---- ParGo-specific ----
        num_global_queries=8,
        num_partial_queries=24,
        pargo_num_layers=2,
        use_pretrained_bert=True,
        pargo_dropout=0.0,
        # ---- Standard config ----
        bert_type="bert-base-uncased",
        num_query_tokens=32,
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

    print(f"\n[4/9] Loading base Qwen model from HuggingFace...")
    model = VLMQwenForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        ignore_mismatched_sizes=True,
    )
    print(f"  ✓ Base Qwen model loaded")

    # ---- Pre-set config attributes that build_mm_projector / initialize_vision_modules need ----
    print(f"\n  Pre-setting config attributes for projector builder...")
    model.config.mm_projector_type = "single_scale_pargo"
    model.config.low_input_size = 384
    model.config.high_input_size = 768
    model.config.mm_hidden_size = 768
    model.config.mm_mlp_depth = 2
    model.config.proj_out_num = 32
    model.config.num_global_queries = 8
    model.config.num_partial_queries = 24
    model.config.pargo_num_layers = 2
    model.config.use_pretrained_bert = True
    model.config.pargo_dropout = 0.0
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

    # Diagnostic: show what's in the checkpoint
    print(f"\n  Sample parameter keys:")
    for i, key in enumerate(list(consolidated_state_dict.keys())[:5]):
        print(f"    - {key}: shape {consolidated_state_dict[key].shape}")

    print(f"\n  Parameter key prefixes in checkpoint:")
    prefixes = {}
    for key in consolidated_state_dict.keys():
        parts = key.split(".")
        prefix = ".".join(parts[:3]) if len(parts) >= 3 else key
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
        print(f"    {p}: {c} params")

    print(f"\n[7/9] Loading checkpoint weights into model...")
    model_dict = model.state_dict()
    loaded_count = 0
    skipped_not_in_model = 0
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
        else:
            skipped_not_in_model += 1
            if skipped_not_in_model <= 5:
                print(f"  ⚠ Key not in model: {key}")

    if skipped_not_in_model > 5:
        print(f"  ... and {skipped_not_in_model - 5} more keys not in model")

    print(f"  Loading state dict into model...")
    missing, unexpected = model.load_state_dict(model_dict, strict=False)

    print(f"\n{'=' * 60}")
    print(f"WEIGHT LOADING SUMMARY:")
    print(f"  ✓ Successfully loaded: {loaded_count} parameters")
    print(f"  ⚠ Shape mismatches: {len(shape_mismatch_keys)}")
    print(f"  ⚠ Not in model: {skipped_not_in_model}")
    if missing:
        print(f"  Missing keys (not in checkpoint): {len(missing)}")
        print(f"  First few missing: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        print(f"  First few unexpected: {unexpected[:5]}")
    print(f"{'=' * 60}\n")

    # Fix shared tensor issue — SingleScaleParGo uses BERT
    print(f"[8/9] Fixing shared tensors in BERT...")
    projector = model.model.mm_projector
    if hasattr(projector, "bert"):
        bert = projector.bert
        # Check for the word_embeddings -> cls.predictions.decoder tie
        if hasattr(bert, "cls") and hasattr(bert.cls, "predictions"):
            bert.cls.predictions.decoder.weight = torch.nn.Parameter(
                bert.cls.predictions.decoder.weight.clone()
            )
            bert.cls.predictions.decoder.bias = torch.nn.Parameter(
                bert.cls.predictions.decoder.bias.clone()
            )
            print(f"  ✓ Fixed BERT cls.predictions shared tensors")
        else:
            print(f"  ℹ No BERT cls.predictions found")

        # Also break the embeddings tie if present
        if hasattr(bert, "embeddings") and hasattr(bert.embeddings, "word_embeddings"):
            bert.embeddings.word_embeddings.weight = torch.nn.Parameter(
                bert.embeddings.word_embeddings.weight.clone()
            )
            print(f"  ✓ Cloned BERT word embeddings to break potential ties")
    else:
        print(f"  ℹ No BERT found in projector")

    print(f"\n[9/9] Saving complete model...")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.config.architectures = ["VLMQwenForCausalLM"]
    model.config.mm_projector_type = "single_scale_pargo"
    model.config.vision_tower = "dcformer"
    model.config.bert_type = "bert-base-uncased"
    model.config.num_query_tokens = 32
    model.config.proj_out_num = 32
    model.config.num_global_queries = 8
    model.config.num_partial_queries = 24
    model.config.pargo_num_layers = 2
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