# save_complete_modified_pargo_model.py
import os
import torch
from transformers import AutoTokenizer
from src.model.llm.qwen import VLMQwenForCausalLM
from types import SimpleNamespace
import glob
from safetensors.torch import load_file

def main():
    print("="*80)
    print("SAVING COMPLETE MODIFIED PARGO MODEL")
    print("="*80)
    
    # Paths - UPDATED FOR MODIFIED PARGO
    base_checkpoint = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/modified_pargo_stage1"
    output_dir = "./output2/Med3DVLM-Qwen-2.5-7B-Modified-ParGo-Complete"
    
    print(f"\n[1/9] Configuration")
    print(f"  Base checkpoint: {base_checkpoint}")
    print(f"  Output directory: {output_dir}")
    print(f"  Checking if checkpoint exists: {os.path.exists(base_checkpoint)}")
    
    # Find the latest checkpoint folder
    checkpoint_dirs = sorted(glob.glob(os.path.join(base_checkpoint, "checkpoint-*")))
    if checkpoint_dirs:
        latest_checkpoint = checkpoint_dirs[-1]
        print(f"  Found {len(checkpoint_dirs)} checkpoints")
        print(f"  Using latest: {os.path.basename(latest_checkpoint)}")
        base_checkpoint = latest_checkpoint
    else:
        print(f"  No checkpoint-* folders found, using base directory")
    
    # Load tokenizer from base checkpoint
    print(f"\n[2/9] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_checkpoint,
        padding_side="right",
        use_fast=False,
    )
    print(f"  ✓ Tokenizer loaded successfully")
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Special token <im_patch> ID: {tokenizer.convert_tokens_to_ids('<im_patch>')}")
    
    # Model arguments - UPDATED FOR MODIFIED PARGO
    print(f"\n[3/9] Setting up model arguments...")
    model_args = SimpleNamespace(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        vision_tower="dcformer",
        vision_select_layer=-2,
        vision_select_feature="cls_patch",
        pretrain_vision_model=None,
        pretrain_clip_model=None,
        freeze_vision_tower=False,
        
        # CRITICAL: Set to modified_pargo
        mm_projector_type="modified_pargo",
        
        # Modified ParGo specific parameters
        n_low_tokens=144,
        n_high_tokens=32,
        low_level_hidden_size=384,
        pargo_num_layers=6,
        use_cross_scale_attention=False,
        use_positional_embedding=True,
        
        # Standard parameters
        bert_type="bert-base-uncased",
        num_query_tokens=304,  # Legacy, overridden by modified_pargo
        proj_out_num=176,      # 144 + 32
        img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
        vocab_size=len(tokenizer),
        dim=768,
        depth=12,
        input_size=(256, 256, 128),
        patch_size=(16, 16, 16),
        num_new_tokens=1,
        mm_mlp_depth=2,
        low_output_size=[192, 128],
        high_output_size=[64, 128],
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=None
    )
    print(f"  ✓ Model arguments configured")
    print(f"  Projector type: {model_args.mm_projector_type}")
    print(f"  Vision tower: {model_args.vision_tower}")
    print(f"  Output tokens: {model_args.proj_out_num}")
    print(f"  Low-level tokens: {model_args.n_low_tokens}")
    print(f"  High-level tokens: {model_args.n_high_tokens}")
    
    print(f"\n[4/9] Loading base Qwen model from HuggingFace...")
    model = VLMQwenForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        ignore_mismatched_sizes=True
    )
    print(f"  ✓ Base Qwen model loaded")
    print(f"  Model type: {type(model).__name__}")
    
    print(f"\n[5/9] Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args)
    print(f"  ✓ Vision modules initialized")
    print(f"  Projector class: {type(model.get_model().mm_projector).__name__}")
    model.initialize_vision_tokenizer(model_args, tokenizer)
    print(f"  ✓ Vision tokenizer initialized")
    
    # Resize embeddings
    original_embedding_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"  ✓ Token embeddings resized: {original_embedding_size} → {new_embedding_size}")
    
    print(f"\n[6/9] Loading checkpoint from safetensors...")
    # Load all safetensor shards
    consolidated_state_dict = {}
    
    shard_files = sorted(glob.glob(os.path.join(base_checkpoint, "model-*.safetensors")))
    
    if shard_files:
        print(f"  Found {len(shard_files)} safetensor shards:")
        for idx, shard_file in enumerate(shard_files, 1):
            print(f"    [{idx}/{len(shard_files)}] Loading {os.path.basename(shard_file)}...")
            shard_dict = load_file(shard_file)
            print(f"         → Loaded {len(shard_dict)} parameters from this shard")
            consolidated_state_dict.update(shard_dict)
        print(f"  ✓ Total parameters loaded from checkpoint: {len(consolidated_state_dict)}")
    else:
        # Try loading from mm_projector folder (tune_mm_mlp_adapter=True mode)
        print(f"  No safetensor shards found in checkpoint folder")
        print(f"  Checking for mm_projector weights...")
        
        mm_projector_folder = os.path.join(os.path.dirname(base_checkpoint), "mm_projector")
        if os.path.exists(mm_projector_folder):
            checkpoint_name = os.path.basename(base_checkpoint)
            mm_projector_file = os.path.join(mm_projector_folder, f"{checkpoint_name}.bin")
            
            if os.path.exists(mm_projector_file):
                print(f"  Found mm_projector weights: {mm_projector_file}")
                mm_projector_weights = torch.load(mm_projector_file, map_location="cpu")
                consolidated_state_dict.update(mm_projector_weights)
                print(f"  ✓ Loaded {len(mm_projector_weights)} parameters from mm_projector")
            else:
                print(f"  ⚠ mm_projector file not found: {mm_projector_file}")
        else:
            raise FileNotFoundError(f"No safetensor files or mm_projector folder found!")
    
    # Check for positional embedding parameters
    print(f"\n  Checking for positional embedding parameters...")
    pos_embed_keys = [k for k in consolidated_state_dict.keys() if 'pos_embed' in k.lower() or 'position' in k.lower()]
    if pos_embed_keys:
        print(f"  ✓ Found {len(pos_embed_keys)} positional embedding related parameters:")
        for key in pos_embed_keys[:5]:  # Show first 5
            print(f"    - {key}: shape {consolidated_state_dict[key].shape}")
    else:
        print(f"  ℹ No explicit 'pos_embed' keys found (embedded in Sinusoidal3DPositionalEncoding)")
    
    # Check for modified_pargo specific keys
    print(f"\n  Checking for Modified ParGo parameters...")
    pargo_keys = [k for k in consolidated_state_dict.keys() if 'pargo' in k.lower() or 'mm_projector' in k.lower()]
    if pargo_keys:
        print(f"  ✓ Found {len(pargo_keys)} Modified ParGo related parameters")
        # Show structure
        low_level_keys = [k for k in pargo_keys if 'low_level' in k]
        high_level_keys = [k for k in pargo_keys if 'high_level' in k]
        branch_keys = [k for k in pargo_keys if 'branch' in k]
        print(f"    - Low-level branch: {len(low_level_keys)} parameters")
        print(f"    - High-level branch: {len(high_level_keys)} parameters")
        print(f"    - Plane branches: {len(branch_keys)} parameters")
        print(f"  Sample keys:")
        for key in pargo_keys[:3]:
            print(f"    - {key}")
    else:
        print(f"  ⚠ No Modified ParGo keys found!")
    
    print(f"\n[7/9] Loading checkpoint weights into model...")
    # Load the state dict into model
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
    
    print(f"\n{'='*60}")
    print(f"WEIGHT LOADING SUMMARY:")
    print(f"  ✓ Successfully loaded: {loaded_count} parameters")
    print(f"  ⚠ Skipped: {skipped_count} parameters")
    if shape_mismatch_keys:
        print(f"  Shape mismatches: {len(shape_mismatch_keys)}")
        for key in shape_mismatch_keys[:5]:
            print(f"    - {key}")
    if missing:
        print(f"  Missing keys in checkpoint: {len(missing)}")
        print(f"  First few missing: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys in checkpoint: {len(unexpected)}")
        print(f"  First few unexpected: {unexpected[:5]}")
    print(f"{'='*60}\n")
    
    # No BERT to fix in modified_pargo
    print(f"[8/9] Checking for shared tensors...")
    print(f"  ℹ Modified ParGo doesn't use BERT, skipping shared tensor fix")
    
    print(f"\n[9/9] Saving complete model...")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  ✓ Output directory created/verified")
    
    # Update config with Modified ParGo parameters
    print(f"  Updating model config...")
    model.config.architectures = ["VLMQwenForCausalLM"]
    model.config.mm_projector_type = "modified_pargo"
    model.config.vision_tower = "dcformer"
    model.config.vision_select_layer = -2
    model.config.vision_select_feature = "cls_patch"
    
    # Modified ParGo specific config
    model.config.n_low_tokens = 144
    model.config.n_high_tokens = 32
    model.config.proj_out_num = 176
    model.config.low_level_hidden_size = 384
    model.config.pargo_num_layers = 6
    model.config.use_cross_scale_attention = False
    model.config.use_positional_embedding = True
    model.config.llm_hidden_size = 3584
    model.config.vision_hidden_size = 768
    
    model.config.save_pretrained(output_dir)
    print(f"  ✓ Config saved with Modified ParGo parameters")
    
    # Save the model in safetensors format
    print(f"  Saving model weights in safetensors format...")
    print(f"  (This may take a few minutes...)")
    model.save_pretrained(
        output_dir, 
        safe_serialization=True,
        max_shard_size="5GB"
    )
    print(f"  ✓ Model weights saved")
    
    # Save tokenizer
    print(f"  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Tokenizer saved")
    
    # Verify
    print(f"\n  Verifying saved files...")
    safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    config_exists = os.path.exists(os.path.join(output_dir, "config.json"))
    tokenizer_exists = os.path.exists(os.path.join(output_dir, "tokenizer_config.json"))
    
    print(f"\n{'='*80}")
    print(f"FINAL VERIFICATION:")
    print(f"  ✓ Created {len(safetensor_files)} safetensor files")
    for sf in safetensor_files:
        size_mb = os.path.getsize(sf) / (1024**2)
        print(f"    - {os.path.basename(sf)}: {size_mb:.2f} MB")
    
    total_size = sum(os.path.getsize(f) for f in safetensor_files)
    print(f"  ✓ Total model size: {total_size / 1e9:.2f} GB")
    
    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print(f"  ✓ Index file created successfully")
    
    print(f"  ✓ Config file exists: {config_exists}")
    print(f"  ✓ Tokenizer files exist: {tokenizer_exists}")
    
    # Print config for verification
    print(f"\n  Model config verification:")
    print(f"    - mm_projector_type: {model.config.mm_projector_type}")
    print(f"    - proj_out_num: {model.config.proj_out_num}")
    print(f"    - n_low_tokens: {model.config.n_low_tokens}")
    print(f"    - n_high_tokens: {model.config.n_high_tokens}")
    
    print(f"\n{'='*80}")
    print(f"✓✓✓ COMPLETE MODIFIED PARGO MODEL SAVED SUCCESSFULLY! ✓✓✓")
    print(f"Output location: {os.path.abspath(output_dir)}")
    print(f"{'='*80}\n")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Test inference with: python test_inference_modified_pargo.py")
    print(f"  2. Evaluate on validation set")
    print(f"  3. Compare with baseline (MLP-Mixer) results")
    print(f"\n")

if __name__ == "__main__":
    main()