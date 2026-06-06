import os
import torch
import glob
from transformers import AutoTokenizer
from src.model.llm.qwen import VLMQwenForCausalLM
from peft import LoraConfig, get_peft_model
from types import SimpleNamespace

def find_all_linear_names(model):
    """Find all linear layers for LoRA (exclude vision/projector)"""
    cls = torch.nn.Linear
    lora_module_names = set()
    ignore_keywords = ["vision_tower", "mm_projector", "embed_tokens", "lm_head"]
    
    for name, module in model.named_modules():
        if any(kw in name for kw in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    
    return list(lora_module_names)

def main():
    print("="*80)
    print("MERGING LORA - Following Med3DVLM Approach Exactly")
    print("="*80)
    
    # Paths
    lora_checkpoint_dir = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Med3DVLM-Qwen-2.5-7B-ParGo-finetune-with-pos"
    pytorch_full = os.path.join(lora_checkpoint_dir, "pytorch_model_full.bin")
    output_dir = "./output3/Med3DVLM-Qwen-2.5-7B-ParGo-Finetuned-Complete-with-pos"
    
    print(f"\nPaths:")
    print(f"  LoRA checkpoint: {lora_checkpoint_dir}")
    print(f"  Full state dict: {pytorch_full}")
    print(f"  Output: {output_dir}")
    
    if not os.path.exists(pytorch_full):
        print("\n❌ pytorch_model_full.bin not found!")
        return
    
    # Step 1: Tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_checkpoint_dir,
        padding_side="right",
        use_fast=False,
    )
    
    # Add special tokens (important!)
    special_token = {"additional_special_tokens": ["<im_patch>"]}
    tokenizer.add_special_tokens(special_token)
    
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    print(f"  ✓ Tokenizer loaded (vocab: {len(tokenizer)})")
    
    # Step 2: Model arguments
    print("\n[2/6] Setting up model arguments...")
    model_args = SimpleNamespace(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        model_type="vlm_qwen",
        vision_tower="dcformer",
        vision_select_layer=-2,
        vision_select_feature="cls_patch",
        pretrain_vision_model=None,
        pretrain_clip_model=None,
        freeze_vision_tower=False,
        mm_projector_type="pargo",
        bert_type="bert-base-uncased",
        num_query_tokens=304,
        proj_out_num=304,
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
        tune_mm_mlp_adapter=False,
        use_positional_embedding=False,
    )
    print("  ✓ Model arguments set")
    
    # Step 3: Load base model
    print("\n[3/6] Loading base Qwen model...")
    model = VLMQwenForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float32,  # Use float32 for merging
    )
    print("  ✓ Base Qwen loaded")
    
    # Step 4: Initialize vision modules
    print("\n[4/6] Initializing vision modules...")
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    
    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)
    print("  ✓ Vision modules initialized")
    
    # Step 5: Apply PEFT LoRA wrapper (CRITICAL!)
    print("\n[5/6] Applying PEFT LoRA wrapper...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("  Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    print("  ✓ PEFT LoRA wrapper applied")
    model.print_trainable_parameters()
    
    # Step 6: Load the full state dict (with LoRA weights)
    print("\n[6/6] Loading weights from pytorch_model_full.bin...")
    state_dict = torch.load(pytorch_full, map_location="cpu")
    print(f"  Loaded {len(state_dict)} keys from checkpoint")
    
    # Check what's in it
    vision_keys = [k for k in state_dict.keys() if "vision" in k.lower()]
    projector_keys = [k for k in state_dict.keys() if "mm_projector" in k or "projector" in k]
    lora_keys = [k for k in state_dict.keys() if "lora" in k.lower()]
    
    print(f"  Contents:")
    print(f"    Vision keys: {len(vision_keys)}")
    print(f"    Projector keys: {len(projector_keys)}")
    print(f"    LoRA keys: {len(lora_keys)}")
    
    # Load state dict
    print("\n  Loading state dict into model...")
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  ✓ Loaded with strict=True")
    except Exception as e:
        print(f"  Strict loading failed: {e}")
        print("  Trying with strict=False...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
        if missing:
            print(f"    Missing keys (first 10): {missing[:10]}")
    
    # Merge and unload LoRA
    print("\n  Merging LoRA weights...")
    model = model.merge_and_unload()
    print("  ✓ LoRA merged and unloaded")
    
    # Get final state dict
    state_dict = model.state_dict()
    print(f"  Final state dict has {len(state_dict)} keys")
    
    # Verify vision/projector are present
    final_vision = [k for k in state_dict.keys() if "vision" in k.lower()]
    final_projector = [k for k in state_dict.keys() if "mm_projector" in k]
    print(f"  Final vision keys: {len(final_vision)}")
    print(f"  Final projector keys: {len(final_projector)}")
    
    if len(final_vision) == 0 or len(final_projector) == 0:
        print("\n  ❌ ERROR: Vision or projector missing from final model!")
        return
    
    # Fix shared tensors
    print("\n  Fixing BERT shared tensors...")
    if hasattr(model.model, 'mm_projector') and hasattr(model.model.mm_projector, 'bert'):
        bert = model.model.mm_projector.bert
        if hasattr(bert, 'cls') and hasattr(bert.cls, 'predictions'):
            bert.cls.predictions.decoder.weight = torch.nn.Parameter(
                bert.cls.predictions.decoder.weight.clone()
            )
            bert.cls.predictions.decoder.bias = torch.nn.Parameter(
                bert.cls.predictions.decoder.bias.clone()
            )
            print("  ✓ Fixed")
    
    # Save
    print(f"\n  Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config
    model.model.config.architectures = [model.__class__.__name__]
    model._name_or_path = output_dir
    model.config.mm_projector_type = "pargo"
    model.config.vision_tower = "dcformer"
    model.config.bert_type = "bert-base-uncased"
    model.config.num_query_tokens = 304
    model.config.proj_out_num = 304
    model.config.vision_select_layer = -2
    model.config.vision_select_feature = "cls_patch"
    model.config.use_positional_embedding = False
    
    print("  Saving config...")
    model.config.save_pretrained(output_dir)
    
    print("  Saving model (this will take several minutes)...")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    
    print("  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    
    # Also save vision tower separately (like Med3DVLM does)
    print("  Saving vision tower separately...")
    vision_tower = model.get_model().vision_tower.state_dict()
    torch.save(vision_tower, os.path.join(output_dir, "vision_tower.bin"))
    
    # Verify
    print("\n" + "="*80)
    print("VERIFICATION:")
    safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)
    
    print(f"  ✓ Created {len(safetensor_files)} safetensor files")
    for sf in sorted(safetensor_files):
        print(f"    - {os.path.basename(sf)}: {os.path.getsize(sf) / 1e9:.2f} GB")
    
    print(f"\n  ✓ Total: {total_size / 1e9:.2f} GB (expected ~30-33 GB)")
    
    # Check index
    import json
    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        mm_proj_keys = [k for k in index["weight_map"].keys() if "mm_projector" in k]
        vision_keys = [k for k in index["weight_map"].keys() if "vision" in k]
        print(f"\n  Index file check:")
        print(f"    mm_projector keys: {len(mm_proj_keys)}")
        print(f"    vision_tower keys: {len(vision_keys)}")
    
    print("="*80)
    print("✓✓✓ MERGE COMPLETE!")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("="*80)

if __name__ == "__main__":
    main()