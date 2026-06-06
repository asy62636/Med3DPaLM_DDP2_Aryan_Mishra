# # # # save_complete_model.py
# # # import os
# # # import torch
# # # from transformers import AutoTokenizer
# # # from src.model.llm.qwen import VLMQwenForCausalLM
# # # from types import SimpleNamespace

# # # def main():
# # #     # Paths
# # #     checkpoint_dir = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/TEST_SAVE_CHECK_20250925_180125/checkpoint-10"
# # #     output_dir = "./output/Med3DVLM-Qwen-2.5-7B-ParGo-Complete"
    
# # #     # Load tokenizer
# # #     tokenizer = AutoTokenizer.from_pretrained(
# # #         checkpoint_dir,
# # #         padding_side="right",
# # #         use_fast=False,
# # #     )
    
# # #     # Model arguments - include ALL required attributes
# # #     model_args = SimpleNamespace(
# # #         model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
# # #         vision_tower="dcformer",
# # #         vision_select_layer=-2,
# # #         vision_select_feature="cls_patch",
# # #         pretrain_vision_model=None,
# # #         pretrain_clip_model=None,
# # #         freeze_vision_tower=False,
# # #         mm_projector_type="pargo",
# # #         bert_type="bert-base-uncased",
# # #         num_query_tokens=304,
# # #         proj_out_num=304,
# # #         img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
# # #         vocab_size=len(tokenizer),
# # #         dim=768,
# # #         depth=12,
# # #         input_size=(256, 256, 128),
# # #         patch_size=(16, 16, 16),
# # #         num_new_tokens=1,
# # #         # Add these MLP-related attributes (won't be used for ParGo but needed by the code)
# # #         mm_mlp_depth=2,
# # #         proj_layer_type="mlp",
# # #         proj_layer_num=2,
# # #         proj_pooling_type="spatial",
# # #         proj_pooling_size=2,
# # #         proj_residual=False,
# # #         low_output_size=[192, 128],
# # #         high_output_size=[64, 128],
# # #         pretrain_mm_mlp_adapter=None,
# # #         tune_mm_mlp_adapter = None
# # #     )
    
# # #     print("Loading base Qwen model...")
# # #     model = VLMQwenForCausalLM.from_pretrained(
# # #         "Qwen/Qwen2.5-7B-Instruct",
# # #         ignore_mismatched_sizes=True
# # #     )
    
# # #     print("Initializing vision modules...")
# # #     model.get_model().initialize_vision_modules(model_args=model_args)
# # #     model.initialize_vision_tokenizer(model_args, tokenizer)
    
# # #     # Resize embeddings to match your trained tokenizer
# # #     model.resize_token_embeddings(len(tokenizer))
    
# # #     print("Loading trained weights...")
# # #     trained_weights = torch.load(
# # #         os.path.join(checkpoint_dir, "complete_model.bin"),
# # #         map_location="cpu"
# # #     )
    
# # #     if "module" in trained_weights:
# # #         trained_weights = trained_weights["module"]
    
# # #     # Load only the trained components (projector, vision tower, embeddings)
# # #     model_dict = model.state_dict()
# # #     trained_components = {}
    
# # #     for key, value in trained_weights.items():
# # #         if any(component in key for component in ["mm_projector", "vision_tower", "embed_tokens", "lm_head"]):
# # #             if key in model_dict and model_dict[key].shape == value.shape:
# # #                 trained_components[key] = value
# # #                 print(f"Loading trained: {key[:50]}...")
    
# # #     # Update model with trained components
# # #     model_dict.update(trained_components)
# # #     model.load_state_dict(model_dict, strict=False)
    
# # #     print(f"Saving complete model to {output_dir}...")
# # #     os.makedirs(output_dir, exist_ok=True)
    
# # #     # Save config with correct architecture
# # #     model.config.architectures = ["VLMQwenForCausalLM"]
# # #     model.config.mm_projector_type = "pargo"
# # #     model.config.bert_type = "bert-base-uncased"
# # #     model.config.num_query_tokens = 304
# # #     model.config.proj_out_num = 304
# # #     model.config.save_pretrained(output_dir)
    
# # #     # Save the complete model
# # #     # model.save_pretrained(output_dir, safe_serialization=True)

# # #     try:
# # #         model.save_pretrained(output_dir, safe_serialization=False)
# # #     except Exception as e:
# # #         print("Save failed:", e)
    
# # #     # Save tokenizer
# # #     tokenizer.save_pretrained(output_dir)
    
# # #     # Verify the saved model size
# # #     import glob
# # #     safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
# # #     total_size = sum(os.path.getsize(f) for f in safetensor_files)
# # #     print(f"Total model size: {total_size / 1e9:.2f} GB")
    
# # #     print("Complete model saved successfully!")

# # # if __name__ == "__main__":
# # #     main()


# # # save_complete_model_safetensors.py
# # # save_complete_model_safetensors.py
# # import os
# # import torch
# # from transformers import AutoTokenizer
# # from src.model.llm.qwen import VLMQwenForCausalLM
# # from types import SimpleNamespace
# # import copy

# # def main():
# #     # Paths
# #     checkpoint_dir = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/Pargo_without_pos_embedding/checkpoint-78354"
# #     output_dir = "./output/Med3DVLM-Qwen-2.5-7B-ParGo-Complete-Safe-Pargo-without-pos-embedding"
    
# #     # Load tokenizer
# #     tokenizer = AutoTokenizer.from_pretrained(
# #         checkpoint_dir,
# #         padding_side="right",
# #         use_fast=False,
# #     )
    
# #     # Model arguments
# #     model_args = SimpleNamespace(
# #         model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
# #         vision_tower="dcformer",
# #         vision_select_layer=-2,
# #         vision_select_feature="cls_patch",
# #         pretrain_vision_model=None,
# #         pretrain_clip_model=None,
# #         freeze_vision_tower=False,
# #         mm_projector_type="pargo",
# #         bert_type="bert-base-uncased",
# #         num_query_tokens=304,
# #         proj_out_num=304,
# #         img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
# #         vocab_size=len(tokenizer),
# #         dim=768,
# #         depth=12,
# #         input_size=(256, 256, 128),
# #         patch_size=(16, 16, 16),
# #         num_new_tokens=1,
# #         mm_mlp_depth=2,
# #         proj_layer_type="mlp",
# #         proj_layer_num=2,
# #         proj_pooling_type="spatial",
# #         proj_pooling_size=2,
# #         proj_residual=False,
# #         low_output_size=[192, 128],
# #         high_output_size=[64, 128],
# #         pretrain_mm_mlp_adapter=None,
# #         tune_mm_mlp_adapter=None
# #     )
    
# #     print("Loading base Qwen model...")
# #     model = VLMQwenForCausalLM.from_pretrained(
# #         "Qwen/Qwen2.5-7B-Instruct",
# #         ignore_mismatched_sizes=True
# #     )
    
# #     print("Initializing vision modules...")
# #     model.get_model().initialize_vision_modules(model_args=model_args)
# #     model.initialize_vision_tokenizer(model_args, tokenizer)
    
# #     # Resize embeddings
# #     model.resize_token_embeddings(len(tokenizer))
    
# #     print("Loading trained weights...")
# #     trained_weights = torch.load(
# #         os.path.join(checkpoint_dir, "complete_model.bin"),
# #         map_location="cpu"
# #     )
    
# #     if "module" in trained_weights:
# #         trained_weights = trained_weights["module"]
    
# #     # Load trained components
# #     model_dict = model.state_dict()
# #     trained_components = {}
    
# #     for key, value in trained_weights.items():
# #         if any(component in key for component in ["mm_projector", "vision_tower", "embed_tokens", "lm_head"]):
# #             if key in model_dict and model_dict[key].shape == value.shape:
# #                 trained_components[key] = value
# #                 print(f"Loading trained: {key[:50]}...")
    
# #     model_dict.update(trained_components)
# #     model.load_state_dict(model_dict, strict=False)
    
# #     # Fix shared tensor issue by cloning
# #     print("Fixing shared tensors in BERT...")
# #     if hasattr(model.model, 'mm_projector') and hasattr(model.model.mm_projector, 'bert'):
# #         bert = model.model.mm_projector.bert
# #         if hasattr(bert, 'cls') and hasattr(bert.cls, 'predictions'):
# #             # Clone the shared tensors to break the sharing
# #             bert.cls.predictions.decoder.weight = torch.nn.Parameter(
# #                 bert.cls.predictions.decoder.weight.clone()
# #             )
# #             bert.cls.predictions.decoder.bias = torch.nn.Parameter(
# #                 bert.cls.predictions.decoder.bias.clone()
# #             )
    
# #     print(f"Saving complete model to {output_dir}...")
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # Update config
# #     model.config.architectures = ["VLMQwenForCausalLM"]
# #     model.config.mm_projector_type = "pargo"
# #     model.config.vision_tower = "dcformer"
# #     model.config.bert_type = "bert-base-uncased"
# #     model.config.num_query_tokens = 304
# #     model.config.proj_out_num = 304
# #     model.config.vision_select_layer = -2
# #     model.config.vision_select_feature = "cls_patch"
# #     model.config.save_pretrained(output_dir)
    
# #     # Save the model in safetensors format with sharding
# #     print("Saving model in safetensors format...")
# #     model.save_pretrained(
# #         output_dir, 
# #         safe_serialization=True,
# #         max_shard_size="5GB"
# #     )
    
# #     # Save tokenizer
# #     tokenizer.save_pretrained(output_dir)
    
# #     # Verify the saved files
# #     import glob
# #     safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
# #     print(f"Created {len(safetensor_files)} safetensor files")
# #     total_size = sum(os.path.getsize(f) for f in safetensor_files)
# #     print(f"Total model size: {total_size / 1e9:.2f} GB")
    
# #     # Check for index file
# #     index_file = os.path.join(output_dir, "model.safetensors.index.json")
# #     if os.path.exists(index_file):
# #         print("✓ Index file created successfully")
    
# #     print("Complete model saved successfully in safetensors format!")

# # if __name__ == "__main__":
# #     main()

# import os
# import torch
# from transformers import AutoTokenizer
# from src.model.llm.qwen import VLMQwenForCausalLM
# from types import SimpleNamespace
# import glob

# def main():
#     # Paths
#     base_checkpoint = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/Pargo_without_pos_embedding/checkpoint-78354"
#     consolidated_checkpoint = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/Pargo_without_pos_embedding/checkpoint-78354-consolidated"
#     output_dir = "./output/Med3DVLM-Qwen-2.5-7B-ParGo-Complete-Pargo-without-pos-embedding"
    
#     # Load tokenizer from base checkpoint
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_checkpoint,
#         padding_side="right",
#         use_fast=False,
#     )
    
#     # Model arguments
#     model_args = SimpleNamespace(
#         model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
#         vision_tower="dcformer",
#         vision_select_layer=-2,
#         vision_select_feature="cls_patch",
#         pretrain_vision_model=None,
#         pretrain_clip_model=None,
#         freeze_vision_tower=False,
#         mm_projector_type="pargo",
#         bert_type="bert-base-uncased",
#         num_query_tokens=304,
#         proj_out_num=304,
#         img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
#         vocab_size=len(tokenizer),
#         dim=768,
#         depth=12,
#         input_size=(256, 256, 128),
#         patch_size=(16, 16, 16),
#         num_new_tokens=1,
#         mm_mlp_depth=2,
#         proj_layer_type="mlp",
#         proj_layer_num=2,
#         proj_pooling_type="spatial",
#         proj_pooling_size=2,
#         proj_residual=False,
#         low_output_size=[192, 128],
#         high_output_size=[64, 128],
#         pretrain_mm_mlp_adapter=None,
#         tune_mm_mlp_adapter=None
#     )
    
#     print("Loading base Qwen model...")
#     model = VLMQwenForCausalLM.from_pretrained(
#         "Qwen/Qwen2.5-7B-Instruct",
#         ignore_mismatched_sizes=True
#     )
    
#     print("Initializing vision modules...")
#     model.get_model().initialize_vision_modules(model_args=model_args)
#     model.initialize_vision_tokenizer(model_args, tokenizer)
    
#     # Resize embeddings
#     model.resize_token_embeddings(len(tokenizer))
    
#     print("Loading consolidated checkpoint...")
#     # Load all .bin shards from consolidated checkpoint
#     consolidated_state_dict = {}
    
#     shard_files = sorted(glob.glob(os.path.join(consolidated_checkpoint, "pytorch_model-*.bin")))
    
#     if shard_files:
#         print(f"Found {len(shard_files)} checkpoint shards")
#         for shard_file in shard_files:
#             print(f"Loading {os.path.basename(shard_file)}...")
#             shard_dict = torch.load(shard_file, map_location="cpu")
#             consolidated_state_dict.update(shard_dict)
#     else:
#         # Fallback to single file
#         single_file = os.path.join(consolidated_checkpoint, "pytorch_model.bin")
#         if os.path.exists(single_file):
#             print("Loading single pytorch_model.bin...")
#             consolidated_state_dict = torch.load(single_file, map_location="cpu")
#         else:
#             raise FileNotFoundError("No checkpoint files found!")
    
#     print(f"Loaded {len(consolidated_state_dict)} parameters from consolidated checkpoint")
    
#     # Load the state dict into model
#     model_dict = model.state_dict()
#     loaded_count = 0
#     skipped_count = 0
    
#     for key, value in consolidated_state_dict.items():
#         if key in model_dict:
#             if model_dict[key].shape == value.shape:
#                 model_dict[key] = value
#                 loaded_count += 1
#             else:
#                 print(f"Shape mismatch for {key}: model={model_dict[key].shape}, checkpoint={value.shape}")
#                 skipped_count += 1
#         else:
#             print(f"Key not in model: {key[:70]}")
#             skipped_count += 1
    
#     missing, unexpected = model.load_state_dict(model_dict, strict=False)
    
#     print(f"\n{'='*60}")
#     print(f"Loaded {loaded_count} parameters")
#     print(f"Skipped {skipped_count} parameters")
#     if missing:
#         print(f"Missing keys: {len(missing)}")
#     if unexpected:
#         print(f"Unexpected keys: {len(unexpected)}")
#     print(f"{'='*60}\n")
    
#     # Fix shared tensor issue
#     print("Fixing shared tensors in BERT...")
#     if hasattr(model.model, 'mm_projector') and hasattr(model.model.mm_projector, 'bert'):
#         bert = model.model.mm_projector.bert
#         if hasattr(bert, 'cls') and hasattr(bert.cls, 'predictions'):
#             bert.cls.predictions.decoder.weight = torch.nn.Parameter(
#                 bert.cls.predictions.decoder.weight.clone()
#             )
#             bert.cls.predictions.decoder.bias = torch.nn.Parameter(
#                 bert.cls.predictions.decoder.bias.clone()
#             )
#             print("✓ Fixed BERT shared tensors")
    
#     print(f"\nSaving complete model to {output_dir}...")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Update config
#     model.config.architectures = ["VLMQwenForCausalLM"]
#     model.config.mm_projector_type = "pargo"
#     model.config.vision_tower = "dcformer"
#     model.config.bert_type = "bert-base-uncased"
#     model.config.num_query_tokens = 304
#     model.config.proj_out_num = 304
#     model.config.vision_select_layer = -2
#     model.config.vision_select_feature = "cls_patch"
#     model.config.save_pretrained(output_dir)
    
#     # Save the model in safetensors format
#     print("Saving model in safetensors format...")
#     model.save_pretrained(
#         output_dir, 
#         safe_serialization=True,
#         max_shard_size="5GB"
#     )
    
#     # Save tokenizer
#     tokenizer.save_pretrained(output_dir)
    
#     # Verify
#     safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
#     print(f"\n{'='*60}")
#     print(f"✓ Created {len(safetensor_files)} safetensor files")
#     total_size = sum(os.path.getsize(f) for f in safetensor_files)
#     print(f"✓ Total model size: {total_size / 1e9:.2f} GB")
    
#     index_file = os.path.join(output_dir, "model.safetensors.index.json")
#     if os.path.exists(index_file):
#         print("✓ Index file created successfully")
    
#     print(f"{'='*60}")
#     print("Complete model saved successfully!")
#     print(f"Output: {output_dir}")
#     print(f"{'='*60}\n")

# if __name__ == "__main__":
#     main()

import os
import torch
from transformers import AutoTokenizer
from src.model.llm.qwen import VLMQwenForCausalLM
from types import SimpleNamespace
import glob
from safetensors.torch import load_file

def main():
    print("="*80)
    print("STARTING MODEL CONSOLIDATION AND SAVING PROCESS")
    print("="*80)
    
    # Paths - UPDATED FOR YOUR NEW MODEL
    base_checkpoint = "/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Pargo_with_pos_embedding/checkpoint-78354"
    output_dir = "./output/Med3DVLM-Qwen-2.5-7B-ParGo-Complete-Pargo-with-pos-embedding"
    
    print(f"\n[1/9] Configuration")
    print(f"  Base checkpoint: {base_checkpoint}")
    print(f"  Output directory: {output_dir}")
    
    # Verify checkpoint exists
    if not os.path.exists(base_checkpoint):
        print(f"\n❌ ERROR: Checkpoint directory does not exist!")
        print(f"  Path: {base_checkpoint}")
        return
    
    print(f"  ✓ Checkpoint directory exists")
    
    # Check for required files
    required_files = ["config.json", "tokenizer_config.json", "vocab.json"]
    print(f"  Verifying required files...")
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
    
    # Check for safetensor files
    shard_files = glob.glob(os.path.join(base_checkpoint, "model-*.safetensors"))
    if not shard_files:
        print(f"\n❌ ERROR: No safetensor files found in checkpoint!")
        return
    print(f"  ✓ Found {len(shard_files)} safetensor files")
    
    # Load tokenizer
    print(f"\n[2/9] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_checkpoint,
            padding_side="right",
            use_fast=False,
        )
        print(f"  ✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"  Special token <im_patch> ID: {tokenizer.convert_tokens_to_ids('<im_patch>')}")
    except Exception as e:
        print(f"❌ ERROR loading tokenizer: {e}")
        return
    
    # Model arguments
    print(f"\n[3/9] Setting up model arguments...")
    model_args = SimpleNamespace(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
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
        tune_mm_mlp_adapter=None,
        use_positional_embedding=False  # No pos embedding for this model
    )
    print(f"  ✓ Model arguments configured")
    print(f"  Projector type: {model_args.mm_projector_type}")
    print(f"  Vision tower: {model_args.vision_tower}")
    print(f"  Use positional embedding: {model_args.use_positional_embedding}")
    
    print(f"\n[4/9] Loading base Qwen model from HuggingFace...")
    model = VLMQwenForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        ignore_mismatched_sizes=True
    )
    print(f"  ✓ Base Qwen model loaded")
    
    print(f"\n[5/9] Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args)
    print(f"  ✓ Vision modules initialized")
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
    
    print(f"  Found {len(shard_files)} safetensor shards:")
    for idx, shard_file in enumerate(shard_files, 1):
        print(f"    [{idx}/{len(shard_files)}] Loading {os.path.basename(shard_file)}...")
        shard_dict = load_file(shard_file)
        print(f"         → Loaded {len(shard_dict)} parameters from this shard")
        consolidated_state_dict.update(shard_dict)
    
    print(f"  ✓ Total parameters loaded from checkpoint: {len(consolidated_state_dict)}")
    
    # Print some sample keys
    print(f"\n  Sample parameter keys:")
    for i, key in enumerate(list(consolidated_state_dict.keys())[:5]):
        print(f"    - {key}: shape {consolidated_state_dict[key].shape}")
    
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
    if missing:
        print(f"  Missing keys in checkpoint: {len(missing)}")
        print(f"  First few missing: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys in checkpoint: {len(unexpected)}")
        print(f"  First few unexpected: {unexpected[:5]}")
    print(f"{'='*60}\n")
    
    # Fix shared tensor issue
    print(f"[8/9] Fixing shared tensors in BERT...")
    if hasattr(model.model, 'mm_projector') and hasattr(model.model.mm_projector, 'bert'):
        bert = model.model.mm_projector.bert
        if hasattr(bert, 'cls') and hasattr(bert.cls, 'predictions'):
            bert.cls.predictions.decoder.weight = torch.nn.Parameter(
                bert.cls.predictions.decoder.weight.clone()
            )
            bert.cls.predictions.decoder.bias = torch.nn.Parameter(
                bert.cls.predictions.decoder.bias.clone()
            )
            print(f"  ✓ Fixed BERT shared tensors")
        else:
            print(f"  ℹ No BERT cls.predictions found")
    else:
        print(f"  ℹ No BERT module found in mm_projector")
    
    print(f"\n[9/9] Saving complete model...")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  ✓ Output directory created/verified")
    
    # Update config
    print(f"  Updating model config...")
    model.config.architectures = ["VLMQwenForCausalLM"]
    model.config.mm_projector_type = "pargo"
    model.config.vision_tower = "dcformer"
    model.config.bert_type = "bert-base-uncased"
    model.config.num_query_tokens = 304
    model.config.proj_out_num = 304
    model.config.vision_select_layer = -2
    model.config.vision_select_feature = "cls_patch"
    model.config.use_positional_embedding = False
    model.config.save_pretrained(output_dir)
    print(f"  ✓ Config saved")
    
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
    
    print(f"\n{'='*80}")
    print(f"✓✓✓ COMPLETE MODEL SAVED SUCCESSFULLY! ✓✓✓")
    print(f"Output location: {os.path.abspath(output_dir)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()