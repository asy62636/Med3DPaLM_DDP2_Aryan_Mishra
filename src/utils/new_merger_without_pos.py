import os
import torch
import transformers
from transformers import AutoTokenizer
from peft import PeftModel
from src.model.llm import VLMQwenForCausalLM
from dataclasses import dataclass, field
from typing import Optional, List

# Copy the EXACT ModelArguments from your training script
@dataclass
class ModelArguments:
    wb_name: Optional[str] = field(default="MLLM")
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")
    
    # Path to PEFT checkpoint - CHANGED TO WITHOUT POS
    peft_model_path: Optional[str] = field(
        default="./output2/Med3DVLM-Qwen-2.5-7B-ParGo-finetune-without-pos/checkpoint-174265"
    )

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(
        default=False,
        metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"},
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained mm_projector and embed_tokens."},
    )

    # image
    input_size: tuple = field(default=(256, 256, 128))
    patch_size: int = field(default=(16, 16, 16))
    dim: int = field(default=768)
    depth: int = field(default=12)

    # vision
    vision_tower: Optional[str] = field(default="dcformer")
    vision_select_layer: Optional[int] = field(default=-2)
    vision_select_feature: Optional[str] = field(default="cls_patch")
    pretrain_vision_model: str = field(
        default="./output2/DCFormer_SigLIP/pretrained_ViT.bin", 
        metadata={"help": "Path to pretrained model for ViT."}
    )
    pretrain_clip_model: str = field(
        default=None, metadata={"help": "Path to pretrained model for CLIP."}
    )
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default="pargo")
    mm_mlp_depth: int = field(
        default=2, metadata={"help": "Depth of MLP in projector."}
    )

    low_output_size: List[int] = field(
        default_factory=lambda: [192, 128],
        metadata={"help": "Output size of low feature."},
    )
    high_output_size: List[int] = field(
        default_factory=lambda: [64, 128],
        metadata={"help": "Output size of high feature."},
    )

    bert_type: str = field(default="bert-base-uncased")
    num_query_tokens: int = field(default=304)
    proj_out_num: int = field(default=304)

    use_positional_embedding: bool = field(default=False)  # FALSE for without-pos!
    pos_embed_dim: int = field(default=3)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = "./output2/Med3DVLM-Qwen-2.5-7B-ParGo-MERGED-without-pos"  # Changed output dir


def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    print("="*50)
    print("Merging WITHOUT Positional Embeddings")
    print("="*50)
    print("Tokenizer preparation")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )

    special_token = {"additional_special_tokens": ["<im_patch>"]}
    tokenizer.add_special_tokens(special_token)

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    print("vocab_size:", model_args.vocab_size)
    print("use_positional_embedding:", model_args.use_positional_embedding)

    print("="*50)
    print("Model preparation")
    print("="*50)
    
    # Set projector config (from training script)
    if model_args.mm_projector_type == "pargo":
        model_args.proj_out_num = model_args.num_query_tokens
        model_args.vision_width = 768

    # Load base model
    print(f"Loading base model from: {model_args.model_name_or_path}")
    model = VLMQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir
    )

    # Configure model (from training script)
    model.config.llm_hidden_size = 3584
    model.config.vision_hidden_size = 768
    model.config.bert_type = model_args.bert_type
    model.config.num_query_tokens = model_args.num_query_tokens
    model.config.proj_out_num = model_args.proj_out_num
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.vocab_size = model_args.vocab_size
    model.config.use_positional_embedding = model_args.use_positional_embedding  # FALSE
    model.config.pos_embed_dim = model_args.pos_embed_dim

    # Initialize vision modules
    print("Initializing vision modules...")
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)

    print("="*50)
    print("Loading PEFT checkpoint and merging")
    print("="*50)
    
    # Load the PEFT model
    print(f"Loading PEFT weights from: {model_args.peft_model_path}")
    model = PeftModel.from_pretrained(
        model,
        model_args.peft_model_path,
        is_trainable=False
    )
    
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    
    print("="*50)
    print("Fixing shared tensors in BERT")
    print("="*50)
    
    # Fix the shared tensor issue in BERT
    if hasattr(model.model, 'mm_projector') and hasattr(model.model.mm_projector, 'bert'):
        bert = model.model.mm_projector.bert
        print("Untying BERT shared weights...")
        
        # Untie the decoder weight and bias
        if hasattr(bert, 'cls') and hasattr(bert.cls, 'predictions'):
            # Clone the decoder weight to make it independent
            bert.cls.predictions.decoder.weight = torch.nn.Parameter(
                bert.cls.predictions.decoder.weight.clone()
            )
            # Clone the decoder bias to make it independent  
            bert.cls.predictions.decoder.bias = torch.nn.Parameter(
                bert.cls.predictions.decoder.bias.clone()
            )
            print("✓ BERT shared tensors untied")
    
    print("="*50)
    print("Saving merged model")
    print("="*50)
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    print(f"Saving to: {training_args.output_dir}")
    
    # Save everything with safe_serialization
    model.config.save_pretrained(training_args.output_dir)
    print("Saving model weights (this may take a few minutes)...")
    model.save_pretrained(
        training_args.output_dir,
        safe_serialization=True,  # Use safetensors
        max_shard_size="5GB"  # Split into multiple files
    )
    tokenizer.save_pretrained(training_args.output_dir)

    # Save vision tower separately
    print("Saving vision tower separately...")
    vision_tower = model.get_model().vision_tower.state_dict()
    torch.save(vision_tower, os.path.join(training_args.output_dir, "vision_tower.bin"))

    print("="*50)
    print("✅ Merge completed successfully!")
    print(f"✅ Merged model (WITHOUT pos embeddings) saved to: {training_args.output_dir}")
    
    # Check output size
    import glob
    safetensor_files = glob.glob(os.path.join(training_args.output_dir, "*.safetensors"))
    if safetensor_files:
        total_size = sum(os.path.getsize(f) for f in safetensor_files)
        print(f"✅ Total model size: {total_size / 1e9:.2f} GB")
    print("="*50)


if __name__ == "__main__":
    main()