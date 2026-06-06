import os
import torch
import transformers
from transformers import AutoTokenizer
from src.model.llm import VLMQwenForCausalLM
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    wb_name: Optional[str] = field(default="MLLM")
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")
    
    # Path to your trained projector checkpoint
    trained_projector_path: Optional[str] = field(
        default="./output2/Pargo_with_modified_pos_embedding"
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

    use_positional_embedding: bool = field(default=True)  # Your new sinusoidal pos embed


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = "./output2/Pargo_modified_pos_COMPLETE"


def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    print("="*50)
    print("Combining Trained ParGo with Base LLM")
    print("="*50)
    print("Tokenizer preparation")
    print("="*50)
    
    # Load tokenizer from your trained checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.trained_projector_path,
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
    
    # Set projector config
    if model_args.mm_projector_type == "pargo":
        model_args.proj_out_num = model_args.num_query_tokens
        model_args.vision_width = 768

    # Load base Qwen model (this gives us the LLM)
    print(f"Loading base LLM from: {model_args.model_name_or_path}")
    model = VLMQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir
    )

    # Configure model
    model.config.llm_hidden_size = 3584
    model.config.vision_hidden_size = 768
    model.config.bert_type = model_args.bert_type
    model.config.num_query_tokens = model_args.num_query_tokens
    model.config.proj_out_num = model_args.proj_out_num
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.vocab_size = model_args.vocab_size
    model.config.use_positional_embedding = model_args.use_positional_embedding

    # Initialize vision modules (creates empty projector and vision tower)
    print("Initializing vision modules...")
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)

    print("="*50)
    print("Loading trained projector and vision weights")
    print("="*50)
    
    # Load mm_projector weights
    mm_projector_path = os.path.join(model_args.trained_projector_path, "mm_projector.bin")
    if os.path.exists(mm_projector_path):
        print(f"Loading mm_projector from: {mm_projector_path}")
        mm_projector_weights = torch.load(mm_projector_path, map_location="cpu")
        
        # Clean keys if needed
        cleaned_weights = {}
        for k, v in mm_projector_weights.items():
            new_key = k
            if new_key.startswith("model.mm_projector."):
                new_key = new_key.replace("model.mm_projector.", "", 1)
            elif new_key.startswith("mm_projector."):
                new_key = new_key.replace("mm_projector.", "", 1)
            cleaned_weights[new_key] = v
        
        missing, unexpected = model.get_model().mm_projector.load_state_dict(
            cleaned_weights, strict=False
        )
        print(f"✓ mm_projector loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    else:
        print(f"⚠ WARNING: mm_projector.bin not found at {mm_projector_path}")
    
    # Load vision encoder weights
    vision_encoder_path = os.path.join(model_args.trained_projector_path, "vision_encoder.bin")
    if os.path.exists(vision_encoder_path):
        print(f"Loading vision_encoder from: {vision_encoder_path}")
        vision_weights = torch.load(vision_encoder_path, map_location="cpu")
        
        # Clean keys if needed
        cleaned_vision = {}
        for k, v in vision_weights.items():
            new_key = k
            if new_key.startswith("model.vision_tower."):
                new_key = new_key.replace("model.vision_tower.", "", 1)
            elif new_key.startswith("vision_tower."):
                new_key = new_key.replace("vision_tower.", "", 1)
            cleaned_vision[new_key] = v
        
        missing, unexpected = model.get_model().vision_tower.load_state_dict(
            cleaned_vision, strict=False
        )
        print(f"✓ vision_tower loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    else:
        print(f"⚠ WARNING: vision_encoder.bin not found at {vision_encoder_path}")
    
    print("="*50)
    print("Fixing shared tensors in BERT")
    print("="*50)
    
    # Fix the shared tensor issue in BERT
    if hasattr(model.model, 'mm_projector') and hasattr(model.model.mm_projector, 'bert'):
        bert = model.model.mm_projector.bert
        print("Untying BERT shared weights...")
        
        if hasattr(bert, 'cls') and hasattr(bert.cls, 'predictions'):
            bert.cls.predictions.decoder.weight = torch.nn.Parameter(
                bert.cls.predictions.decoder.weight.clone()
            )
            bert.cls.predictions.decoder.bias = torch.nn.Parameter(
                bert.cls.predictions.decoder.bias.clone()
            )
            print("✓ BERT shared tensors untied")
    
    print("="*50)
    print("Saving complete model")
    print("="*50)
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    print(f"Saving to: {training_args.output_dir}")
    
    # Save everything
    model.config.save_pretrained(training_args.output_dir)
    print("Saving model weights (this may take a few minutes)...")
    model.save_pretrained(
        training_args.output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(training_args.output_dir)

    # Save vision tower separately
    print("Saving vision tower separately...")
    vision_tower = model.get_model().vision_tower.state_dict()
    torch.save(vision_tower, os.path.join(training_args.output_dir, "vision_tower.bin"))

    print("="*50)
    print("✅ Merge completed successfully!")
    print(f"✅ Complete model saved to: {training_args.output_dir}")
    
    # Check output size
    import glob
    safetensor_files = glob.glob(os.path.join(training_args.output_dir, "*.safetensors"))
    if safetensor_files:
        total_size = sum(os.path.getsize(f) for f in safetensor_files)
        print(f"✅ Total model size: {total_size / 1e9:.2f} GB (expected ~30-33 GB)")
    print("="*50)


if __name__ == "__main__":
    main()