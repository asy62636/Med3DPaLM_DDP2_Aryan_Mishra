import torch.nn as nn

from .mhsa import MultiHeadSelfAttention
from .mlp import LowHighHybridMLP, MixerLowHighHybridMLP, MultiModalProjector
from .pargo import ParGoProjector
from .modified_pargo import MultiScaleDecomposed3DParGo
from .single_scale_pargo import SingleScaleParGo

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "mlp":
        return MultiModalProjector(
            input_size=config.mm_hidden_size,
            output_size=config.hidden_size,
            mlp_depth=config.mm_mlp_depth,
            proj_out_num=config.proj_out_num,
        )
    elif projector_type == "low_high_mlp":
        return LowHighHybridMLP(
            low_input_size=config.low_input_size,
            high_input_size=config.high_input_size,
            output_size=config.hidden_size,
            mlp_depth=config.mm_mlp_depth,
            proj_out_num=config.proj_out_num,
        )
        
    elif projector_type == "pargo":
        # Ensure vision_hidden_size exists (you can derive it from config.dim)
        if not hasattr(config, "vision_hidden_size"):
            config.vision_hidden_size = getattr(config, "dim", 768)

        if not hasattr(config, "hidden_size"):
            config.hidden_size = 768  # or get from BERT config

        return ParGoProjector(config)

    elif projector_type == "modified_pargo":
        # Set default values for required config parameters
        # These will be overridden if you set them explicitly in train_vlm_2.py
        
        # Low-level feature dimension (DCFormer penultimate layer)
        if not hasattr(config, "low_level_hidden_size"):
            config.low_level_hidden_size = 384
        
        # High-level feature dimension (DCFormer final layer)
        if not hasattr(config, "vision_hidden_size"):
            config.vision_hidden_size = getattr(config, "dim", 768)
        
        # LLM hidden dimension (Qwen-7B)
        if not hasattr(config, "llm_hidden_size"):
            config.llm_hidden_size = getattr(config, "hidden_size", 3584)
        
        # Token allocation
        if not hasattr(config, "n_low_tokens"):
            config.n_low_tokens = 144
        if not hasattr(config, "n_high_tokens"):
            config.n_high_tokens = 32
        
        # Total output tokens (must match n_low_tokens + n_high_tokens)
        config.proj_out_num = config.n_low_tokens + config.n_high_tokens
        
        # 3D positional embeddings
        if not hasattr(config, "use_positional_embedding"):
            config.use_positional_embedding = True
        
        # Number of ParGo layers per scale
        if not hasattr(config, "pargo_num_layers"):
            config.pargo_num_layers = 6
        
        # Cross-scale attention (disable for first experiments)
        if not hasattr(config, "use_cross_scale_attention"):
            config.use_cross_scale_attention = False
        
        print("=" * 60)
        print("Initializing Multi-Scale Decomposed 3D ParGo Projector")
        print("=" * 60)
        print(f"Config parameters:")
        print(f"  Low-level hidden size: {config.low_level_hidden_size}")
        print(f"  High-level hidden size: {config.vision_hidden_size}")
        print(f"  LLM hidden size: {config.llm_hidden_size}")
        print(f"  Low-level tokens: {config.n_low_tokens}")
        print(f"  High-level tokens: {config.n_high_tokens}")
        print(f"  Total output tokens: {config.proj_out_num}")
        print(f"  Use positional embedding: {config.use_positional_embedding}")
        print(f"  ParGo layers per scale: {config.pargo_num_layers}")
        print(f"  Use cross-scale attention: {config.use_cross_scale_attention}")
        print("=" * 60)
        
        return MultiScaleDecomposed3DParGo(config)

    elif projector_type == "single_scale_pargo":
        if not hasattr(config, "vision_hidden_size"):
            config.vision_hidden_size = getattr(config, "dim", 768)
        if not hasattr(config, "llm_hidden_size"):
            config.llm_hidden_size = getattr(config, "hidden_size", 3584)
        
        num_global = getattr(config, "num_global_queries", 8)
        num_partial = getattr(config, "num_partial_queries", 24)
        config.proj_out_num = num_global + num_partial  # 32
        
        print("=" * 60)
        print("Initializing Single-Scale ParGo Projector (Experiment 2)")
        print("=" * 60)
        print(f"  vision_hidden_size : {config.vision_hidden_size}")
        print(f"  llm_hidden_size    : {config.llm_hidden_size}")
        print(f"  num_global_queries : {num_global}")
        print(f"  num_partial_queries: {num_partial}")
        print(f"  num_layers         : {getattr(config, 'pargo_num_layers', 2)}")
        print(f"  proj_out_num       : {config.proj_out_num}")
        print(f"  use_pretrained_bert: {getattr(config, 'use_pretrained_bert', True)}")
        print("=" * 60)
        
        return SingleScaleParGo(
            num_input_tokens=32,
            vision_hidden_size=config.vision_hidden_size,
            llm_hidden_size=config.llm_hidden_size,
            num_global_queries=num_global,
            num_partial_queries=num_partial,
            num_layers=getattr(config, "pargo_num_layers", 2),
            use_pretrained_bert=getattr(config, "use_pretrained_bert", True),
            dropout=getattr(config, "pargo_dropout", 0.0),
        )

    elif projector_type == "dual_scale_pargo":
        from .dual_scale_pargo import DualScaleParGo
        mm_projector = DualScaleParGo(
            # Low-level branch: penultimate layer features
            low_num_input_tokens=256,
            low_vision_hidden_size=384,
            low_num_global_queries=56,
            low_num_partial_queries=200,
            low_num_layers=2,
            # High-level branch: final layer features
            high_num_input_tokens=32,
            high_vision_hidden_size=768,
            high_num_global_queries=8,
            high_num_partial_queries=24,
            high_num_layers=2,
            # Shared
            hidden_size=768,
            llm_hidden_size=3584,  # Qwen2.5-7B
            bert_model_name="bert-base-uncased",
            use_pretrained_bert=True,
        )
        return mm_projector

    elif projector_type == "mixer":
        return MixerLowHighHybridMLP(
            low_input_size=config.low_input_size,
            low_output_size=config.low_output_size,
            high_input_size=config.high_input_size,
            high_output_size=config.high_output_size,
            output_dim=config.hidden_size,
            depth=len(config.low_output_size),
            mlp_depth=config.mm_mlp_depth,
            proj_out_num=config.proj_out_num,
        )
    elif projector_type == "mhsa":
        return MultiHeadSelfAttention(
            embed_dim=config.mm_hidden_size,
            output_dim=config.hidden_size,
            num_heads=hasattr(config, "num_heads") and config.num_heads or 8,
            proj_out_num=config.proj_out_num,
        )
    elif projector_type == "identity":
        return IdentityMap()
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
