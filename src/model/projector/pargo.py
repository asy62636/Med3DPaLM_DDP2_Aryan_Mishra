# import torch
# import torch.nn as nn
# from .qformer_bert import BertLMHeadModel, BertConfig

# class ParGoProjector(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         # Number of output tokens
#         self.proj_out_num = int(getattr(config, "proj_out_num", 304))
        
#         # Vision and LLM dimensions
#         vision_width = int(getattr(config, "vision_hidden_size", 768))
#         llm_hidden_size = int(getattr(config, "llm_hidden_size", 3584))
#         self.use_pos_embed = getattr(config, "use_positional_embedding", False)
#         self.pos_embed_dim = getattr(config, "pos_embed_dim", 3)
        
#         # Load BERT config
#         bert_cfg = BertConfig.from_pretrained(config.bert_type)
#         bert_hidden = bert_cfg.hidden_size  # 768
        
#         # Vision to BERT projection
#         input_dim = vision_width + self.pos_embed_dim if self.use_pos_embed else vision_width
#         self.vision_proj = nn.Linear(input_dim, bert_hidden)
        
#         # BERT to LLM projection  
#         self.output_proj = nn.Linear(bert_hidden, llm_hidden_size)
        
#         # Configure BERT for cross-attention
#         bert_cfg.add_cross_attention = True
#         bert_cfg.cross_attention_freq = 2
#         bert_cfg.encoder_width = bert_hidden
#         bert_cfg.query_length = self.proj_out_num
#         bert_cfg.use_cache = False
#         bert_cfg.local_query_length = [0]  # FIX: Add this line
        
#         # Initialize BERT
#         self.bert = BertLMHeadModel.from_pretrained(
#             config.bert_type, 
#             config=bert_cfg
#         )
        
#         # Resize token embeddings if needed
#         if hasattr(config, "vocab_size") and config.vocab_size:
#             self.bert.resize_token_embeddings(config.vocab_size)
        
#         # Learnable query tokens
#         self.query_tokens = nn.Parameter(
#             torch.zeros(1, self.proj_out_num, bert_hidden)
#         )
#         nn.init.normal_(self.query_tokens, std=0.02)

#     def generate_3d_positional_embedding(self, B, N, D=2, H=4, W=4, device='cuda'):
#         """
#         Generate normalized 3D positional embeddings
#         B: batch size
#         N: number of tokens (should be D*H*W)
#         """
#         assert N == D * H * W, f"Expected {D*H*W} tokens, got {N}"
        
#         # Generate 3D grid coordinates
#         d_coords = torch.arange(D, device=device).view(-1, 1, 1).expand(D, H, W)
#         h_coords = torch.arange(H, device=device).view(1, -1, 1).expand(D, H, W)
#         w_coords = torch.arange(W, device=device).view(1, 1, -1).expand(D, H, W)
        
#         # Flatten to match feature ordering
#         d_coords = d_coords.reshape(-1, 1).float() / D  # Normalize to [0, 1]
#         h_coords = h_coords.reshape(-1, 1).float() / H
#         w_coords = w_coords.reshape(-1, 1).float() / W
        
#         # Concatenate positional info (N, 3)
#         pos_embedding = torch.cat([d_coords, h_coords, w_coords], dim=1)
        
#         # Expand for batch (B, N, 3)
#         pos_embedding = pos_embedding.unsqueeze(0).expand(B, -1, -1)
        
#         return pos_embedding
        
#     def forward(self, visual_features, spatial_dims = None):
#         if spatial_dims is None:
#             spatial_dims = (2, 4, 4) 

#         # Handle list/tuple input - take last layer features
#         if isinstance(visual_features, (list, tuple)):
#             visual_features = visual_features[-1]
#         print("*"*20 + "Incoming visual features shape = ", visual_features.shape)
#         B, N, D = visual_features.shape
        
#         if self.use_pos_embed: # Default DCFormer output dimensions
#             pos_embed = self.generate_3d_positional_embedding(
#                 B, N, 
#                 D=spatial_dims[0], 
#                 H=spatial_dims[1], 
#                 W=spatial_dims[2],
#                 device=visual_features.device
#             )
#             pos_embed = pos_embed.to(visual_features.dtype)
#             print(f"Positional embedding shape: {pos_embed.shape}")
            
#             # Concatenate along feature dimension
#             visual_features = torch.cat([visual_features, pos_embed], dim=-1)  # (B, N, 768+3)
#             print(f"Features after pos concat: {visual_features.shape}")

#         # Project vision features to BERT space
#         print(visual_features.dtype)
#         visual_features = self.vision_proj(visual_features)  # (B, N, bert_hidden)
        
#         # Expand query tokens
#         query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, proj_out_num, bert_hidden)
        
#         # Create attention mask for visual tokens
#         encoder_mask = torch.ones(
#             B, N, 
#             dtype=torch.long, 
#             device=visual_features.device
#         )
        
#         # BERT cross-attention
#         outputs = self.bert.bert(
#             input_ids=None,
#             attention_mask=None,
#             query_embeds=query_tokens,
#             encoder_hidden_states=visual_features,
#             encoder_attention_mask=encoder_mask,
#             return_dict=True,
#             is_decoder=True,
#         )
        
#         # Extract query outputs
#         query_output = outputs.last_hidden_state[:, :self.proj_out_num, :]
        
#         # Project to LLM space
#         image_features = self.output_proj(query_output)  # (B, proj_out_num, llm_hidden)
        
#         return image_features
    
#     def load_state_dict(self, state_dict, strict=False):
#         """Flexible state dict loading"""
#         # Filter out incompatible keys
#         own_state = self.state_dict()
#         compatible_state = {}
        
#         for k, v in state_dict.items():
#             if k in own_state and own_state[k].shape == v.shape:
#                 compatible_state[k] = v
#             else:
#                 print(f"Skipping incompatible key: {k}")
        
#         return super().load_state_dict(compatible_state, strict=False)

import torch
import torch.nn as nn
import math
from .qformer_bert import BertLMHeadModel, BertConfig

class ParGoProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Number of output tokens
        self.proj_out_num = int(getattr(config, "proj_out_num", 304))
        
        # Vision and LLM dimensions
        vision_width = int(getattr(config, "vision_hidden_size", 768))
        llm_hidden_size = int(getattr(config, "llm_hidden_size", 3584))
        self.use_pos_embed = getattr(config, "use_positional_embedding", False)
        
        # Load BERT config
        bert_cfg = BertConfig.from_pretrained(config.bert_type)
        bert_hidden = bert_cfg.hidden_size  # 768
        
        # Vision to BERT projection (no longer needs pos_embed_dim)
        self.vision_proj = nn.Linear(vision_width, bert_hidden)
        
        # BERT to LLM projection  
        self.output_proj = nn.Linear(bert_hidden, llm_hidden_size)
        
        # Configure BERT for cross-attention
        bert_cfg.add_cross_attention = True
        bert_cfg.cross_attention_freq = 2
        bert_cfg.encoder_width = bert_hidden
        bert_cfg.query_length = self.proj_out_num
        bert_cfg.use_cache = False
        bert_cfg.local_query_length = [0]
        
        # Initialize BERT
        self.bert = BertLMHeadModel.from_pretrained(
            config.bert_type, 
            config=bert_cfg
        )
        
        # Resize token embeddings if needed
        if hasattr(config, "vocab_size") and config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.proj_out_num, bert_hidden)
        )
        nn.init.normal_(self.query_tokens, std=0.02)

    def generate_3d_sinusoidal_embedding(self, B, N, D=2, H=4, W=4, embed_dim=768, device='cuda'):
        """
        Generate 3D sinusoidal positional embeddings following Transformer convention.
        
        Args:
            B: batch size
            N: number of tokens (should be D*H*W)
            D, H, W: depth, height, width of the 3D grid
            embed_dim: embedding dimension (768)
            device: torch device
            
        Returns:
            pos_embedding: (B, N, embed_dim) tensor
        """
        assert N == D * H * W, f"Expected {D*H*W} tokens, got {N}"
        assert embed_dim % 3 == 0, f"embed_dim must be divisible by 3, got {embed_dim}"
        
        # Dimensions per axis
        dim_per_axis = embed_dim // 3  # 256 for each of x, y, z
        assert dim_per_axis % 2 == 0, "dim_per_axis must be even"
        
        # Generate 3D grid coordinates
        d_coords = torch.arange(D, device=device).view(-1, 1, 1).expand(D, H, W)
        h_coords = torch.arange(H, device=device).view(1, -1, 1).expand(D, H, W)
        w_coords = torch.arange(W, device=device).view(1, 1, -1).expand(D, H, W)
        
        # Flatten to match feature ordering (N,)
        x_pos = w_coords.reshape(-1).float()  # Width
        y_pos = h_coords.reshape(-1).float()  # Height
        z_pos = d_coords.reshape(-1).float()  # Depth
        
        # Normalize to [0, 1]
        x_pos = x_pos / max(W - 1, 1)
        y_pos = y_pos / max(H - 1, 1)
        z_pos = z_pos / max(D - 1, 1)
        
        # Initialize positional embedding tensor
        pos_embedding = torch.zeros(N, embed_dim, device=device)
        
        # Number of frequency bands (sine/cosine pairs)
        num_bands = dim_per_axis // 2  # 128
        
        # Generate frequency bands
        # freq = 1 / (10000 ^ (2i / dim_per_axis))
        div_term = torch.exp(torch.arange(0, dim_per_axis, 2, device=device).float() * 
                             -(math.log(10000.0) / dim_per_axis))
        
        # X axis encoding (dimensions 0:256)
        pos_embedding[:, 0:dim_per_axis:2] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pos_embedding[:, 1:dim_per_axis:2] = torch.cos(x_pos.unsqueeze(1) * div_term)
        
        # Y axis encoding (dimensions 256:512)
        pos_embedding[:, dim_per_axis:2*dim_per_axis:2] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pos_embedding[:, dim_per_axis+1:2*dim_per_axis:2] = torch.cos(y_pos.unsqueeze(1) * div_term)
        
        # Z axis encoding (dimensions 512:768)
        pos_embedding[:, 2*dim_per_axis:3*dim_per_axis:2] = torch.sin(z_pos.unsqueeze(1) * div_term)
        pos_embedding[:, 2*dim_per_axis+1:3*dim_per_axis:2] = torch.cos(z_pos.unsqueeze(1) * div_term)
        
        # Expand for batch (B, N, embed_dim)
        pos_embedding = pos_embedding.unsqueeze(0).expand(B, -1, -1)
        
        return pos_embedding
        
    def forward(self, visual_features, spatial_dims=None):
        if spatial_dims is None:
            spatial_dims = (2, 4, 4)  # Default DCFormer output dimensions (D, H, W)

        # Handle list/tuple input - take last layer features
        if isinstance(visual_features, (list, tuple)):
            visual_features = visual_features[-1]
            
        print("*"*20 + " Incoming visual features shape = ", visual_features.shape)
        B, N, D_feat = visual_features.shape
        
        if self.use_pos_embed:
            # Generate sinusoidal positional embeddings
            pos_embed = self.generate_3d_sinusoidal_embedding(
                B, N, 
                D=spatial_dims[0], 
                H=spatial_dims[1], 
                W=spatial_dims[2],
                embed_dim=D_feat,  # Same as visual features (768)
                device=visual_features.device
            )
            pos_embed = pos_embed.to(visual_features.dtype)
            print(f"Positional embedding shape: {pos_embed.shape}")
            
            # ADD positional embeddings (like in Transformer, not concatenate!)
            visual_features = visual_features + pos_embed
            print(f"Features after pos addition: {visual_features.shape}")

        # Project vision features to BERT space
        print(f"Visual features dtype: {visual_features.dtype}")
        visual_features = self.vision_proj(visual_features)  # (B, N, bert_hidden)
        
        # Expand query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, proj_out_num, bert_hidden)
        
        # Create attention mask for visual tokens
        encoder_mask = torch.ones(
            B, N, 
            dtype=torch.long, 
            device=visual_features.device
        )
        
        # BERT cross-attention
        outputs = self.bert.bert(
            input_ids=None,
            attention_mask=None,
            query_embeds=query_tokens,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=encoder_mask,
            return_dict=True,
            is_decoder=True,
        )
        
        # Extract query outputs
        query_output = outputs.last_hidden_state[:, :self.proj_out_num, :]
        
        # Project to LLM space
        image_features = self.output_proj(query_output)  # (B, proj_out_num, llm_hidden)
        
        return image_features
    
    def load_state_dict(self, state_dict, strict=False):
        """Flexible state dict loading"""
        # Filter out incompatible keys
        own_state = self.state_dict()
        compatible_state = {}
        
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                compatible_state[k] = v
            else:
                print(f"Skipping incompatible key: {k}")
        
        return super().load_state_dict(compatible_state, strict=False)