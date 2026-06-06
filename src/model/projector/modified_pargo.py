import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional

try:
    from .qformer_bert import BertLMHeadModel, BertConfig
except:
    # If relative import fails, try absolute import
    try:
        from qformer_bert import BertLMHeadModel, BertConfig
    except:
        print("Warning: Could not import BertLMHeadModel. Using fallback.")
        BertLMHeadModel = None
        BertConfig = None


class MultiScaleDecomposed3DParGo(nn.Module):
    """
    Multi-Scale Decomposed 3D ParGo Projector.
    
    Combines:
    1. Multi-scale fusion from original ParGo (low-level + high-level features)
    2. 3D decomposition for volumetric medical imaging (axial/coronal/sagittal)
    3. Partial-global attention mechanism
    
    Architecture:
        DCFormer → [Low features (B,256,384), High features (B,32,768)]
                ↓
        Decomposed 3D ParGo (process each scale separately)
                ↓
        [Low tokens (B,144,llm_dim), High tokens (B,32,llm_dim)]
                ↓
        Concatenate → (B, 176, llm_dim)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions from DCFormer
        # Penultimate layer (low-level): spatial details
        self.low_dim = int(getattr(config, "low_level_hidden_size", 384))
        # Final layer (high-level): semantic features
        self.high_dim = int(getattr(config, "vision_hidden_size", 768))
        
        # LLM dimension (Qwen-7B: 3584)
        self.llm_hidden_size = int(getattr(config, "llm_hidden_size", 3584))
        
        # Token allocation for multi-scale
        # Low-level: more tokens for spatial details
        self.n_low_tokens = int(getattr(config, "n_low_tokens", 144))
        # High-level: fewer tokens for semantic info
        self.n_high_tokens = int(getattr(config, "n_high_tokens", 32))
        
        # Total output tokens
        self.proj_out_num = self.n_low_tokens + self.n_high_tokens  # 176
        
        # Use positional embeddings (recommended for 3D)
        self.use_pos_embed = getattr(config, "use_positional_embedding", True)
        
        # Number of ParGo layers per scale
        self.num_layers = int(getattr(config, "pargo_num_layers", 6))
        
        # Build decomposed 3D ParGo for each scale
        print(f"Building Multi-Scale Decomposed 3D ParGo:")
        print(f"  Low-level: {self.low_dim}D → {self.n_low_tokens} tokens")
        print(f"  High-level: {self.high_dim}D → {self.n_high_tokens} tokens")
        print(f"  Total output: {self.proj_out_num} tokens → {self.llm_hidden_size}D")
        
        # Low-level feature processor
        self.low_level_pargo = Decomposed3DParGoBlock(
            in_dim=self.low_dim,
            out_dim=self.llm_hidden_size,
            n_partial=self.n_low_tokens - 8,  # 136 partial tokens
            n_global=8,                        # 8 global tokens
            num_layers=self.num_layers,
            use_pos_embed=self.use_pos_embed
        )
        
        # High-level feature processor  
        self.high_level_pargo = Decomposed3DParGoBlock(
            in_dim=self.high_dim,
            out_dim=self.llm_hidden_size,
            n_partial=self.n_high_tokens - 4,  # 28 partial tokens
            n_global=4,                         # 4 global tokens
            num_layers=self.num_layers,
            use_pos_embed=self.use_pos_embed
        )
        
        # Optional: Cross-scale interaction
        # Enable this if you want low and high tokens to interact
        self.use_cross_scale = getattr(config, "use_cross_scale_attention", False)
        if self.use_cross_scale:
            print("  Using cross-scale attention")
            self.cross_scale_attn = nn.MultiheadAttention(
                self.llm_hidden_size, 
                num_heads=8, 
                batch_first=True,
                dropout=0.1
            )
            self.cross_scale_norm = nn.LayerNorm(self.llm_hidden_size)
    
    def forward(self, visual_features, spatial_dims=None):
        """
        Args:
            visual_features: List/tuple of features from DCFormer
                - If list: [layer1, layer2, ..., layer_n]
                - We use: visual_features[-2] (low) and visual_features[-1] (high)
            spatial_dims: Tuple (D, H, W) - spatial dimensions after encoder
                - If None, auto-detected from feature shape
        
        Returns:
            image_features: (B, proj_out_num, llm_hidden_size)
        """
        # Extract multi-scale features
        if isinstance(visual_features, (list, tuple)):
            if len(visual_features) < 2:
                raise ValueError(
                    f"Expected at least 2 feature levels, got {len(visual_features)}"
                )
            # Penultimate layer: low-level spatial details
            low_features = visual_features[-2]
            # Final layer: high-level semantic features
            high_features = visual_features[-1]
        else:
            # Fallback: use same features for both scales
            print("Warning: Single feature tensor provided, using for both scales")
            low_features = visual_features
            high_features = visual_features
        
        B = low_features.shape[0]
        N_low = low_features.shape[1]  # Number of tokens in low-level features
        N_high = high_features.shape[1]  # Number of tokens in high-level features
        
        # Auto-detect spatial dimensions if not provided
        if spatial_dims is None:
            # For 256 tokens: most likely 4×8×8 or 16×4×4
            # For 32 tokens: most likely 2×4×4
            # We'll use a simple cube root approximation
            import math
            
            # Try to factorize N_low into (D, H, W)
            # Prefer shapes like (D, H, H) or (D, H, 2H)
            if N_low == 256:
                spatial_dims = (4, 8, 8)  # Most common for DCFormer
            elif N_low == 32:
                spatial_dims = (2, 4, 4)
            elif N_low == 128:
                spatial_dims = (2, 8, 8)
            elif N_low == 512:
                spatial_dims = (8, 8, 8)
            else:
                # Generic fallback: cube root
                side = int(math.pow(N_low, 1/3))
                spatial_dims = (side, side, side)
                print(f"Warning: Using approximated spatial dims {spatial_dims} for {N_low} tokens")
        
        print("*" * 50)
        print(f"Multi-Scale Decomposed 3D ParGo Forward:")
        print(f"  Low features shape: {low_features.shape}")
        print(f"  High features shape: {high_features.shape}")
        print(f"  Auto-detected spatial dims: {spatial_dims}")
        print(f"  Expected tokens: {spatial_dims[0] * spatial_dims[1] * spatial_dims[2]}")
        
        # Process through decomposed 3D ParGo
        low_tokens = self.low_level_pargo(
            low_features, spatial_dims
        )  # (B, n_low_tokens, llm_dim)
        
        # For high-level features, use simpler spatial dims
        # High-level usually has fewer tokens (32 vs 256)
        if N_high != N_low:
            # Recalculate spatial dims for high features
            if N_high == 32:
                high_spatial_dims = (2, 4, 4)
            elif N_high == 128:
                high_spatial_dims = (2, 8, 8)
            elif N_high == 256:
                high_spatial_dims = (4, 8, 8)
            else:
                import math
                side = int(math.pow(N_high, 1/3))
                high_spatial_dims = (side, side, side)
            print(f"  High-level spatial dims: {high_spatial_dims}")
        else:
            high_spatial_dims = spatial_dims
        
        high_tokens = self.high_level_pargo(
            high_features, high_spatial_dims
        )  # (B, n_high_tokens, llm_dim)
        
        print(f"  Output low tokens: {low_tokens.shape}")
        print(f"  Output high tokens: {high_tokens.shape}")
        
        # Optional: Cross-scale attention
        if self.use_cross_scale:
            all_tokens = torch.cat([low_tokens, high_tokens], dim=1)
            attended, _ = self.cross_scale_attn(
                all_tokens, all_tokens, all_tokens
            )
            image_features = self.cross_scale_norm(attended + all_tokens)
        else:
            # Simple concatenation (recommended for first experiments)
            image_features = torch.cat([low_tokens, high_tokens], dim=1)
        
        print(f"  Final output shape: {image_features.shape}")
        print("*" * 50)
        
        return image_features
    
    def load_state_dict(self, state_dict, strict=False):
        """Flexible state dict loading for backwards compatibility"""
        own_state = self.state_dict()
        compatible_state = {}
        
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                compatible_state[k] = v
            else:
                print(f"Skipping incompatible key: {k}")
        
        return super().load_state_dict(compatible_state, strict=False)



class ParGoLayer2D(nn.Module):
    """
    Single ParGo layer implementing:
    1. Cross-attention from queries to visual features (partial-global)
    2. Cascaded Partial Perception (self-attention among partial tokens)
    3. Feed-forward network
    """
    
    def __init__(self, dim: int, n_partial: int, n_global: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_partial = n_partial
        self.n_global = n_global
        self.n_tokens = n_partial + n_global
        
        # Cross-attention: queries attend to visual features
        self.cross_attn = nn.MultiheadAttention(
            dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Cascaded Partial Perception: partial tokens interact
        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(
        self, 
        queries: torch.Tensor, 
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, n_tokens, dim) - partial + global query tokens
            visual_features: (B, N, dim) - features from one anatomical plane
        
        Returns:
            queries: (B, n_tokens, dim) - updated tokens
        """
        # 1. Cross-attention: all queries attend to visual features
        #    Global tokens see everything, partial tokens see subregions
        attended, _ = self.cross_attn(
            self.norm1(queries),
            visual_features,
            visual_features
        )
        queries = queries + attended
        
        # 2. Cascaded Partial Perception (CPP)
        #    Only partial tokens interact with each other
        #    Global tokens remain independent
        partial_tokens = queries[:, :self.n_partial]
        global_tokens = queries[:, self.n_partial:]
        
        partial_updated, _ = self.self_attn(
            self.norm2(partial_tokens),
            partial_tokens,
            partial_tokens
        )
        partial_tokens = partial_tokens + partial_updated
        
        # Concatenate updated partial tokens with unchanged global tokens
        # IMPORTANT: Use torch.cat instead of in-place assignment
        queries = torch.cat([partial_tokens, global_tokens], dim=1)
        
        # 3. Feed-forward network
        queries = queries + self.ffn(self.norm3(queries))
        
        return queries


class ParGoBranch(nn.Module):
    """Container for a single ParGo branch (one anatomical plane)."""
    
    def __init__(self, n_partial: int, n_global: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.n_partial = n_partial
        self.n_global = n_global
        
        # Learnable query tokens
        self.partial_queries = nn.Parameter(
            torch.randn(n_partial, hidden_dim) * 0.02
        )
        self.global_queries = nn.Parameter(
            torch.randn(n_global, hidden_dim) * 0.02
        )
        
        # Stack of ParGo layers
        self.layers = nn.ModuleList([
            ParGoLayer2D(
                dim=hidden_dim,
                n_partial=n_partial,
                n_global=n_global,
                num_heads=8
            )
            for _ in range(num_layers)
        ])


class Decomposed3DParGoBlock(nn.Module):
    """
    Single-scale Decomposed 3D ParGo Block.
    
    Processes features across three anatomical planes:
    - Axial: H-W plane (looking down from top)
    - Coronal: D-H plane (looking from front)
    - Sagittal: D-W plane (looking from side)
    
    Each plane uses ParGo-style partial-global attention,
    then results are fused adaptively.
    """
    
    def __init__(
        self, 
        in_dim: int,
        out_dim: int, 
        n_partial: int,
        n_global: int,
        num_layers: int = 6,
        use_pos_embed: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_partial = n_partial
        self.n_global = n_global
        self.n_tokens = n_partial + n_global
        self.num_layers = num_layers
        self.use_pos_embed = use_pos_embed
        
        # Project input to common hidden dimension
        self.hidden_dim = 768  # Standard BERT hidden size
        self.input_proj = nn.Linear(in_dim, self.hidden_dim)
        
        # 3D positional embedding generator
        if self.use_pos_embed:
            self.pos_embed_generator = Sinusoidal3DPositionalEncoding(
                dim=in_dim,
                temperature=10000
            )
        
        # Build three plane-specific ParGo branches
        # Each branch has its own parameters for maximum flexibility
        self.axial_branch = self._build_pargo_branch("axial")
        self.coronal_branch = self._build_pargo_branch("coronal")
        self.sagittal_branch = self._build_pargo_branch("sagittal")
        
        # Adaptive fusion: learn how to weight three planes
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        
        # Final projection to LLM dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, out_dim)
        )
        
        # RMS normalization for token norm alignment
        self.rms_norm = RMSNorm(out_dim)
    
    def _build_pargo_branch(self, plane_name: str):
        """Build a ParGo branch for one anatomical plane."""
        return ParGoBranch(
            n_partial=self.n_partial,
            n_global=self.n_global,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
    
    def forward(self, features: torch.Tensor, spatial_dims: Tuple[int, int, int]):
        """
        Args:
            features: (B, N, in_dim) where N = D*H*W
            spatial_dims: (D, H, W) - spatial dimensions
        
        Returns:
            tokens: (B, n_tokens, out_dim)
        """
        B, N, C = features.shape
        D, H, W = spatial_dims
        
        # Verify spatial dimensions match
        assert N == D * H * W, f"Feature count {N} != spatial size {D}*{H}*{W}={D*H*W}"
        
        # Add 3D positional embeddings
        if self.use_pos_embed:
            pos_embed = self.pos_embed_generator(B, N, D, H, W, C, features.device)
            pos_embed = pos_embed.to(features.dtype)
            features = features + pos_embed
            print(f"  Applied 3D positional embedding: {pos_embed.shape}")
        
        # Project to hidden dimension
        features = self.input_proj(features)  # (B, N, hidden_dim)
        
        # Reshape to 3D volume for plane extraction
        features_3d = features.view(B, D, H, W, self.hidden_dim)
        
        # Process each anatomical plane
        # Axial plane: average over depth, process H-W
        axial_features = features_3d.mean(dim=1)  # (B, H, W, C)
        axial_features = axial_features.reshape(B, H * W, self.hidden_dim)
        axial_tokens = self._process_plane(axial_features, self.axial_branch)
        
        # Coronal plane: average over width, process D-H
        coronal_features = features_3d.mean(dim=3)  # (B, D, H, C)
        coronal_features = coronal_features.reshape(B, D * H, self.hidden_dim)
        coronal_tokens = self._process_plane(coronal_features, self.coronal_branch)
        
        # Sagittal plane: average over height, process D-W
        sagittal_features = features_3d.mean(dim=2)  # (B, D, W, C)
        sagittal_features = sagittal_features.reshape(B, D * W, self.hidden_dim)
        sagittal_tokens = self._process_plane(sagittal_features, self.sagittal_branch)
        
        # Adaptive fusion with learned weights
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (
            weights[0] * axial_tokens + 
            weights[1] * coronal_tokens + 
            weights[2] * sagittal_tokens
        )
        fused = self.fusion_norm(fused)
        
        print(f"    Fusion weights: axial={weights[0]:.3f}, coronal={weights[1]:.3f}, sagittal={weights[2]:.3f}")
        
        # Project to LLM dimension
        output = self.output_proj(fused)  # (B, n_tokens, out_dim)
        
        # Apply RMS normalization for token norm alignment
        output = self.rms_norm(output)
        
        return output
    
    def _process_plane(
        self, 
        plane_features: torch.Tensor, 
        branch: nn.Module
    ) -> torch.Tensor:
        """
        Process one 2D anatomical plane through ParGo layers.
        
        Args:
            plane_features: (B, N_plane, hidden_dim)
            branch: Module containing queries and layers
        
        Returns:
            tokens: (B, n_tokens, hidden_dim)
        """
        B = plane_features.shape[0]
        
        # Initialize with learnable query tokens
        queries = torch.cat([
            branch.partial_queries,  # (n_partial, hidden_dim)
            branch.global_queries     # (n_global, hidden_dim)
        ], dim=0)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_tokens, hidden_dim)
        
        # Process through ParGo layers with partial-global attention
        output = queries
        for layer_idx, layer in enumerate(branch.layers):
            output = layer(output, plane_features)
        
        return output


class Sinusoidal3DPositionalEncoding(nn.Module):
    """
    3D sinusoidal positional encoding for volumetric data.
    
    Extends standard transformer positional encoding to 3 spatial dimensions.
    Handles anisotropic grids (e.g., 128×256×256) by normalizing coordinates.
    """
    
    def __init__(self, dim: int, temperature: float = 10000):
        super().__init__()
        assert dim % 6 == 0, f"Embedding dim must be divisible by 6, got {dim}"
        self.dim = dim
        self.temperature = temperature
        self.dim_per_axis = dim // 3  # Split equally across D, H, W
    
    def forward(
        self, 
        B: int, 
        N: int, 
        D: int, 
        H: int, 
        W: int,
        embed_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate 3D positional embeddings.
        
        Args:
            B: batch size
            N: number of tokens (should equal D*H*W)
            D, H, W: depth, height, width
            embed_dim: embedding dimension
            device: torch device
        
        Returns:
            pos_embedding: (B, N, embed_dim)
        """
        assert N == D * H * W, f"Token count {N} != spatial size {D*H*W}"
        assert embed_dim % 3 == 0, f"embed_dim must be divisible by 3"
        
        dim_per_axis = embed_dim // 3
        
        # Generate 3D coordinate grid
        d_coords = torch.arange(D, device=device).view(-1, 1, 1).expand(D, H, W)
        h_coords = torch.arange(H, device=device).view(1, -1, 1).expand(D, H, W)
        w_coords = torch.arange(W, device=device).view(1, 1, -1).expand(D, H, W)
        
        # Flatten and normalize to [0, 1] (handles anisotropic grids)
        z_pos = d_coords.reshape(-1).float() / max(D - 1, 1)  # Depth
        y_pos = h_coords.reshape(-1).float() / max(H - 1, 1)  # Height
        x_pos = w_coords.reshape(-1).float() / max(W - 1, 1)  # Width
        
        # Compute frequency bands
        omega = 1.0 / (
            self.temperature ** (torch.arange(0, dim_per_axis, 2, device=device).float() / dim_per_axis)
        )
        
        # Initialize positional embedding
        pos_embedding = torch.zeros(N, embed_dim, device=device)
        
        # Encode each axis with sin/cos
        # X axis (width): dimensions 0 to dim_per_axis
        pos_embedding[:, 0:dim_per_axis:2] = torch.sin(x_pos.unsqueeze(1) * omega)
        pos_embedding[:, 1:dim_per_axis:2] = torch.cos(x_pos.unsqueeze(1) * omega)
        
        # Y axis (height): dimensions dim_per_axis to 2*dim_per_axis
        pos_embedding[:, dim_per_axis:2*dim_per_axis:2] = torch.sin(y_pos.unsqueeze(1) * omega)
        pos_embedding[:, dim_per_axis+1:2*dim_per_axis:2] = torch.cos(y_pos.unsqueeze(1) * omega)
        
        # Z axis (depth): dimensions 2*dim_per_axis to 3*dim_per_axis
        pos_embedding[:, 2*dim_per_axis:3*dim_per_axis:2] = torch.sin(z_pos.unsqueeze(1) * omega)
        pos_embedding[:, 2*dim_per_axis+1:3*dim_per_axis:2] = torch.cos(z_pos.unsqueeze(1) * omega)
        
        # Expand for batch
        pos_embedding = pos_embedding.unsqueeze(0).expand(B, -1, -1)
        
        return pos_embedding


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler and more stable than LayerNorm for aligning token magnitudes.
    Helps with vision-text token norm alignment.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim)
        Returns:
            normalized: (B, N, dim)
        """
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x / rms


# Backward compatibility: keep original class name as alias
ParGoProjector = MultiScaleDecomposed3DParGo