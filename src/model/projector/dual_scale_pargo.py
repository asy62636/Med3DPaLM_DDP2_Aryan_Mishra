"""
Dual-Scale ParGo Projector for Experiment 3.
Processes both low-level (penultimate) and high-level (final) DCFormer features
with independent ParGo branches, then concatenates outputs.

Low-level branch:  (B, 256, 384) -> 256 queries (200 partial + 56 global) -> (B, 256, 3584)
High-level branch: (B, 32, 768)  -> 32 queries  (24 partial + 8 global)   -> (B, 32, 3584)
Combined output:   (B, 288, 3584) — matches Med3DVLM's token count.
"""

import logging
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

logger = logging.getLogger(__name__)


def _log_tensor(name, t, level="debug"):
    """Helper to log tensor stats."""
    fn = getattr(logger, level)
    if t is None:
        fn(f"  {name}: None")
        return
    if t.is_floating_point():
        fn(
            f"  {name}: shape={list(t.shape)}, dtype={t.dtype}, "
            f"device={t.device}, min={t.min().item():.6f}, "
            f"max={t.max().item():.6f}, mean={t.mean().item():.6f}, "
            f"std={t.std().item():.6f}"
        )
    else:
        fn(f"  {name}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}")


class ParGoBranch(nn.Module):
    """
    A single ParGo branch: cross-attention with partial-global masks + cascaded self-attention.

    This is the core building block shared by both the low-level and high-level branches.
    Each branch has its own BERT instance, queries, input projection, and masks.
    """

    def __init__(
        self,
        branch_name,
        num_input_tokens,
        vision_hidden_size,
        hidden_size=768,
        num_global_queries=8,
        num_partial_queries=24,
        num_layers=2,
        bert_config=None,
        use_pretrained_bert=True,
        bert_model_name="bert-base-uncased",
    ):
        super().__init__()
        self.branch_name = branch_name
        self.num_input_tokens = num_input_tokens
        self.num_global = num_global_queries
        self.num_partial = num_partial_queries
        self.num_queries = num_global_queries + num_partial_queries
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        logger.info(f"  [{branch_name}] Initializing ParGoBranch")
        logger.info(f"    num_input_tokens   = {num_input_tokens}")
        logger.info(f"    vision_hidden_size  = {vision_hidden_size}")
        logger.info(f"    num_global_queries  = {num_global_queries}")
        logger.info(f"    num_partial_queries = {num_partial_queries}")
        logger.info(f"    num_queries (total) = {self.num_queries}")
        logger.info(f"    num_layers          = {num_layers}")

        # --- Input projection (project vision features to BERT's hidden size) ---
        if vision_hidden_size != hidden_size:
            self.input_proj = nn.Linear(vision_hidden_size, hidden_size)
            logger.info(f"    input_proj: Linear({vision_hidden_size} -> {hidden_size})")
        else:
            self.input_proj = nn.Identity()
            logger.info(f"    input_proj: Identity (vision dim already {hidden_size})")

        # --- Learnable query tokens ---
        self.global_queries = nn.Parameter(torch.zeros(1, num_global_queries, hidden_size))
        self.partial_queries = nn.Parameter(torch.zeros(1, num_partial_queries, hidden_size))
        nn.init.normal_(self.global_queries, std=0.02)
        nn.init.normal_(self.partial_queries, std=0.02)

        # --- BERT (cross-attention enabled decoder) ---
        cfg = BertConfig.from_pretrained(bert_model_name)
        cfg.num_hidden_layers = num_layers
        cfg.add_cross_attention = True
        cfg.is_decoder = True
        cfg.hidden_dropout_prob = 0.0
        cfg.attention_probs_dropout_prob = 0.0

        if use_pretrained_bert:
            logger.info(f"    [{branch_name}] Loading pretrained BertModel...")
            self.bert = BertModel.from_pretrained(bert_model_name, config=cfg)
            self._log_bert_weight_status()
        else:
            logger.info(f"    [{branch_name}] Initializing BertModel with RANDOM weights.")
            self.bert = BertModel(cfg)

        # --- Build masks ---
        self._build_masks()

        params = sum(p.numel() for p in self.parameters())
        logger.info(f"    [{branch_name}] Total parameters: {params:,}")

    def _log_bert_weight_status(self):
        cross_attn_params = sum(
            p.numel() for n, p in self.bert.named_parameters() if "crossattention" in n
        )
        self_attn_params = sum(
            p.numel() for n, p in self.bert.named_parameters() if "crossattention" not in n
        )
        logger.info(f"    [{self.branch_name}] BERT self-attn/FFN params (pretrained): {self_attn_params:,}")
        logger.info(f"    [{self.branch_name}] BERT cross-attn params (random init): {cross_attn_params:,}")

    def _build_masks(self):
        """Build partial-global cross-attention and cascaded self-attention masks."""
        nv = self.num_input_tokens
        np_ = self.num_partial
        ng = self.num_global
        nq = self.num_queries

        logger.info(f"    [{self.branch_name}] Building masks: nq={nq} (ng={ng}, np={np_}), nv={nv}")

        # =================================================================
        # Cross-attention mask (queries x visual_tokens)
        # 0.0 = attend, -inf = masked
        # =================================================================
        cross_mask = torch.zeros(nq, nv)

        # Global queries: attend to ALL visual tokens (already 0.0)
        # Partial queries: each attends to a contiguous subset
        tokens_per_partial = nv / np_
        logger.info(f"    [{self.branch_name}] tokens_per_partial = {tokens_per_partial:.4f}")

        partial_assignments = []
        for i in range(np_):
            start = int(i * tokens_per_partial)
            end = int((i + 1) * tokens_per_partial)
            end = max(end, start + 1)
            end = min(end, nv)

            cross_mask[ng + i, :] = float("-inf")
            cross_mask[ng + i, start:end] = 0.0
            partial_assignments.append((start, end))

        # Verify coverage
        covered = set()
        for s, e in partial_assignments:
            covered.update(range(s, e))
        uncovered = set(range(nv)) - covered
        if uncovered:
            logger.warning(f"    [{self.branch_name}] WARNING: visual tokens {uncovered} not covered!")
        else:
            logger.info(f"    [{self.branch_name}] All {nv} visual tokens covered ✓")

        n_attend_global = (cross_mask[:ng] == 0.0).sum(dim=1).float().mean().item()
        n_attend_partial = (cross_mask[ng:] == 0.0).sum(dim=1).float().mean().item()
        logger.info(
            f"    [{self.branch_name}] Avg tokens attended — global: {n_attend_global:.1f}, partial: {n_attend_partial:.1f}"
        )

        self.register_buffer("cross_attention_mask", cross_mask.unsqueeze(0).unsqueeze(0))

        # =================================================================
        # Cascaded self-attention masks (one per layer)
        # =================================================================
        k = max(np_ // self.num_layers, 1)
        logger.info(f"    [{self.branch_name}] CPP cascade step k = {k}")

        for layer_idx in range(self.num_layers):
            self_mask = torch.zeros(nq, nq)

            # Global tokens can always see all other global tokens (rows 0..ng-1)
            # (already 0.0)

            # Partial tokens: cascading visibility
            n_vis = k * layer_idx  # number of neighbors per side

            for i in range(np_):
                # Start by masking all partial-to-partial
                self_mask[ng + i, ng:] = float("-inf")
                # Unmask self
                self_mask[ng + i, ng + i] = 0.0
                # Unmask neighbors
                for j in range(1, n_vis + 1):
                    if i - j >= 0:
                        self_mask[ng + i, ng + i - j] = 0.0
                    if i + j < np_:
                        self_mask[ng + i, ng + i + j] = 0.0

            # Partial tokens can always see global tokens
            self_mask[ng:, :ng] = 0.0

            n_attend_avg = (self_mask[ng:] == 0.0).sum(dim=1).float().mean().item()
            logger.info(
                f"    [{self.branch_name}] Layer {layer_idx} self-attn: "
                f"avg visible per partial = {n_attend_avg:.1f} (of {nq})"
            )

            buf_name = f"self_attention_mask_layer_{layer_idx}"
            self.register_buffer(buf_name, self_mask.unsqueeze(0).unsqueeze(0))

        logger.info(f"    [{self.branch_name}] Mask building complete ✓")

    def forward(self, visual_features):
        """
        Args:
            visual_features: (B, num_input_tokens, vision_hidden_size)
        Returns:
            (B, num_queries, hidden_size=768)
        """
        B = visual_features.shape[0]

        # Project visual features to BERT hidden size
        visual_tokens = self.input_proj(visual_features)

        # Prepare queries
        queries = torch.cat([
            self.global_queries.expand(B, -1, -1),
            self.partial_queries.expand(B, -1, -1),
        ], dim=1)  # (B, num_queries, 768)

        # Cross-attention mask: (1, 1, nq, nv) -> (B, 1, nq, nv)
        cross_mask = self.cross_attention_mask.expand(B, -1, -1, -1)

        # Run through BERT layers manually (to use per-layer self-attention masks)
        hidden_states = queries
        for i, layer_module in enumerate(self.bert.encoder.layer):
            layer_self_mask = getattr(self, f"self_attention_mask_layer_{i}")
            layer_self_mask = layer_self_mask.expand(B, -1, -1, -1)  # (B, 1, nq, nq)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=layer_self_mask,
                encoder_hidden_states=visual_tokens,
                encoder_attention_mask=cross_mask,
            )
            hidden_states = layer_outputs[0]

        return hidden_states  # (B, num_queries, 768)


class DualScaleParGo(nn.Module):
    """
    Dual-Scale ParGo Projector — Experiment 3.

    Two independent ParGo branches process features from different DCFormer stages:
      - Low-level:  penultimate layer (B, 256, 384) -> 256 query tokens
      - High-level: final layer       (B, 32, 768)  -> 32 query tokens

    Outputs are projected to LLM dim and concatenated: (B, 288, llm_hidden_size).
    """

    def __init__(
        self,
        # Low-level branch config
        low_num_input_tokens=256,
        low_vision_hidden_size=384,
        low_num_global_queries=56,
        low_num_partial_queries=200,
        low_num_layers=2,
        # High-level branch config
        high_num_input_tokens=32,
        high_vision_hidden_size=768,
        high_num_global_queries=8,
        high_num_partial_queries=24,
        high_num_layers=2,
        # Shared config
        hidden_size=768,
        llm_hidden_size=3584,
        bert_model_name="bert-base-uncased",
        use_pretrained_bert=True,
    ):
        super().__init__()

        logger.info("=" * 60)
        logger.info("Initializing DualScaleParGo (Experiment 3)")
        logger.info("=" * 60)

        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size

        # --- Low-level branch ---
        self.low_branch = ParGoBranch(
            branch_name="LOW",
            num_input_tokens=low_num_input_tokens,
            vision_hidden_size=low_vision_hidden_size,
            hidden_size=hidden_size,
            num_global_queries=low_num_global_queries,
            num_partial_queries=low_num_partial_queries,
            num_layers=low_num_layers,
            use_pretrained_bert=use_pretrained_bert,
            bert_model_name=bert_model_name,
        )

        # --- High-level branch ---
        self.high_branch = ParGoBranch(
            branch_name="HIGH",
            num_input_tokens=high_num_input_tokens,
            vision_hidden_size=high_vision_hidden_size,
            hidden_size=hidden_size,
            num_global_queries=high_num_global_queries,
            num_partial_queries=high_num_partial_queries,
            num_layers=high_num_layers,
            use_pretrained_bert=use_pretrained_bert,
            bert_model_name=bert_model_name,
        )

        # --- Output projections (each branch gets its own) ---
        low_out_tokens = low_num_global_queries + low_num_partial_queries
        high_out_tokens = high_num_global_queries + high_num_partial_queries
        self.proj_out_num = low_out_tokens + high_out_tokens

        self.low_output_proj = nn.Linear(hidden_size, llm_hidden_size)
        self.high_output_proj = nn.Linear(hidden_size, llm_hidden_size)

        logger.info(f"  low_output_proj:  Linear({hidden_size} -> {llm_hidden_size})")
        logger.info(f"  high_output_proj: Linear({hidden_size} -> {llm_hidden_size})")
        logger.info(f"  proj_out_num = {self.proj_out_num} "
                     f"(low={low_out_tokens} + high={high_out_tokens})")

        # --- Parameter count ---
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Total parameters     : {total_params:,}")
        logger.info(f"  Trainable parameters : {trainable_params:,}")
        logger.info("=" * 60)

    def forward(self, image_features, **kwargs):
        """
        Args:
            image_features: list of tensors from DCFormer.
                image_features[0] = low-level  (B, 256, 384) from penultimate layer
                image_features[1] = high-level (B, 32, 768)  from final layer

                OR if a single tensor is passed (backward compat), treat as high-level only.
                The Med3DVLM vision tower returns a list when vision_select_layer is configured
                for dual-scale.

        Returns:
            (B, proj_out_num, llm_hidden_size) — concatenated low + high tokens
        """
        # Handle different input formats
        if isinstance(image_features, (list, tuple)):
            if len(image_features) == 2:
                low_features = image_features[0]   # (B, 256, 384)
                high_features = image_features[1]   # (B, 32, 768)
            elif len(image_features) == 1:
                # Fallback: single feature — treat as high-level only
                raise ValueError(
                    "DualScaleParGo requires 2 feature maps (low + high). "
                    f"Got list of length {len(image_features)}. "
                    "Check your vision_select_layer config."
                )
            else:
                # More than 2 — take penultimate and last
                low_features = image_features[-2]
                high_features = image_features[-1]
        else:
            raise ValueError(
                "DualScaleParGo requires a list of [low_features, high_features]. "
                f"Got {type(image_features)}."
            )

        # Run branches
        low_out = self.low_branch(low_features)     # (B, 256, 768)
        high_out = self.high_branch(high_features)   # (B, 32, 768)

        # Project to LLM dim
        low_out = self.low_output_proj(low_out)      # (B, 256, 3584)
        high_out = self.high_output_proj(high_out)    # (B, 32, 3584)

        # Concatenate: low tokens first, then high tokens
        output = torch.cat([low_out, high_out], dim=1)  # (B, 288, 3584)

        return output


# ===========================================================================
# Standalone test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("STANDALONE TEST: DualScaleParGo")
    print("=" * 70 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- 1. Instantiate ---
    print("--- Step 1: Instantiate DualScaleParGo ---")
    model = DualScaleParGo(
        low_num_input_tokens=256,
        low_vision_hidden_size=384,
        low_num_global_queries=56,
        low_num_partial_queries=200,
        low_num_layers=2,
        high_num_input_tokens=32,
        high_vision_hidden_size=768,
        high_num_global_queries=8,
        high_num_partial_queries=24,
        high_num_layers=2,
        hidden_size=768,
        llm_hidden_size=3584,
        bert_model_name="bert-base-uncased",
        use_pretrained_bert=True,
    ).to(device)

    # --- 2. Parameter summary ---
    print("\n--- Step 2: Parameter summary ---")
    total = 0
    groups = {}
    for name, p in model.named_parameters():
        total += p.numel()
        # Group by top-level component
        parts = name.split(".")
        group = f"{parts[0]}.{parts[1]}" if len(parts) > 1 else parts[0]
        groups[group] = groups.get(group, 0) + p.numel()

    for g, c in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g:35s}: {c:>12,} params ({100 * c / total:.1f}%)")
    print(f"  {'TOTAL':35s}: {total:>12,} params")

    # --- 3. Forward pass ---
    print("\n--- Step 3: Forward pass (batch=2) ---")
    low_input = torch.randn(2, 256, 384, device=device)
    high_input = torch.randn(2, 32, 768, device=device)
    print(f"  Low input shape:  {list(low_input.shape)}")
    print(f"  High input shape: {list(high_input.shape)}")

    with torch.no_grad():
        output = model([low_input, high_input])

    print(f"  Output shape: {list(output.shape)}")
    print(f"  Expected:     [2, 288, 3584]")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Has NaN: {output.isnan().any().item()}")
    print(f"  Has Inf: {output.isinf().any().item()}")

    assert output.shape == (2, 288, 3584), f"Shape mismatch: {output.shape}"

    # --- 4. Gradient check ---
    print("\n--- Step 4: Gradient check ---")
    model.zero_grad()
    low_input2 = torch.randn(2, 256, 384, device=device)
    high_input2 = torch.randn(2, 32, 768, device=device)
    output2 = model([low_input2, high_input2])
    loss = output2.sum()
    loss.backward()

    grads_ok = True
    no_grad_names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                no_grad_names.append(name)
                grads_ok = False
            elif p.grad.isnan().any():
                print(f"  WARNING: {name} has NaN gradient!")
                grads_ok = False

    if grads_ok:
        print("  All parameters have valid gradients ✓")
    else:
        print(f"  WARNING: {len(no_grad_names)} params have no gradient:")
        for n in no_grad_names[:10]:
            print(f"    {n}")

    # --- 5. Mask sanity check ---
    print("\n--- Step 5: Mask sanity checks ---")
    for branch_name, branch in [("LOW", model.low_branch), ("HIGH", model.high_branch)]:
        cm = branch.cross_attention_mask.squeeze()
        print(f"  [{branch_name}] Cross-attention mask shape: {list(cm.shape)}")
        print(f"  [{branch_name}] Global rows all-attend: {(cm[:branch.num_global] == 0.0).all().item()}")
        n_attend = (cm[branch.num_global:] == 0.0).sum(dim=1).float().mean().item()
        print(f"  [{branch_name}] Avg partial tokens attended: {n_attend:.1f}")

    # --- 6. Memory estimate ---
    print("\n--- Step 6: Memory estimate ---")
    param_mb = total * 4 / (1024 ** 2)  # float32
    param_mb_bf16 = total * 2 / (1024 ** 2)  # bfloat16
    print(f"  Parameter memory (fp32):  {param_mb:.1f} MB")
    print(f"  Parameter memory (bf16):  {param_mb_bf16:.1f} MB")

    print("\n" + "=" * 70)
    print("TEST COMPLETE ✓")
    print("=" * 70)