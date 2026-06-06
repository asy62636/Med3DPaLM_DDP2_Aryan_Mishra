"""
Single-Scale ParGo Projector for Experiment 2.
Minimal implementation: 2 layers, 32 queries, BERT-initialized.

WITH COMPREHENSIVE DEBUG LOGGING — remove/disable for production training.
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
            f"std={t.std().item():.6f}, has_nan={t.isnan().any().item()}, "
            f"has_inf={t.isinf().any().item()}"
        )
    else:
        fn(f"  {name}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}")


class SingleScaleParGo(nn.Module):
    """
    Minimal ParGo operating on DCFormer's final-layer features.

    Input:  (B, 32, 768) from DCFormer stage 4
    Output: (B, 32, 3584) for Qwen LLM
    """

    def __init__(
        self,
        num_input_tokens=32,
        vision_hidden_size=768,
        llm_hidden_size=3584,
        num_global_queries=8,
        num_partial_queries=24,
        num_layers=2,
        bert_model_name="bert-base-uncased",
        use_pretrained_bert=True,
        dropout=0.0,
    ):
        super().__init__()

        logger.info("=" * 60)
        logger.info("Initializing SingleScaleParGo")
        logger.info("=" * 60)
        logger.info(f"  num_input_tokens   = {num_input_tokens}")
        logger.info(f"  vision_hidden_size = {vision_hidden_size}")
        logger.info(f"  llm_hidden_size    = {llm_hidden_size}")
        logger.info(f"  num_global_queries = {num_global_queries}")
        logger.info(f"  num_partial_queries= {num_partial_queries}")
        logger.info(f"  num_layers         = {num_layers}")
        logger.info(f"  bert_model_name    = {bert_model_name}")
        logger.info(f"  use_pretrained_bert= {use_pretrained_bert}")
        logger.info(f"  dropout            = {dropout}")

        self.num_input_tokens = num_input_tokens
        self.num_global = num_global_queries
        self.num_partial = num_partial_queries
        self.num_queries = num_global_queries + num_partial_queries
        self.num_layers = num_layers
        self.hidden_size = 768

        # Proj out num — used by dataset/tokenizer to know how many <im_patch> tokens
        self.proj_out_num = self.num_queries
        logger.info(f"  proj_out_num       = {self.proj_out_num}")

        # --- Input projection ---
        if vision_hidden_size != 768:
            self.input_proj = nn.Linear(vision_hidden_size, 768)
            logger.info(f"  input_proj: Linear({vision_hidden_size} -> 768)")
        else:
            self.input_proj = nn.Identity()
            logger.info("  input_proj: Identity (vision dim already 768)")

        # --- Learnable query tokens ---
        self.global_queries = nn.Parameter(torch.zeros(1, num_global_queries, 768))
        self.partial_queries = nn.Parameter(torch.zeros(1, num_partial_queries, 768))
        nn.init.normal_(self.global_queries, std=0.02)
        nn.init.normal_(self.partial_queries, std=0.02)
        logger.info(
            f"  global_queries : shape={list(self.global_queries.shape)}, "
            f"init mean={self.global_queries.data.mean().item():.6f}"
        )
        logger.info(
            f"  partial_queries: shape={list(self.partial_queries.shape)}, "
            f"init mean={self.partial_queries.data.mean().item():.6f}"
        )

        # --- BERT config ---
        logger.info(f"  Loading BertConfig from '{bert_model_name}'...")
        bert_config = BertConfig.from_pretrained(bert_model_name)
        bert_config.num_hidden_layers = num_layers
        bert_config.add_cross_attention = True
        bert_config.is_decoder = True
        bert_config.hidden_dropout_prob = dropout
        bert_config.attention_probs_dropout_prob = dropout

        logger.info(f"  BertConfig overrides:")
        logger.info(f"    num_hidden_layers          = {bert_config.num_hidden_layers}")
        logger.info(f"    hidden_size                = {bert_config.hidden_size}")
        logger.info(f"    num_attention_heads         = {bert_config.num_attention_heads}")
        logger.info(f"    intermediate_size           = {bert_config.intermediate_size}")
        logger.info(f"    add_cross_attention         = {bert_config.add_cross_attention}")
        logger.info(f"    is_decoder                  = {bert_config.is_decoder}")
        logger.info(f"    hidden_dropout_prob         = {bert_config.hidden_dropout_prob}")
        logger.info(f"    attention_probs_dropout_prob= {bert_config.attention_probs_dropout_prob}")

        # --- Load BERT ---
        if use_pretrained_bert:
            logger.info(f"  Loading pretrained BertModel from '{bert_model_name}'...")
            self.bert = BertModel.from_pretrained(bert_model_name, config=bert_config)
            logger.info("  Pretrained BERT loaded successfully.")

            # Log which layers were actually loaded vs randomly init
            self._log_bert_weight_status()
        else:
            logger.info("  Initializing BertModel with RANDOM weights.")
            self.bert = BertModel(bert_config)

        # --- Output projection ---
        self.output_proj = nn.Linear(768, llm_hidden_size)
        logger.info(f"  output_proj: Linear(768 -> {llm_hidden_size})")

        # --- Build masks ---
        logger.info("  Building attention masks...")
        self._build_masks()

        # --- Parameter count ---
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Total parameters     : {total_params:,}")
        logger.info(f"  Trainable parameters : {trainable_params:,}")
        logger.info("=" * 60)

    def _log_bert_weight_status(self):
        """Check which BERT layers got pretrained weights vs random init."""
        for name, param in self.bert.named_parameters():
            # Cross-attention layers are newly added — will be randomly init
            if "crossattention" in name:
                logger.debug(f"    [RANDOM INIT] {name}: shape={list(param.shape)}")
            elif "embeddings" in name:
                logger.debug(f"    [PRETRAINED?] {name}: shape={list(param.shape)}")
            else:
                logger.debug(f"    [PRETRAINED ] {name}: shape={list(param.shape)}")

        # Count cross-attention params
        cross_attn_params = sum(
            p.numel()
            for n, p in self.bert.named_parameters()
            if "crossattention" in n
        )
        self_attn_params = sum(
            p.numel()
            for n, p in self.bert.named_parameters()
            if "crossattention" not in n
        )
        logger.info(f"    BERT self-attn/FFN params (pretrained): {self_attn_params:,}")
        logger.info(f"    BERT cross-attn params (random init)  : {cross_attn_params:,}")

    def _build_masks(self):
        """Build partial-global cross-attention and cascaded self-attention masks."""
        nv = self.num_input_tokens  # 32
        np_ = self.num_partial  # 24
        ng = self.num_global  # 8
        nq = self.num_queries  # 32

        logger.info(f"    Building masks: nq={nq} (ng={ng}, np={np_}), nv={nv}")

        # =====================================================================
        # Cross-attention mask  (queries x visual_tokens)
        # 0.0 = attend,  -inf = masked
        # =====================================================================
        cross_mask = torch.zeros(nq, nv)

        tokens_per_partial = nv / np_
        logger.info(f"    tokens_per_partial = {tokens_per_partial:.4f}")

        partial_assignments = []
        for i in range(np_):
            start = int(i * tokens_per_partial)
            end = int((i + 1) * tokens_per_partial)
            end = max(end, start + 1)
            end = min(end, nv)

            cross_mask[ng + i, :] = float("-inf")
            cross_mask[ng + i, start:end] = 0.0
            partial_assignments.append((start, end))

        logger.info(f"    Partial query assignments (token ranges):")
        for i, (s, e) in enumerate(partial_assignments):
            logger.debug(f"      partial[{i:2d}] -> visual tokens [{s}:{e}] ({e - s} tokens)")

        # Verify: every visual token is seen by at least one partial query
        covered = set()
        for s, e in partial_assignments:
            covered.update(range(s, e))
        uncovered = set(range(nv)) - covered
        if uncovered:
            logger.warning(f"    WARNING: visual tokens {uncovered} not covered by any partial query!")
        else:
            logger.info(f"    All {nv} visual tokens covered by partial queries ✓")

        # Verify global rows are all zeros (attend to everything)
        global_mask_ok = (cross_mask[:ng] == 0.0).all().item()
        logger.info(f"    Global queries attend to all visual tokens: {global_mask_ok} ✓")

        # Count attend/mask per query type
        n_attend_global = (cross_mask[:ng] == 0.0).sum(dim=1).float().mean().item()
        n_attend_partial = (cross_mask[ng:] == 0.0).sum(dim=1).float().mean().item()
        logger.info(
            f"    Avg tokens attended — global: {n_attend_global:.1f}, partial: {n_attend_partial:.1f}"
        )

        self.register_buffer("cross_attention_mask", cross_mask.unsqueeze(0).unsqueeze(0))
        logger.info(f"    cross_attention_mask registered: shape={list(self.cross_attention_mask.shape)}")

        # =====================================================================
        # Cascaded self-attention masks (one per layer)
        # =====================================================================
        k = np_ // self.num_layers
        logger.info(f"    CPP cascade step k = {k} (np_={np_} / num_layers={self.num_layers})")

        for layer_idx in range(self.num_layers):
            self_mask = torch.zeros(nq, nq)

            n_vis = k * layer_idx
            logger.info(f"    Layer {layer_idx}: each partial token sees {n_vis} neighbors per side")

            for i in range(np_):
                self_mask[ng + i, ng:] = float("-inf")
                self_mask[ng + i, ng + i] = 0.0
                for j in range(1, n_vis + 1):
                    if i - j >= 0:
                        self_mask[ng + i, ng + i - j] = 0.0
                    if i + j < np_:
                        self_mask[ng + i, ng + i + j] = 0.0

            # Partial tokens can always see global tokens
            self_mask[ng:, :ng] = 0.0

            n_attend_partial_self = (self_mask[ng:] == 0.0).sum(dim=1).float().mean().item()
            logger.info(
                f"    Layer {layer_idx} self-attn: avg tokens visible per partial query = "
                f"{n_attend_partial_self:.1f} (of {nq})"
            )

            buf_name = f"self_attention_mask_layer_{layer_idx}"
            self.register_buffer(buf_name, self_mask.unsqueeze(0).unsqueeze(0))
            logger.info(f"    {buf_name} registered: shape={list(getattr(self, buf_name).shape)}")

        logger.info("    Mask building complete ✓")

    # def forward(self, image_features, **kwargs):
    #     B = image_features.shape[0]
        
    #     # Handle list input from vision tower
    #     if isinstance(image_features, list):
    #         if len(image_features) == 1:
    #             image_features = image_features[0]
    #         else:
    #             image_features = image_features[-1]

    def forward(self, image_features, **kwargs):
        if isinstance(image_features, (list, tuple)):
            if len(image_features) == 1:
                image_features = image_features[0]
            else:
                image_features = image_features[-1]
        B = image_features.shape[0]            # now it's a tensor
        
        visual_tokens = self.input_proj(image_features)
        
        queries = torch.cat([
            self.global_queries.expand(B, -1, -1),
            self.partial_queries.expand(B, -1, -1),
        ], dim=1)
        
        # === Key fix: bypass get_extended_attention_mask ===
        # BERT attention layers expect: (B, num_heads, tgt_len, src_len)
        # Our masks are (1, 1, nq, nq) and (1, 1, nq, nv)
        # Expand to (B, 1, nq, nq) — the 1 broadcasts across heads
        
        num_heads = self.bert.config.num_attention_heads  # 12
        
        last_layer_idx = self.num_layers - 1
        self_mask = getattr(self, f'self_attention_mask_layer_{last_layer_idx}')
        # (1, 1, nq, nq) -> (B, 1, nq, nq) — broadcasts over heads
        self_mask_expanded = self_mask.expand(B, -1, -1, -1)
        
        cross_mask = self.cross_attention_mask.expand(B, -1, -1, -1)
        # (B, 1, nq, nv)
        
        # Pass masks through BERT manually to bypass get_extended_attention_mask
        # We feed through each layer ourselves
        hidden_states = queries
        
        for i, layer_module in enumerate(self.bert.encoder.layer):
            # Get per-layer self-attention mask
            layer_self_mask = getattr(self, f'self_attention_mask_layer_{i}')
            layer_self_mask = layer_self_mask.expand(B, -1, -1, -1)  # (B, 1, nq, nq)
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=layer_self_mask,           # (B, 1, nq, nq)
                encoder_hidden_states=visual_tokens,
                encoder_attention_mask=cross_mask,         # (B, 1, nq, nv)
            )
            hidden_states = layer_outputs[0]
        
        output = self.output_proj(hidden_states)
        return output


# ===========================================================================
# Standalone test — run this file directly to validate the module
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("STANDALONE TEST: SingleScaleParGo")
    print("=" * 70 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- 1. Instantiate ---
    print("--- Step 1: Instantiate SingleScaleParGo ---")
    model = SingleScaleParGo(
        num_input_tokens=32,
        vision_hidden_size=768,
        llm_hidden_size=3584,
        num_global_queries=8,
        num_partial_queries=24,
        num_layers=2,
        bert_model_name="bert-base-uncased",
        use_pretrained_bert=True,
        dropout=0.0,
    ).to(device)

    # --- 2. Print parameter summary ---
    print("\n--- Step 2: Parameter summary ---")
    total = 0
    groups = {}
    for name, p in model.named_parameters():
        total += p.numel()
        group = name.split(".")[0]
        groups[group] = groups.get(group, 0) + p.numel()

    for g, c in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g:25s}: {c:>12,} params ({100 * c / total:.1f}%)")
    print(f"  {'TOTAL':25s}: {total:>12,} params")

    # --- 3. Forward pass with dummy data ---
    print("\n--- Step 3: Forward pass (batch=2) ---")
    dummy_input = torch.randn(2, 32, 768, device=device)
    print(f"  Input shape: {list(dummy_input.shape)}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Has NaN: {output.isnan().any().item()}")
    print(f"  Has Inf: {output.isinf().any().item()}")

    # --- 4. Gradient check ---
    print("\n--- Step 4: Gradient check ---")
    model.zero_grad()
    dummy_input2 = torch.randn(2, 32, 768, device=device, requires_grad=False)
    output2 = model(dummy_input2)
    loss = output2.sum()
    loss.backward()

    grads_ok = True
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                print(f"  WARNING: {name} has NO gradient!")
                grads_ok = False
            elif p.grad.isnan().any():
                print(f"  WARNING: {name} has NaN gradient!")
                grads_ok = False
            elif (p.grad == 0).all():
                print(f"  WARNING: {name} has ALL-ZERO gradient!")
                grads_ok = False

    if grads_ok:
        print("  All parameters have valid gradients ✓")

    # Show grad norms for key components
    print("\n  Gradient norms:")
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            if gn > 0:
                print(f"    {name:50s}: grad_norm={gn:.6f}")

    # --- 5. Mask sanity check ---
    print("\n--- Step 5: Mask sanity checks ---")
    cm = model.cross_attention_mask.squeeze()  # (nq, nv)
    print(f"  Cross-attention mask shape: {list(cm.shape)}")
    print(f"  Global rows (should be all 0.0):")
    for i in range(model.num_global):
        vals = cm[i].unique().tolist()
        print(f"    global[{i}]: unique values = {vals}")

    print(f"  Partial rows (should have mix of 0.0 and -inf):")
    for i in range(min(5, model.num_partial)):
        n_attend = (cm[model.num_global + i] == 0.0).sum().item()
        n_mask = (cm[model.num_global + i] == float("-inf")).sum().item()
        print(f"    partial[{i}]: attends to {n_attend} tokens, masks {n_mask}")

    # --- 6. Wrong input shape test ---
    print("\n--- Step 6: Wrong input shape handling ---")
    try:
        bad_input = torch.randn(2, 64, 768, device=device)
        with torch.no_grad():
            _ = model(bad_input)
        print("  64-token input: passed (but may have mask issues — check warnings)")
    except Exception as e:
        print(f"  64-token input: raised {type(e).__name__}: {e}")

    try:
        bad_input = torch.randn(2, 32, 512, device=device)
        with torch.no_grad():
            _ = model(bad_input)
        print("  512-dim input: passed (unexpected if input_proj is Identity)")
    except Exception as e:
        print(f"  512-dim input: raised {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)