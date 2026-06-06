# eval_test.py
"""
Robust evaluation script for Med3DVLM ParGo-complete model.
This file:
- loads tokenizer + model
- attempts to (re)initialize mm_projector if missing
- adapts image shapes and channels
- handles generation fallback via inputs_embeds
- attaches a temporary dummy projector if no saved projector weights exist
"""

import argparse
import csv
import os
import random
import json
import traceback
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import evaluate

# local imports (assumes PYTHONPATH=. when running)
from src.model.llm.qwen import VLMQwenForCausalLM
from src.dataset.mllm_dataset import CapDataset
from types import SimpleNamespace
print("imports done")
# metrics
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

print("metrics set")
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output3/Med3DVLM-Qwen-2.5-7B-ParGo-Finetuned-Complete-with-pos",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--cap_data_path",
        type=str,
        default="/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/data/M3D_Cap_npy/M3D_Cap_subset.json",
    )
    parser.add_argument("--output_dir", type=str, default="./output/eval_caption_pargo_complete/")
    parser.add_argument("--proj_out_num", type=int, default=304)
    parser.add_argument("--test_size", type=int, default=None)

    # If true, attach a dummy (untrained) mm_projector when none is present.
    parser.add_argument("--attach_dummy_projector", action="store_true", default=True)

    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def safe_get_text(x):
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else ""
    if isinstance(x, torch.Tensor):
        try:
            return x.item()
        except Exception:
            try:
                return x.tolist()
            except Exception:
                return str(x)
    return x


def find_first_conv_in_channels(module):
    for name, p in module.named_parameters():
        shp = tuple(p.shape)
        if len(shp) == 5 and shp[1] >= 1:
            return shp[1], name, shp
        if len(shp) == 4 and shp[1] >= 1:
            return shp[1], name, shp
    return None, None, None


def safe_adapt_images_for_vision_tower(images: torch.Tensor, vision_tower) -> (torch.Tensor, str):
    if not isinstance(images, torch.Tensor):
        return images, "[SKIP] images not a tensor"

    if images.dim() == 4:
        images = images.unsqueeze(0)

    img_dim = images.dim()
    expected_in_channels, param_name, param_shape = find_first_conv_in_channels(vision_tower)
    if expected_in_channels is None:
        return images, "[UNKNOWN VISION TOWER - no conv param found]"

    if img_dim == 5:
        B, C, D, H, W = images.shape
        if expected_in_channels == C:
            return images, f"[OK] images shape matches vision tower (B,C,D,H,W)={images.shape}, expected in_channels={expected_in_channels} (param: {param_name}{param_shape})"
        if expected_in_channels == D and C == 1:
            new_images = images.squeeze(1)
            return new_images, f"[FIX] squeezed singleton channel axis. New shape {new_images.shape}"
        if expected_in_channels == 3 and C == 1:
            new_images = images.repeat(1, 3, 1, 1, 1)
            return new_images, f"[FIX] replicated single channel -> 3 channels. New shape {new_images.shape}"
        if expected_in_channels == 1 and C > 1:
            new_images = images.mean(dim=1, keepdim=True)
            return new_images, f"[FIX] reduced channels by mean -> {new_images.shape}"
        return images, f"[MISMATCH] images C={C}, expected_in_channels={expected_in_channels}, param={param_name}{param_shape}"

    elif img_dim == 4:
        B, C, H, W = images.shape
        if expected_in_channels == C:
            return images, f"[OK] images shape matches vision tower (B,C,H,W)={images.shape}"
        if expected_in_channels == 3 and C == 1:
            new_images = images.repeat(1, 3, 1, 1)
            return new_images, f"[FIX] replicated single channel -> 3 channels. New shape {new_images.shape}"
        return images, f"[MISMATCH-4D] images C={C}, expected_in_channels={expected_in_channels}, param={param_name}{param_shape}"
    else:
        return images, f"[UNHANDLED_DIM] images.dim()={img_dim}"


def ensure_mm_projector(model: VLMQwenForCausalLM, tokenizer: AutoTokenizer) -> bool:
    """
    Attempt to initialize mm_projector in-place using model.config and the
    model's initialize_vision_modules / initialize_vision_tokenizer helpers.
    Returns True if mm_projector is callable after the attempt.
    
    UPDATED FOR MODIFIED PARGO SUPPORT
    """
    gm = model.get_model()
    mm = getattr(gm, "mm_projector", None)
    if mm is not None and callable(mm):
        print("[ensure_mm_projector] mm_projector already present and callable.")
        return True

    print("[ensure_mm_projector] mm_projector missing or None — attempting to initialize vision modules using model.config...")

    cfg = model.config

    def cfg_get(n, d):
        return getattr(cfg, n, d)

    # CRITICAL: Check if this is modified_pargo
    mm_projector_type = cfg_get("mm_projector_type", "pargo")
    
    model_args = SimpleNamespace(
        model_name_or_path=cfg_get("model_name_or_path", cfg_get("name_or_path", "Qwen/Qwen2.5-7B-Instruct")),
        vision_tower=cfg_get("vision_tower", "dcformer"),
        vision_select_layer=cfg_get("vision_select_layer", -2),
        vision_select_feature=cfg_get("vision_select_feature", "cls_patch"),
        pretrain_vision_model=cfg_get("pretrain_vision_model", None),
        pretrain_clip_model=cfg_get("pretrain_clip_model", None),
        freeze_vision_tower=cfg_get("freeze_vision_tower", False),
        mm_projector_type=mm_projector_type,
        bert_type=cfg_get("bert_type", "bert-base-uncased"),
        num_query_tokens=cfg_get("num_query_tokens", 304),
        proj_out_num=cfg_get("proj_out_num", 304),
        img_token_id=cfg_get("img_token_id", tokenizer.convert_tokens_to_ids("<im_patch>")),
        vocab_size=cfg_get("vocab_size", len(tokenizer)),
        dim=cfg_get("dim", 768),
        depth=cfg_get("depth", 12),
        input_size=cfg_get("input_size", (256, 256, 128)),
        patch_size=cfg_get("patch_size", (16, 16, 16)),
        num_new_tokens=cfg_get("num_new_tokens", 1),
        mm_mlp_depth=cfg_get("mm_mlp_depth", 2),
        proj_layer_type=cfg_get("proj_layer_type", "mlp"),
        proj_layer_num=cfg_get("proj_layer_num", 2),
        proj_pooling_type=cfg_get("proj_pooling_type", "spatial"),
        proj_pooling_size=cfg_get("proj_pooling_size", 2),
        proj_residual=cfg_get("proj_residual", False),
        low_output_size=cfg_get("low_output_size", [192, 128]),
        high_output_size=cfg_get("high_output_size", [64, 128]),
        pretrain_mm_mlp_adapter=cfg_get("pretrain_mm_mlp_adapter", None),
        tune_mm_mlp_adapter=cfg_get("tune_mm_mlp_adapter", False),
    )
    
    # ADD MODIFIED PARGO SPECIFIC PARAMETERS
    if mm_projector_type == "modified_pargo":
        print("[ensure_mm_projector] Detected modified_pargo, adding specific parameters...")
        model_args.n_low_tokens = cfg_get("n_low_tokens", 144)
        model_args.n_high_tokens = cfg_get("n_high_tokens", 32)
        model_args.low_level_hidden_size = cfg_get("low_level_hidden_size", 384)
        model_args.pargo_num_layers = cfg_get("pargo_num_layers", 6)
        model_args.use_cross_scale_attention = cfg_get("use_cross_scale_attention", False)
        model_args.use_positional_embedding = cfg_get("use_positional_embedding", True)
        model_args.proj_out_num = cfg_get("proj_out_num", 176)  # Override to 176
        print(f"  Modified ParGo config: low={model_args.n_low_tokens}, high={model_args.n_high_tokens}, total={model_args.proj_out_num}")

    try:
        gm.initialize_vision_modules(model_args=model_args)
        model.initialize_vision_tokenizer(model_args, tokenizer)
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

        mm_after = getattr(gm, "mm_projector", None)
        
        # CRITICAL: Move projector to same device as model
        if mm_after is not None:
            device = next(model.parameters()).device
            mm_after = mm_after.to(device)
            gm.mm_projector = mm_after
        
        print("[ensure_mm_projector] mm_projector after init:", type(mm_after).__name__ if mm_after else None)
        print("[ensure_mm_projector] mm_projector device:", next(mm_after.parameters()).device if mm_after else None)
        print("[ensure_mm_projector] callable:", callable(mm_after))
        return callable(mm_after)
    except Exception as e:
        print("[ensure_mm_projector] failed to init mm_projector:", repr(e))
        traceback.print_exc()
        return False

def find_mm_projector_keys_in_index(model_path: str):
    idx = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(idx):
        try:
            d = json.load(open(idx, "r"))
            weight_map = d.get("weight_map", d)
            mm_names = [n for n in weight_map.keys() if "mm_projector" in n]
            return mm_names
        except Exception as e:
            print("Failed to read index JSON:", e)
            return []
    else:
        return []


def attach_dummy_mm_projector(model: VLMQwenForCausalLM, sample_image_tensor: torch.Tensor):
    """
    Attach a small untrained mm_projector module to model.get_model().mm_projector
    using a sample image to infer feature dimension.
    """
    gm = model.get_model()
    print("[attach_dummy_mm_projector] Attaching a small untrained mm_projector (for debugging only).")
    # Ensure tensor on same device as model
    if isinstance(sample_image_tensor, np.ndarray):
        sample_image_tensor = torch.from_numpy(sample_image_tensor)
    sample_image_tensor = sample_image_tensor.to(next(model.parameters()).device)
    if sample_image_tensor.dim() == 4:
        sample_image_tensor = sample_image_tensor.unsqueeze(0)
    # get vision tower outputs
    with torch.no_grad():
        try:
            raw = gm.vision_tower(sample_image_tensor)
        except Exception:
            # try encode_images if available
            raw = None
            try:
                raw = model.encode_images(sample_image_tensor)
            except Exception:
                raw = None

    if raw is None:
        raise RuntimeError("vision_tower/encode_images returned None; cannot infer feature dim for dummy projector")

    # raw may be tuple/list/dict
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if isinstance(raw, dict):
        # try common keys
        for k in ("last_hidden_state", "features", "output"):
            if k in raw:
                raw = raw[k]
                break

    if not isinstance(raw, torch.Tensor):
        raise RuntimeError("Could not extract tensor from vision_tower output for dummy projector")

    feat_dim = raw.shape[-1]
    out_dim = getattr(model.config, "proj_out_num", 304)
    print(f"[attach_dummy_mm_projector] inferred feat_dim={feat_dim}, proj_out={out_dim}")

    dummy = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.GELU(),
        nn.Linear(feat_dim, out_dim)
    )
    gm.mm_projector = dummy
    print("[attach_dummy_mm_projector] dummy mm_projector attached.")


def build_inputs_embeds_from_image_features(model: VLMQwenForCausalLM, input_ids: torch.Tensor, image_features: torch.Tensor):
    """
    Recreate the inputs_embeds used by the model's multimodal prepare function.
    image_features: (B, num_img_toks, hidden)
    input_ids: (B, seq_len)
    """
    gm = model.get_model()
    # token embeddings
    if not hasattr(gm, "embed_tokens") and hasattr(gm, "get_input_embeddings"):
        token_embs = gm.get_input_embeddings()(input_ids)
    else:
        token_embs = gm.embed_tokens(input_ids)

    B = token_embs.shape[0]
    num_img_toks = image_features.shape[1]
    seq_len = token_embs.shape[1]

    # Safely create new_inputs_embeds
    if seq_len <= (num_img_toks + 1):
        new_inputs_embeds = torch.cat((token_embs[:, :1, :], image_features), dim=1)
    else:
        new_inputs_embeds = torch.cat(
            (token_embs[:, :1, :], image_features, token_embs[:, (num_img_toks + 1):, :]), dim=1
        )

    # build attention mask: keep first token mask, ones for image tokens, tail mask
    # For attention_mask we assume 1s for image tokens
    return new_inputs_embeds

# def load_pargo_weights_manually(model, checkpoint_path):
#     """Manually load ParGo projector weights from checkpoint"""
#     import glob
#     from safetensors import safe_open
    
#     safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    
#     if not safetensor_files:
#         print("No safetensors files found!")
#         return False
    
#     # Get model dtype
#     model_dtype = next(model.parameters()).dtype
#     device = next(model.parameters()).device
    
#     mm_projector_state = {}
    
#     for sf_file in safetensor_files:
#         with safe_open(sf_file, framework="pt", device="cpu") as f:
#             for key in f.keys():
#                 if "mm_projector" in key:
#                     clean_key = key.replace("model.mm_projector.", "")
#                     # Load and convert to correct dtype
#                     tensor = f.get_tensor(key)
#                     mm_projector_state[clean_key] = tensor.to(device=device, dtype=model_dtype)
    
#     if not mm_projector_state:
#         print("No mm_projector weights found in checkpoint!")
#         return False
    
#     if hasattr(model.model, 'mm_projector') and model.model.mm_projector is not None:
#         try:
#             # Ensure the projector is also in correct dtype
#             model.model.mm_projector = model.model.mm_projector.to(dtype=model_dtype)
#             model.model.mm_projector.load_state_dict(mm_projector_state, strict=False)
#             print(f"Manually loaded {len(mm_projector_state)} ParGo weights in dtype {model_dtype}")
#             return True
#         except Exception as e:
#             print(f"Failed to load ParGo weights: {e}")
#             return False
    
#     return False

def load_pargo_weights_manually(model, checkpoint_path):
    """Manually load ParGo projector weights from checkpoint - UPDATED FOR MODIFIED PARGO"""
    import glob
    from safetensors import safe_open
    
    safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    
    if not safetensor_files:
        print("No safetensors files found!")
        return False
    
    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    
    mm_projector_state = {}
    
    print(f"Loading weights from {len(safetensor_files)} safetensor files...")
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mm_projector" in key:
                    # Handle both "model.mm_projector." and "mm_projector." prefixes
                    if key.startswith("model.mm_projector."):
                        clean_key = key.replace("model.mm_projector.", "")
                    elif key.startswith("mm_projector."):
                        clean_key = key.replace("mm_projector.", "")
                    else:
                        clean_key = key
                    
                    # Load and convert to correct dtype
                    tensor = f.get_tensor(key)
                    mm_projector_state[clean_key] = tensor.to(device=device, dtype=model_dtype)
                    
                    # Print first few keys to verify
                    if len(mm_projector_state) <= 3:
                        print(f"  Loaded: {key} -> {clean_key}, shape={tensor.shape}")
    
    if not mm_projector_state:
        print("No mm_projector weights found in checkpoint!")
        return False
    
    print(f"Found {len(mm_projector_state)} mm_projector parameters")
    
    # Check for modified_pargo specific keys
    pargo_keys = [k for k in mm_projector_state.keys() if 'pargo' in k.lower() or 'branch' in k.lower()]
    if pargo_keys:
        print(f"  Detected Modified ParGo weights ({len(pargo_keys)} ParGo-specific keys)")
        print(f"  Sample keys: {pargo_keys[:3]}")
    
    if hasattr(model.model, 'mm_projector') and model.model.mm_projector is not None:
        try:
            # Ensure the projector is also in correct dtype
            model.model.mm_projector = model.model.mm_projector.to(dtype=model_dtype)
            missing, unexpected = model.model.mm_projector.load_state_dict(mm_projector_state, strict=False)
            
            print(f"✓ Loaded {len(mm_projector_state)} projector weights in dtype {model_dtype}")
            if missing:
                print(f"  ⚠ Missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                print(f"  ⚠ Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
            
            return True
        except Exception as e:
            print(f"Failed to load projector weights: {e}")
            traceback.print_exc()
            return False
    else:
        print("mm_projector is None or doesn't exist!")
        return False

def main():
    seed_everything(42)
    args = parse_args()

    # device = torch.device("cpu")
    # print("Running on CPU - this will be slow!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return

    print(f"Loading complete model from {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("[INFO] tokenizer.pad_token set to tokenizer.eos_token to ensure padding behavior.")
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print("[INFO] Added <pad> token to tokenizer and will resize embeddings if needed.")

    print("Loading model with all weights (this will take time)...")
    # model = VLMQwenForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.float32,
    #     device_map="cpu",
    #     low_cpu_mem_usage=True,
    # )

    if device.type == "cuda":
        model = VLMQwenForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,  # Use fp16 for GPU
            device_map="auto",  # Automatic device mapping
            low_cpu_mem_usage=True,
        ) #simple
    else:
        model = VLMQwenForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # quick diagnostics
    print("=== DIAGNOSTICS: mm_projector / model internals ===")
    try:
        gm = model.get_model()
    except Exception as e:
        print("get_model() failed:", e)
        gm = None

    print("has mm_projector attr (get_model):", hasattr(gm, "mm_projector") if gm is not None else None)
    print("model.model.mm_projector:", getattr(model.model, "mm_projector", None))
    print("type(mm_projector):", type(getattr(gm, "mm_projector", None)))
    print("callable(mm_projector):", callable(getattr(gm, "mm_projector", None)))
    print("model.config.mm_projector_type:", getattr(model.config, "mm_projector_type", None))
    print("model.config keys (subset):", [k for k in model.config.__dict__.keys() if "proj" in k or "mm" in k or "vision" in k][:50])

    mm_state_keys = [k for k in model.state_dict().keys() if "mm_projector" in k]
    print("Number of state_dict keys containing 'mm_projector':", len(mm_state_keys))
    print("Example mm_projector keys:", mm_state_keys[:50])
    print("=== END DIAGNOSTICS ===")

    # Try to initialize mm_projector using config if missing
    init_ok = ensure_mm_projector(model, tokenizer)
    if init_ok:
        print("[MAIN] mm_projector initialized successfully.")
    else:
        print("[MAIN] ensure_mm_projector returned False or mm_projector is still not callable.")
        mm_keys = find_mm_projector_keys_in_index(args.model_path)
        if len(mm_keys) > 0:
            print(f"[MAIN] Found {len(mm_keys)} mm_projector entries in index JSON (will attempt to re-init and then load if possible). Example keys:", mm_keys[:40])
        else:
            print("[MAIN] No mm_projector keys found in index JSON or state dict - projector weights absent in checkpoint.")

    print(f"MM projector exists: {hasattr(model.get_model(), 'mm_projector')}")
    print("Model ready. Entering evaluation...")

    model.eval()

    init_ok = ensure_mm_projector(model, tokenizer)
    if init_ok:
        loaded = load_pargo_weights_manually(model, args.model_path)
        if loaded:
            print("ParGo weights loaded successfully!")
        else:
            print("Warning: ParGo weights not loaded, using random initialization")

    # Load dataset
    args.proj_out_num = 304
    if args.test_size is None:
        print("Evaluating on ALL test samples (no size limit)")
    else:
        print(f"Evaluating on {args.test_size} test samples")
    test_dataset = CapDataset(args, tokenizer=tokenizer, mode="test", test_size=args.test_size)
    print(f"Test dataset has {len(test_dataset)} samples")
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=False, shuffle=False, drop_last=False)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "pargo_eval_complete.csv")

    all_scores = {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []}

    with open(output_path, mode="w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth", "Prediction", "BLEU", "ROUGE1", "METEOR", "BERTScore_F1"])

        for idx, sample in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            raw_question = sample.get("question", "")
            raw_answer = sample.get("answer", "")

            question_text = safe_get_text(raw_question)
            if isinstance(question_text, (list, tuple)):
                question_text = question_text[0] if len(question_text) > 0 else ""
            question_text = str(question_text)

            answer_text = safe_get_text(raw_answer)
            if isinstance(answer_text, (list, tuple)):
                answer_text = answer_text[0] if len(answer_text) > 0 else ""
            answer_text = str(answer_text)

            tok_out = tokenizer(
                question_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_attention_mask=True,
            )

            input_ids = tok_out["input_ids"].to(device)
            attention_mask = tok_out.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
                attention_mask = (input_ids != pad_id).long().to(device)

            # prepare image
            image = sample.get("image")
            if image is None:
                print(f"[WARN] No image found in sample {idx}. Skipping generation.")
                generated_texts = ["[No Image]"]
            else:
                # if isinstance(image, np.ndarray):
                #     image = torch.from_numpy(image)
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                else:
                    image = image.float()
                image = image.to(device)

                model_dtype = next(model.parameters()).dtype
                image = image.to(dtype=model_dtype)
                print(f"[DEBUG] Image dtype: {image.dtype}, Model dtype: {model_dtype}")

                if image.dim() == 4:
                    image = image.unsqueeze(0)

                # adapt image channels/dims
                vision_tower = getattr(model.get_model(), "vision_tower", None)
                images_before_shape = tuple(image.shape)
                images, adapt_msg = safe_adapt_images_for_vision_tower(image, vision_tower) if vision_tower is not None else (image, "[SKIP] no vision_tower")
                print(f"[IMG DIAG] before={images_before_shape}, after={tuple(images.shape)}, adapt_msg={adapt_msg}")

                if adapt_msg.startswith("[MISMATCH]") or adapt_msg.startswith("[MISMATCH-4D]") or adapt_msg.startswith("[UNHANDLED_DIM]") or adapt_msg.startswith("[AMBIGUOUS]"):
                    print(f"[ERROR] Image shape mismatch could not be auto-adapted safely. adapt_msg={adapt_msg}")
                    generated_texts = ["[Image shape mismatch - skipped]"]
                else:
                    # ensure projector exists before encode_images
                    gm = model.get_model()
                    mm = getattr(gm, "mm_projector", None)
                    if not (mm is not None and callable(mm)):
                        print("[MAIN LOOP] mm_projector missing at generation time.")
                        # try one more init attempt
                        if ensure_mm_projector(model, tokenizer):
                            print("[MAIN LOOP] mm_projector created by ensure_mm_projector.")
                            mm = getattr(model.get_model(), "mm_projector", None)

                    # if still missing, optionally attach dummy projector (if flag)
                    if not (mm is not None and callable(mm)):
                        print("[MAIN LOOP] mm_projector still missing. Checking index JSON for mm keys...")
                        mm_keys = find_mm_projector_keys_in_index(args.model_path)
                        if len(mm_keys) > 0:
                            print(f"[MAIN LOOP] Found {len(mm_keys)} mm_projector keys in index JSON, but they weren't loaded. You may need to manually load them from shards.")
                        else:
                            print("[MAIN LOOP] No mm_projector keys found in checkpoint.")
                        if args.attach_dummy_projector:
                            try:
                                # attach dummy using this sample's images
                                attach_dummy_mm_projector(model, images)
                                mm = getattr(model.get_model(), "mm_projector", None)
                            except Exception as e:
                                print("[MAIN LOOP] Failed to attach dummy projector:", e)
                                traceback.print_exc()

                    # prepare generation kwargs
                    gen_kwargs = dict(
                        inputs=input_ids,  # Use 'inputs' instead of 'input_ids'
                        attention_mask=attention_mask,
                        images=images,
                        max_new_tokens=min(args.max_new_tokens, 128),
                        do_sample=args.do_sample,
                    )
                    if args.do_sample:
                        if args.top_p is not None:
                            gen_kwargs["top_p"] = args.top_p
                        gen_kwargs["temperature"] = args.temperature
                    if not args.do_sample and (args.top_p is not None or args.temperature != 1.0):
                        print("[WARN] top_p/temperature provided but do_sample is False. These sampling params will be ignored.")

                    # Attempt custom generate -> fallback to inputs_embeds path if it fails
                    try:
                        with torch.no_grad():
                            generation = model.generate(**gen_kwargs)

                        if isinstance(generation, (list, tuple)):
                            gen_ids = generation[0]
                        else:
                            gen_ids = generation

                        if isinstance(gen_ids, torch.Tensor):
                            gen_ids = gen_ids.cpu()
                        generated_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    except Exception as e:
                        print(f"[WARN] model.generate failed: {e}")
                        traceback.print_exc()

                        # fallback: try to compute image_features then inputs_embeds
                        try:
                            gm = model.get_model()
                            # Prefer model.encode_images if exists
                            if hasattr(model, "encode_images"):
                                image_features = model.encode_images(images)
                            else:
                                # call vision_tower and mm_projector manually
                                raw_feat = gm.vision_tower(images)
                                if isinstance(raw_feat, (tuple, list)):
                                    raw_feat = raw_feat[0]
                                image_features = gm.mm_projector(raw_feat)

                            if image_features is None:
                                raise RuntimeError("image_features is None after compute")

                            # Build inputs_embeds
                            new_inputs_embeds = build_inputs_embeds_from_image_features(model, input_ids, image_features)

                            # Build new attention mask (ones for image tokens)
                            num_img_toks = image_features.shape[1]
                            B = input_ids.shape[0]
                            ones_for_images = torch.ones((B, num_img_toks), dtype=attention_mask.dtype, device=attention_mask.device)
                            if attention_mask.shape[1] <= (num_img_toks + 1):
                                new_attention_mask = torch.cat((attention_mask[:, :1], ones_for_images), dim=1)
                            else:
                                new_attention_mask = torch.cat((attention_mask[:, :1], ones_for_images, attention_mask[:, (num_img_toks + 1):]), dim=1)

                            # Call underlying HF LM generate using inputs_embeds (avoid wrapper's custom prepare)
                            with torch.no_grad():
                                hf_gen = model.model.generate(
                                    inputs_embeds=new_inputs_embeds,
                                    attention_mask=new_attention_mask,
                                    max_new_tokens=min(args.max_new_tokens, 128),
                                    do_sample=args.do_sample,
                                    **({"top_p": args.top_p, "temperature": args.temperature} if args.do_sample else {}),
                                )

                            if isinstance(hf_gen, (list, tuple)):
                                gen_ids = hf_gen[0]
                            else:
                                gen_ids = hf_gen
                            if isinstance(gen_ids, torch.Tensor):
                                gen_ids = gen_ids.cpu()
                            generated_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                        except Exception as e2:
                            print("[ERROR] Fallback inputs_embeds generation also failed:", e2)
                            traceback.print_exc()
                            generated_texts = ["[Generation Error]"]

            # compute metrics
            result = {}
            decoded_preds, decoded_labels = postprocess_text(generated_texts, [answer_text])

            try:
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                result["bleu"] = float(bleu_score.get("bleu", 0.0))
                # result["bleu"] = 0.0
            except Exception:
                result["bleu"] = 0.0

            try:
                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rouge1"])
                result["rouge1"] = float(rouge_score.get("rouge1", 0.0))
                # result["rouge1"] = 0.0
            except Exception:
                result["rouge1"] = 0.0

            try:
                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = float(meteor_score.get("meteor", 0.0))
                # result["meteor"] = 0.0
            except Exception:
                result["meteor"] = 0.0

            try:
                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en", device="cpu")
                f1s = bert_score.get("f1", [])
                result["bert_f1"] = float(sum(f1s) / len(f1s)) if len(f1s) > 0 else 0.0
                # result["bert_f1"] = 0.0
            except Exception:
                result["bert_f1"] = 0.0

            all_scores["bleu"].append(result["bleu"])
            all_scores["rouge1"].append(result["rouge1"])
            all_scores["meteor"].append(result["meteor"])
            all_scores["bert_f1"].append(result["bert_f1"])

            print("Generated text = ", generated_texts[0] if len(generated_texts) > 0 else "Generational error")
            writer.writerow([
                question_text,
                answer_text,
                generated_texts[0] if len(generated_texts) > 0 else "",
                f"{result['bleu']:.4f}",
                f"{result['rouge1']:.4f}",
                f"{result['meteor']:.4f}",
                f"{result['bert_f1']:.4f}",
            ])

            # print(f"Sample {idx+1}/{args.test_size}: BLEU={result['bleu']:.3f}")
            if idx == 0:
                print(f"Question: {question_text[:200]}")
                print(f"Generated: {generated_texts[0][:200]}")

    print(f"\n{'='*60}")
    print("AVERAGE SCORES ACROSS ALL TEST SAMPLES:")
    print(f"{'='*60}")
    for metric, values in all_scores.items():
        if len(values) > 0:
            avg = sum(values) / len(values)
            print(f"{metric.upper()}: {avg:.4f}")
        else:
            print(f"{metric.upper()}: N/A")

    print(f"{'='*60}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
