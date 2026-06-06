"""
Evaluation Script for Med3DVLM Caption Generation
===================================================
Clean evaluation for complete saved models (LLM + vision tower + projector).

Supports:
  - Experiment 1: low_high_mlp baseline     (proj_out_num=288)
  - Experiment 2: single_scale_pargo        (proj_out_num=32)
  - Experiment 3: dual_scale_pargo          (proj_out_num=288)

Usage:
  python eval_caption.py \
      --model_path /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/exp2-single-pargo-stage3-complete \
      --output_dir /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/eval_exp2

  python eval_caption.py \
      --model_path /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/exp3-dual-pargo-stage3-complete \
      --output_dir /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/eval_exp3
"""

import argparse
import csv
import os
import random
import json
import traceback

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging
import evaluate
from types import SimpleNamespace
import glob
from safetensors import safe_open

from src.model.llm.qwen import VLMQwenForCausalLM
from src.dataset.mllm_dataset import CapDataset

print("Imports done")

# ============================================================================
# Metrics
# ============================================================================
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")
print("Metrics loaded")


# ============================================================================
# Helpers
# ============================================================================

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Med3DVLM on caption generation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the complete saved model directory")
    parser.add_argument("--output_dir", type=str, default="./output/eval/",
                        help="Directory to save evaluation results")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cap_data_path", type=str,
                        default="./data/M3D_Cap_npy/M3D_Cap.json")
    parser.add_argument("--test_size", type=int, default=None,
                        help="Number of test samples (None = all 2000)")
    parser.add_argument("--batch_size", type=int, default=1)
    # Optional: local BERT path for Exp2 on nodes without internet
    parser.add_argument("--bert_model_path", type=str, default="bert-base-uncased",
                        help="Local path or HuggingFace name for BERT (Exp2 only)")
    return parser.parse_args()


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_all_metrics(decoded_preds, decoded_labels):
    results = {}
    try:
        bleu_score = bleu_metric.compute(predictions=decoded_preds,
                                          references=decoded_labels, max_order=1)
        results["bleu"] = float(bleu_score.get("bleu", 0.0))
    except Exception:
        results["bleu"] = 0.0

    try:
        rouge_score = rouge_metric.compute(predictions=decoded_preds,
                                            references=decoded_labels,
                                            rouge_types=["rouge1"])
        results["rouge1"] = float(rouge_score.get("rouge1", 0.0))
    except Exception:
        results["rouge1"] = 0.0

    try:
        meteor_score = meteor_metric.compute(predictions=decoded_preds,
                                              references=decoded_labels)
        results["meteor"] = float(meteor_score.get("meteor", 0.0))
    except Exception:
        results["meteor"] = 0.0

    try:
        bert_score = bertscore_metric.compute(predictions=decoded_preds,
                                               references=decoded_labels,
                                               lang="en", device="cpu")
        f1s = bert_score.get("f1", [])
        results["bert_f1"] = float(sum(f1s) / len(f1s)) if f1s else 0.0
    except Exception:
        results["bert_f1"] = 0.0

    return results


# ============================================================================
# Model loading
# ============================================================================

def read_config_from_saved_model(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {model_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_projector_weights_from_safetensors(model, model_path):
    """
    Load mm_projector and vision_tower weights from safetensor shards.
    Called AFTER initialize_vision_modules has created the module structure.
    """
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        print(f"  WARNING: No safetensors files found in {model_path}")
        return 0, -1, 0

    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    projector_weights = {}
    vision_tower_weights = {}

    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mm_projector" in key:
                    # Strip outer prefix to get projector-local keys
                    clean_key = key[len("model.mm_projector."):] if key.startswith("model.mm_projector.") else key
                    projector_weights[clean_key] = f.get_tensor(key).to(device=device, dtype=model_dtype)
                elif "vision_tower" in key:
                    clean_key = key[len("model.vision_tower."):] if key.startswith("model.vision_tower.") else key
                    vision_tower_weights[clean_key] = f.get_tensor(key).to(device=device, dtype=model_dtype)

    print(f"  Found {len(projector_weights)} mm_projector tensors in safetensors")
    print(f"  Found {len(vision_tower_weights)} vision_tower tensors in safetensors")

    # Load projector
    loaded, missing_count = 0, 0
    gm = model.get_model()

    if projector_weights and hasattr(gm, "mm_projector") and gm.mm_projector is not None:
        gm.mm_projector = gm.mm_projector.to(dtype=model_dtype)
        missing_keys, unexpected_keys = gm.mm_projector.load_state_dict(
            projector_weights, strict=False
        )
        loaded = len(projector_weights) - len(unexpected_keys)
        missing_count = len(missing_keys)
        if missing_keys:
            print(f"  Projector missing keys ({len(missing_keys)}): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"  Projector unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}")
        print(f"  ✓ Loaded {loaded} projector weights")
    else:
        print("  ⚠ WARNING: mm_projector not found or empty — projector weights NOT loaded!")

    # Load vision tower
    if vision_tower_weights and hasattr(gm, "vision_tower") and gm.vision_tower is not None:
        gm.vision_tower = gm.vision_tower.to(dtype=model_dtype)
        vt_missing, vt_unexpected = gm.vision_tower.load_state_dict(
            vision_tower_weights, strict=False
        )
        vt_loaded = len(vision_tower_weights) - len(vt_unexpected)
        if vt_missing:
            print(f"  Vision tower missing keys ({len(vt_missing)}): {vt_missing[:3]}")
        print(f"  ✓ Loaded {vt_loaded} vision tower weights")
    else:
        print("  ⚠ WARNING: vision_tower not found or empty — vision tower weights NOT loaded!")

    return loaded, missing_count, 0


def load_complete_model(model_path, device, bert_model_path="bert-base-uncased"):
    """
    Load a complete saved Med3DVLM model (any experiment).

    Phase 1: Load LLM weights via from_pretrained
             (projector/VT keys in safetensors are skipped here — expected)
    Phase 2: initialize_vision_modules to create projector + VT structure
    Phase 3: Load projector + VT weights manually from safetensors
    """
    print("=" * 60)
    print(f"Loading complete model from: {model_path}")
    print("=" * 60)

    # --- Read config ---
    config_dict = read_config_from_saved_model(model_path)
    mm_projector_type = config_dict.get("mm_projector_type", "low_high_mlp")
    proj_out_num      = config_dict.get("proj_out_num", 288)
    vision_select_layer = config_dict.get("vision_select_layer", -2)
    use_pretrained_bert = config_dict.get("use_pretrained_bert", False)

    print(f"  Projector type      : {mm_projector_type}")
    print(f"  proj_out_num        : {proj_out_num}")
    print(f"  vision_select_layer : {vision_select_layer}")
    print(f"  use_pretrained_bert : {use_pretrained_bert}")

    # --- Tokenizer ---
    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "<pad>"
    if "<im_patch>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>"]})
    print(f"  Vocab size   : {len(tokenizer)}")
    print(f"  <im_patch> ID: {tokenizer.convert_tokens_to_ids('<im_patch>')}")

    # --- Phase 1: Load LLM weights ---
    # Suppress the expected flood of "unexpected key" warnings for
    # model.mm_projector.* and model.vision_tower.* — these are intentionally
    # loaded manually in Phase 3.
    print("\n  [Phase 1] Loading LLM weights via from_pretrained...")
    print("  (Warnings about unexpected mm_projector/vision_tower keys are expected)")
    hf_logging.set_verbosity_error()

    if device.type == "cuda":
        model = VLMQwenForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
        )
    else:
        model = VLMQwenForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
        )

    hf_logging.set_verbosity_warning()
    print("  ✓ LLM weights loaded")

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # --- Phase 2: Initialize vision module structure ---
    print(f"\n  [Phase 2] Initializing vision modules ({mm_projector_type})...")

    # For Exp2 (single_scale_pargo), use_pretrained_bert=True will try to
    # download bert-base-uncased. Pass --bert_model_path to a local cache
    # if the compute node has no internet access.
    if use_pretrained_bert:
        print(f"  NOTE: BERT required for {mm_projector_type}. Using: {bert_model_path}")
        print(f"  If this hangs, re-run with --bert_model_path /path/to/local/bert-base-uncased")

    model_args = SimpleNamespace(
        model_name_or_path=model_path,
        vision_tower=config_dict.get("vision_tower", "dcformer"),
        vision_select_layer=vision_select_layer,
        vision_select_feature=config_dict.get("vision_select_feature", "cls_patch"),
        pretrain_vision_model=None,
        pretrain_clip_model=None,
        freeze_vision_tower=True,
        mm_projector_type=mm_projector_type,
        proj_out_num=proj_out_num,
        mm_hidden_size=config_dict.get("mm_hidden_size", 768),
        low_input_size=config_dict.get("low_input_size", 384),
        high_input_size=config_dict.get("high_input_size", 768),
        bert_type=bert_model_path,           # use local path if provided
        num_query_tokens=proj_out_num,
        img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
        vocab_size=len(tokenizer),
        dim=config_dict.get("dim", 768),
        depth=config_dict.get("depth", 12),
        input_size=tuple(config_dict.get("input_size", [256, 256, 128])),
        patch_size=tuple(config_dict.get("patch_size", [16, 16, 16])),
        num_new_tokens=1,
        mm_mlp_depth=config_dict.get("mm_mlp_depth", 2),
        proj_layer_type=config_dict.get("proj_layer_type", "mlp"),
        proj_layer_num=config_dict.get("proj_layer_num", 2),
        proj_pooling_type=config_dict.get("proj_pooling_type", "spatial"),
        proj_pooling_size=config_dict.get("proj_pooling_size", 2),
        proj_residual=config_dict.get("proj_residual", False),
        low_output_size=config_dict.get("low_output_size", [192, 128]),
        high_output_size=config_dict.get("high_output_size", [64, 128]),
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
        use_positional_embedding=config_dict.get("use_positional_embedding", False),
        num_global_queries=config_dict.get("num_global_queries", 8),
        num_partial_queries=config_dict.get("num_partial_queries", 24),
        pargo_num_layers=config_dict.get("pargo_num_layers", 2),
        use_pretrained_bert=use_pretrained_bert,
        pargo_dropout=config_dict.get("pargo_dropout", 0.0),
    )

    # Mirror config values so build_mm_projector reads them correctly
    for attr in ["mm_projector_type", "mm_hidden_size", "low_input_size",
                 "high_input_size", "mm_mlp_depth", "proj_out_num",
                 "low_output_size", "high_output_size", "num_global_queries",
                 "num_partial_queries", "pargo_num_layers", "use_pretrained_bert",
                 "pargo_dropout"]:
        setattr(model.config, attr, getattr(model_args, attr))

    gm = model.get_model()
    gm.initialize_vision_modules(model_args=model_args)
    model.initialize_vision_tokenizer(model_args, tokenizer)
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    print("  ✓ Vision module structure created")

    # Move newly created modules to same device/dtype as LLM
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    print(f"  Moving vision modules to {model_device} (dtype={model_dtype})...")

    if hasattr(gm, "mm_projector") and gm.mm_projector is not None:
        gm.mm_projector = gm.mm_projector.to(device=model_device, dtype=model_dtype)
        print("  ✓ mm_projector moved")
    if hasattr(gm, "vision_tower") and gm.vision_tower is not None:
        gm.vision_tower = gm.vision_tower.to(device=model_device, dtype=model_dtype)
        print("  ✓ vision_tower moved")

    # --- Phase 3: Load trained projector + VT weights from safetensors ---
    print("\n  [Phase 3] Loading projector & vision tower weights from safetensors...")
    loaded, missing, _ = load_projector_weights_from_safetensors(model, model_path)

    if loaded == 0:
        raise RuntimeError(
            "No projector weights were loaded! Check that model_path contains "
            "the correct safetensor shards and that mm_projector keys are present."
        )

    model.eval()
    print(f"\n  ✓ Model ready")
    print(f"    Projector: {mm_projector_type} | tokens: {proj_out_num}")
    print("=" * 60)

    return model, tokenizer, config_dict


# ============================================================================
# Main evaluation loop
# ============================================================================

def main():
    seed_everything(42)
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return

    # ---- Load model ----
    model, tokenizer, config_dict = load_complete_model(
        args.model_path, device, bert_model_path=args.bert_model_path
    )
    proj_out_num = config_dict.get("proj_out_num", 288)

    # ---- Load dataset ----
    print(f"\nLoading test dataset...")
    data_args = SimpleNamespace(
        data_root=args.data_root,
        cap_data_path=args.cap_data_path,
        max_length=args.max_length,
        proj_out_num=proj_out_num,
        test_size=args.test_size,
    )
    test_dataset = CapDataset(data_args, tokenizer=tokenizer, mode="test",
                               test_size=args.test_size)
    print(f"Test dataset: {len(test_dataset)} samples")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    # ---- Evaluate ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv     = os.path.join(args.output_dir, "eval_caption_results.csv")
    output_summary = os.path.join(args.output_dir, "eval_summary.json")

    all_scores = {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []}
    num_errors = 0

    with open(output_csv, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Index", "Question", "Ground_Truth", "Prediction",
                         "BLEU", "ROUGE1", "METEOR", "BERTScore_F1"])

        for idx, sample in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            raw_question = sample.get("question", "")
            raw_answer   = sample.get("answer", "")

            question_text = raw_question[0] if isinstance(raw_question, (list, tuple)) else str(raw_question)
            answer_text   = raw_answer[0]   if isinstance(raw_answer,   (list, tuple)) else str(raw_answer)

            if isinstance(question_text, torch.Tensor):
                question_text = str(question_text.item()) if question_text.dim() == 0 else str(question_text)
            if isinstance(answer_text, torch.Tensor):
                answer_text = str(answer_text.item()) if answer_text.dim() == 0 else str(answer_text)

            tok_out = tokenizer(
                question_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_attention_mask=True,
            )
            input_ids      = tok_out["input_ids"].to(device)
            attention_mask = tok_out["attention_mask"].to(device)

            image = sample.get("image")
            if image is None:
                print(f"  [WARN] Sample {idx}: no image, skipping")
                num_errors += 1
                continue

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)

            model_dtype = next(model.parameters()).dtype
            image = image.to(device=device, dtype=model_dtype)
            if image.dim() == 4:
                image = image.unsqueeze(0)

            gen_kwargs = dict(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=image,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )
            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p

            try:
                with torch.no_grad():
                    generation = model.generate(**gen_kwargs)

                gen_ids = generation[0] if isinstance(generation, (list, tuple)) else generation
                if isinstance(gen_ids, torch.Tensor):
                    gen_ids = gen_ids.cpu()

                generated_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                prediction = generated_texts[0] if generated_texts else ""

            except Exception as e:
                print(f"  [ERROR] Sample {idx}: {e}")
                if idx == 0:
                    traceback.print_exc()
                prediction = "[Generation Error]"
                num_errors += 1

            decoded_preds, decoded_labels = postprocess_text([prediction], [answer_text])
            scores = compute_all_metrics(decoded_preds, decoded_labels)

            for k in all_scores:
                all_scores[k].append(scores[k])

            writer.writerow([
                idx,
                question_text[:200],
                answer_text[:200],
                prediction[:200],
                f"{scores['bleu']:.4f}",
                f"{scores['rouge1']:.4f}",
                f"{scores['meteor']:.4f}",
                f"{scores['bert_f1']:.4f}",
            ])

            if idx < 3:
                print(f"\n  --- Sample {idx} ---")
                print(f"  Q:    {question_text[:120]}")
                print(f"  GT:   {answer_text[:120]}")
                print(f"  Pred: {prediction[:120]}")
                print(f"  BLEU={scores['bleu']:.3f} ROUGE={scores['rouge1']:.3f} "
                      f"METEOR={scores['meteor']:.3f} BERT={scores['bert_f1']:.3f}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Model    : {args.model_path}")
    print(f"Projector: {config_dict.get('mm_projector_type', 'unknown')}")
    print(f"Samples  : {len(all_scores['bleu'])}  |  Errors: {num_errors}")
    print()

    summary = {
        "model_path":     args.model_path,
        "projector_type": config_dict.get("mm_projector_type", "unknown"),
        "proj_out_num":   proj_out_num,
        "num_samples":    len(all_scores["bleu"]),
        "num_errors":     num_errors,
        "metrics": {},
    }

    for metric, values in all_scores.items():
        avg = sum(values) / len(values) if values else 0.0
        summary["metrics"][metric] = avg
        print(f"  {metric.upper():>12s}: {avg:.4f}")

    print(f"\n  Results CSV : {output_csv}")
    print(f"  Summary JSON: {output_summary}")
    print(f"{'=' * 60}")

    with open(output_summary, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()