"""
VQA Evaluation Script — Med3DVLM (All Experiments)
====================================================
Supports open-ended and closed-ended VQA for:
  Exp 1: low_high_mlp  baseline  (proj_out_num=288)
  Exp 2: single_scale_pargo      (proj_out_num=32)
  Exp 3: dual_scale_pargo        (proj_out_num=288)

Usage:
  # Open-ended
  python eval_vqa.py --model_path ./output/exp2-single-pargo-stage3-complete \
      --output_dir ./output/eval_vqa_exp2 --test_size 100

  # Closed-ended
  python eval_vqa.py --model_path ./output/exp2-single-pargo-stage3-complete \
      --output_dir ./output/eval_vqa_exp2 --close_ended --test_size 100
"""

import argparse
import csv
import os
import random
import json
import traceback
import glob

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging
import evaluate
from types import SimpleNamespace
from safetensors import safe_open

from src.model.llm.qwen import VLMQwenForCausalLM
from src.dataset.mllm_dataset import VQADataset

print("Imports done")

# ============================================================================
# Metrics
# ============================================================================
bleu      = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor    = evaluate.load("meteor")
rouge     = evaluate.load("rouge")
print("Metrics loaded")


# ============================================================================
# Helpers
# ============================================================================

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate Med3DVLM on open/closed-ended VQA"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/eval_vqa/")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--vqa_data_test_path", type=str,
                        default="./data/M3D-VQA/M3D_VQA_test.csv")
    parser.add_argument("--close_ended", action="store_true", default=False)
    parser.add_argument("--test_size", type=int, default=None)
    # For Exp2: local BERT path if compute node has no internet
    parser.add_argument("--bert_model_path", type=str, default="bert-base-uncased",
                        help="Local path or HF name for BERT (Exp2 only)")
    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds  = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def safe_get_text(x):
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else ""
    if isinstance(x, torch.Tensor):
        try:
            return x.item()
        except Exception:
            return str(x.tolist())
    return x


def find_first_conv_in_channels(module):
    for _, p in module.named_parameters():
        shp = tuple(p.shape)
        if len(shp) == 5 and shp[1] >= 1:
            return shp[1]
        if len(shp) == 4 and shp[1] >= 1:
            return shp[1]
    return None


def safe_adapt_images_for_vision_tower(images: torch.Tensor, vision_tower):
    if not isinstance(images, torch.Tensor):
        return images
    if images.dim() == 4:
        images = images.unsqueeze(0)
    expected_in_channels = find_first_conv_in_channels(vision_tower)
    if expected_in_channels is None:
        return images
    B, C, D, H, W = images.shape
    if expected_in_channels == C:
        return images
    if expected_in_channels == D and C == 1:
        return images.squeeze(1)
    if expected_in_channels == 3 and C == 1:
        return images.repeat(1, 3, 1, 1, 1)
    if expected_in_channels == 1 and C > 1:
        return images.mean(dim=1, keepdim=True)
    return images


# ============================================================================
# Model loading
# ============================================================================

def read_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {model_path}")
    with open(config_path) as f:
        return json.load(f)


def _load_weights_from_safetensors(model, model_path, model_dtype, model_device):
    """Load mm_projector and vision_tower weights from all safetensor shards."""
    sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not sf_files:
        print("  WARNING: No safetensors files found — weights not loaded!")
        return

    proj_weights = {}
    vt_weights   = {}

    for sf_file in sf_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key).to(device=model_device, dtype=model_dtype)
                if "mm_projector" in key:
                    clean = key[len("model.mm_projector."):] if key.startswith("model.mm_projector.") else key
                    proj_weights[clean] = tensor
                elif "vision_tower" in key:
                    clean = key[len("model.vision_tower."):] if key.startswith("model.vision_tower.") else key
                    vt_weights[clean] = tensor

    gm = model.get_model()

    if proj_weights and hasattr(gm, "mm_projector") and gm.mm_projector is not None:
        gm.mm_projector = gm.mm_projector.to(dtype=model_dtype)
        missing, unexpected = gm.mm_projector.load_state_dict(proj_weights, strict=False)
        loaded = len(proj_weights) - len(unexpected)
        print(f"  ✓ Projector: loaded {loaded}/{len(proj_weights)} weights")
        if missing:
            print(f"    Missing   ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"    Unexpected ({len(unexpected)}): {unexpected[:3]}")
        if loaded == 0:
            raise RuntimeError(
                "No projector weights loaded! Check safetensor keys match "
                "model.mm_projector.* prefix."
            )
    else:
        print("  WARNING: Could not load projector weights")

    if vt_weights and hasattr(gm, "vision_tower") and gm.vision_tower is not None:
        gm.vision_tower = gm.vision_tower.to(dtype=model_dtype)
        missing, unexpected = gm.vision_tower.load_state_dict(vt_weights, strict=False)
        loaded = len(vt_weights) - len(unexpected)
        print(f"  ✓ Vision tower: loaded {loaded}/{len(vt_weights)} weights")
    else:
        print("  INFO: No vision tower weights loaded")


def load_complete_model(model_path, device, bert_model_path="bert-base-uncased"):
    """
    Three-phase loading:
      Phase 1 — LLM weights via from_pretrained
      Phase 2 — initialize_vision_modules to create projector + VT structure
      Phase 3 — load projector + VT weights manually from safetensors
    """
    print("=" * 60)
    print(f"Loading model from: {model_path}")
    print("=" * 60)

    config_dict        = read_config(model_path)
    mm_projector_type  = config_dict.get("mm_projector_type", "low_high_mlp")
    proj_out_num       = config_dict.get("proj_out_num", 288)
    vision_select_layer = config_dict.get("vision_select_layer", -2)
    use_pretrained_bert = config_dict.get("use_pretrained_bert", False)

    print(f"  Projector type      : {mm_projector_type}")
    print(f"  proj_out_num        : {proj_out_num}")
    print(f"  vision_select_layer : {vision_select_layer}")
    print(f"  use_pretrained_bert : {use_pretrained_bert}")

    # Tokenizer
    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=512, padding_side="right", use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "<pad>"
    if "<im_patch>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>"]})
    print(f"  Vocab size   : {len(tokenizer)}")
    print(f"  <im_patch> ID: {tokenizer.convert_tokens_to_ids('<im_patch>')}")

    # --- Phase 1: Load LLM weights ---
    # Suppress expected flood of "unexpected key" warnings for
    # model.mm_projector.* and model.vision_tower.* keys —
    # these are loaded manually in Phase 3.
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
    # NOTE: Always run this — do NOT short-circuit based on mm_projector key
    # count. Even if keys exist in the state dict, the module structure
    # (nn.Module objects) may not have been created by from_pretrained.
    print(f"\n  [Phase 2] Initializing vision modules ({mm_projector_type})...")

    if use_pretrained_bert:
        print(f"  NOTE: BERT required for {mm_projector_type}. Using: {bert_model_path}")
        print(f"  If this hangs, re-run with --bert_model_path /path/to/local/bert")

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
        bert_type=bert_model_path,
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

    for attr in [
        "mm_projector_type", "mm_hidden_size", "low_input_size", "high_input_size",
        "mm_mlp_depth", "proj_out_num", "low_output_size", "high_output_size",
        "num_global_queries", "num_partial_queries", "pargo_num_layers",
        "use_pretrained_bert", "pargo_dropout", "use_positional_embedding",
    ]:
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
    model_dtype  = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    print(f"  Moving vision modules to {model_device} ({model_dtype})...")
    if hasattr(gm, "mm_projector") and gm.mm_projector is not None:
        gm.mm_projector = gm.mm_projector.to(device=model_device, dtype=model_dtype)
        print("  ✓ mm_projector moved")
    if hasattr(gm, "vision_tower") and gm.vision_tower is not None:
        gm.vision_tower = gm.vision_tower.to(device=model_device, dtype=model_dtype)
        print("  ✓ vision_tower moved")

    # --- Phase 3: Load trained projector + VT weights from safetensors ---
    print("\n  [Phase 3] Loading projector & vision tower weights from safetensors...")
    _load_weights_from_safetensors(model, model_path, model_dtype, model_device)

    model.eval()
    print(f"\n  ✓ Model ready | projector: {mm_projector_type} | tokens: {proj_out_num}")
    print("=" * 60)
    return model, tokenizer, config_dict


# ============================================================================
# Metrics helpers
# ============================================================================

def compute_open_metrics(preds, labels):
    result = {}
    try:
        result["bleu"] = float(
            bleu.compute(predictions=preds, references=labels, max_order=1).get("bleu", 0.0)
        )
    except Exception:
        result["bleu"] = 0.0
    try:
        result["rouge1"] = float(
            rouge.compute(predictions=preds, references=labels,
                          rouge_types=["rouge1"]).get("rouge1", 0.0)
        )
    except Exception:
        result["rouge1"] = 0.0
    try:
        result["meteor"] = float(
            meteor.compute(predictions=preds, references=labels).get("meteor", 0.0)
        )
    except Exception:
        result["meteor"] = 0.0
    try:
        f1s = bertscore.compute(
            predictions=preds, references=labels, lang="en", device="cpu"
        ).get("f1", [])
        result["bert_f1"] = float(sum(f1s) / len(f1s)) if f1s else 0.0
    except Exception:
        result["bert_f1"] = 0.0
    return result


def generate_answer(model, tokenizer, input_ids, attention_mask, images, args, device):
    gen_kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=images,
        max_new_tokens=min(args.max_new_tokens, 128),
        do_sample=args.do_sample,
    )
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        generation = model.generate(**gen_kwargs)
    gen_ids = generation[0] if isinstance(generation, (list, tuple)) else generation
    texts = tokenizer.batch_decode(gen_ids.cpu(), skip_special_tokens=True)
    return texts[0] if texts else ""


# ============================================================================
# Evaluation loops
# ============================================================================

Question_Type = {1: "Plane", 2: "Phase", 3: "Organ", 4: "Abnormality", 5: "Location"}


def eval_closed(model, tokenizer, dataloader, args, device, output_dir):
    print("\n" + "=" * 60)
    print("EVALUATING CLOSED-ENDED VQA")
    print("=" * 60)

    output_path = os.path.join(output_dir, "eval_close_vqa.csv")

    with open(output_path, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question Type", "Question", "Answer",
                         "Answer Choice", "Pred", "Correct"])

        for idx, sample in enumerate(tqdm(dataloader, desc="Closed-Ended")):
            question      = safe_get_text(sample["question"])
            question_type = sample["question_type"].item()
            answer_choice = safe_get_text(sample["answer_choice"])
            answer        = safe_get_text(sample["answer"])

            tok_out   = tokenizer(question, return_tensors="pt", padding=True,
                                  truncation=True, max_length=args.max_length)
            input_ids = tok_out["input_ids"].to(device)
            attn_mask = tok_out.get(
                "attention_mask",
                (input_ids != tokenizer.pad_token_id).long()
            ).to(device)

            image = sample.get("image")
            image = torch.from_numpy(image).float() if isinstance(image, np.ndarray) else image.float()
            image = image.to(device=device, dtype=next(model.parameters()).dtype)
            if image.dim() == 4:
                image = image.unsqueeze(0)
            image = safe_adapt_images_for_vision_tower(image, model.get_model().vision_tower)

            try:
                pred = generate_answer(model, tokenizer, input_ids, attn_mask,
                                       image, args, device)
            except Exception as e:
                print(f"  [WARN] Sample {idx} generation failed: {e}")
                if idx == 0:
                    traceback.print_exc()
                pred = "[Error]"

            correct = 1 if (answer_choice + ".") in pred else 0
            writer.writerow([question_type, question, answer, answer_choice, pred, correct])

            if idx == 0:
                print(f"\n  Sample 0:")
                print(f"  Q: {question[:150]}")
                print(f"  GT choice: {answer_choice}  |  Pred: {pred[:100]}")
                print(f"  Correct: {correct}")

    # Accuracy by type
    total         = [0] * 5
    correct_count = [0] * 5
    with open(output_path) as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            i = int(row["Question Type"]) - 1
            total[i] += 1
            if row["Correct"] == "1":
                correct_count[i] += 1

    summary = {"projector_type": args._projector_type, "task": "closed_vqa", "metrics": {}}
    print("\n" + "=" * 60)
    print("CLOSED-ENDED VQA ACCURACY:")
    print("=" * 60)
    for i in range(5):
        if total[i] > 0:
            acc = correct_count[i] / total[i]
            summary["metrics"][Question_Type[i + 1]] = round(acc, 4)
            print(f"  {Question_Type[i+1]:14s}: {acc:.4f}  ({correct_count[i]}/{total[i]})")
    overall = sum(correct_count) / sum(total) if sum(total) > 0 else 0
    summary["metrics"]["Overall"] = round(overall, 4)
    print(f"  {'Overall':14s}: {overall:.4f}  ({sum(correct_count)}/{sum(total)})")
    print("=" * 60)

    summary_path = os.path.join(output_dir, "eval_close_vqa_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results CSV : {output_path}")
    print(f"  Summary JSON: {summary_path}")


def eval_open(model, tokenizer, dataloader, args, device, output_dir):
    print("\n" + "=" * 60)
    print("EVALUATING OPEN-ENDED VQA")
    print("=" * 60)

    output_path    = os.path.join(output_dir, "eval_open_vqa.csv")
    scores_by_type = {str(i): {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []}
                      for i in range(1, 6)}

    with open(output_path, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question Type", "Question", "Answer", "Pred",
                         "BLEU", "ROUGE1", "METEOR", "BERTScore_F1"])

        for idx, sample in enumerate(tqdm(dataloader, desc="Open-Ended")):
            question      = safe_get_text(sample["question"])
            question_type = sample["question_type"].item()
            answer        = safe_get_text(sample["answer"])

            tok_out   = tokenizer(question, return_tensors="pt", padding=True,
                                  truncation=True, max_length=args.max_length)
            input_ids = tok_out["input_ids"].to(device)
            attn_mask = tok_out.get(
                "attention_mask",
                (input_ids != tokenizer.pad_token_id).long()
            ).to(device)

            image = sample.get("image")
            image = torch.from_numpy(image).float() if isinstance(image, np.ndarray) else image.float()
            image = image.to(device=device, dtype=next(model.parameters()).dtype)
            if image.dim() == 4:
                image = image.unsqueeze(0)
            image = safe_adapt_images_for_vision_tower(image, model.get_model().vision_tower)

            try:
                pred = generate_answer(model, tokenizer, input_ids, attn_mask,
                                       image, args, device)
            except Exception as e:
                print(f"  [WARN] Sample {idx} generation failed: {e}")
                if idx == 0:
                    traceback.print_exc()
                pred = "[Error]"

            decoded_preds, decoded_labels = postprocess_text([pred], [answer])
            scores = compute_open_metrics(decoded_preds, decoded_labels)

            type_key = str(question_type)
            for m in scores:
                scores_by_type[type_key][m].append(scores[m])

            writer.writerow([
                question_type, question, answer, pred,
                f"{scores['bleu']:.4f}", f"{scores['rouge1']:.4f}",
                f"{scores['meteor']:.4f}", f"{scores['bert_f1']:.4f}",
            ])

            if idx == 0:
                print(f"\n  Sample 0:")
                print(f"  Q: {question[:150]}")
                print(f"  GT: {answer[:100]}")
                print(f"  Pred: {pred[:100]}")

    # Results by type
    summary = {"projector_type": args._projector_type, "task": "open_vqa", "metrics": {}}
    overall = {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []}

    print("\n" + "=" * 60)
    print("OPEN-ENDED VQA SCORES BY TYPE:")
    print("=" * 60)
    for i in range(1, 6):
        key    = str(i)
        scores = scores_by_type[key]
        if scores["bleu"]:
            print(f"\n  {Question_Type[i]}:")
            summary["metrics"][Question_Type[i]] = {}
            for m in ["bleu", "rouge1", "meteor", "bert_f1"]:
                if scores[m]:
                    avg = sum(scores[m]) / len(scores[m])
                    print(f"    {m.upper():12s}: {avg:.4f}")
                    summary["metrics"][Question_Type[i]][m] = round(avg, 4)
                    overall[m].extend(scores[m])

    print(f"\n  Overall:")
    summary["metrics"]["Overall"] = {}
    for m in ["bleu", "rouge1", "meteor", "bert_f1"]:
        if overall[m]:
            avg = sum(overall[m]) / len(overall[m])
            print(f"    {m.upper():12s}: {avg:.4f}")
            summary["metrics"]["Overall"][m] = round(avg, 4)
    print("=" * 60)

    summary_path = os.path.join(output_dir, "eval_open_vqa_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results CSV : {output_path}")
    print(f"  Summary JSON: {summary_path}")


# ============================================================================
# Main
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

    model, tokenizer, config_dict = load_complete_model(
        args.model_path, device, bert_model_path=args.bert_model_path
    )

    args._projector_type = config_dict.get("mm_projector_type", "unknown")
    proj_out_num         = config_dict.get("proj_out_num", 288)

    print(f"\nLoading VQA test dataset...")
    task_args = SimpleNamespace(
        data_root=args.data_root,
        vqa_data_test_path=args.vqa_data_test_path,
        max_length=args.max_length,
        proj_out_num=proj_out_num,
    )
    test_dataset = VQADataset(
        task_args,
        tokenizer=tokenizer,
        close_ended=args.close_ended,
        mode="test",
    )
    if args.test_size is not None:
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(min(args.test_size, len(test_dataset)))
        )
    print(f"Test samples: {len(test_dataset)}")

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.close_ended:
        eval_closed(model, tokenizer, test_dataloader, args, device, args.output_dir)
    else:
        eval_open(model, tokenizer, test_dataloader, args, device, args.output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()