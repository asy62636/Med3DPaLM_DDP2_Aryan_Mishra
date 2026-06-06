# eval_vqa_test.py
"""
VQA evaluation script for Med3DVLM ParGo model with positional embeddings.
Handles both open-ended and closed-ended VQA tasks.
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
from types import SimpleNamespace

from src.model.llm.qwen import VLMQwenForCausalLM
from src.dataset.mllm_dataset import VQADataset
print("imports done")

# Load metrics
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
print("metrics done")

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        # FIXED: Removed duplicate path
        default="/home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/Med3DVLM-Qwen-2.5-7B-ParGo-Complete-Pargo-with-pos-embedding",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--vqa_data_test_path",
        type=str,
        default="./data/M3D-VQA/M3D_VQA_test_subset.csv",
    )
    parser.add_argument("--close_ended", action="store_true", default=False)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/eval_vqa_pargo_with_pos/",
    )
    parser.add_argument("--proj_out_num", type=int, default=304)
    parser.add_argument("--test_size", type=int, default=None)

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

def safe_adapt_images_for_vision_tower(images: torch.Tensor, vision_tower):
    if not isinstance(images, torch.Tensor):
        return images, "[SKIP] images not a tensor"

    if images.dim() == 4:
        images = images.unsqueeze(0)

    img_dim = images.dim()
    expected_in_channels, param_name, param_shape = find_first_conv_in_channels(vision_tower)
    
    if expected_in_channels is None:
        return images, "[UNKNOWN VISION TOWER]"

    if img_dim == 5:
        B, C, D, H, W = images.shape
        if expected_in_channels == C:
            return images, f"[OK] shape matches"
        if expected_in_channels == D and C == 1:
            return images.squeeze(1), f"[FIX] squeezed channel"
        if expected_in_channels == 3 and C == 1:
            return images.repeat(1, 3, 1, 1, 1), f"[FIX] replicated to 3 channels"
        if expected_in_channels == 1 and C > 1:
            return images.mean(dim=1, keepdim=True), f"[FIX] mean channels"
    
    return images, f"[MISMATCH] C={images.shape[1]}, expected={expected_in_channels}"

def ensure_mm_projector(model, tokenizer):
    """Initialize mm_projector with support for positional embeddings"""
    gm = model.get_model()
    mm = getattr(gm, "mm_projector", None)
    if mm is not None and callable(mm):
        print("[ensure_mm_projector] mm_projector already present and callable.")
        return True

    print("[ensure_mm_projector] Initializing mm_projector from config...")
    cfg = model.config
    
    # UPDATED: Added all missing parameters including use_positional_embedding
    model_args = SimpleNamespace(
        model_name_or_path=getattr(cfg, "model_name_or_path", getattr(cfg, "name_or_path", "Qwen/Qwen2.5-7B-Instruct")),
        vision_tower=getattr(cfg, "vision_tower", "dcformer"),
        vision_select_layer=getattr(cfg, "vision_select_layer", -2),
        vision_select_feature=getattr(cfg, "vision_select_feature", "cls_patch"),
        pretrain_vision_model=getattr(cfg, "pretrain_vision_model", None),
        pretrain_clip_model=getattr(cfg, "pretrain_clip_model", None),
        freeze_vision_tower=getattr(cfg, "freeze_vision_tower", False),
        mm_projector_type=getattr(cfg, "mm_projector_type", "pargo"),
        bert_type=getattr(cfg, "bert_type", "bert-base-uncased"),
        num_query_tokens=getattr(cfg, "num_query_tokens", 304),
        proj_out_num=getattr(cfg, "proj_out_num", 304),
        dim=getattr(cfg, "dim", 768),
        depth=getattr(cfg, "depth", 12),
        input_size=getattr(cfg, "input_size", (256, 256, 128)),
        patch_size=getattr(cfg, "patch_size", (16, 16, 16)),
        num_new_tokens=getattr(cfg, "num_new_tokens", 1),
        mm_mlp_depth=getattr(cfg, "mm_mlp_depth", 2),
        proj_layer_type=getattr(cfg, "proj_layer_type", "mlp"),
        proj_layer_num=getattr(cfg, "proj_layer_num", 2),
        proj_pooling_type=getattr(cfg, "proj_pooling_type", "spatial"),
        proj_pooling_size=getattr(cfg, "proj_pooling_size", 2),
        proj_residual=getattr(cfg, "proj_residual", False),
        low_output_size=getattr(cfg, "low_output_size", [192, 128]),
        high_output_size=getattr(cfg, "high_output_size", [64, 128]),
        pretrain_mm_mlp_adapter=getattr(cfg, "pretrain_mm_mlp_adapter", None),
        tune_mm_mlp_adapter=getattr(cfg, "tune_mm_mlp_adapter", False),
        img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
        vocab_size=len(tokenizer),
        use_positional_embedding=getattr(cfg, "use_positional_embedding", True),  # ADDED
    )

    try:
        gm.initialize_vision_modules(model_args=model_args)
        model.initialize_vision_tokenizer(model_args, tokenizer)
        
        mm_after = getattr(gm, "mm_projector", None)
        if mm_after is not None:
            device = next(model.parameters()).device
            mm_after = mm_after.to(device)
            gm.mm_projector = mm_after
            print(f"[ensure_mm_projector] mm_projector moved to {device}")
        
        return callable(mm_after)
    except Exception as e:
        print(f"[ERROR] Failed to init mm_projector: {e}")
        traceback.print_exc()
        return False

def load_pargo_weights_manually(model, checkpoint_path):
    """Manually load ParGo projector weights from checkpoint"""
    import glob
    from safetensors import safe_open
    
    safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    if not safetensor_files:
        print("No safetensors files found!")
        return False
    
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    mm_projector_state = {}
    
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mm_projector" in key:
                    clean_key = key.replace("model.mm_projector.", "")
                    tensor = f.get_tensor(key)
                    mm_projector_state[clean_key] = tensor.to(device=device, dtype=model_dtype)
    
    if not mm_projector_state:
        print("No mm_projector weights found in checkpoint!")
        return False
    
    if hasattr(model.model, 'mm_projector') and model.model.mm_projector is not None:
        try:
            model.model.mm_projector = model.model.mm_projector.to(dtype=model_dtype)
            model.model.mm_projector.load_state_dict(mm_projector_state, strict=False)
            print(f"✓ Manually loaded {len(mm_projector_state)} ParGo weights in dtype {model_dtype}")
            return True
        except Exception as e:
            print(f"Failed to load ParGo weights: {e}")
            traceback.print_exc()
            return False
    
    return False

def build_inputs_embeds_from_image_features(model, input_ids, image_features):
    """Build inputs_embeds for fallback generation"""
    gm = model.get_model()
    if hasattr(gm, "embed_tokens"):
        token_embs = gm.embed_tokens(input_ids)
    else:
        token_embs = gm.get_input_embeddings()(input_ids)

    num_img_toks = image_features.shape[1]
    seq_len = token_embs.shape[1]

    if seq_len <= (num_img_toks + 1):
        new_inputs_embeds = torch.cat((token_embs[:, :1, :], image_features), dim=1)
    else:
        new_inputs_embeds = torch.cat(
            (token_embs[:, :1, :], image_features, token_embs[:, (num_img_toks + 1):, :]), 
            dim=1
        )
    
    return new_inputs_embeds

def main():
    seed_everything(42)
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("[INFO] Set pad_token to eos_token")
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print("[INFO] Added <pad> token")

    print("Loading model...")
    if device.type == "cuda":
        model = VLMQwenForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
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

    # ADDED: Diagnostics
    print("\n=== DIAGNOSTICS: mm_projector / model internals ===")
    try:
        gm = model.get_model()
    except Exception as e:
        print("get_model() failed:", e)
        gm = None

    print("has mm_projector attr:", hasattr(gm, "mm_projector") if gm is not None else None)
    print("model.model.mm_projector:", getattr(model.model, "mm_projector", None))
    print("callable(mm_projector):", callable(getattr(gm, "mm_projector", None)) if gm else None)
    print("model.config.mm_projector_type:", getattr(model.config, "mm_projector_type", None))
    print("model.config.use_positional_embedding:", getattr(model.config, "use_positional_embedding", None))
    
    mm_state_keys = [k for k in model.state_dict().keys() if "mm_projector" in k]
    print("Number of mm_projector keys in state_dict:", len(mm_state_keys))
    print("Example keys:", mm_state_keys[:10])
    print("=== END DIAGNOSTICS ===\n")

    # Initialize and load projector
    init_ok = ensure_mm_projector(model, tokenizer)
    if init_ok:
        loaded = load_pargo_weights_manually(model, args.model_path)
        if loaded:
            print("✓ ParGo weights loaded successfully!")
        else:
            print("⚠ Warning: ParGo weights not loaded, using random initialization")
    else:
        print("⚠ Warning: mm_projector initialization failed")

    model.eval()
    print("Model ready for evaluation\n")

    # Load dataset
    test_dataset = VQADataset(
        args, 
        tokenizer=tokenizer, 
        close_ended=args.close_ended, 
        mode="test"
    )

    if args.test_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(args.test_size, len(test_dataset))))
        print(f"Limited to {len(test_dataset)} samples for testing")
    
    print(f"Test dataset: {len(test_dataset)} samples")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=0, 
        shuffle=False, 
        drop_last=False
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    Question_Type = {1: "Plane", 2: "Phase", 3: "Organ", 4: "Abnormality", 5: "Location"}

    if args.close_ended:
        print("="*60)
        print("EVALUATING CLOSE-ENDED VQA")
        print("="*60)
        output_path = os.path.join(args.output_dir, "pargo_eval_close_vqa.csv")
        
        with open(output_path, mode="w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow([
                "Question Type", "Question", "Answer", 
                "Answer Choice", "Pred", "Correct"
            ])
            
            for idx, sample in enumerate(tqdm(test_dataloader, desc="Evaluating Close-Ended")):
                question = safe_get_text(sample["question"])
                question_type = sample["question_type"].item()
                answer_choice = safe_get_text(sample["answer_choice"])
                answer = safe_get_text(sample["answer"])

                # Tokenize
                tok_out = tokenizer(
                    question,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                input_ids = tok_out["input_ids"].to(device)
                attention_mask = tok_out.get("attention_mask", 
                    (input_ids != tokenizer.pad_token_id).long()).to(device)

                # Prepare image
                image = sample.get("image")
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                else:
                    image = image.float()
                
                image = image.to(device)
                model_dtype = next(model.parameters()).dtype
                image = image.to(dtype=model_dtype)

                if image.dim() == 4:
                    image = image.unsqueeze(0)

                # Adapt image
                vision_tower = model.get_model().vision_tower
                images, adapt_msg = safe_adapt_images_for_vision_tower(image, vision_tower)

                # Generate
                gen_kwargs = dict(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    max_new_tokens=min(args.max_new_tokens, 128),
                    do_sample=args.do_sample,
                )
                if args.do_sample:
                    if args.top_p is not None:
                        gen_kwargs["top_p"] = args.top_p
                    gen_kwargs["temperature"] = args.temperature

                try:
                    with torch.no_grad():
                        generation = model.generate(**gen_kwargs)
                    
                    if isinstance(generation, (list, tuple)):
                        gen_ids = generation[0]
                    else:
                        gen_ids = generation
                    
                    generated_texts = tokenizer.batch_decode(
                        gen_ids.cpu(), skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"[WARN] Generation failed for sample {idx}: {e}")
                    generated_texts = ["[Error]"]

                # Check correctness
                correct = 1 if answer_choice + "." in generated_texts[0] else 0

                writer.writerow([
                    question_type, question, answer,
                    answer_choice, generated_texts[0], correct
                ])

        # Compute accuracy by type
        with open(output_path, mode="r") as infile:
            reader = csv.DictReader(infile)
            total = [0] * 5
            correct_count = [0] * 5
            
            for row in reader:
                idx = int(row["Question Type"]) - 1
                total[idx] += 1
                if row["Correct"] == "1":
                    correct_count[idx] += 1

        print("\n" + "="*60)
        print("CLOSED-ENDED VQA ACCURACY:")
        print("="*60)
        for i in range(5):
            if total[i] > 0:
                acc = correct_count[i] / total[i]
                print(f"{Question_Type[i+1]:12s}: {acc:.4f} ({correct_count[i]}/{total[i]})")
        
        overall_acc = sum(correct_count) / sum(total) if sum(total) > 0 else 0
        print(f"{'Overall':12s}: {overall_acc:.4f} ({sum(correct_count)}/{sum(total)})")
        print("="*60)

    else:
        print("="*60)
        print("EVALUATING OPEN-ENDED VQA")
        print("="*60)
        output_path = os.path.join(args.output_dir, "pargo_eval_open_vqa.csv")
        
        all_scores_by_type = {str(i): {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []} 
                              for i in range(1, 6)}
        
        with open(output_path, mode="w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow([
                "Question Type", "Question", "Answer", "Pred",
                "BLEU", "ROUGE1", "METEOR", "BERTScore_F1"
            ])
            
            for idx, sample in enumerate(tqdm(test_dataloader, desc="Evaluating Open-Ended")):
                question = safe_get_text(sample["question"])
                question_type = sample["question_type"].item()
                answer = safe_get_text(sample["answer"])

                # Tokenize
                tok_out = tokenizer(
                    question,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                input_ids = tok_out["input_ids"].to(device)
                attention_mask = tok_out.get("attention_mask",
                    (input_ids != tokenizer.pad_token_id).long()).to(device)

                # Prepare image
                image = sample.get("image")
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                else:
                    image = image.float()
                
                image = image.to(device)
                model_dtype = next(model.parameters()).dtype
                image = image.to(dtype=model_dtype)

                if image.dim() == 4:
                    image = image.unsqueeze(0)

                # Adapt image
                vision_tower = model.get_model().vision_tower
                images, adapt_msg = safe_adapt_images_for_vision_tower(image, vision_tower)

                # Generate
                gen_kwargs = dict(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    max_new_tokens=min(args.max_new_tokens, 128),
                    do_sample=args.do_sample,
                )
                if args.do_sample:
                    if args.top_p is not None:
                        gen_kwargs["top_p"] = args.top_p
                    gen_kwargs["temperature"] = args.temperature

                try:
                    with torch.no_grad():
                        generation = model.generate(**gen_kwargs)
                    
                    if isinstance(generation, (list, tuple)):
                        gen_ids = generation[0]
                    else:
                        gen_ids = generation
                    
                    generated_texts = tokenizer.batch_decode(
                        gen_ids.cpu(), skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"[WARN] Generation failed for sample {idx}: {e}")
                    generated_texts = ["[Error]"]

                # Compute metrics
                result = {}
                decoded_preds, decoded_labels = postprocess_text(
                    generated_texts, [answer]
                )

                try:
                    bleu_score = bleu.compute(
                        predictions=decoded_preds, 
                        references=decoded_labels, 
                        max_order=1
                    )
                    result["bleu"] = float(bleu_score.get("bleu", 0.0))
                except:
                    result["bleu"] = 0.0

                try:
                    rouge_score = rouge.compute(
                        predictions=decoded_preds,
                        references=decoded_labels,
                        rouge_types=["rouge1"]
                    )
                    result["rouge1"] = float(rouge_score.get("rouge1", 0.0))
                except:
                    result["rouge1"] = 0.0

                try:
                    meteor_score = meteor.compute(
                        predictions=decoded_preds, 
                        references=decoded_labels
                    )
                    result["meteor"] = float(meteor_score.get("meteor", 0.0))
                except:
                    result["meteor"] = 0.0

                try:
                    bert_score = bertscore.compute(
                        predictions=decoded_preds,
                        references=decoded_labels,
                        lang="en",
                        device="cpu"
                    )
                    f1s = bert_score.get("f1", [])
                    result["bert_f1"] = float(sum(f1s) / len(f1s)) if len(f1s) > 0 else 0.0
                except:
                    result["bert_f1"] = 0.0

                # Store by type
                type_key = str(question_type)
                all_scores_by_type[type_key]["bleu"].append(result["bleu"])
                all_scores_by_type[type_key]["rouge1"].append(result["rouge1"])
                all_scores_by_type[type_key]["meteor"].append(result["meteor"])
                all_scores_by_type[type_key]["bert_f1"].append(result["bert_f1"])

                writer.writerow([
                    question_type, question, answer, generated_texts[0],
                    f"{result['bleu']:.4f}",
                    f"{result['rouge1']:.4f}",
                    f"{result['meteor']:.4f}",
                    f"{result['bert_f1']:.4f}",
                ])

                # Print first sample for debugging
                if idx == 0:
                    print(f"\nSample output:")
                    print(f"Q: {question[:200]}")
                    print(f"A: {answer[:200]}")
                    print(f"P: {generated_texts[0][:200]}\n")

        # Print results by type
        print("\n" + "="*60)
        print("OPEN-ENDED VQA SCORES BY TYPE:")
        print("="*60)
        
        overall_scores = {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []}
        
        for type_idx in range(1, 6):
            type_key = str(type_idx)
            scores = all_scores_by_type[type_key]
            
            if len(scores["bleu"]) > 0:
                print(f"\n{Question_Type[type_idx]}:")
                for metric in ["bleu", "rouge1", "meteor", "bert_f1"]:
                    if len(scores[metric]) > 0:
                        avg = sum(scores[metric]) / len(scores[metric])
                        print(f"  {metric.upper():12s}: {avg:.4f}")
                        overall_scores[metric].extend(scores[metric])
        
        # Overall averages
        print(f"\n{'Overall':}:")
        for metric in ["bleu", "rouge1", "meteor", "bert_f1"]:
            if len(overall_scores[metric]) > 0:
                avg = sum(overall_scores[metric]) / len(overall_scores[metric])
                print(f"  {metric.upper():12s}: {avg:.4f}")
        
        print("="*60)

    print(f"\nResults saved to {output_path}")
    print("Evaluation complete!")

if __name__ == "__main__":
    main()