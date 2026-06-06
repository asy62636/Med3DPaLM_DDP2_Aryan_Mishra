"""
Experiment 1: Clean Baseline Training Script
=============================================
Based on the original Med3DVLM train_vlm.py with minimal improvements:
  - Better save logic (tokenizer + config always saved)
  - Optional wandb (controlled by flag)
  - Checkpoint resume logic

This script supports ALL original projector types:
  - "mlp"           -> 32 tokens  (single-scale MLP)
  - "low_high_mlp"  -> 288 tokens (2x MLP-Mixer-H, Med3DVLM's best)
  - "mhsa"          -> 32 tokens  (multi-head self-attention)

Usage:
  Stage 2 (Projector Pretraining):
    bash scripts/exp1_stage2_pretrain.sh
  Stage 3 (LoRA Fine-tuning):
    bash scripts/exp1_stage3_finetune.sh
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

from src.dataset.mllm_dataset import CapDataset, TextDatasets, TextYNDatasets
from src.model.llm.qwen import VLMQwenForCausalLM
from src.train.trainer import MLLMTrainer


# ============================================================================
# Utility functions (unchanged from original)
# ============================================================================

def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args):
    if is_rank_zero():
        print(*args)


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, tokenizer=None):
    """Collects the state dict and dump to disk. Improved: also saves tokenizer and config."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Stage 2: Only save projector and embed_tokens
        keys_to_match = ["mm_projector", "embed_tokens", "embeddings"]

        weight_to_save = get_mm_projector_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, "mm_projector.bin")
                )
            
            # Save tokenizer alongside projector weights
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        trainer.model.config.save_pretrained(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        trainer.model.config.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = [
        "vision_tower",
        "mm_projector",
        "embed_tokens",
        "lm_head",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


# ============================================================================
# Arguments — matching original Med3DVLM exactly
# ============================================================================

@dataclass
class ModelArguments:
    wb_name: Optional[str] = field(default="MLLM")
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")

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
        default=None, metadata={"help": "Path to pretrained model for ViT."}
    )
    pretrain_clip_model: str = field(
        default=None, metadata={"help": "Path to pretrained model for CLIP."}
    )
    freeze_vision_tower: bool = field(default=False)

    # projector — defaults match Med3DVLM's best config
    mm_projector_type: Optional[str] = field(default="low_high_mlp")
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

    # wandb control
    use_wandb: bool = field(default=False, metadata={"help": "Enable wandb logging"})


@dataclass
class DataArguments:
    data_root: str = field(
        default="./data/", metadata={"help": "Root directory for all data."}
    )

    # caption data — FULL dataset by default (not subset!)
    cap_data_path: str = field(
        default="./data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption data."},
    )

    # VQA data — FULL dataset by default (not subset!)
    vqa_data_train_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_train.csv",
        metadata={"help": "Path to training VQA data."},
    )
    vqa_data_val_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_val.csv",
        metadata={"help": "Path to validation VQA data."},
    )
    vqa_data_test_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_test.csv",
        metadata={"help": "Path to testing VQA data."},
    )

    vqa_yn_data_train_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_yn_train.csv",
        metadata={"help": "Path to training VQA Yes or No data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # Defaults here are just fallbacks — override everything from bash scripts
    bf16: bool = True
    output_dir: str = "./output/exp1-baseline"
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


# ============================================================================
# Data Collator (unchanged from original)
# ============================================================================

@dataclass
class DataCollator:
    def __call__(self, batch: list) -> dict:
        images, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch]
            for key in ("image", "input_id", "label", "attention_mask")
        )

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        return return_dict


# ============================================================================
# Main — closely follows original Med3DVLM train_vlm.py
# ============================================================================

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("=" * 60)
    rank0_print("EXPERIMENT 1: BASELINE REPRODUCTION (Med3DVLM low_high_mlp)")
    rank0_print("=" * 60)

    # ---- Tokenizer ----
    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_token = {"additional_special_tokens": ["<im_patch>"]}
    tokenizer.add_special_tokens(special_token)

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if "llama3" in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)

    # ---- Projector token count (matches original logic exactly) ----
    if model_args.mm_projector_type is not None:
        if model_args.mm_projector_type == "low_high_mlp":
            model_args.proj_out_num = 288
        elif model_args.mm_projector_type in ("mlp", "mhsa"):
            model_args.proj_out_num = 32
        else:
            model_args.proj_out_num = 256
    else:
        raise ValueError(f"Unknown Projector Type {model_args.mm_projector_type}")

    rank0_print(f"Projector type: {model_args.mm_projector_type}")
    rank0_print(f"Projector output tokens: {model_args.proj_out_num}")

    # ---- Model ----
    rank0_print("=" * 20 + " Model preparation " + "=" * 20)
    if model_args.vision_tower is not None:
        if "qwen" in model_args.model_type:
            model = VLMQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path, cache_dir=training_args.cache_dir
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Initialize vision modules (projector is built inside here)
    # This matches the ORIGINAL order — no manual config injection before this call
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
        model_args.tune_mm_mlp_adapter
    )
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)

    # ---- Load pretrained projector (for Stage 3) ----
    if model_args.pretrain_mm_mlp_adapter is not None:
        rank0_print(f"Loading pretrained mm_projector from: {model_args.pretrain_mm_mlp_adapter}")
        mm_projector_weights = torch.load(
            model_args.pretrain_mm_mlp_adapter, map_location="cpu"
        )
        # The original code doesn't clean keys — it expects them to match directly
        # If your save logic uses prefixes, uncomment the cleaning below:
        # mm_projector_weights = {
        #     k.replace("model.mm_projector.", "").replace("mm_projector.", ""): v
        #     for k, v in mm_projector_weights.items()
        # }
        missing, unexpected = model.get_model().mm_projector.load_state_dict(
            mm_projector_weights, strict=False
        )
        rank0_print(f"  Loaded projector weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            rank0_print(f"  Missing keys (first 5): {missing[:5]}")
        if unexpected:
            rank0_print(f"  Unexpected keys (first 5): {unexpected[:5]}")

    # ---- Load full pretrained MLLM (if provided) ----
    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("Loaded pretrained MLLM weights.")

    # ---- LoRA (for Stage 3) ----
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(
                x in n
                for x in ["vision_tower", "mm_projector", "embed_tokens", "lm_head"]
            ):
                p.requires_grad = True

        model.print_trainable_parameters()

    # ---- Print trainable parameter summary ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: {total_params:,}")
    rank0_print(f"Trainable parameters: {trainable_params:,}")
    rank0_print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # ---- Dataset ----
    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("Vision tokens output from projector: ", data_args.proj_out_num)

    # Verify data files exist
    for path_name, path_val in [
        ("cap_data_path", data_args.cap_data_path),
        ("vqa_data_train_path", data_args.vqa_data_train_path),
        ("vqa_data_val_path", data_args.vqa_data_val_path),
    ]:
        if not os.path.exists(path_val):
            rank0_print(f"WARNING: {path_name} not found: {path_val}")

    if model_args.tune_mm_mlp_adapter:
        # Stage 2: Caption + VQA (excluding yes/no)
        train_dataset = TextDatasets(data_args, tokenizer, mode="train")
    else:
        # Stage 3: Caption + VQA (including yes/no)
        train_dataset = TextYNDatasets(data_args, tokenizer, mode="train")

    eval_dataset = CapDataset(data_args, tokenizer, mode="validation")
    data_collator = DataCollator()

    rank0_print(f"Training dataset size: {len(train_dataset)}")
    rank0_print(f"Eval dataset size: {len(eval_dataset)}")

    # ---- Training ----
    rank0_print("=" * 20 + " Training " + "=" * 20)
    rank0_print(f"Output dir: {training_args.output_dir}")
    rank0_print(f"Epochs: {training_args.num_train_epochs}")
    rank0_print(f"Batch size: {training_args.per_device_train_batch_size}")
    rank0_print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    rank0_print(f"Learning rate: {training_args.learning_rate}")
    rank0_print(f"Model max length: {training_args.model_max_length}")
    rank0_print(f"LoRA enabled: {training_args.lora_enable}")

    trainer = MLLMTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Optional wandb
    if model_args.use_wandb and is_rank_zero():
        import wandb
        wandb.login()
        wandb.init(project="MLLM", name=model_args.wb_name)

    # Resume from checkpoint if available
    if os.path.exists(training_args.output_dir):
        checkpoints = sorted(
            [
                d
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(training_args.output_dir, d))
            ],
            key=lambda x: int(x.split("-")[-1]) if "-" in x else 0,
        )
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            resume_ckpt = os.path.join(training_args.output_dir, last_checkpoint)
            rank0_print(f"Resuming from checkpoint: {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            trainer.train()
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    # ---- Save ----
    rank0_print("=" * 20 + " Save model " + "=" * 20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(
            state_dict_with_lora,
            os.path.join(training_args.output_dir, "model_with_lora.bin"),
        )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir, tokenizer=tokenizer
        )

    # Always save tokenizer and config
    tokenizer.save_pretrained(training_args.output_dir)
    model.config.save_pretrained(training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")

    if model_args.use_wandb and is_rank_zero():
        import wandb
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()