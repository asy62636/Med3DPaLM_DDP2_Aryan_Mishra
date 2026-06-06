#!/bin/bash
# ==============================================================================
# EXPERIMENT 3 — Stage 3: Dual-Scale ParGo LoRA Fine-tuning
# ==============================================================================
# All paths, GPU count, batch sizes, and hyperparameters mirror expt3.sh.
# Only changes from Stage 2:
#   - tune_mm_mlp_adapter False  (train LLM via LoRA, not just projector)
#   - lora_enable True           (activate LoRA on LLM linear layers)
#   - freeze_vision_tower False  (allow vision tower to update)
#   - pretrain_mm_mlp_adapter    (load Stage 2 projector weights)
#   - learning_rate 5e-5         (lower LR for fine-tuning)
#   - num_train_epochs 5         (matches original Med3DVLM finetune)
#   - Full dataset (not subset)
# ==============================================================================

source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments

echo "============================================"
echo "Experiment 3 — Stage 3: Dual-Scale ParGo LoRA Fine-tuning"
echo "  Projector: dual_scale_pargo (loaded from Stage 2)"
echo "  LoRA: r=16, alpha=32, dropout=0.05"
echo "  vision_select_layer: -2 (matches Stage 2)"
echo "  Effective batch: ~18 (2 x 3 x 3 GPUs)"
echo "  LR: 5e-5, Epochs: 5"
echo "  model_max_length: 2048"
echo "============================================"

# --- Paths (identical to expt3.sh) ---
VISION_ENCODER="./output2/DCFormer_SigLIP/pretrained_ViT.bin"
STAGE2_PROJECTOR="./output/exp3-dual-pargo/stage2/mm_projector.bin"
OUTPUT_DIR="./output/exp3-dual-pargo/stage3"
DATA_ROOT="./data"
DS_CONFIG="./scripts/experiment3/ds_zero2.json"
TRAIN_SCRIPT="src/train/train_dual_scale_pargo.py"

# --- Verify prerequisites ---
if [ ! -f "$VISION_ENCODER" ]; then
    echo "ERROR: Vision encoder not found at $VISION_ENCODER"
    exit 1
fi

if [ ! -f "$STAGE2_PROJECTOR" ]; then
    echo "ERROR: Stage 2 projector weights not found at $STAGE2_PROJECTOR"
    echo "Run first:"
    echo "  python scripts/extract_projector_from_checkpoint.py \\"
    echo "      --checkpoint ./output/exp3-dual-pargo/stage2/checkpoint-130590 \\"
    echo "      --output ${STAGE2_PROJECTOR}"
    exit 1
fi

if [ ! -f "$DS_CONFIG" ]; then
    echo "ERROR: DeepSpeed config not found at $DS_CONFIG"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p training_logs

echo "All prerequisites verified. Starting Stage 3 LoRA fine-tuning..."
echo ""
export MASTER_PORT=$((29500 + RANDOM % 1000))

# --- Launch ---
PYTHONPATH=. deepspeed --num_gpus=3 $TRAIN_SCRIPT \
    --deepspeed $DS_CONFIG \
    --wb_name EXP3_DUAL_PARGO_STAGE3_LORA \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --pretrain_vision_model $VISION_ENCODER \
    --mm_projector_type "dual_scale_pargo" \
    --vision_select_layer -2 \
    --pretrain_mm_mlp_adapter $STAGE2_PROJECTOR \
    --tune_mm_mlp_adapter False \
    --freeze_vision_tower False \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --data_root $DATA_ROOT \
    --cap_data_path "${DATA_ROOT}/M3D_Cap_npy/M3D_Cap.json" \
    --vqa_data_train_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_train.csv" \
    --vqa_data_val_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_val.csv" \
    --vqa_data_test_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_test.csv" \
    --vqa_yn_data_train_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_yn_train.csv" \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --eval_strategy "no" \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 8e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --max_grad_norm 1.0 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard" \
    --seed 42 \
    > training_logs/exp3_dual_pargo_stage3_lora.txt 2>&1

echo ""
echo "============================================"
echo "Experiment 3 — Stage 3 complete!"
echo "============================================"
echo "Check log:  tail -100 training_logs/exp3_dual_pargo_stage3_lora.txt"
echo "LoRA model: $OUTPUT_DIR/model_with_lora.bin"
echo "============================================"