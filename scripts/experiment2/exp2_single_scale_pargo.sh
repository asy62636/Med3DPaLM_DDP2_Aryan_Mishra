#!/bin/bash
# ==============================================================================
# EXPERIMENT 2 — Stage 2: Single-Scale ParGo Projector Pretraining
# ==============================================================================
# Single A100 GPU, DeepSpeed ZeRO-2
#
# Key differences from Experiment 1 baseline:
#   Projector: single_scale_pargo (BERT-initialized, 2 layers)
#   Output tokens: 32 (8 global + 24 partial)
#   Params: ~20M (vs 47M for mixer)
#   vision_select_layer: -1 (final layer only, not -2)
#   model_max_length: 2048 (not 512 — critical for report generation)
#
# Batch math (1 GPU):
#   per_device_train_batch_size × gradient_accumulation_steps = 16
#   8 × 2 = 16 ✓
# ==============================================================================

source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments

echo "============================================"
echo "Experiment 2 — Stage 2: Single-Scale ParGo Pretraining"
echo "  Projector: single_scale_pargo"
echo "  Queries: 32 (8 global + 24 partial)"
echo "  BERT layers: 2 (pretrained init)"
echo "  Trainable params: ~20M"
echo "  vision_select_layer: -1 (final only)"
echo "  Effective batch: 16 (8 × 2)"
echo "  LR: 1e-4, Epochs: 3"
echo "  model_max_length: 2048"
echo "============================================"

# --- Paths ---
VISION_ENCODER="./output2/DCFormer_SigLIP/pretrained_ViT.bin"
OUTPUT_DIR="./output/exp2-single-pargo/stage2"
DATA_ROOT="./data"
DS_CONFIG="./scripts/experiment2/ds_zero2.json"
TRAIN_SCRIPT="src/train/train_single_scale_pargo.py"

# --- Verify prerequisites ---
if [ ! -f "$VISION_ENCODER" ]; then
    echo "ERROR: Vision encoder not found at $VISION_ENCODER"
    exit 1
fi

if [ ! -f "${DATA_ROOT}/M3D_Cap_npy/M3D_Cap.json" ]; then
    echo "ERROR: Caption data not found. Check data symlink:"
    echo "  ls -la ${DATA_ROOT}"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$DS_CONFIG" ]; then
    echo "ERROR: DeepSpeed config not found at $DS_CONFIG"
    exit 1
fi

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p training_logs

echo "All prerequisites verified. Starting training..."
echo ""
export MASTER_PORT=$((29500 + RANDOM % 1000))
# --- Launch training ---
PYTHONPATH=. deepspeed --num_gpus=3 $TRAIN_SCRIPT \
    --deepspeed $DS_CONFIG \
    --wb_name EXP2_SINGLE_PARGO_STAGE2 \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --pretrain_vision_model $VISION_ENCODER \
    --mm_projector_type "single_scale_pargo" \
    --vision_select_layer -1 \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --data_root $DATA_ROOT \
    --cap_data_path "${DATA_ROOT}/M3D_Cap_npy/M3D_Cap_subset.json" \
    --vqa_data_train_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_train_subset.csv" \
    --vqa_data_val_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_val_subset.csv" \
    --vqa_data_test_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_test_subset.csv" \
    --vqa_yn_data_train_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_yn_train_subset.csv" \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
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
    > training_logs/exp2_single_pargo_stage2.txt 2>&1

echo ""
echo "============================================"
echo "Experiment 2 — Stage 2 complete!"
echo "============================================"
echo "Check results:"
echo "  tail -100 training_logs/exp2_single_pargo_stage2.txt"
echo "Check loss:"
echo "  grep \"{'loss':\" training_logs/exp2_single_pargo_stage2.txt"
echo "Projector saved at:"
echo "  $OUTPUT_DIR/mm_projector.bin"
echo "============================================"