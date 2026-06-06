#!/bin/bash
# ==============================================================================
# EXPERIMENT 1 — Stage 2: Projector Pretraining (Baseline Reproduction)
# ==============================================================================
# Single A100 GPU, DeepSpeed ZeRO-2
#
# Med3DVLM paper settings:
#   Projector: low_high_mlp (2×MLP-Mixer-H), 288 tokens, ~47M params
#   Effective batch size: 16
#   Learning rate: 1e-4
#   Epochs: 3
#   Warmup: 0.03
#   Scheduler: cosine
#   Trainable: projector only (vision encoder + LLM frozen)
#
# Batch math (1 GPU):
#   per_device_train_batch_size × gradient_accumulation_steps = 16
#   8 × 2 = 16 ✓
#
# If you get OOM, reduce to 4 × 4 = 16
# ==============================================================================

source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments

echo "============================================"
echo "Experiment 1 — Stage 2: Baseline Projector Pretraining"
echo "  Projector: low_high_mlp (2×MLP-Mixer-H)"
echo "  Trainable params: ~47M"
echo "  Output tokens: 288"
echo "  Effective batch: 16 (8 × 2)"
echo "  LR: 1e-4, Epochs: 3"
echo "============================================"

# --- Paths ---
VISION_ENCODER="./output2/DCFormer_SigLIP/pretrained_ViT.bin"
OUTPUT_DIR="./output/exp1-baseline-stage2"
DATA_ROOT="./data"
DS_CONFIG="./scripts/experiment1/ds_zero2.json"
TRAIN_SCRIPT="src/train/train_baseline.py"
CHECKPOINT="./output/exp1-baseline-stage2/checkpoint-166000"

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
    echo "  cp train_baseline.py src/train/train_baseline.py"
    exit 1
fi

if [ ! -f "$DS_CONFIG" ]; then
    echo "ERROR: DeepSpeed config not found at $DS_CONFIG"
    exit 1
fi

# Create log directory
mkdir -p training_logs

echo "All prerequisites verified. Starting training..."
echo ""

# --- Launch training ---
PYTHONPATH=. deepspeed $TRAIN_SCRIPT \
    --deepspeed $DS_CONFIG \
    --resume_from_checkpoint $CHECKPOINT \
    --wb_name EXP1_BASELINE_STAGE2 \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --pretrain_vision_model $VISION_ENCODER \
    --mm_projector_type "mixer" \
    --vision_select_layer -2 \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --data_root $DATA_ROOT \
    --cap_data_path "${DATA_ROOT}/M3D_Cap_npy/M3D_Cap.json" \
    --vqa_data_train_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_train.csv" \
    --vqa_data_val_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_val.csv" \
    --vqa_data_test_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_test.csv" \
    --vqa_yn_data_train_path "${DATA_ROOT}/M3D-VQA/M3D_VQA_yn_train.csv" \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --max_grad_norm 1.0 \
    --model_max_length 512 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard" \
    --seed 42 \
    >> training_logs/exp1_baseline_stage2.txt 2>&1

echo ""
echo "============================================"
echo "Stage 2 complete!"
echo "============================================"
echo "Check results:"
echo "  tail -100 training_logs/exp1_baseline_stage2.txt"
echo "Check loss:"
echo "  grep \"{'loss':\" training_logs/exp1_baseline_stage2.txt"
echo "Projector saved at:"
echo "  $OUTPUT_DIR/mm_projector.bin"
echo "============================================"