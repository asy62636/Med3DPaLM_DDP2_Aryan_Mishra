#!/bin/bash
# ==============================================================================
# Evaluate Experiment 2: Single-Scale ParGo (32 tokens)
# ==============================================================================

source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments

set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="./output/exp2-single-pargo-complete"
OUTPUT_DIR="./output/eval_exp2_single_pargo"
CAP_DATA="./data/M3D_Cap_npy/M3D_Cap.json"

# Use --test_size for quick sanity check, remove for full eval
PYTHONPATH=. python /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/src/eval/eval_caption.py \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --cap_data_path ${CAP_DATA} \
    --max_length 512 \
    --max_new_tokens 256 \
    --test_size 100 > /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/test_logs/exp2_cap.txt 2>&1