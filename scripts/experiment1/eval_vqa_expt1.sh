#!/bin/bash
# ==============================================================================
# Evaluate Experiment 1: Baseline low_high_mlp (288 tokens)
# Runs open-ended VQA then closed-ended VQA sequentially.
# ==============================================================================

source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments

set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="./output/exp1-baseline-complete"
VQA_DATA="./data/M3D-VQA/M3D_VQA_test.csv"
LOG_DIR="/home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/test_logs"
mkdir -p ${LOG_DIR}

# ------------------------------------------------------------------------------
# Open-ended VQA
# Remove --test_size for full evaluation
# ------------------------------------------------------------------------------
echo "=== Exp 1: Open-ended VQA ===" | tee ${LOG_DIR}/exp1_vqa_open.txt
PYTHONPATH=. python /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/src/eval/eval_vqa_good_code.py \
    --model_path   ${MODEL_PATH} \
    --output_dir   ./output/eval_exp1_vqa/open \
    --vqa_data_test_path ${VQA_DATA} \
    --max_length   512 \
    --max_new_tokens 256 \
    --test_size    100 \
    >> ${LOG_DIR}/exp1_vqa_open.txt 2>&1

echo "Open-ended done. Log: ${LOG_DIR}/exp1_vqa_open.txt"

# ------------------------------------------------------------------------------
# Closed-ended VQA
# ------------------------------------------------------------------------------
echo "=== Exp 1: Closed-ended VQA ===" | tee ${LOG_DIR}/exp1_vqa_close.txt
PYTHONPATH=. python /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/src/eval/eval_vqa_good_code.py \
    --model_path   ${MODEL_PATH} \
    --output_dir   ./output/eval_exp1_vqa/closed \
    --vqa_data_test_path ${VQA_DATA} \
    --max_length   512 \
    --max_new_tokens 256 \
    --close_ended \
    --test_size    100 \
    >> ${LOG_DIR}/exp1_vqa_close.txt 2>&1

echo "Closed-ended done. Log: ${LOG_DIR}/exp1_vqa_close.txt"
echo "=== Exp 1 VQA evaluation complete ==="