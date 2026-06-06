#!/bin/bash -l
#SBATCH --job-name=VQA_Exp3_S3
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

echo "Running on host: $(hostname)"
echo "Starting job at: $(date)"
nvidia-smi

source ~/.bashrc
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments
conda activate Med3DVLM

BASE=/home/medal/ankit_k/Med3DVLM_and_Pargo/experiments
MODEL_PATH=${BASE}/output/exp3-dual-pargo-stage3-complete
VQA_DATA=${BASE}/data/M3D-VQA/M3D_VQA_test.csv
LOG_DIR=${BASE}/training_logs

echo "=== Exp3 Stage3: Open-ended VQA ==="
PYTHONPATH=. python ${BASE}/src/eval/eval_vqa_good_code_stage_3.py \
    --model_path     ${MODEL_PATH} \
    --output_dir     ${BASE}/output/eval_exp3_stage3_vqa/open \
    --vqa_data_test_path ${VQA_DATA} \
    --data_root      ${BASE}/data \
    --max_length     512 \
    --max_new_tokens 256 \
    > ${LOG_DIR}/eval_vqa_exp3_stage3_open.txt 2>&1

echo "Open-ended done at: $(date)"

echo "=== Exp3 Stage3: Closed-ended VQA ==="
PYTHONPATH=. python ${BASE}/src/eval/eval_vqa.py \
    --model_path     ${MODEL_PATH} \
    --output_dir     ${BASE}/output/eval_exp3_stage3_vqa/closed \
    --vqa_data_test_path ${VQA_DATA} \
    --data_root      ${BASE}/data \
    --max_length     512 \
    --max_new_tokens 256 \
    --close_ended \
    >> ${LOG_DIR}/eval_vqa_exp3_stage3_open.txt 2>&1

echo "Closed-ended done at: $(date)"
echo "Job finished at: $(date)"