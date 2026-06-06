#!/bin/bash -l
#SBATCH --job-name=EvalExp2
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err
#SBATCH --time=30:00:00
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

PYTHONPATH=. python src/eval/eval_caption_stage3.py \
    --model_path /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/exp2-single-pargo-stage3-complete \
    --output_dir /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/output/eval_exp2_stage3_caption \
    --data_root ./data \
    --cap_data_path ./data/M3D_Cap_npy/M3D_Cap.json \
    > /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/test_logs/eval_exp2_stage3_caption.txt 2>&1

echo "Job finished at: $(date)"