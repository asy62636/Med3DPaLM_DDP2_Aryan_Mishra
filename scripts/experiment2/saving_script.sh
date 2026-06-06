#!/bin/bash -l
#SBATCH --job-name=Exp2Save
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err
#SBATCH --time=12:00:00
#SBATCH --partition=dgx # dgx(12.4_dgx), a40(12.8), l40(12.8)
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

echo "Running on host: $(hostname)"d
echo "Starting job at: $(date)"

# Check GPU and CUDA version
nvidia-smi

#Run my bash script
source ~/.bashrc
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments
conda activate Med3DVLM
PYTHONPATH=. python /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/src/eval/merge_lora_and_save_exp2.py > /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/training_logs/save_expt2.txt 2>&1

echo "Job finished at: $(date)"