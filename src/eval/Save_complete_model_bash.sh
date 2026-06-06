#!/bin/bash -l
#SBATCH --job-name=Exp2Stg2
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err
#SBATCH --time=144:00:00
#SBATCH --partition=dgx # dgx(12.4_dgx), a40(12.8), l40(12.8)
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

echo "Running on host: $(hostname)"d
echo "Starting job at: $(date)"

# Check GPU and CUDA version
nvidia-smi

source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments

echo "Begininning to save entire model!"

#Run my bash script
PYTHONPATH=. python /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/src/eval/save_complete_model_expt2.py > /home/medal/ankit_k/Med3DVLM_and_Pargo/experiments/training_logs/saving_model.txt 2>&1

echo "Job finished at: $(date)"