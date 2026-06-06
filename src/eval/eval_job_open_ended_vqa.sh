#!/bin/bash -l
#SBATCH --job-name=evalPOSO
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err
#SBATCH --time=4:00:00
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
bash /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/src/eval/evaluation_script_open.sh

echo "Job finished at: $(date)"
