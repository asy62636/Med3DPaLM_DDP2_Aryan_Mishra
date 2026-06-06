#!/bin/bash

source ~/.bashrc
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM || exit 1
conda activate Med3DVLM

echo "================================"
echo "Saving Complete Finetuned Model"
echo "================================"

PYTHONPATH=. python /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/src/utils/merge_lora_and_save_hf_model.py > /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/training_logs/merging.txt 2>&1

echo "================================"
echo "Complete!"
echo "================================"