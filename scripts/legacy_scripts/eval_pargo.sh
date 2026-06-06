#!/bin/bash
source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM
echo "Beginning evaluation process!"
which python

PYTHONPATH=. python src/eval/eval_pargo.py \
  --checkpoint-dir /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180 \
  --config /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output/Med3DVLM-Qwen-2.5-7B-pretrain/config.json \
  --device cuda:0 \
  --save-small --save-small-path /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/save_proj_ckpt/pargo_vision.pt

