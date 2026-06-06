#!/bin/bash
source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM
echo "Beginning LoRA finetuning process WITHOUT positional embeddings!"

PYTHONPATH=. deepspeed src/train/train_vlm_1.py \
    --deepspeed ./scripts/zero5.json \
    --wb_name Med3DVLM-Qwen-2.5-7B-ParGo-finetune-without-pos \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --mm_projector_type "pargo" \
    --bert_type "bert-base-uncased" \
    --num_query_tokens 304 \
    --use_positional_embedding False \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --vision_select_layer -2 \
    --pretrain_vision_model ./output2/DCFormer_SigLIP/pretrained_ViT.bin \
    --pretrain_mm_mlp_adapter ./output2/Pargo_without_pos_embedding/mm_projector.bin \
    --data_root ./data \
    --bf16 True \
    --output_dir ./output2/Med3DVLM-Qwen-2.5-7B-ParGo-finetune-without-pos \
    --num_train_epochs 5 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --optim adamw_torch \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 2 > /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/training_logs/finetune_lora_without_pos_emb.txt 2>&1

echo "Training completed! Model saved to ./output2/Med3DVLM-Qwen-2.5-7B-ParGo-finetune-without-pos"