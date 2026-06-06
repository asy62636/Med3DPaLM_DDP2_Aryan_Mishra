#!/bin/bash
source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM
echo "QUICK TEST: 100 steps to verify fixed hyperparameters work!"

# Output directory for QUICK TEST
TEST_DIR="./output2/modified_pargo_TEST_100steps"

PYTHONPATH=. deepspeed src/train/train_vlm_2.py \
    --deepspeed ./scripts/zero5.json \
    --wb_name MODIFIED_PARGO_TEST_100STEPS \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --pretrain_vision_model ./output2/DCFormer_SigLIP/pretrained_ViT.bin \
    --mm_projector_type "modified_pargo" \
    --vision_select_layer -2 \
    --tune_mm_mlp_adapter True \
    --data_root ./data \
    --bf16 True \
    --output_dir $TEST_DIR \
    --num_train_epochs 0.05 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_grad_norm 2.0 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --bert_type "bert-base-uncased" \
    --n_low_tokens 144 \
    --n_high_tokens 32 \
    --low_level_hidden_size 384 \
    --pargo_num_layers 6 \
    --use_cross_scale_attention False \
    --use_positional_embedding True \
    > /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/training_logs/modified_pargo_TEST_100steps.txt 2>&1

echo ""
echo "=========================================="
echo "QUICK TEST COMPLETE!"
echo "=========================================="
echo ""
echo "Check results with:"
echo "  tail -100 training_logs/modified_pargo_TEST_100steps.txt"
echo ""
echo "Check loss progression:"
echo "  grep \"{'loss':\" training_logs/modified_pargo_TEST_100steps.txt"
echo ""
echo "WHAT TO LOOK FOR:"
echo "  ✓ GOOD:  Loss drops from ~7.0 to ~3.5-4.5 in 100 steps"
echo "  ✗ BAD:   Loss stays at 6.5-7.0 (not learning)"
echo "  ✗ BAD:   Loss becomes NaN (exploding)"
echo ""
echo "If loss drops to ~4.0 after 100 steps:"
echo "  → Fix works! Run full 3-epoch training"
echo ""
echo "If loss stays high (>6.0):"
echo "  → Need to increase LR to 3e-3 or 5e-3"
echo ""