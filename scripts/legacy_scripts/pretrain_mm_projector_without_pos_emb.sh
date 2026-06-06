#!/bin/bash
source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM
echo "Beginning training process with Positional embeddings!"

# Create a test output directory
TEST_DIR="./output2/Pargo_without_pos_embedding"

PYTHONPATH=. deepspeed src/train/train_vlm_1.py \
    --deepspeed ./scripts/zero3.json \
    --wb_name TEST_SAVE_CHECK \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --pretrain_vision_model ./output2/DCFormer_SigLIP/pretrained_ViT.bin \
    --mm_projector_type "pargo" \
    --vision_select_layer -2 \
    --tune_mm_mlp_adapter True \
    --data_root ./data \
    --bf16 True \
    --output_dir $TEST_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --bert_type "bert-base-uncased" \
    --num_query_tokens 304 > /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/training_logs/pargo_without_pos_emb.txt 2>&1

echo "Training complete. Checking saved files..."

# Check what files were saved
echo "=== Files in output directory ==="
ls -la $TEST_DIR/

echo "=== Files in checkpoint-5 ==="
ls -la $TEST_DIR/checkpoint-5/ 2>/dev/null || echo "No checkpoint-5 found"

echo "=== Files in checkpoint-10 ==="
ls -la $TEST_DIR/checkpoint-10/ 2>/dev/null || echo "No checkpoint-10 found"

# Check for critical files
echo "=== Checking for critical files ==="
for file in "config.json" "tokenizer_config.json" "special_tokens_map.json" "added_tokens.json" "training_args.bin"; do
    if [ -f "$TEST_DIR/$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
    fi
done

# Check model weights
if [ -f "$TEST_DIR/pytorch_model.bin" ] || [ -f "$TEST_DIR/model.safetensors" ]; then
    echo "✓ Found model weights"
else
    echo "✗ Missing model weights"
fi

echo "Test complete! Check the output above."