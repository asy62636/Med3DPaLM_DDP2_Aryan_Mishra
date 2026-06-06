#!/bin/bash
source ~/.bashrc
conda activate Med3DVLM
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM
echo "Beginning TEST training process!"

# Create a test output directory
TEST_DIR="./output2/TEST_SAVE_CHECK_$(date +%Y%m%d_%H%M%S)"

PYTHONPATH=. deepspeed src/train/train_vlm_2.py \
    --deepspeed ./scripts/zero4.json \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_steps 10 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 0 \
    --bert_type "bert-base-uncased" \
    --num_query_tokens 304

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