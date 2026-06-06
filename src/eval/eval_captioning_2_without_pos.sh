#!/bin/bash

# Caption Evaluation Script for Med3DVLM Merged Model with Positional Embeddings
# ============================================================================

# Step 1: Source bashrc
source ~/.bashrc

# Step 2: Change directory
cd /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM || exit 1

# Step 3: Activate conda environment
conda activate Med3DVLM

# Step 4: Create log directory if it doesn't exist
mkdir -p ./test_logs

# Step 5: Echo status and show GPU info
echo "================================"
echo "Environment setup complete!"
echo "Current directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Date: $(date)"
echo "================================"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo "================================"
echo "Starting Caption Evaluation with Merged Model (With Pos Embeddings)..."
echo "Model Path: /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output3/Med3DVLM-Qwen-2.5-7B-ParGo-MERGED"
echo "================================"
echo ""

# Step 6: Run the evaluation (output redirected to log file only)
PYTHONPATH=. python src/eval/eval_test.py \
    --model_path /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output3/Med3DVLM-Qwen-2.5-7B-ParGo-MERGED-without-pos \
    --output_dir ./output3/eval_caption_samples_merged_without_pos \
    --max_length 512 \
    --max_new_tokens 256 \
    > ./test_logs/merged_model_caption_without_pos_embedding_$(date +%Y%m%d_%H%M%S).log 2>&1

# Step 7: Check exit status
EXIT_CODE=$?

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Caption Evaluation Complete!"
else
    echo "✗ Evaluation failed with exit code $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "Log location: ./test_logs/"
echo "Results location: ./output3/eval_caption_5_samples_merged_with_pos/"
echo "================================"

exit $EXIT_CODE