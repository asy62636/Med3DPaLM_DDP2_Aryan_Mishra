#!/bin/bash

# VQA Close-Ended Evaluation Script for Med3DVLM with Positional Embeddings
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
echo "Starting VQA Closed-Ended Evaluation With Pos Embeddings..."
echo "================================"
echo ""

# Step 6: Run the evaluation (output redirected to log file only)
PYTHONPATH=. python src/eval/eval_vqa_test.py \
    --model_path ./output/Med3DVLM-Qwen-2.5-7B-ParGo-Complete-Pargo-with-pos-embedding \
    --vqa_data_test_path ./data/M3D-VQA/M3D_VQA_test.csv \
    --output_dir ./output/eval_vqa_closed_test_with_pos/ \
    --close_ended \
    > ./test_logs/pargo_with_pos_embedding_close_ended_$(date +%Y%m%d_%H%M%S).log 2>&1

# Step 7: Check exit status
EXIT_CODE=$?

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Close-Ended Evaluation Complete!"
else
    echo "✗ Evaluation failed with exit code $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "Log location: ./test_logs/"
echo "Results location: ./output/eval_vqa_closed_test_with_pos/"
echo "================================"

exit $EXIT_CODE