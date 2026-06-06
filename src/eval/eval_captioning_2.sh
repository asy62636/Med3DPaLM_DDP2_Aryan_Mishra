#!/bin/bash

# Caption Evaluation Script for Med3DVLM with Modified Pos Embeddings (ALL samples)
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
echo "Starting Caption Evaluation with Modified Pos Embeddings (ALL samples)..."
echo "Model Path: /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Pargo_modified_pos_COMPLETE"
echo "================================"
echo ""

# Step 6: Run the evaluation on ALL samples (no --test_size argument)
PYTHONPATH=. python src/eval/eval_test.py \
    --model_path /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Pargo_modified_pos_COMPLETE \
    --output_dir ./output3/eval_caption_all_samples_modified_pos \
    --max_length 512 \
    --max_new_tokens 256 \
    > ./test_logs/modified_pos_embedding_all_samples_$(date +%Y%m%d_%H%M%S).log 2>&1

# Step 7: Check exit status
EXIT_CODE=$?

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Caption Evaluation Complete (ALL samples)!"
else
    echo "✗ Evaluation failed with exit code $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "Log location: ./test_logs/"
echo "Results location: ./output3/eval_caption_all_samples_modified_pos/"
echo "================================"

exit $EXIT_CODE