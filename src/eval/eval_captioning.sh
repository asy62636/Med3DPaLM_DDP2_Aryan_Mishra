#!/bin/bash

# Caption Evaluation Script for Med3DVLM with Modified ParGo (5 samples for quick test)
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
echo "Starting Caption Evaluation with Modified ParGo (5 samples quick test)..."
echo "Model Path: /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Med3DVLM-Qwen-2.5-7B-Modified-ParGo-Complete"
echo "================================"
echo ""

# Step 6: Run the evaluation on 5 samples for quick test
PYTHONPATH=. python src/eval/eval_test2.py \
    --model_path /home/medal/ankit_k/Med3DVLM_and_Pargo/Med3DVLM/output2/Med3DVLM-Qwen-2.5-7B-Modified-ParGo-Complete \
    --output_dir ./output3/eval_caption_modified_pargo_5samples \
    --max_length 512 \
    --max_new_tokens 256 \
    --test_size 5 \
    > ./test_logs/modified_pargo_5samples_$(date +%Y%m%d_%H%M%S).log 2>&1

# Step 7: Check exit status
EXIT_CODE=$?

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Caption Evaluation Complete (5 samples)!"
    echo ""
    echo "Quick results preview:"
    tail -20 ./test_logs/modified_pargo_5samples_*.log 2>/dev/null | grep -E "BLEU|ROUGE|METEOR|BERT" || echo "  (Check log file for results)"
else
    echo "✗ Evaluation failed with exit code $EXIT_CODE"
    echo ""
    echo "Error details (last 30 lines):"
    tail -30 ./test_logs/modified_pargo_5samples_*.log 2>/dev/null || echo "  (No log file found)"
fi
echo "Finished at: $(date)"
echo "Log location: ./test_logs/"
echo "Results location: ./output3/eval_caption_modified_pargo_5samples/"
echo "================================"

exit $EXIT_CODE