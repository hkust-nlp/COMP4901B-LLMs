#!/bin/bash

# GSM8K Evaluation Script
# Usage: bash run_gsm8k_eval.sh <model_path> [options]

set -e  # Exit on error

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: bash run_gsm8k_eval.sh <model_path> [options]"
    echo ""
    echo "Required:"
    echo "  model_path              Path to the model"
    echo ""
    echo "Optional:"
    echo "  --output_dir DIR        Output directory (default: results)"
    echo "  --max_tokens N          Max tokens to generate (default: 512)"
    echo "  --temperature T         Sampling temperature (default: 0.0)"
    echo "  --top_p P               Nucleus sampling parameter (default: 1.0)"
    echo "  --top_k K               Top-k sampling parameter (default: -1, disabled)"
    echo "  --n_rollouts N          Number of rollouts per question (default: 1)"
    echo "  --tensor_parallel N     Number of GPUs (default: 1)"
    echo "  --gpu_memory_util F     GPU memory utilization (default: 0.9)"
    echo "  --system_message MSG    System message for chat template"
    echo "  --use_few_shot          Use few-shot prompting (8 examples, default: zero-shot)"
    echo "  --no_chat_template      Disable chat template and use raw prompts"
    echo "  --enable_thinking       Enable thinking mode for Qwen3 models (default: disabled)"
    echo "  --split SPLIT           Dataset split to use: train or test (default: test)"
    echo "  --n_queries N           Number of queries to use (-1 for all, default: -1)"
    echo ""
    echo "Example:"
    echo "  bash run_gsm8k_eval.sh /path/to/model --output_dir my_results --use_few_shot"
    exit 1
fi

MODEL_PATH=$1
shift

# Default values
OUTPUT_DIR="results"
MAX_TOKENS=512
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=-1
N_ROLLOUTS=1
TENSOR_PARALLEL=1
GPU_MEMORY_UTIL=0.8
SYSTEM_MESSAGE=""
USE_FEW_SHOT=false
NO_CHAT_TEMPLATE=false
ENABLE_THINKING=false
SPLIT="test"
N_QUERIES=-1

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --n_rollouts)
            N_ROLLOUTS="$2"
            shift 2
            ;;
        --tensor_parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --gpu_memory_util)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --system_message)
            SYSTEM_MESSAGE="$2"
            shift 2
            ;;
        --use_few_shot)
            USE_FEW_SHOT=true
            shift 1
            ;;
        --no_chat_template)
            NO_CHAT_TEMPLATE=true
            shift 1
            ;;
        --enable_thinking)
            ENABLE_THINKING=true
            shift 1
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --n_queries)
            N_QUERIES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Extract model name from path
MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${MODEL_NAME}_${TIMESTAMP}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Output files
INFERENCE_OUTPUT="$OUTPUT_DIR/${RUN_NAME}_inference.jsonl"
EVAL_OUTPUT="$OUTPUT_DIR/${RUN_NAME}_evaluation.jsonl"
LOG_FILE="$OUTPUT_DIR/${RUN_NAME}.log"

echo "========================================" | tee "$LOG_FILE"
echo "GSM8K Evaluation Pipeline" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Run Name: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Dataset Split: $SPLIT" | tee -a "$LOG_FILE"
echo "Number of Queries: $([ "$N_QUERIES" -eq -1 ] && echo 'All' || echo $N_QUERIES)" | tee -a "$LOG_FILE"
echo "Max Tokens: $MAX_TOKENS" | tee -a "$LOG_FILE"
echo "Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "Top-p: $TOP_P" | tee -a "$LOG_FILE"
echo "Top-k: $TOP_K" | tee -a "$LOG_FILE"
echo "Number of Rollouts: $N_ROLLOUTS" | tee -a "$LOG_FILE"
echo "Tensor Parallel: $TENSOR_PARALLEL" | tee -a "$LOG_FILE"
echo "Mode: $([ "$USE_FEW_SHOT" = true ] && echo 'Few-shot (8 examples)' || echo 'Zero-shot')" | tee -a "$LOG_FILE"
echo "Chat Template: $([ "$NO_CHAT_TEMPLATE" = true ] && echo 'Disabled' || echo 'Enabled')" | tee -a "$LOG_FILE"
echo "Thinking Mode: $([ "$ENABLE_THINKING" = true ] && echo 'Enabled' || echo 'Disabled')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 1: Run inference
echo "[1/2] Running inference..." | tee -a "$LOG_FILE"
echo "Output will be saved to: $INFERENCE_OUTPUT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

INFERENCE_CMD="python inference_vllm.py \
    --model_path \"$MODEL_PATH\" \
    --output_path \"$INFERENCE_OUTPUT\" \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --n_rollouts $N_ROLLOUTS \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --gpu_memory_utilization $GPU_MEMORY_UTIL \
    --split $SPLIT \
    --n_queries $N_QUERIES"

if [ -n "$SYSTEM_MESSAGE" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --system_message \"$SYSTEM_MESSAGE\""
fi

if [ "$USE_FEW_SHOT" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --use_few_shot"
fi

if [ "$NO_CHAT_TEMPLATE" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --no_chat_template"
fi

if [ "$ENABLE_THINKING" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --enable_thinking"
fi

echo "Running: $INFERENCE_CMD" >> "$LOG_FILE"
eval $INFERENCE_CMD 2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$INFERENCE_OUTPUT" ]; then
    echo "ERROR: Inference failed - output file not created" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "Inference completed successfully!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 2: Run evaluation
echo "[2/2] Running evaluation..." | tee -a "$LOG_FILE"
echo "Evaluation results will be saved to: $EVAL_OUTPUT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python evaluate_model_outputs.py \
    --input_jsonl "$INFERENCE_OUTPUT" \
    --output_jsonl "$EVAL_OUTPUT" 2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$EVAL_OUTPUT" ]; then
    echo "ERROR: Evaluation failed - output file not created" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Evaluation Pipeline Completed!" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Results:" | tee -a "$LOG_FILE"
echo "  - Inference outputs: $INFERENCE_OUTPUT" | tee -a "$LOG_FILE"
echo "  - Evaluation results: $EVAL_OUTPUT" | tee -a "$LOG_FILE"
echo "  - Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
