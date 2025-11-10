#!/bin/bash

# GSM8K Self-Improving Training Pipeline
# Usage: bash self_train_gsm8k.sh <initial_model_path> [options]

set -e  # Exit on error

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: bash self_train_gsm8k.sh <initial_model_path> [options]"
    echo ""
    echo "Required:"
    echo "  initial_model_path      Path to the initial model"
    echo ""
    echo "Optional:"
    echo "  --output_dir DIR        Output directory (default: self_training_results)"
    echo "  --run_name NAME         Custom run name (default: auto-generated with timestamp)"
    echo "  --num_iterations N      Number of self-training iterations (default: 3)"
    echo "  --max_tokens N          Max tokens to generate (default: 512)"
    echo "  --temperature T         Sampling temperature for inference (default: 0.0)"
    echo "  --top_p P               Nucleus sampling parameter (default: 1.0)"
    echo "  --top_k K               Top-k sampling parameter (default: -1, disabled)"
    echo "  --n_rollouts N          Number of rollouts per question (default: 1)"
    echo "  --tensor_parallel N     Number of GPUs (default: 1)"
    echo "  --gpu_memory_util F     GPU memory utilization (default: 0.9)"
    echo "  --system_message MSG    System message for chat template"
    echo "  --use_few_shot          Use few-shot prompting (8 examples, default: zero-shot)"
    echo "  --no_chat_template      Disable chat template and use raw prompts"
    echo "  --enable_thinking       Enable thinking mode for Qwen3 models (default: disabled)"
    echo "  --split SPLIT           Dataset split to use: train or test (default: train)"
    echo "  --n_queries N           Number of queries to use (-1 for all, default: -1)"
    echo ""
    echo "Training parameters:"
    echo "  --learning_rate LR      Learning rate for training (default: 2e-5)"
    echo "  --total_batch_size N    Total effective batch size (default: 16)"
    echo "  --batch_size_per_dev N  Batch size per device (default: 4)"
    echo "  --num_epochs N          Number of training epochs per iteration (default: 1)"
    echo "  --save_steps N          Save checkpoint every N steps (default: 100)"
    echo "  --lora_r N              LoRA rank (default: 64)"
    echo ""
    echo "Note: gradient_accumulation_steps is automatically calculated as total_batch_size / batch_size_per_dev"
    echo "Note: model_max_length is automatically set to max_tokens + 200"
    echo ""
    echo "Example:"
    echo "  bash self_train_gsm8k.sh /path/to/model --output_dir my_self_training --num_iterations 5"
    echo "  bash self_train_gsm8k.sh /path/to/model --learning_rate 1e-5 --total_batch_size 32 --batch_size_per_dev 8"
    exit 1
fi

INITIAL_MODEL_PATH=$1
shift

# Default values
OUTPUT_DIR="ckpt"
RUN_NAME="first-run"
NUM_ITERATIONS=1
MAX_TOKENS=512
TEMPERATURE=1.0
TOP_P=1
TOP_K=-1
N_ROLLOUTS=8
TENSOR_PARALLEL=1
GPU_MEMORY_UTIL=0.8
SYSTEM_MESSAGE=""
USE_FEW_SHOT=false
NO_CHAT_TEMPLATE=false
ENABLE_THINKING=false
SPLIT="train"
N_QUERIES=2000

# Training parameters
LEARNING_RATE=2e-5
TOTAL_BATCH_SIZE=128
BATCH_SIZE_PER_DEV=1
NUM_EPOCHS=1
SAVE_STEPS=30
LORA_R=64

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --num_iterations)
            NUM_ITERATIONS="$2"
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
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --total_batch_size)
            TOTAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --batch_size_per_dev)
            BATCH_SIZE_PER_DEV="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate training parameters
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / BATCH_SIZE_PER_DEV))
MODEL_MAX_LENGTH=$((MAX_TOKENS + 200))

# Generate run name if not provided
if [ -z "$RUN_NAME" ]; then
    MODEL_NAME=$(basename "$INITIAL_MODEL_PATH")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="${MODEL_NAME}_self_training_${TIMESTAMP}"
fi

# Create output directory structure
mkdir -p "$OUTPUT_DIR"
PIPELINE_DIR="$OUTPUT_DIR/$RUN_NAME"
mkdir -p "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR/models"
mkdir -p "$PIPELINE_DIR/data"
mkdir -p "$PIPELINE_DIR/logs"

# Main log file
MAIN_LOG="$PIPELINE_DIR/logs/pipeline.log"

echo "========================================" | tee "$MAIN_LOG"
echo "GSM8K Self-Improving Training Pipeline" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Run Name: $RUN_NAME" | tee -a "$MAIN_LOG"
echo "Initial Model: $INITIAL_MODEL_PATH" | tee -a "$MAIN_LOG"
echo "Pipeline Directory: $PIPELINE_DIR" | tee -a "$MAIN_LOG"
echo "Number of Iterations: $NUM_ITERATIONS" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Inference Settings:" | tee -a "$MAIN_LOG"
echo "  Dataset Split: $SPLIT" | tee -a "$MAIN_LOG"
echo "  Number of Queries: $([ "$N_QUERIES" -eq -1 ] && echo 'All' || echo $N_QUERIES)" | tee -a "$MAIN_LOG"
echo "  Max Tokens: $MAX_TOKENS" | tee -a "$MAIN_LOG"
echo "  Temperature: $TEMPERATURE" | tee -a "$MAIN_LOG"
echo "  Top-p: $TOP_P" | tee -a "$MAIN_LOG"
echo "  Top-k: $TOP_K" | tee -a "$MAIN_LOG"
echo "  Number of Rollouts: $N_ROLLOUTS" | tee -a "$MAIN_LOG"
echo "  Tensor Parallel: $TENSOR_PARALLEL" | tee -a "$MAIN_LOG"
echo "  Mode: $([ "$USE_FEW_SHOT" = true ] && echo 'Few-shot (8 examples)' || echo 'Zero-shot')" | tee -a "$MAIN_LOG"
echo "  Chat Template: $([ "$NO_CHAT_TEMPLATE" = true ] && echo 'Disabled' || echo 'Enabled')" | tee -a "$MAIN_LOG"
echo "  Thinking Mode: $([ "$ENABLE_THINKING" = true ] && echo 'Enabled' || echo 'Disabled')" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Training Settings:" | tee -a "$MAIN_LOG"
echo "  Learning Rate: $LEARNING_RATE" | tee -a "$MAIN_LOG"
echo "  Total Batch Size: $TOTAL_BATCH_SIZE" | tee -a "$MAIN_LOG"
echo "  Batch Size Per Device: $BATCH_SIZE_PER_DEV" | tee -a "$MAIN_LOG"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS (auto-calculated)" | tee -a "$MAIN_LOG"
echo "  Number of Epochs: $NUM_EPOCHS" | tee -a "$MAIN_LOG"
echo "  Save Steps: $SAVE_STEPS" | tee -a "$MAIN_LOG"
echo "  LoRA Rank: $LORA_R" | tee -a "$MAIN_LOG"
echo "  Model Max Length: $MODEL_MAX_LENGTH (auto-calculated: MAX_TOKENS + 200)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Initialize current model path
CURRENT_MODEL_PATH="$INITIAL_MODEL_PATH"

# Start self-training iterations
for ((iter=0; iter<NUM_ITERATIONS; iter++)); do
    ITER_DIR="$PIPELINE_DIR/iteration_$iter"
    mkdir -p "$ITER_DIR"

    echo "========================================"
    echo "ITERATION $iter / $((NUM_ITERATIONS-1))"
    echo "========================================"
    echo "Model: $CURRENT_MODEL_PATH"
    echo ""

    echo "========================================"  >> "$MAIN_LOG"
    echo "ITERATION $iter / $((NUM_ITERATIONS-1))" >> "$MAIN_LOG"
    echo "========================================" >> "$MAIN_LOG"
    echo "Model: $CURRENT_MODEL_PATH" >> "$MAIN_LOG"
    echo "" >> "$MAIN_LOG"

    # File paths for this iteration
    INFERENCE_OUTPUT="$ITER_DIR/inference.jsonl"
    EVAL_OUTPUT="$ITER_DIR/evaluation.jsonl"
    CORRECT_OUTPUT="$ITER_DIR/correct_examples.jsonl"
    ITER_LOG="$ITER_DIR/iteration.log"

    # Step 1: Run inference
    echo "[Step 1/4] Running inference..." | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "Output: $INFERENCE_OUTPUT" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    INFERENCE_CMD="python inference_vllm.py \
        --model_path \"$CURRENT_MODEL_PATH\" \
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

    echo "Command: $INFERENCE_CMD" >> "$ITER_LOG"
    eval $INFERENCE_CMD 2>&1 | tee -a "$ITER_LOG"

    if [ ! -f "$INFERENCE_OUTPUT" ]; then
        echo "ERROR: Inference failed - output file not created" | tee -a "$MAIN_LOG" "$ITER_LOG"
        exit 1
    fi

    echo "✓ Inference completed" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    # Step 2: Run evaluation
    echo "[Step 2/4] Running evaluation..." | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "Output: $EVAL_OUTPUT" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    python evaluate_model_outputs.py \
        --input_jsonl "$INFERENCE_OUTPUT" \
        --output_jsonl "$EVAL_OUTPUT" 2>&1 | tee -a "$ITER_LOG"

    if [ ! -f "$EVAL_OUTPUT" ]; then
        echo "ERROR: Evaluation failed - output file not created" | tee -a "$MAIN_LOG" "$ITER_LOG"
        exit 1
    fi

    echo "✓ Evaluation completed" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    # Step 3: Filter correct examples
    echo "[Step 3/4] Filtering correct examples..." | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "Output: $CORRECT_OUTPUT" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    python filter_correct_examples.py \
        --input_jsonl "$EVAL_OUTPUT" \
        --output_jsonl "$CORRECT_OUTPUT" 2>&1 | tee -a "$ITER_LOG"

    if [ ! -f "$CORRECT_OUTPUT" ]; then
        echo "ERROR: Filtering failed - output file not created" | tee -a "$MAIN_LOG" "$ITER_LOG"
        exit 1
    fi

    # Count correct examples
    NUM_CORRECT=$(wc -l < "$CORRECT_OUTPUT" | tr -d ' ')
    echo "Number of correct examples: $NUM_CORRECT" | tee -a "$MAIN_LOG" "$ITER_LOG"

    echo "✓ Filtering completed" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    # Step 4: Train on correct examples
    echo "[Step 4/4] Training on correct examples..." | tee -a "$MAIN_LOG" "$ITER_LOG"

    NEW_MODEL_PATH="$PIPELINE_DIR/models/model_iter_$((iter+1))"

    echo "Training data: $CORRECT_OUTPUT" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "New model will be saved to: $NEW_MODEL_PATH" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    # Check if we have enough correct examples to train
    if [ "$NUM_CORRECT" -lt 10 ]; then
        echo "⚠ WARNING: Only $NUM_CORRECT correct examples - too few to train effectively" | tee -a "$MAIN_LOG" "$ITER_LOG"
        echo "Skipping training and stopping pipeline" | tee -a "$MAIN_LOG" "$ITER_LOG"
        echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"
        break
    fi



    # Force single GPU training (multi-GPU not yet supported)
    export CUDA_VISIBLE_DEVICES="0"

    TRAIN_CMD="python train_gsm8k_self_training_lora.py \
        --model_name_or_path \"$CURRENT_MODEL_PATH\" \
        --data_path \"$CORRECT_OUTPUT\" \
        --output_dir \"$NEW_MODEL_PATH\" \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE_PER_DEV \
        --per_device_eval_batch_size $BATCH_SIZE_PER_DEV \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps $SAVE_STEPS \
        --save_total_limit 2 \
        --bf16 \
        --model_max_length $MODEL_MAX_LENGTH \
        --lora_r $LORA_R \
        --gradient_checkpointing"

    echo "Training command: $TRAIN_CMD" >> "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    eval $TRAIN_CMD 2>&1 | tee -a "$ITER_LOG"

    if [ ! -d "$NEW_MODEL_PATH" ]; then
        echo "ERROR: Training failed - model directory not created" | tee -a "$MAIN_LOG" "$ITER_LOG"
        exit 1
    fi

    # Update current model path for next iteration
    # Check if merged model exists (for LoRA training)
    MERGED_MODEL_PATH="${NEW_MODEL_PATH}-merged"
    if [ -d "$MERGED_MODEL_PATH" ]; then
        echo "Using merged LoRA model: $MERGED_MODEL_PATH" | tee -a "$MAIN_LOG" "$ITER_LOG"
        CURRENT_MODEL_PATH="$MERGED_MODEL_PATH"
    else
        echo "No merged model found, using base checkpoint: $NEW_MODEL_PATH" | tee -a "$MAIN_LOG" "$ITER_LOG"
        CURRENT_MODEL_PATH="$NEW_MODEL_PATH"
    fi
    echo "✓ Training completed" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"

    echo "✓ Iteration $iter completed" | tee -a "$MAIN_LOG" "$ITER_LOG"
    echo "" | tee -a "$MAIN_LOG" "$ITER_LOG"
done

# Final summary
echo "========================================" | tee -a "$MAIN_LOG"
echo "Self-Training Pipeline Summary" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Pipeline Directory: $PIPELINE_DIR" | tee -a "$MAIN_LOG"
echo "Iterations Completed: $iter / $NUM_ITERATIONS" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Results by iteration:" | tee -a "$MAIN_LOG"

for ((i=0; i<iter; i++)); do
    ITER_DIR="$PIPELINE_DIR/iteration_$i"
    if [ -f "$ITER_DIR/correct_examples.jsonl" ]; then
        NUM_CORRECT=$(wc -l < "$ITER_DIR/correct_examples.jsonl" | tr -d ' ')
        echo "  Iteration $i: $NUM_CORRECT correct examples" | tee -a "$MAIN_LOG"
    fi
done

echo "" | tee -a "$MAIN_LOG"
echo "Logs:" | tee -a "$MAIN_LOG"
echo "  - Main log: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "  - Iteration logs: $PIPELINE_DIR/iteration_*/iteration.log" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

if [ "$iter" -lt "$NUM_ITERATIONS" ]; then
    echo "⚠ Note: Pipeline stopped early (completed $iter of $NUM_ITERATIONS iterations)" | tee -a "$MAIN_LOG"
    echo "   This may be due to insufficient correct examples or training failures" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
fi

if [ "$iter" -eq "$NUM_ITERATIONS" ]; then
    echo "✓ All iterations completed successfully!" | tee -a "$MAIN_LOG"
    if [ -n "$CURRENT_MODEL_PATH" ] && [ -d "$CURRENT_MODEL_PATH" ]; then
        echo "✓ Final model saved to: $CURRENT_MODEL_PATH" | tee -a "$MAIN_LOG"
    fi
    echo "" | tee -a "$MAIN_LOG"
fi

echo "========================================" | tee -a "$MAIN_LOG"
echo "Pipeline finished!" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
