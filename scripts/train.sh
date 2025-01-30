#!/bin/bash

function print_usage() {
    echo "Usage: bash scripts/train.sh [OPTIONS]"
    echo "Options:"
    echo "  --task_name         Task name (default: humaneval)"
    echo "  --model             Model name (default: infly/OpenCoder-1.5B-Instruct)"
    echo "  --train_batch_size  Training batch size (default: 8)"
    echo "  --num_epochs        Number of epochs (default: 3)"
    echo "  --gpus              GPU IDs (e.g., '0,1' or single number)"
    echo "  --port              Main process port (default: 29523)"
    echo "  --augmentation      Enable data augmentation (default: false)"
    echo "  --limit             Limit the number of training examples"
    echo "  --alpha_distill     Distillation loss weight (default: 0.6)"
    echo "  --alpha_ce          Cross entropy loss weight (default: 0.2)"
    echo "  --alpha_switch      Switch loss weight (default: 0.2)"
    echo "  --context_width     Context window width (default: 4)"
    echo "  --use_cache         Use cached data (default: false)"
    exit 1
}

# Default values
TASK_NAME="humaneval"
MODEL="infly/OpenCoder-1.5B-Instruct"
TRAIN_BATCH_SIZE=64
NUM_EPOCHS=3
OUTPUT_DIR="./train"
GPUS="0"
PORT=29523
AUGMENTATION="false"
LIMIT=0
ALPHA_DISTILL=0.6
ALPHA_CE=0.2
ALPHA_SWITCH=0.2
CONTEXT_WIDTH=4
USE_CACHE="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_name) TASK_NAME="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --augmentation) AUGMENTATION="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --alpha_distill) ALPHA_DISTILL="$2"; shift 2 ;;
        --alpha_ce) ALPHA_CE="$2"; shift 2 ;;
        --alpha_switch) ALPHA_SWITCH="$2"; shift 2 ;;
        --context_width) CONTEXT_WIDTH="$2"; shift 2 ;;
        --use_cache) USE_CACHE="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Environment variables
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"

# Determine if multi-GPU
if [[ $GPUS == *","* ]]; then
    ACCELERATE_ARGS=(
        "launch"
        "--multi_gpu"
        "--num_processes" "2"
        "--gpu_ids" "$GPUS"
        "--main_process_port" "$PORT"
    )
else
    ACCELERATE_ARGS=(
        "launch"
        "--num_processes" "1"
        "--gpu_ids" "$GPUS"
        "--main_process_port" "$PORT"
    )
fi

# Base arguments for the training script
BASE_ARGS=(
    "src/train.py"
    "--task_name" "$TASK_NAME"
    "--model" "$MODEL"
    "--output_dir" "$OUTPUT_DIR"
    "--train_batch_size" "$TRAIN_BATCH_SIZE"
    "--num_epochs" "$NUM_EPOCHS"
    "--augmentation" "$AUGMENTATION"
    "--limit" "$LIMIT"
    "--alpha_distill" "$ALPHA_DISTILL"
    "--alpha_ce" "$ALPHA_CE"
    "--alpha_switch" "$ALPHA_SWITCH"
    "--context_width" "$CONTEXT_WIDTH"
    "--use_cache" "$USE_CACHE"
)


# Combine base arguments with accelerate arguments
FINAL_ARGS=("${ACCELERATE_ARGS[@]}" "${BASE_ARGS[@]}")

echo "Running training script with arguments: ${FINAL_ARGS[@]}"

# Run the training script with combined arguments
accelerate "${FINAL_ARGS[@]}"
