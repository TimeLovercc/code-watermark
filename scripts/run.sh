#!/bin/bash

function print_usage() {
    echo "Usage: bash run.sh [OPTIONS]"
    echo "  --model             Model name/path (default: infly/OpenCoder-1.5B-Instruct)"
    echo "  --task_name         Task name (humaneval, mbpp, ds1000)"
    echo "  --wm                Watermarking method (wllm, sweet, code, no)"
    echo "  --gpus              GPU IDs (e.g., '0,1' or single number)"
    echo "  --limit             Number of samples to generate (default: 5)"
    echo "  --code_model        Path to code model (optional)"
    echo "  --switch_threshold  Switch threshold for code model (default: -10.0)"
    echo "  --seed              Random seed (default:"
    echo "  --delta             Delta value for watermarking (default: 2.0)"
    echo "  --context_width     Context width for code model (default: 4)"
    echo "  --entropy_threshold  Entropy threshold for code model (default: 0.5)"
    exit 1
}

# Default values
MODEL="infly/OpenCoder-1.5B-Instruct"
TASK="humaneval"
WM="wllm"
GPUS="0"
CODE_MODEL=""
BEST=false
SWITCH_THRESHOLD="-10.0"
SEED=42
DELTA=2.0
CONTEXT_WIDTH=1
ENTROPY_THRESHOLD=1.2

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --task_name) TASK="$2"; shift 2 ;;
        --wm) WM="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --limit) LIMIT="5"; shift ;;
        --code_model) CODE_MODEL="$2"; shift 2 ;;
        --switch_threshold) SWITCH_THRESHOLD="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --delta) DELTA="$2"; shift 2 ;;
        --context_width) CONTEXT_WIDTH="$2"; shift 2 ;;
        --entropy_threshold) ENTROPY_THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; print_usage ;;
    esac
done

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=${GPUS}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export HF_ALLOW_CODE_EVAL=1
export CONDA_PREFIX=/gpu02home/zzg5107/miniconda3/envs/torch
export CPATH="$CONDA_PREFIX/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Run the main script with HumanEval configuration
python src/main.py \
    --model ${MODEL} \
    --task_name ${TASK} \
    --output_dir ./outputs \
    --batch_size 20 \
    --max_length 2048 \
    --precision "bf16" \
    --temperature 0.2 \
    --top_p 0.95 \
    --n_samples 20 \
    --wm ${WM} \
    --gamma 0.5 \
    --delta ${DELTA} \
    --postprocess true \
    --allow_code_execution true \
    --detection_z_threshold 4.0 \
    --context_width ${CONTEXT_WIDTH} \
    --switch_threshold ${SWITCH_THRESHOLD} \
    --entropy_threshold ${ENTROPY_THRESHOLD} \
    --seed ${SEED} \
    ${LIMIT:+--limit ${LIMIT}} \
    --code_model ${CODE_MODEL} 
