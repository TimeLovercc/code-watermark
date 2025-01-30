# ACW Project

This repository contains a Python-based project with various utilities and helper modules for model training and evaluation.

## Project Structure

```
.
├── src/                    # Source code directory
│   ├── main.py            # Main application entry point
│   ├── train.py           # Training module
│   ├── modelhelper.py     # Model-related utilities
│   ├── datahelper.py      # Data processing and handling
│   ├── utils.py           # General utility functions
│   ├── sfthelper.py       # SFT (Supervised Fine-Tuning) helper functions
│   └── asthelper.py       # AST (Abstract Syntax Tree) processing utilities
│
├── scripts/               # Shell scripts for running experiments
│   ├── train.sh          # Training script
│   └── run.sh            # Evaluation script
├── lm_eval/               # Language Model evaluation directory
├── exp_utils/             # Experiment utilities
└── models/                # Model storage directory
```

## Main Components

- **main.py**: Entry point of the application
- **train.py**: Contains training logic and procedures
- **modelhelper.py**: Provides model-related utilities and helper functions
- **datahelper.py**: Handles data processing, loading, and manipulation
- **utils.py**: Contains general utility functions used across the project
- **sfthelper.py**: Implements supervised fine-tuning helper functions
- **asthelper.py**: Provides functionality for working with Abstract Syntax Trees

## Getting Started

To use this project, make sure you have Python installed and the required dependencies set up. The project contains various utilities for model training, evaluation, and experimentation.

### Workflow

1. **Data Generation**:
   First, generate the required data using the datahelper script:
   ```bash
   CUDA_VISIBLE_DEVICES=5 python src/datahelper.py
   ```

2. **Model Training**:
   Train the model using the training script. Example:
   ```bash
   bash scripts/train.sh \
       --model "deepseek-ai/deepseek-coder-1.3b-instruct" \
       --task_name gen_mbpp \
       --train_batch_size 512 \
       --num_epochs 1000 \
       --gpus 5 \
       --augmentation false \
       --alpha_distill 0.1 \
       --alpha_ce 0.1 \
       --alpha_switch 8.0 \
       --context_width 2 \
       --use_cache false
   ```

3. **Model Evaluation/Running**:
   After training, run the model using the run script:
   ```bash
   bash scripts/run.sh \
       --model "infly/OpenCoder-1.5B-Instruct" \
       --task_name humanevalsynthesize-cpp \
       --wm wllm \
       --gpus 1 \
       --code_model none \
       --entropy_threshold 1.2 \
       --switch_threshold 0.7 \
       --seed 42 \
       --delta 2.0
   ```

For more detailed information about specific components, please refer to the individual files. 
