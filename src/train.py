import os
import sys
sys.path.append(os.getcwd())

import logging
import argparse
import torch
from transformers import get_cosine_schedule_with_warmup

from src.datahelper import create_sft_dataloaders
from src.sfthelper import TrainingHelper
from src.modelhelper import CodeModel
from src.utils import ( 
    str_to_bool_or_none,
    reserve_gpu_memory,
    setup_accelerator,
    setup_output_dir, 
    setup_logging, 
    setup_seed, 
    get_tokenizer,
    save_checkpoint
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument("--model", type=str, default="infly/OpenCoder-1.5B-Instruct")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--trust_remote_code", type=str_to_bool_or_none, default=True)
    parser.add_argument("--use_auth_token", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp32")
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./train")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--entropy", type=str_to_bool_or_none, default=False)
    parser.add_argument("--context_width", type=int, default=4)
    parser.add_argument("--augmentation", type=str_to_bool_or_none, default=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_cache", type=str_to_bool_or_none, default=False)
    
    # Loss weights
    parser.add_argument("--alpha_distill", type=float, default=0.6)
    parser.add_argument("--alpha_ce", type=float, default=0.2)
    parser.add_argument("--alpha_switch", type=float, default=0.2)
    
    args = parser.parse_args()
    for arg in vars(args):
        setattr(args, arg, str_to_bool_or_none(getattr(args, arg)))
    
    return args

def main(args):
    # Setup accelerator
    accelerator = setup_accelerator(args)
    device = accelerator.device
    
    # Setup output directory and logging
    output_dir = setup_output_dir(args)
    log_file = os.path.join(output_dir['logs'], 'train.log')
    setup_logging(log_file)
    
    # Set random seed
    setup_seed(args.seed)
    
    # Initialize tokenizer and model
    tokenizer = get_tokenizer(args.model)
    vocab_size = len(tokenizer)
    
    code_model_config = {
        "vocab_size": vocab_size,
        "alpha_distill": args.alpha_distill,  # Distillation loss weight
        "alpha_ce": args.alpha_ce,      # Cross entropy loss weight 
        "alpha_switch": args.alpha_switch,  # Switch loss weight
    }
    code_model = CodeModel(code_model_config)
    
    # Get data
    train_dataloader, val_dataloader = create_sft_dataloaders(args.task_name, tokenizer, device, args)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        code_model.parameters(),
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dataloader) * args.num_epochs,
        num_cycles=0.5
    )
    
    # Prepare for distributed training
    code_model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        code_model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Train model
    code_model = TrainingHelper.train(
        model=code_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        args=args,
        output_dirs=output_dir  # Pass output directories to training function
    )
    
    # Save final model
    if accelerator.is_local_main_process:
        save_checkpoint(
            model=code_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.num_epochs - 1,
            save_dir=os.path.join(output_dir['checkpoints'], "final"),
            metrics={},
            accelerator=accelerator
        )
    
    accelerator.end_training()

if __name__ == '__main__':
    args = parse_args()
    reserve_gpu_memory(20)
    main(args)