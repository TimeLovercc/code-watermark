import os
import gc
import logging
from typing import Optional, List, Tuple, Dict, Any, Callable
import os
import json
import wandb
from datetime import datetime
import ipdb
import multiprocessing
import hashlib
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    StoppingCriteria,
    LogitsProcessor
)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import sklearn.metrics as metrics

from models.code import CodeLogitsProcessor, CodeDetector
from models.sweet import SweetLogitsProcessor, SweetDetector
from models.wllm import WatermarkLogitsProcessor, WatermarkDetector
from models.sweetcode import SweetCodeLogitsProcessor, SweetCodeDetector
from models.exp import EXPLogitsProcessor, EXPDetector

logger = logging.getLogger(__name__)

def str_to_bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, str):
        value = value.lower()
        if value == "true":
            return True
        elif value == "false":
            return False
        elif value == "none":
            return None
    return value

def reserve_gpu_memory(
    amount: float, 
    device_id: int = 0,
    max_retries: int = 5,
    retry_delay: float = 1.0
) -> None:
    """
    Reserve GPU memory to prevent other frameworks from consuming all available memory.
    
    Args:
        amount: Memory to reserve (>1: in GB, <1: ratio of total memory)
        device_id: GPU device ID
        max_retries: Maximum number of allocation attempts
        retry_delay: Delay between retries in seconds
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            device = torch.device(f'cuda:{device_id}')
            free, total = torch.cuda.mem_get_info(device)
            
            if amount <= 0:
                raise ValueError("Amount must be positive")
            
            # Determine if amount is GB or ratio
            if amount < 1:  # ratio
                reserve_bytes = int(total * amount)
                amount_str = f"{amount:.1%}"
            else:  # GB
                reserve_bytes = int(amount * (1024 ** 3))
                amount_str = f"{amount:.1f}GB"
                
            # Calculate tensor size (using float32 - 4 bytes per element)
            num_elements = reserve_bytes // 4
            side_length = int(num_elements ** (1/3))  # Cube root for 3D tensor
            
            # Allocate and immediately delete tensor - memory remains in allocator cache
            x = torch.rand((side_length, side_length, side_length), device=device)
            del x
            logger.info(f"Successfully reserved {amount_str} of GPU memory")
            return  # Success - exit function
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            
    # If we get here, all attempts failed
    logger.error(f"Failed to reserve GPU memory after {max_retries} attempts: {last_error}")
    raise RuntimeError(f"Failed to reserve GPU memory after {max_retries} attempts: {last_error}")


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)
    
    
def find_code_start(tokenized_prefix, tokenized_code):
    # Convert tensors to lists if they're not already
    prefix_tokens = tokenized_prefix.tolist() if hasattr(tokenized_prefix, 'tolist') else tokenized_prefix
    code_tokens = tokenized_code.tolist() if hasattr(tokenized_code, 'tolist') else tokenized_code
    
    # Look for the first n tokens of code in prefix
    n = 10  # Number of tokens to match
    for i in range(len(prefix_tokens) - n):
        if prefix_tokens[i:i+n] == code_tokens[:n]:
            # print(f"Found code start at token position: {i}")
            # print("\nVerifying first {n} tokens match:")
            # for j in range(n):
            #     print(f"Token {j}: prefix[{i+j}] = {prefix_tokens[i+j]}, code[{j}] = {code_tokens[j]}")
            return i
            
    return -1

    
def get_roc_auc(human_z, machine_z):
    assert len(human_z) == len(machine_z)

    baseline_z_scores = np.array(human_z)
    watermark_z_scores = np.array(machine_z)
    all_scores = np.concatenate([baseline_z_scores, watermark_z_scores])

    baseline_labels = np.zeros_like(baseline_z_scores)
    watermarked_labels = np.ones_like(watermark_z_scores)
    all_labels = np.concatenate([baseline_labels, watermarked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    return roc_auc, fpr, tpr, thresholds

def get_tpr(fpr, tpr, error_rate):
    assert len(fpr) == len(tpr)

    value = None
    for f, t in zip(fpr, tpr):
        if f <= error_rate:
            value = t
        else:
            assert value is not None
            return value
        
    assert value == 1.0
    return value

def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    https://github.com/OpenLMLab/MOSS/blob/e088f438d1a95d424c6dffef0d73134ebe62cb72/models_jittor/generation.py#L146
    """
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))
        
    return cum_logits

def setup_accelerator(args):
    """Setup accelerator with robust wandb initialization and fallback options"""
    # Initialize accelerator with desired settings upfront
    global accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="wm_train",  # Set your desired project name here
            config=vars(args)
        )
    return accelerator


def synchronize_if_distributed():
    if accelerator.use_distributed:
        accelerator.wait_for_everyone()

def setup_logging(log_file: str = 'app.log', level: int = logging.INFO) -> None:
    """
    Set up logging configuration with both file and console handlers.
    
    Args:
        log_file: Path to the log file
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
def setup_output_dir(args):
    """Set up output directory with timestamp and subdirectories
    
    Args:
        args: Arguments from argument parser
        logger: Logger instance
        
    Returns:
        dict: Dictionary containing paths to various output directories
    """
    
    # Create shorter but still unique directory name
    timestamp = datetime.now().strftime("%m%d_%H%M")  # Shorter timestamp format
    pid = multiprocessing.current_process().pid
    
    # Create a short hash from pid and timestamp
    hash_input = f"{timestamp}{pid}".encode()
    short_hash = hashlib.md5(hash_input).hexdigest()[:4]
    
    # Final directory name will be like "0629_1423_a4f2"
    dir_name = f"{timestamp}_{short_hash}"
    output_dir = os.path.join(args.output_dir, dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Create subdirectories
    checkpoints_dir = os.path.join(output_dir, "checkpoints")  # For model checkpoints
    logs_dir = os.path.join(output_dir, "logs")  # For training logs
    results_dir = os.path.join(output_dir, "results")  # For evaluation results
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Created checkpoint directory: {checkpoints_dir}")
    logger.info(f"Created log directory: {logs_dir}")
    logger.info(f"Created results directory: {results_dir}")
    
    # Return dictionary of paths
    return {
        "base_dir": output_dir,
        "checkpoints": checkpoints_dir,
        "logs": logs_dir,
        "results": results_dir
    }

def setup_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across Python's random module, NumPy, and PyTorch.
    
    Args:
        seed (int): The random seed to use
        deterministic (bool, optional): If True, set PyTorch to use deterministic algorithms. 
                                      Defaults to True.
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # All GPUs
        torch.cuda.manual_seed(seed)      # Current GPU
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    logger.info(f"Using seed: {seed}")

def get_tokenizer(
    model_name: str,
    revision: str = "main",
    trust_remote_code: bool = True,
    use_auth_token: bool = True,
    truncation_side: str = "left",
    padding_side: str = "right",
) -> PreTrainedTokenizer:
    """
    Initialize and configure a tokenizer with standard settings.
    
    Args:
        model_name: Name or path of the model to load tokenizer from
        revision: Model revision to use
        trust_remote_code: Whether to trust remote code
        use_auth_token: HuggingFace auth token
        
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        token=use_auth_token,
        truncation_side=truncation_side,
        padding_side=padding_side,
    )
    
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loaded tokenizer from {model_name}")
    return tokenizer

class TensorDataset(Dataset):
    """
    Generic dataset class for tensor data with flexible field handling.
    """
    def __init__(self, encodings: Dict[str, torch.Tensor]):
        self.encodings = encodings
        
    def __len__(self) -> int:
        return len(next(iter(self.encodings.values())))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

def create_dataloaders(
    dataset: Dataset,
    train_ratio: float,
    batch_size: int,
    seed: int,
    collate_fn: Optional[callable] = None,
    val_batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: Input dataset
        train_ratio: Ratio of data to use for training
        batch_size: Training batch size
        seed: Random seed for reproducibility
        collate_fn: Optional custom collate function
        val_batch_size: Optional separate validation batch size
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
        
    val_batch_size = val_batch_size or batch_size
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader

def calculate_entropy(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> List[float]:
    """
    Calculate entropy for each position in the input sequence.
    
    Args:
        model: Language model to use for predictions
        input_ids: Input token IDs
        attention_mask: Optional attention mask
        
    Returns:
        List of entropy values for each position
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            return_dict=True
        )
        probs = torch.softmax(outputs.logits, dim=-1)
        entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        return entropy[0].cpu().tolist()

def get_scores(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Get logits for each position in the input sequence.
    
    Args:
        model: Language model to use for predictions
        input_ids: Input token IDs
        
    Returns:
        Tensor of logits with shape (sequence_length, vocab_size)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            return_dict=True
        )
        # Extract logits and remove batch dimension
        logits = outputs.logits[0]
        return logits

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    save_dir: str,
    metrics: Optional[Dict[str, float]] = None,
    accelerator: Optional[Accelerator] = None
) -> None:
    """
    Save a model checkpoint with associated training state.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch number
        save_dir: Directory to save checkpoint in
        metrics: Optional metrics to save
        accelerator: Optional accelerator for distributed training
    """
    os.makedirs(save_dir, exist_ok=True)
    
    save_function = accelerator.save if accelerator else torch.save
    is_main_process = not accelerator or accelerator.is_main_process
    
    if is_main_process:
        # Save model
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(
                save_dir,
                is_main_process=is_main_process,
                save_function=save_function
            )
        else:
            save_function(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            
        # Save training state
        training_state = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'metrics': metrics or {}
        }
        save_function(training_state, os.path.join(save_dir, "training_state.pt"))
        
        logger.info(f"Saved checkpoint to {save_dir}")
        
    if accelerator:
        accelerator.wait_for_everyone()

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    load_dir: str,
    accelerator: Optional[Accelerator] = None
) -> Tuple[nn.Module, torch.optim.Optimizer, Any, Dict[str, Any]]:
    """
    Load a model checkpoint and associated training state.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        load_dir: Directory containing checkpoint
        accelerator: Optional accelerator for distributed training
        
    Returns:
        Tuple of (model, optimizer, scheduler, training_state)
    """
    load_function = accelerator.load if accelerator else torch.load
    
    # Load model weights
    if hasattr(model, 'from_pretrained'):
        model = type(model).from_pretrained(load_dir)
    else:
        model.load_state_dict(load_function(os.path.join(load_dir, "pytorch_model.bin")))
        
    # Load training state
    training_state = load_function(os.path.join(load_dir, "training_state.pt"))
    optimizer.load_state_dict(training_state['optimizer'])
    if scheduler and training_state['scheduler']:
        scheduler.load_state_dict(training_state['scheduler'])
        
    logger.info(f"Loaded checkpoint from {load_dir}")
    
    return model, optimizer, scheduler, training_state


class PretrainedModelMixin:
    """
    Mixin class that adds functionality for saving and loading pretrained models.
    
    This mixin provides methods to:
    - Save model weights and configuration to disk
    - Load model weights and configuration from disk
    - Handle custom save/load functions
    """
    
    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        **kwargs
    ) -> None:
        """
        Save model weights and configuration to the specified directory.
        
        Args:
            save_directory (str): Path to directory where model should be saved
            is_main_process (bool, optional): Whether this is the main process in distributed training.
                                            Defaults to True.
            save_function (Callable, optional): Custom function to save model weights.
                                              If None, uses torch.save. Defaults to None.
            **kwargs: Additional arguments to pass to the save function
        """
        if not hasattr(self, 'config'):
            raise AttributeError(
                "Model must have a 'config' attribute containing model configuration"
            )
            
        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            
            # Save model weights
            model_path = os.path.join(save_directory, "pytorch_model.bin")
            if save_function is not None:
                save_function(self.state_dict(), model_path)
            else:
                torch.save(self.state_dict(), model_path)
            
            # Save configuration
            config_path = os.path.join(save_directory, "config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        load_function: Optional[Callable] = None,
        map_location: Optional[str] = None,
        **kwargs
    ) -> 'PretrainedModelMixin':
        """
        Load a model from pretrained weights and configuration.
        
        Args:
            load_directory (str): Path to directory containing the pretrained model
            load_function (Callable, optional): Custom function to load model weights.
                                              If None, uses torch.load. Defaults to None.
            map_location (str, optional): Device to map model to when loading.
                                        Defaults to None.
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            PretrainedModelMixin: Loaded model instance
            
        Raises:
            FileNotFoundError: If config.json or pytorch_model.bin not found
        """
        # Load configuration
        config_path = os.path.join(load_directory, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}"
            )
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Update config with any override kwargs
        config.update(kwargs)
        
        # Initialize model with loaded config
        model = cls(**config)
        
        # Load weights
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}"
            )
            
        if load_function is not None:
            state_dict = load_function(model_path)
        else:
            state_dict = torch.load(model_path, map_location=map_location)
            
        model.load_state_dict(state_dict)
        return model
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration dictionary
        
        Raises:
            AttributeError: If config not found
        """
        if not hasattr(self, '_config'):
            raise AttributeError(
                "Model must have a '_config' attribute containing model configuration"
            )
        return self._config
    
    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        """
        Set model configuration.
        
        Args:
            value (Dict[str, Any]): Configuration dictionary to set
        """
        self._config = value
        

def analyze_results(results, context_width: int, wm: str):
    """Analyze and compute metrics from generation results"""
    logger.info("Analyzing results")

    # Calculate basic statistics
    total_count = len(results)

    # Calculate pass rates
    pass_1 = sum([r['metrics']["pass@1"] for r in results.values()]) / len(results)
    pass_10 = sum([r['metrics']["pass@10"] for r in results.values()]) / len(results)
    
    if "z_score" not in results[list(results.keys())[0]]["metrics"]:
        return {
            "avg_pass_1": pass_1,
            "avg_pass_10": pass_10,
            "total_count": total_count,
        }
        
    # Calculate length statistics
    len_list = [r['metrics']["len"] for r in results.values()]
    avg_len = sum(len_list) / len(len_list)
    
    # Calculate entropy statistics
    entropy_list = [r['metrics']["entropy"] for r in results.values()]
    flattened_entropy_list = [item for sublist in entropy_list for item in sublist]
    avg_entropy = sum(flattened_entropy_list) / len(flattened_entropy_list)
    
    positive_count = sum([r["metrics"]["prediction"] for r in results.values()])
    
    # Calculate watermark statistics
    num_len = [r['metrics']['len'] for r in results.values()]
    if wm == "exp":
        # Extract scores for ROC analysis
        human_z = [-r["human_metrics"]["z_score"] for r in results.values()]
        machine_z = [-r["metrics"]["z_score"] for r in results.values()]
        roc_auc, fpr, tpr, _ = get_roc_auc(human_z, machine_z)
        
        # Calculate TPR at different FPR thresholds
        tpr_at_fpr_0 = get_tpr(fpr, tpr, 0)
        tpr_at_fpr_01 = get_tpr(fpr, tpr, 0.01)
        tpr_at_fpr_05 = get_tpr(fpr, tpr, 0.05)
        
        avg_p_value = sum([r['metrics']['z_score'] for r in results.values()]) / len(results)
        avg_human_p_value = sum([r['human_metrics']['z_score'] for r in results.values()]) / len(results)
        return {
            "avg_pass_1": pass_1,
            "avg_pass_10": pass_10,
            "total_count": total_count,
            "avg_p_value": avg_p_value,
            "avg_human_p_value": avg_human_p_value,
            "avg_len": avg_len,
            "roc_auc": roc_auc,
            "tpr_at_fpr_0": tpr_at_fpr_0,
            "tpr_at_fpr_01": tpr_at_fpr_01,
            "tpr_at_fpr_05": tpr_at_fpr_05,
            "avg_entropy": avg_entropy,
            "positive_count": positive_count,
        }
    
    # Extract scores for ROC analysis
    human_z = [r["human_metrics"]["z_score"] for r in results.values()]
    machine_z = [r["metrics"]["z_score"] for r in results.values()]
    roc_auc, fpr, tpr, _ = get_roc_auc(human_z, machine_z)
    
    # Calculate TPR at different FPR thresholds
    tpr_at_fpr_0 = get_tpr(fpr, tpr, 0)
    tpr_at_fpr_01 = get_tpr(fpr, tpr, 0.01)
    tpr_at_fpr_05 = get_tpr(fpr, tpr, 0.05)
    
    num_tokens_scored = [r['metrics']['num_tokens_scored'] for r in results.values()]
    num_green_tokens = [r['metrics']['num_green_tokens'] for r in results.values()]

    watermark_fraction_list = [num_tokens_scored[i] / num_len[i] for i in range(len(num_green_tokens)) if num_len[i] > 0]
    avg_watermark_fraction = sum(watermark_fraction_list) / len(watermark_fraction_list)
    
    green_fraction_list = [num_green_tokens[i] / num_tokens_scored[i] for i in range(len(num_green_tokens)) if num_tokens_scored[i] > 0]
    avg_green_fraction = sum(green_fraction_list) / len(green_fraction_list) if len(green_fraction_list) > 0 else 0
    
    avg_z_score = sum(machine_z) / len(machine_z) if len(machine_z) > 0 else 0
    avg_human_z_score = sum(human_z) / len(human_z) if len(human_z) > 0 else 0
    
    return {
        "avg_pass_1": pass_1,
        "avg_pass_10": pass_10,
        "avg_entropy": avg_entropy,
        "total_count": total_count,
        "positive_count": positive_count,
        "avg_len": avg_len,
        "roc_auc": roc_auc,
        "tpr_at_fpr_0": tpr_at_fpr_0,
        "tpr_at_fpr_01": tpr_at_fpr_01,
        "tpr_at_fpr_05": tpr_at_fpr_05,
        "avg_watermark_fraction": avg_watermark_fraction,
        "avg_green_fraction": avg_green_fraction,
        "avg_z_score": avg_z_score,
        "avg_human_z_score": avg_human_z_score
    }

def load_processors_and_detectors(
    device: torch.device,
    wm: str,
    vocab_size: int,
    gamma: float,
    delta: float,
    switch_threshold: float = -10.0,
    entropy_threshold: float = 0.5,
    z_threshold: float = 4,
    key_length: int = 512,
    detection_p_threshold: float = 0.1,
    block_size: int = None,
    tokenizer: Optional[AutoTokenizer] = None,
    code_model_path: Optional[str] = None,
    context_width: int = 4,
    output_dir: str = "./outputs"
) -> Tuple[Optional[LogitsProcessor], Optional[Any]]:
    """Load logits processors and detectors based on watermarking configuration"""
    
    logger.info(f"Loading processors and detectors for watermarking method: {wm}")
    processor = None
    detector = None
    
    if wm == "no":
        logger.info("No watermarking selected")
        pass
    elif wm == "wllm":
        logger.info("Loading WLLM watermarking")
        processor = WatermarkLogitsProcessor(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta
        )
        detector = WatermarkDetector(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            z_threshold=z_threshold,
            tokenizer=tokenizer,
        )
    elif wm == "sweet":
        logger.info("Loading SWEET watermarking")
        processor = SweetLogitsProcessor(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            entropy_threshold=entropy_threshold
        )
        detector = SweetDetector(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            entropy_threshold=entropy_threshold,
            z_threshold=z_threshold,
            tokenizer=tokenizer,
        )
    elif wm == "exp":
        logger.info("Loading EXP watermarking")
        processor = EXPLogitsProcessor(
            vocab_size=vocab_size,
            n = key_length,
        )
        detector = EXPDetector(
            vocab_size=vocab_size,
            n = key_length,
            detection_p_threshold=detection_p_threshold,
            k = block_size,
        )
    
    elif wm == "code":
        logger.info("Loading CODE watermarking")
        from src.modelhelper import CodeModel
        
        code_model_config = {
            "vocab_size": vocab_size, 
        }
        code_model = CodeModel(code_model_config).to(device)
        
        if code_model_path is not None:
            code_model_path = os.path.join("./train", code_model_path, "pytorch_model.bin")
            code_model.load_state_dict(torch.load(code_model_path))
            logger.info(f"=== Code model loaded from {code_model_path} ===")
        else:
            logger.info("Code model is None, so we use a random model")
        model_save_path = f"{output_dir}/pytorch_model.bin"
        torch.save(code_model.state_dict(), model_save_path)
        
        processor = CodeLogitsProcessor(
            model=code_model,
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            context_width=context_width,
            switch_threshold=switch_threshold
        )
        detector = CodeDetector(
            vocab_size=vocab_size,
            context_width=context_width,
            device=device,
            gamma=gamma,
            tokenizer=tokenizer,
            z_threshold=z_threshold,
            model=code_model,
            switch_threshold=switch_threshold
            )
    elif wm == "sweetcode":     
        logger.info("Loading sweet code watermarking")
        processor = SweetCodeLogitsProcessor(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            entropy_threshold=entropy_threshold
        )
        detector = SweetCodeDetector(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            entropy_threshold=entropy_threshold,
            z_threshold=z_threshold,
            tokenizer=tokenizer,
        )
    else:
        logger.error(f"Invalid watermarking method: {wm}")
        raise ValueError(f"Invalid watermarking method: {wm}")
    
    return processor, detector


def print_gpu_memory_status():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print("\nGPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}:")
            print(f"  Total: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated_memory:.2f} GB")
            print(f"  Cached: {cached_memory:.2f} GB")
            print(f"  Free: {total_memory - allocated_memory:.2f} GB")

def clean_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_model_memory_usage(model):
    """Get memory usage of a PyTorch model"""
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return (mem_params + mem_bufs) / 1024**3  # Convert to GB