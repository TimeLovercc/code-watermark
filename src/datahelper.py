import sys
import os
sys.path.append(os.getcwd())

import math
import logging
import collections
import shutil
from collections import Counter
import ipdb
from typing import Dict, Any, List, Optional, Tuple
from itertools import chain, tee
from tqdm import tqdm
import gc
import pickle
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset, random_split, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.asthelper import CodeAugmenter

from lm_eval import tasks

logger = logging.getLogger(__name__)

NUM_AUGMENTATIONS = 3

def ngrams(sequence, n: int, pad_left: bool = False, pad_right: bool = False, pad_symbol: Any = None):
    """Generate n-grams from a sequence"""
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)
    
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)

class TaskDataset(IterableDataset):
    """Dataset class that handles loading, tokenization and iteration of task data"""
    
    def __init__(self, 
                 task_name: str,
                 tokenizer,
                 prefix: str = "",
                 n_samples: int = 1,
                 batch_size: int = 1,
                 max_length: int = 512,
                 limit: Optional[int] = None):
        """
        Initialize dataset with task configuration
        
        Args:
            task_name: Name of the task to load
            tokenizer: Tokenizer to use for processing prompts
            prefix: Optional prefix to add to prompts
            n_samples: Number of samples to generate per prompt
            batch_size: Batch size for determining copies needed
            max_length: Maximum sequence length for tokenization
            limit: Optional limit on number of examples to load
        """
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.n_copies = math.ceil(n_samples / batch_size)
        self.max_length = max_length
        
        # Load task data
        task_data, task = self._load_task_data(limit)
        self._process_prompts(task_data)
        
        self.data = task_data
        self.task = task
        
        logger.info(f"Created TaskDataset with {len(self.data)} items and {self.n_copies} copies each")

    def _load_task_data(self, limit: Optional[int]) -> Tuple[List[Dict], Any]:
        """Load and prepare task data"""
        logger.info(f"Loading task data for {self.task_name}")
        try:
            task = tasks.get_task(self.task_name)
            dataset = task.get_dataset()
        except Exception as e:
            logger.error(f"Failed to load task {self.task_name}: {str(e)}")
            raise

        logger.info(f"Loading {len(dataset)} items from dataset")
        
        task_data = []
        for idx, item in enumerate(dataset):
            task_data.append({
                'task_id': idx,
                'full_data': task.get_full_data(item),
                'reference': task.get_reference(item),
                'solution': task.get_solutions(item),
                'prompt': task.get_prompt(item)
            })
            
        if limit:
            task_data = task_data[10:30]
            
        return task_data, task

    def _process_prompts(self, task_data):
        """Process and tokenize prompts"""
        # Add prefix to prompts if they're strings
        prompts = [
            self.prefix + item['prompt'] if isinstance(item['prompt'], str) else item['prompt']
            for item in task_data
        ]
        
        # Tokenize prompts
        self.tokenized_prompts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_attention_mask=True
        )

    def __iter__(self):
        """Iterate over dataset items with proper copies"""
        for i in range(len(self.data)):
            for _ in range(self.n_copies):
                yield {
                    "task_id": self.data[i]['task_id'],
                    "ids": self.tokenized_prompts.input_ids[i],
                    "input_len": self.tokenized_prompts.attention_mask[i].sum(),
                    "attention_mask": self.tokenized_prompts.attention_mask[i],
                    "prompt": self.data[i]['prompt'],
                    "reference": self.data[i]['reference'],
                    "full_data": self.data[i]['full_data'],
                    "solution": self.data[i]['solution']
                }

def create_task_dataloader(task_name: str, tokenizer: AutoTokenizer, args) -> Tuple[DataLoader, Any, List[Dict]]:
    """
    Create a dataloader for task evaluation
    
    Args:
        task_name: Name of the task to load
        tokenizer: Tokenizer to use for processing prompts
        prefix: Optional prefix to add to prompts
        n_samples: Number of samples to generate per prompt
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length for tokenization
        limit: Optional limit on number of examples to load
    
    Returns:
        Tuple of (DataLoader, task, data)
    """
    # Create dataset
    dataset = TaskDataset(
        task_name=task_name,
        tokenizer=tokenizer,
        prefix=args.prefix,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit
    )
    
    # Define collate function
    def collate_fn(batch):
        return {
            "task_id": torch.tensor([int(item["task_id"]) for item in batch]),
            "ids": torch.stack([item["ids"] for item in batch]),
            "input_len": torch.tensor([item["input_len"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "prompt": [item["prompt"] for item in batch],
            "reference": [item["reference"] for item in batch],
            "full_data": [item["full_data"] for item in batch],
            "solution": [item["solution"] for item in batch]
        }
    
    # Create and return dataloader
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    return dataloader, dataset.task, dataset.data

class UnifiedSFTDataset(Dataset):
    """Unified dataset combining CodeAlpaca and standard SFT functionality"""
    
    TASK_SOURCES = {
        "humaneval": "task",
        "mbpp": "task",
        "gen_humaneval": "generated",
        "gen_mbpp": "generated",
        "gen_humanevalsynthesize-java": "generated",
        "gen_humanevalsynthesize-js": "generated",
        "gen_humanevalsynthesize-cpp": "generated",
        "codealpaca": "alpaca"
    }
    
    def __init__(
        self,
        task_name: str,
        augmentation: bool,
        tokenizer,
        llm_model,
        max_length: int,
        context_width: int = 4,
        limit: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        generated_data_dir: Optional[str] = "outputs/dp_generation"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_width = context_width
        self.llm_model = llm_model
        self.augmentation = augmentation
        self.task_name = task_name
        self.limit = limit
        self.generated_data_dir = generated_data_dir
        
        # Set up caching directory only for CodeAlpaca
        self.cache_dir = Path(cache_dir) if cache_dir else Path(f"cache/{task_name}")
            
        if str(self.cache_dir).endswith('/test'):
            if self.cache_dir.exists():
                logger.info(f"Clearing test cache directory: {self.cache_dir}")
                shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Language patterns for code block identification
        self.language_patterns = {
            'python': r'```\s*(?:python|py)\s*\n',
            'sql': r'```\s*(?:sql|srql)\s*\n',
            'javascript': r'```\s*(?:javascript|js)\s*\n',
            'java': r'```\s*(?:java)\s*\n',
            'cpp': r'```\s*(?:cpp|c\+\+)\s*\n',
            'rust': r'```\s*(?:rust|rs)\s*\n',
            'go': r'```\s*(?:go|golang)\s*\n',
            'ruby': r'```\s*(?:ruby|rb)\s*\n',
            'php': r'```\s*(?:php)\s*\n',
            'typescript': r'```\s*(?:typescript|ts)\s*\n',
            'generic': r'```(?!\s*(?:python|py|sql|srql|javascript|js|java|cpp|c\+\+|rust|rs|go|golang|ruby|rb|php|typescript|ts)\s*\n)'
        }

        if task_name not in self.TASK_SOURCES:
            raise ValueError(f"Unsupported task name: {task_name}. Must be one of {list(self.TASK_SOURCES.keys())}")

        # Process data with or without caching based on task type
        if use_cache:
            self._process_data_with_cache()
        else:
            self._process_data_without_cache()
        
        logger.info(f"Created UnifiedSFTDataset with {len(self.ngram_data)} unique ngrams")

    def _process_data_with_cache(self):
        """Process CodeAlpaca data with caching at each step"""
        # Step 1: Load raw data
        raw_data_cache = self.cache_dir / "raw_data.pkl"
        if raw_data_cache.exists():
            logger.info("Loading cached raw data...")
            data = self._load_cache(raw_data_cache)
        else:
            logger.info("Processing raw data...")
            data = self._load_raw_data()
            self._save_cache(data, raw_data_cache)

        # Step 2: Apply augmentation if enabled
        if self.augmentation:
            aug_data_cache = self.cache_dir / "augmented_data.pkl"
            if aug_data_cache.exists():
                logger.info("Loading cached augmented data...")
                data = self._load_cache(aug_data_cache)
            else:
                logger.info("Applying code augmentation...")
                data = self._augment_data(data)
                self._save_cache(data, aug_data_cache)

        # Step 3: Compute model outputs
        logger.info("Computing model outputs for all examples...")
        # examples_outputs = self._compute_model_outputs_batch(data)
        examples_outputs = []
        for item in tqdm(data, desc="Computing model outputs"):
            outputs = self._compute_model_outputs(item)
            examples_outputs.extend(outputs)
        del data  # Explicitly mark for garbage collection

        logger.info("Processing ngrams from examples...")
        self.ngram_data = self._process_ngrams(examples_outputs)
        del examples_outputs  # Explicitly mark for garbage collection

    def _process_data_without_cache(self):
        """Process non-CodeAlpaca data without caching"""
        logger.info("Processing raw data...")
        data = self._load_raw_data()
        
        if self.augmentation:
            logger.info("Applying code augmentation...")
            data = self._augment_data(data)  # Pass data as parameter
        
        logger.info("Computing model outputs for all examples...")
        # examples_outputs = self._compute_model_outputs_batch(data)
        examples_outputs = []
        for item in tqdm(data, desc="Computing model outputs"):
            outputs = self._compute_model_outputs(item)
            examples_outputs.extend(outputs)
        del data  # Explicitly mark for garbage collection
        
        logger.info("Processing ngrams from examples...")
        self.ngram_data = self._process_ngrams(examples_outputs)
        del examples_outputs  # Explicitly mark for garbage collection

    def _load_cache(self, cache_path: Path) -> Any:
        """Load data from cache file"""
        try:
            with cache_path.open('rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache from {cache_path}: {str(e)}")
            cache_path.unlink(missing_ok=True)
            raise

    def _save_cache(self, data: Any, cache_path: Path) -> None:
        """Save data to cache file"""
        try:
            with cache_path.open('wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache to {cache_path}: {str(e)}")
            cache_path.unlink(missing_ok=True)
            raise

    def _load_raw_data(self) -> List[Dict]:
        """Load raw data based on task source"""
        source_type = self.TASK_SOURCES[self.task_name]
        
        if source_type == "task":
            # Load from original task
            logger.info(f"Loading task data for {self.task_name}")
            data, self.task = self.load_task_data(self.task_name, self.limit)
            
        elif source_type == "generated":
            # Load generated data
            original_task = self.task_name.replace('gen_', '')
            generated_file = Path(self.generated_data_dir) / f'{original_task}_generated_data.pt'
            
            logger.info(f"Loading generated data from {generated_file}")
            if not generated_file.exists():
                raise FileNotFoundError(f"Generated data file not found: {generated_file}")
                
            try:
                generated_data = torch.load(str(generated_file))
                data = []
                
                for task_id, item in generated_data.items():
                    for gen_idx, solution in enumerate(item['generations']):
                        data.append({
                            'task_id': f"{task_id}_{gen_idx}",
                            'solution': solution,
                            'prompt': item['prompt'],
                            'reference': item['reference'],
                            'full_data': item['full_data'][gen_idx]
                        })
                        
                logger.info(f"Loaded {len(data)} generated examples")
                
            except Exception as e:
                logger.error(f"Error loading generated data: {str(e)}")
                raise
                
        elif source_type == "alpaca":
            # Load CodeAlpaca data
            from datasets import load_dataset
            logger.info("Loading CodeAlpaca dataset")
            try:
                dataset = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
                if self.limit:
                    dataset = dataset.shuffle(seed=42).select(range(self.limit))
                    
                data = []
                for idx, example in enumerate(dataset):
                    instruction = example.get('instruction', '')
                    output = example.get('output', '')
                    data.append({
                        'task_id': idx,
                        'prompt': instruction,
                        'full_data': f"{instruction}\n{output}",
                        'solution': output
                    })
                        
                logger.info(f"Loaded {len(data)} valid CodeAlpaca examples")
                
            except Exception as e:
                logger.error(f"Failed to load CodeAlpaca dataset: {str(e)}")
                raise
        
        return data

    def _extract_code_blocks(self, text: str) -> Tuple[str, str, str]:
        """Extract the first code block and its language from text using markdown code block patterns"""
        if self.task_name != 'codealpaca':
            # For non-codealpaca, treat entire text as code with python language
            return ('', text, 'python')
            
        # Find first code block using markdown pattern
        match = re.search(r'```.*?```', text, re.DOTALL)
        
        if match:
            block_text = text[match.start():match.end()]
            for lang, pattern in self.language_patterns.items():
                if re.match(pattern, block_text):
                    # Find the content after the language specification
                    content_start = re.search(r'\n', block_text)
                    if content_start:
                        # Extract code between ``` markers, excluding the language specification and closing ```
                        code = block_text[content_start.end():len(block_text)-3].strip()
                        detected_lang = lang if lang != 'generic' else 'python'
                        pre_text = text[:match.start()].strip()
                        return (pre_text, code, detected_lang)
                
        return (None, None, 'unknown')  # Default case


    def _augment_data(self, data) -> List[Dict]:
        """Apply code augmentation to dataset with proper code block extraction"""
        augmenter = CodeAugmenter()
        success_count = 0
        fail_count = 0
        no_code_count = 0
        filtered_data = []
        
        for item in tqdm(data, desc="Augmenting data"):
            # Get the prompt and solution
            prompt = item.get('prompt', '')
            solution = item.get('solution', '')
            
            # Extract code block and its language from solution
            pre_code_text, code_block, lang = self._extract_code_blocks(solution)
            
            # Skip if no code block found
            if code_block is None:
                no_code_count += 1
                continue
                
            full_pre_code_text = prompt + "\n" + pre_code_text
            prefix_tokens = self.tokenizer(full_pre_code_text)
            item['prefix_len'] = len(prefix_tokens['input_ids'])
            
            # Apply augmentation if it's Python code
            if lang == 'python':
                try:
                    augmented_versions = augmenter.augment(code_block, NUM_AUGMENTATIONS)
                    for aug_idx, aug_code in enumerate(augmented_versions, 1):
                        # Construct full augmented text with prompt
                        full_aug_text = full_pre_code_text + aug_code
                        item[f'aug_code{aug_idx}'] = full_aug_text
                        item[f'aug_type{aug_idx}'] = 'augmented'
                        success_count += 1
                except Exception as e:
                    fail_count += 1
                    
            item['aug_type0'] = 'original'
            item['language'] = lang
            filtered_data.append(item)

        logger.info(f"Successfully augmented code in {success_count} cases")
        if fail_count > 0:
            logger.warning(f"Failed to augment code in {fail_count} cases")
        
        if no_code_count > 0:
            logger.warning(f"No code block found in {no_code_count} cases")
            
        return filtered_data


    def _compute_model_outputs(self, item: Dict) -> List[Tuple[List[int], List[float], torch.Tensor]]:
        """Compute entropy and logits for original and augmented versions of code
        
        Args:
            item: Dictionary containing original and augmented code versions
            
        Returns:
            List of tuples containing (token_ids, entropy_values, logits) for each version
        """
        self.llm_model = self.llm_model.to(dtype=torch.bfloat16)
        self.llm_model.eval()
        outputs_list = []
        
        # Get all code versions (original + augmented)
        code_versions = []
        for key in item.keys():
            if key == 'full_data' or key.startswith('aug_code'):
                code_versions.append((key, item[key]))
        
        with torch.no_grad():
            for key, code in code_versions:
                tokenized = self.tokenizer(
                    code,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length
                ).to(self.llm_model.device)
                
                outputs = self.llm_model(**tokenized, return_dict=True)
                
                # Compute probabilities and entropy
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.where(probs > 0, probs * torch.log(probs), probs.new([0.0])).sum(dim=-1)
                
                # Convert to CPU and extract data
                token_ids = tokenized.input_ids[0].cpu().tolist()
                entropy_values = entropy[0].cpu().tolist()
                logits_values = logits[0].cpu()
                
                # # Create code mask using prefix length
                # code_mask = [0] * item['prefix_len'] + [1] * (len(token_ids) - item['prefix_len'])
                # code_mask = code_mask[:len(token_ids)]  # Truncate if needed
                
                # Get language from the item
                detected_language = item.get('language', 'unknown')
                
                updated_entropy_values = [0.0] + entropy_values[:-1]
                updated_logits_values = torch.cat([torch.zeros_like(logits_values[0]).unsqueeze(0), logits_values[:-1]], dim=0)
                # Store with code version info
                outputs_list.append({
                    'token_ids': token_ids,
                    'full_data': code,
                    'entropy_values': updated_entropy_values,
                    'logits_values': updated_logits_values,
                    'code_type': key,  # Store which version this is
                    'aug_type': item.get(f'aug_type{key[-1]}' if key.startswith('aug_code') else 'aug_type0', 'unknown'),
                    # 'code_mask': code_mask,
                    'language': detected_language
                })
                
        return outputs_list


    def _process_ngrams(self, examples_outputs: List[Dict]) -> List[Dict]:
        """Process ngrams from all examples"""
        # Initialize tracking dictionaries
        ngram_metrics = {}  # Stores metrics for each unique ngram
        prefix_continuations = collections.defaultdict(list)
        ngram_sources = collections.defaultdict(list)
        
        # Process each example
        no_code_count = 0
        for example in examples_outputs:
            token_ids = example['token_ids']
            entropy_values = example['entropy_values']
            logits_values = example['logits_values']  # Get logits values
            code_type = example['code_type']
            aug_type = example['aug_type']
            full_data = example['full_data']
            # code_mask = example['code_mask']
            language = example['language']
            
            seq_len = len(token_ids)
            # no_code_count = no_code_count + 1 if sum(code_mask) == 0 else no_code_count
            
            # Generate ngrams with sliding window
            for pos in range(self.context_width + 1, seq_len + 1):
                ngram_begin = pos - self.context_width - 1
                current_ngram = tuple(token_ids[ngram_begin:pos])
                prefix = current_ngram[:-1]
                next_token = current_ngram[-1]
                
                # Get masks for the ngram
                # ngram_masks = code_mask[ngram_begin:pos]
                
                # Only store if last mask element is 1 (indicating code)
                # if ngram_masks[-1] == 1:
                # Store metrics
                ngram_key = current_ngram
                if ngram_key not in ngram_metrics:
                    ngram_metrics[ngram_key] = {
                        'entropy': entropy_values[pos-1],
                        'logits': logits_values[pos-1],  # Store logits
                        # 'masks': ngram_masks,
                        'language': language
                    }
                
                # Record continuation
                prefix_continuations[prefix].append({
                    'next_token': next_token,
                    'code_type': code_type,
                    'aug_type': aug_type,
                    # 'masks': ngram_masks,
                    'language': language
                })
                
                # Track source
                ngram_sources[current_ngram].append({
                    'code_type': code_type,
                    'aug_type': aug_type,
                    # 'masks': ngram_masks,
                    'language': language
                })
        
        logger.info(f"Found {no_code_count} examples without code blocks")
        
        # Build final processed dataset
        processed_ngrams = []
        seen_ngram_keys = set()
        
        for prefix, continuations in prefix_continuations.items():
            next_tokens = [cont['next_token'] for cont in continuations]
            token_counts = dict(Counter(next_tokens))
            
            for cont in continuations:
                full_ngram = prefix + (cont['next_token'],)
                ngram_key = full_ngram
                
                if ngram_key not in seen_ngram_keys:
                    seen_ngram_keys.add(ngram_key)
                    
                    processed_ngrams.append({
                        'ngram': full_ngram,
                        'prefix': prefix,
                        'next_token': cont['next_token'],
                        'entropy': ngram_metrics[ngram_key]['entropy'],
                        'logits': ngram_metrics[ngram_key]['logits'],  # Include logits
                        'code_type': cont['code_type'],
                        'aug_types': [cont['aug_type']],
                        'source_count': len(ngram_sources[full_ngram]),
                        # 'masks': ngram_metrics[ngram_key]['masks'],
                        'language': ngram_metrics[ngram_key]['language'],
                        'token_counts': token_counts
                    })
        
        logger.info(f"Processed {len(processed_ngrams)} ngrams")
        return processed_ngrams

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single training example"""
        item = self.ngram_data[idx]
        return {
            'input_ids': torch.tensor(item['prefix']),
            'next_token': torch.tensor(item['next_token']),
            'entropy': torch.tensor(item['entropy']),
            'logits': torch.tensor(item['logits']),  # Return logits tensor
            'code_type': item['code_type'],
            'aug_types': item['aug_types'],
            'source_count': item['source_count'],
            # 'masks': torch.tensor(item['masks']),
            'language': item['language'],
            'token_counts': item['token_counts']
        }
        
    def __len__(self) -> int:
        return len(self.ngram_data)

    def load_task_data(self, task_name: str, limit: Optional[int] = None) -> Tuple[List[Dict], Any]:
        """Load and prepare task data"""
        from lm_eval import tasks
        
        logger.info(f"Loading task data for {task_name}")
        try:
            task = tasks.get_task(task_name)
            dataset = task.get_dataset()
        except Exception as e:
            logger.error(f"Failed to load task {task_name}: {str(e)}")
            raise

        logger.info(f"Loading {len(dataset)} items from dataset")
        
        task_data = []
        for idx, item in enumerate(dataset):
            task_data.append({
                'task_id': idx,
                'full_data': task.get_full_data(item),
                'reference': task.get_reference(item),
                'solution': task.get_solutions(item),
                'prompt': task.get_prompt(item)
            })
        if limit:
            task_data = task_data[:limit]
            
        return task_data, task


def create_sft_dataloaders(
    task_name: str, 
    tokenizer: AutoTokenizer, 
    device: torch.device, 
    args
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for supervised fine-tuning
    
    Args:
        task_name: Name of the task or list of task names to load
        tokenizer: Tokenizer for processing text
        device: Device to load model on
        args: Arguments containing training configuration
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if ',' in task_name:
        # Split on commas and strip whitespace
        tasks = [task.strip() for task in task_name.split(',')]
        # Remove any empty strings
        tasks = [task for task in tasks if task]
        task_names = tasks
    logger.info(f"Loading task data for {task_name}")
    
    dict_precisions = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    
    # Initialize model
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dict_precisions[args.precision],
        trust_remote_code=args.trust_remote_code,
        token=args.use_auth_token
    ).to(device)
    
    if len(tokenizer) != llm_model.vocab_size:
        logger.warning("Vocabulary size mismatch! Resizing model embeddings...")
        llm_model.resize_token_embeddings(len(tokenizer))

    # Convert single task to list for uniform processing
    task_names = [task_name] if isinstance(task_name, str) else task_name
    
    # Validate task names
    for task in task_names:
        if task not in UnifiedSFTDataset.TASK_SOURCES:
            raise ValueError(f"Unsupported task: {task}. Must be one of {list(UnifiedSFTDataset.TASK_SOURCES.keys())}")
    
    # Create datasets for each task
    datasets = []
    total_examples = 0
    for task in task_names:
        # Handle per-task sample limits if specified
        task_limit = args.task_limits.get(task, args.limit) if hasattr(args, 'task_limits') else args.limit
        
        dataset = UnifiedSFTDataset(
            task_name=task,
            augmentation=args.augmentation,
            tokenizer=tokenizer,
            llm_model=llm_model,
            max_length=args.max_length,
            context_width=args.context_width,
            limit=task_limit,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else None
        )
        datasets.append(dataset)
        total_examples += len(dataset)
        logger.info(f"Loaded {len(dataset)} examples from {task}")
    
    # Combine datasets if multiple tasks
    if len(datasets) > 1:
        # Get sampling weights for each dataset if specified
        if hasattr(args, 'task_weights') and args.task_weights:
            weights = [args.task_weights.get(task, 1.0) for task in task_names]
            logger.info(f"Using task weights: {dict(zip(task_names, weights))}")
            
            # Adjust dataset sizes based on weights
            weighted_datasets = []
            for dataset, weight in zip(datasets, weights):
                if weight != 1.0:
                    # Sample the dataset according to weight
                    num_samples = int(len(dataset) * weight)
                    indices = torch.randperm(len(dataset))[:num_samples]
                    weighted_dataset = torch.utils.data.Subset(dataset, indices)
                    weighted_datasets.append(weighted_dataset)
                else:
                    weighted_datasets.append(dataset)
            combined_dataset = ConcatDataset(weighted_datasets)
        else:
            combined_dataset = ConcatDataset(datasets)
        
        logger.info(f"Combined {len(datasets)} datasets with {len(combined_dataset)} total examples")
    else:
        combined_dataset = datasets[0]
    
    # Split into train and validation sets
    train_size = int(args.train_ratio * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    def collate_fn(batch):
        """Collate items into batches with proper handling of item types"""
        collated = {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'next_token': torch.stack([item['next_token'] for item in batch]),
            'entropy': torch.stack([item['entropy'] for item in batch]),
            'logits': torch.stack([item['logits'] for item in batch]),
            'code_type': [item['code_type'] for item in batch],
            'aug_types': [item['aug_types'] for item in batch],
            'source_count': torch.tensor([item['source_count'] for item in batch]),
            # 'masks': torch.stack([item['masks'] for item in batch]),
            'language': [item['language'] for item in batch],
            'token_counts': [item['token_counts'] for item in batch]
        }
        return collated

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Log dataset statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total examples: {len(combined_dataset)}")
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Count examples per task in combined dataset
    if len(datasets) > 1:
        task_counts = {task: 0 for task in task_names}
        for idx in range(len(combined_dataset)):
            item = combined_dataset[idx]
            task_type = item['code_type'].split('_')[0]  # Extract task from code_type
            for task in task_names:
                if task in task_type:
                    task_counts[task] += 1
                    break
        
        logger.info("\nExamples per task:")
        for task, count in task_counts.items():
            percentage = (count / len(combined_dataset)) * 100
            logger.info(f"{task}: {count} ({percentage:.1f}%)")
    
    return train_dataloader, val_dataloader


def prepare_and_generate_data(
    task_name: str,
    model_name: str = "infly/OpenCoder-1.5B-Instruct",
    output_dir: str = "outputs/generated_data",
    batch_size: int = 20,
    n_samples: int = 20,
    max_length: int = 2048,
    precision: str = "bf16",
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare environment and generate data for training with enhanced logging.
    
    Args:
        task_name: Name of the task (humaneval or mbpp)
        model_name: Name of the model to use
        output_dir: Directory to save outputs
        batch_size: Batch size for generation
        n_samples: Number of samples to generate per prompt
        max_length: Maximum length of generated sequences
        precision: Model precision (bf16, fp16, fp32)
        seed: Random seed
        limit: Limit on number of examples to process
    
    Returns:
        Dictionary containing all generated data and metadata
    """
    # Setup basic configuration
    from src.utils import setup_seed, setup_logging, get_tokenizer
    from src.main import generate_code
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(
        os.path.join(output_dir, 'generation.log'),
        level=logging.INFO,
    )
    
    # Log configuration
    logger.info("Starting data generation with configuration:")
    logger.info(f"Task: {task_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Samples per prompt: {n_samples}")
    logger.info(f"Precision: {precision}")
    
    # Setup model precision
    dict_precisions = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model from {model_name}")
    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dict_precisions[precision],
        trust_remote_code=True
    ).to(device)
    
    # Adjust vocab size if needed
    if len(tokenizer) != model.vocab_size:
        logger.info("Resizing model embeddings to match tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    
    # Create arguments namespace
    class Args:
        pass
    args = Args()
    args.model = model_name
    args.batch_size = batch_size
    args.max_length = max_length
    args.n_samples = n_samples
    args.temperature = 0.2
    args.top_p = 0.95
    args.top_k = 0
    args.do_sample = True
    args.prefix = ""
    args.postprocess = False
    args.limit = limit
    args.wm = "no"
    
    # Load data
    logger.info("Loading task data")
    dataloader, task, data = create_task_dataloader(task_name, tokenizer, args)
    
    # Generate data
    results = {}
    total_batches = len(data) * math.ceil(args.n_samples / args.batch_size)
    logger.info(f"Starting generation for {total_batches} batches")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), 
                           total=total_batches,
                           desc="Generating data"):
            # Generate code samples
            gen_codes = generate_code(
                tokenizer=tokenizer,
                model=model,
                processor=None,
                task=task,
                batch=batch,
                device=device,
                args=args
            )
            
            # Store results
            task_id = batch['task_id'].item()
            results[f"{task_id}"] = {
                "task_id": task_id,
                "prompt": batch['prompt'][0],
                "reference": batch['reference'][0],
                "generations": gen_codes,
                "full_data": gen_codes
            }
            
            # Log progress periodically
            if (i + 1) % 10 == 0:
                logger.info(f"Completed batch {i + 1}/{total_batches}")
    
    # Save results
    output_file = os.path.join(output_dir, f'{task_name}_generated_data.pt')
    torch.save(results, output_file)
    logger.info(f"Saved generated data to {output_file}")
    
    # Log completion statistics
    logger.info("Generation completed:")
    logger.info(f"Total tasks processed: {len(results)}")
    logger.info(f"Total generations: {sum(len(data['generations']) for data in results.values())}")
    
    return results

if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1" 
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
    
    # Generate data
    generated_data = prepare_and_generate_data(
        task_name="mbpp",
        model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
        output_dir="outputs/dp_generation",
        batch_size=20,
        n_samples=20,
        limit=0
    )
    
    # Process and display results
    print("\nGeneration Results Summary:")
    print("-" * 50)
    for task_id, data in generated_data.items():
        print(f"\nTask {task_id}:")
        print(f"Number of generations: {len(data['generations'])}")
        print(f"First generation preview: {data['generations'][0][:100]}...")