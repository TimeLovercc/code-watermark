import sys
import os
import ipdb
sys.path.append(os.getcwd())

import wandb
import json
from tqdm import tqdm
import argparse
import logging
import math

import torch
from transformers import (
    AutoModelForCausalLM,
    StoppingCriteriaList, 
    LogitsProcessorList, 
)

from lm_eval.tasks import ALL_TASKS

from src.datahelper import create_task_dataloader
from src.utils import (
    str_to_bool_or_none,
    reserve_gpu_memory,
    setup_seed,
    setup_output_dir,
    setup_logging,
    get_tokenizer,
    EndOfFunctionCriteria,
    calculate_entropy,
    get_scores,
    analyze_results,
    load_processors_and_detectors,
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument("--model", default="infly/OpenCoder-1.5B-Instruct")
    parser.add_argument("--code_model", type=str_to_bool_or_none, default=None)
    parser.add_argument("--revision", type=str_to_bool_or_none, default=None)
    parser.add_argument("--use_auth_token", type=str_to_bool_or_none, default=True)
    parser.add_argument("--trust_remote_code", type=str_to_bool_or_none, default=True)

    # Task configuration  
    parser.add_argument("--task_name", choices=ALL_TASKS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--postprocess", type=str_to_bool_or_none, default=True)
    parser.add_argument("--allow_code_execution", type=str_to_bool_or_none, default=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--do_sample", type=str_to_bool_or_none, default=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--eos", type=str, default="<|endoftext|>")
    parser.add_argument("--seed", type=int, default=0)

    # Watermarking configuration
    parser.add_argument("--wm", default="wllm", choices=["wllm", "sweet", "exp", "code", "no", "sweetcode"])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--n_detection", type=int, default=1)
    parser.add_argument("--entropy_threshold", type=float, default=1.2)
    parser.add_argument("--detection_z_threshold", type=float, default=4)
    parser.add_argument("--context_width", type=int, default=4)
    parser.add_argument("--switch_threshold", type=float, default=-10.0)
    
    # EXP-edit parameters
    parser.add_argument("--key_length", type=int, default=100, help="key length for EXP-edit")
    parser.add_argument("--block_size", type=int, default=20, help="block size for EXP-edit")
    parser.add_argument("--n_runs", type=int, default=50, help="EXP-edit p-value testing")
    parser.add_argument("--detection_p_threshold", type=float, default=0.1, help="EXP-edit p-value threshold")
    
    args = parser.parse_args()
    for arg in vars(args):
        setattr(args, arg, str_to_bool_or_none(getattr(args, arg)))
        
    if args.wm != "code":
        args.context_width = 1
    
    return args


def generate_code(tokenizer, model, processor, task, batch, device, args):
    
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_time": 60.0,
    }
    
    input_ids = batch["ids"][:, :batch["input_len"]].to(device)
    
    if task.stop_words:
        if tokenizer.eos_token:
            task.stop_words.append(tokenizer.eos_token)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )
        
        gen_kwargs["stopping_criteria"][0].start_length = (
            batch["input_len"].max().item()
        )
    
    if args.wm != "no":
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
        
    if args.wm == "exp":
        gen_kwargs["logits_processor"][0].preprocess(args.batch_size)

    generated_tokens = model.generate(
        input_ids=input_ids,
        num_return_sequences=args.batch_size,
        **gen_kwargs
    )
    logger.info(f"Generate tokens done")
    generated_tasks = batch["task_id"].repeat(args.batch_size)
    
    generated_codes = []
    for gen_token, gen_task in zip(generated_tokens, generated_tasks):
        if tokenizer.eos_token in task.stop_words:
            decoded = tokenizer.decode(
                gen_token[1:] if gen_token[0] == tokenizer.bos_token_id else gen_token,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
        else:
            decoded = tokenizer.decode(
                gen_token,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        
        if args.prefix:
            decoded = decoded[len(args.prefix):]
        
        if args.postprocess:
            decoded = task.postprocess_generation(decoded, int(gen_task))
        
        generated_codes.append(decoded)
    logger.info(f"Post generation processing done")
    return generated_codes


def evaluate_code(tokenizer, model, device, detector, task, gen_codes, batch, n_detection, args):
    logger.debug("Evaluating generated code")
    
    prompt = batch['prompt']
    reference = batch['reference']
    
    pass_info, pass_results = task.process_results([gen_codes], reference)
    if args.wm == "no":
        return {
            "pass@1": pass_info['pass@1'],
            "pass@10": pass_info.get('pass@10', 0),
            "pass_results": pass_results,
        }
    
    # Tokenize generated codes and move to device
    def tokenize_fun(example):
        inputs = tokenizer(
            example,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        )
        return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
    
    prefix = prompt
    tokenized_prefix = tokenize_fun(prefix)['input_ids'].squeeze().to(device)
    prefix_len = len(tokenized_prefix)
    
    n_detected = 0
    code_len = 0
    detection_result = None
    entropy_list = []
    for idx, code in enumerate(gen_codes):
        if n_detected >= n_detection:
            break

        tokenized_code = tokenize_fun(code)['input_ids'].squeeze().to(device)
        code_len = len(tokenized_code)
        
        if code_len == prefix_len:
            continue
        
        try:
            # First attempt at assertion
            assert torch.equal(tokenized_code[:prefix_len-1], tokenized_prefix[:-1]), "Tokenized code does not match the prefix!"
        except Exception as e:
            # If first attempt fails, try adjusting the tokenization
            try:
                from src.utils import find_code_start
                instruction_len = find_code_start(tokenized_prefix, tokenized_code)
                tokenized_prefix = tokenized_prefix[instruction_len:]
                prefix_len = len(tokenized_prefix)
                
                # Try assertion again with adjusted tokenization
                assert torch.equal(tokenized_code[:prefix_len-1], tokenized_prefix[:-1]), "Tokenized code does not match the prefix!"
            except Exception as e:
                # If both attempts fail
                print(e)
                ipdb.set_trace()

        if code_len == prefix_len:
            continue
            
        tokenized_main = tokenized_code[prefix_len:]
        len_main = len(tokenized_main)
        
        entropy = calculate_entropy(model, tokenized_code)
        entropy = [0] + entropy[:-1]
        
        if args.wm == "sweet":
            detection_result = detector.detect(
                tokenized_text=tokenized_code,
                prefix_len=prefix_len,
                entropy=entropy
            )
        elif args.wm == "exp":
            detection_result = detector.detect(
                generated_tokens=tokenized_main,
                n_runs=args.n_runs,
            )
        elif args.wm == "sweetcode":
            scores = get_scores(model, tokenized_code)
            scores = torch.cat([torch.zeros(1, scores.shape[1], device=device), scores[:-1]], dim=0)
            detection_result = detector.detect(
                tokenized_text=tokenized_code,
                prefix_len=prefix_len,
                entropy=entropy,
                scores=scores
            )
        else:
            detection_result = detector.detect(
                tokenized_text=tokenized_main,
            )

        if detection_result.pop('invalid', False):
            continue
        
        n_detected += 1
        entropy_list += entropy[prefix_len:]
    
    if n_detected == 0:
        return False
    # report both overall and performance of each completion)
    return {
        'pass@1': pass_info['pass@1'],
        'pass@10': pass_info.get('pass@10', 0),
        'pass_results': pass_results,
        "entropy": entropy_list,
        "len": len_main,
        **detection_result,
    }

def main(args):
    # Initialize wandb
    wandb.init(project="wm", config=vars(args))
    device = torch.device("cuda")
    
    # Setup output directory
    output_dir = setup_output_dir(args)
    
    # Setup logging
    log_file = os.path.join(output_dir['logs'], 'eval.log')
    setup_logging(log_file)
    logger.info(f"Output directory: {output_dir['base_dir']}")
    
    setup_seed(args.seed)
    
    # Setup tokenizer
    tokenizer = get_tokenizer(args.model)
    
    dict_precisions = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    
    logger.info(f"Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dict_precisions[args.precision],
        trust_remote_code=args.trust_remote_code,
        token=args.use_auth_token
    ).to(device)
    
    # Check if they match
    if len(tokenizer) != model.vocab_size:
        logger.info("Warning: Vocabulary size mismatch!")
        # Either resize model embeddings to match tokenizer
        model.resize_token_embeddings(len(tokenizer))
        # Or ensure you're using matching model/tokenizer versions
    
    # Load processors and detectors
    processor, detector = load_processors_and_detectors(
        device=device,
        wm=args.wm,
        vocab_size=len(tokenizer),
        gamma=args.gamma,
        delta=args.delta,
        switch_threshold=args.switch_threshold,
        entropy_threshold=args.entropy_threshold,
        detection_p_threshold=args.detection_p_threshold,
        block_size=args.block_size,
        key_length=args.key_length,
        tokenizer=tokenizer,
        code_model_path=args.code_model,
        context_width=args.context_width,
        output_dir=output_dir['base_dir']
    )

    # Load task data and prepare dataloader
    dataloader, task, data = create_task_dataloader(args.task_name, tokenizer, args)
    
    results = {}
    # Main generation and evaluation loop
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), 
                    total=len(data) * math.ceil(args.n_samples / args.batch_size),
                    desc="Generating and evaluating"):
            logger.info(f"Processing batch {i}")
            
            # Generate code samples
            gen_codes = generate_code(
                tokenizer=tokenizer, 
                model=model, 
                processor=processor, 
                task=task, 
                batch=batch, 
                device=device,
                args=args
            )
            
            # Evaluate machine-generated samples
            machine_result = evaluate_code(
                tokenizer=tokenizer,
                model=model,
                device=device,
                detector=detector,
                task=task,
                gen_codes=gen_codes,
                batch=batch,
                n_detection=args.n_detection,
                args=args
            )
            logger.info(f"Evaluate machine done")
            
            # Evaluate human reference samples
            human_result = evaluate_code(
                tokenizer=tokenizer,
                model=model,
                device=device,
                detector=detector,
                task=task,
                gen_codes=batch['full_data'],
                batch=batch,
                n_detection=args.n_detection,
                args=args
            )
            logger.info(f"Evaluate human done")
            if not machine_result or not human_result:
                logger.info(f"skip {i}")
                continue
            
            # Save results for this batch
            task_id = batch['task_id'].item()
            results[f"{task_id}"] = {
                "prompt": batch['prompt'],
                "generations": gen_codes,
                "reference": batch['reference'],
                "solution": batch['solution'],
                "metrics": machine_result,
                "human_metrics": human_result
            }
            
            # Log intermediate results
            wandb.log({
                "batch": i,
                "machine_z_score": machine_result.get("z_score", None),
                "human_z_score": human_result.get("z_score", None),
                "pass@1": machine_result["pass@1"],
                "pass@10": machine_result["pass@10"]
            })
            
            # Save batch results to file
            result_path = os.path.join(output_dir['results'], f"sample_{task_id}.json")
            with open(result_path, "w") as f:
                json.dump(results[f"{task_id}"], f, indent=2)
            wandb.save(result_path)
            
        # Analyze overall results
        overall_results = analyze_results(results, args.context_width, args.wm)
        logger.info(overall_results)
        wandb.log(overall_results)
        
        # Save final metrics
        metrics_path = os.path.join(output_dir['results'], "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(overall_results, f, indent=2)
        wandb.save(metrics_path)
        
    wandb.finish()
    logger.info("Execution completed successfully")


if __name__ == "__main__":
    args = parse_args()
    reserve_gpu_memory(13)
    main(args)
