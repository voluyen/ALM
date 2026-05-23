import argparse
from evaluator import Evaluator
from transformers import AutoModelForCausalLM, HfArgumentParser

import torch
import json
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    extra_parser.add_argument("--model_path", type=str, default=None)
    extra_parser.add_argument("--lora_path", type=str, default=None)
    extra_parser.add_argument("--tokenizer", type=str, default=None)
    extra_parser.add_argument("--student_device", type=str, default="cpu")
    extra_parser.add_argument("--val_batch_size", type=int, default=32)
    extra_parser.add_argument("--output_dir", type=str, default="./eval_results")

    

    extras = extra_parser.parse_args()

    set_seed(extras.seed)

    if extras.lora_path is not None:
        evaluator = Evaluator(
            tokenizer_path=extras.tokenizer,
            model_path=extras.model_path,
            distilled_lora=extras.lora_path,
            device=extras.student_device,
            # seeds=[10, 20, 30, 40, 50]
            seeds=[50]
        )
    else:
        evaluator = Evaluator(
            tokenizer_path=extras.tokenizer,
            model_path=extras.model_path,
            device=extras.student_device,
            # seeds=[10, 20, 30, 40, 50]
            seeds=[50]
        )
    
    evaluator.model.config.output_hidden_states=False
    evaluator.model.config.output_attentions=False

    benchmark_configs = {
                        'dolly': './data/dolly/valid.jsonl',
                        'sni': './data/sinst/11_/valid.jsonl',                        
                        'self_instruct': './data/self-inst/valid.jsonl',
                        'vicuna': './data/vicuna/valid.jsonl'
                        }
    

    # with torch.cuda.amp.autocast(dtype=torch.float16):
    #     results = evaluator.evaluate_multiple_benchmarks(
    #         benchmark_configs=benchmark_configs, 
    #         batch_size=extras.val_batch_size, 
    #         max_seq_length=256, max_new_tokens=512
    #     )

    # with open(extras.output_dir + "/eval.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)

    result = evaluator.evaluate_benchmark_dataset(
            dataset_path='./data/dialog/valid.jsonl',
            dataset_name='dialog', batch_size=extras.val_batch_size, 
            max_seq_length=512, max_new_tokens=384)
    
    dialog_result = {"rouge_l_f1": result, "status": "success"}
    with open(extras.output_dir + "/dialog_result_eval.json", "w", encoding="utf-8") as f:
        json.dump(dialog_result, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    main()