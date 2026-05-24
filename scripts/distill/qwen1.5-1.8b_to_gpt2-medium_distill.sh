#!/bin/bash
# Pair 1: Qwen1.5-1.8B -> GPT2-medium (full fine-tune, 20 epochs, batch 8).
# Requires data/dolly_train_with_spans.jsonl (run precompute_spans.py first).
set -e

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

NAME=qwen1.5-1.8b_to_gpt2-medium

python3 pytorch_cross_tokenizer_distill.py \
    --config=configs/qwen1.5-1.8b_to_gpt2-medium_distill.yaml \
    --overrides \
    max_teacher_length=256 \
    max_student_length=256 \
    n_data_parallel=1 \
    n_model_parallel=1 \
    eval.tasks=[math_500_openmath2,gsm8k_openmath2] \
    eval.lengths=[2048] \
    eval.tokens_per_batch=16384 \
    eval.chat_template_mode=direct_encode_no_force_eos \
    use_chat_template=false \
    chat_template_mode=direct_encode \
    hypernet.architecture=identity \
    eval_at_step_zero=false \
    save_at_step_zero=false \
    skip_lm_eval=true \
    num_workers=8 \
    name=$NAME
