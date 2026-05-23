#!/bin/bash
# Train GPT2-120M student with combined alm_unbiased + MTA loss
# against Qwen1.5-1.8B teacher.
# Requires: data/dolly_train_with_spans.jsonl (run precompute_spans.py first).

set -e

GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

NAME=gpt2_120M_alm_mta_v1

python3 pytorch_cross_tokenizer_distill.py \
    --config=gpt2_120M_alm_mta_distill.yaml \
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
    num_workers=16 \
    name=$NAME
