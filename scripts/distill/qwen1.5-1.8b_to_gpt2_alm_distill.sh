#!/bin/bash
# Pair 0 (ALM-only): VoCuc/Qwen1.5_1.8B_SFT -> openai-community/gpt2 (124M).
# Full fine-tune, 20 epochs, batch 16 (match MTA-effective config).
# Losses: sft + alm_unconstrained (no MTA).
set -e

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

NAME=qwen1.5-1.8b_to_gpt2_alm

python3 pytorch_cross_tokenizer_distill.py \
    --config=configs/qwen1.5-1.8b_to_gpt2_alm_distill.yaml \
    --overrides \
    max_teacher_length=256 \
    max_student_length=256 \
    n_data_parallel=1 \
    n_model_parallel=1 \
    eval.tasks=[math_500_openmath2,gsm8k_openmath2] \
    eval.lengths=[2048] \
    eval.tokens_per_batch=16384 \
    eval.chat_template_mode=direct_encode_no_force_eos \
    log_interval=50 \
    sync_interval=100 \
    use_chat_template=false \
    chat_template_mode=direct_encode \
    hypernet.architecture=identity \
    train_embeddings=true \
    train_model_mode=full \
    eval_at_step_zero=false \
    save_at_step_zero=false \
    skip_lm_eval=true \
    latents_do_project=true \
    num_workers=8 \
    name=$NAME
