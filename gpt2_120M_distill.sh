GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

NAME=gpt2_120m_distill
python3 pytorch_cross_tokenizer_distill.py \
    --config=gpt2_120M_cross_tokenizer_distill.yaml \
    --overrides \
    losses=[sft,alm_unconstrained] \
    alm_mode=merge_by_space_prob+append_space \
    tokenizer_pair_bias_threshold=0.1 \
    max_teacher_length=256 \
    max_student_length=256 \
    n_data_parallel=1 \
    n_model_parallel=1 \
    steps=7200 \
    warmup_steps=500 \
    eval_interval=50000 \
    save_interval=50000 \
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
    num_workers=24 \
    name=$NAME
