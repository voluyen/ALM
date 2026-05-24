#!/bin/bash
# Pair 0: VoCuc/Qwen1.5_1.8B_SFT -> openai-community/gpt2 (124M).
# Full fine-tune, 20 epochs, batch 8 (carried over from legacy gpt2_120M_distill.sh).
NAME=qwen1.5-1.8b_to_gpt2
python3 pytorch_cross_tokenizer_distill.py \
    --config=configs/qwen1.5-1.8b_to_gpt2_distill.yaml \
    --overrides \
    losses=[sft,alm_unconstrained,mta] \
    loss_weights=[1.0,1.0,2.0] \
    mta_mode=true \
    teacher_layer_mapping=[6,12,18,24] \
    student_layer_mapping=[3,6,9,12] \
    split_layer_mapping=[0,1,4,4] \
    entropy_weight=false \
    wo_span_weight=false \
    data.path=data/dolly_train_with_spans.jsonl \
    student_device=cuda:0 \
    teacher_device=cuda:0 \
    alm_mode=merge_by_space_prob+append_space \
    tokenizer_pair_bias_threshold=0.1 \
    max_teacher_length=256 \
    max_student_length=256 \
    n_data_parallel=1 \
    n_model_parallel=1 \
    epochs=20 \
    steps=0 \
    warmup_steps=0 \
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
    num_workers=8 \
    name=$NAME
