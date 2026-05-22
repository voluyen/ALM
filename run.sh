#!/usr/bin/env bash
set -e

NAME=${1:-gpt2_1.5B_cross_tokenizer_distill}

python3 scripts/cross_tokenizer_distill.py \
    --config=configs/gpt2_1.5B_cross_tokenizer_distill.yaml \
    --overrides \
    n_data_parallel=1 \
    n_model_parallel=2 \
    name=$NAME
