#!/bin/bash
# Push a model as Flax version to HuggingFace Hub
# Usage: bash push_flax.sh [model_name_or_path] [hub_user] [model_class] [extra_args]

MODEL_NAME_OR_PATH="${1:-VoCuc/Qwen1.5_1.8B_SFT_Dolly}"
HUB_USER="${2:-VoCuc}"
MODEL_CLASS="${3:-Llama}"
EXTRA_ARGS="${4:-{\"attention_bias\": true, \"max_length\": 8192}}"

python3 scripts/push_flax_version_to_hub.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --hub_user "$HUB_USER" \
    --model_class "$MODEL_CLASS" \
    --extra_args "$EXTRA_ARGS" \
    --use_cpu
