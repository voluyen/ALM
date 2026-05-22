#!/bin/bash
# Push a model as Flax version to HuggingFace Hub
# Usage: bash push_flax.sh [model_name_or_path] [hub_user] [extra_args] [tmp_path]

MODEL_NAME_OR_PATH="${1:-VoCuc/Qwen1.5_1.8B_SFT_Dolly}"
HUB_USER="${2:-baesad}"
EXTRA_ARGS="${3:-{\"attention_bias\": true, \"max_length\": 8192}}"
TMP_PATH="${4:-/tmp/push_flax_model}"

python3 scripts/push_flax_version_to_hub.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --hub_user "$HUB_USER" \
    --extra_args "$EXTRA_ARGS" \
    --tmp_path "$TMP_PATH" \
    --use_cpu
