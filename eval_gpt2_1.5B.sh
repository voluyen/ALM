#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="openai-community/gpt2-xl"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}-ckpt-14290"


mkdir -p ${OUTPUT_DIR}

OPTS=""

# training
OPTS+=" --val_batch_size 32"

# devices
OPTS+=" --student_device cuda:0"

# models
OPTS+=" --output_dir ${OUTPUT_DIR}"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
OPTS+=" --lora_path outputs/gpt2_xl_distill/14290"
OPTS+=" --tokenizer openai-community/gpt2-xl"
# OPTS+=" --tokenizer openai-community/gpt2:source=GPT2"


# ==== Gọi Python ====
python run_eval.py ${OPTS} 
