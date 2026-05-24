#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="outputs/gpt2_120M_distill_v2/7140"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}"


mkdir -p ${OUTPUT_DIR}

OPTS=""

# training
OPTS+=" --val_batch_size 64"

# devices
OPTS+=" --student_device cuda:1"

# models
OPTS+=" --output_dir ${OUTPUT_DIR}"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
# OPTS+=" --lora_path "
OPTS+=" --tokenizer openai-community/gpt2"
# OPTS+=" --tokenizer openai-community/gpt2:source=GPT2"


# ==== Gọi Python ====
python run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log 2>&1
