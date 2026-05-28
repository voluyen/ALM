#!/bin/bash
# One-stop server entrypoint: setup env (if conda) + install deps + run all distill pairs.
# Always invoke from repo root: `bash scripts/run.sh`.
# Output is streamed to terminal AND mirrored to outputs/run_logs/<name>.log.
#
# Optional env vars:
#   ENV_NAME        conda env name (default: alm)
#   PYTHON_VERSION  python version for new conda env (default: 3.12)
#   SKIP_SETUP=1    skip env creation and install step (env already ready)

set -u
set -o pipefail
export PYTHONUNBUFFERED=1   # avoid Python stdout/tqdm buffering when piped through tee

ENV_NAME="${ENV_NAME:-alm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# ── 1. Environment setup ────────────────────────────────────────────────────
if [ "${SKIP_SETUP:-0}" = "1" ]; then
    echo "[setup] SKIP_SETUP=1, using current Python env"
elif command -v conda &>/dev/null; then
    echo "[setup] conda detected: $(conda --version)"
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "[setup] reusing existing conda env '$ENV_NAME'"
    else
        echo "[setup] creating conda env '$ENV_NAME' with python=$PYTHON_VERSION"
        conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    fi

    conda activate "$ENV_NAME"
    echo "[setup] active python: $(which python)"

    echo "[setup] installing deps via scripts/install.sh"
    bash scripts/install.sh
else
    echo "[setup] WARNING: conda not found — using current Python env without install."
    echo "[setup]          If deps are missing, run 'bash scripts/install.sh' manually first."
fi

# ── 2. Run all distillation pairs ───────────────────────────────────────────
PAIRS=(
    "scripts/distill/qwen1.5-1.8b_to_gpt2_distill.sh"
    "scripts/distill/qwen1.5-1.8b_to_gpt2_alm_distill.sh"
    "scripts/distill/qwen1.5-1.8b_to_gpt2_mta_all_word_distill.sh"
    "scripts/distill/qwen1.5-1.8b_to_gpt2_mta_all_phrase_distill.sh"
    "scripts/distill/qwen1.5-1.8b_to_gpt2_mta_no_weight_distill.sh"
    # "scripts/distill/qwen1.5-1.8b_to_gpt2-medium_distill.sh"
    "scripts/distill/qwen1.5-1.8b_to_gpt2-medium_alm_distill.sh"
    # "scripts/distill/qwen2.5-7b_to_gpt2-xl_distill.sh"
    "scripts/distill/qwen2.5-7b_to_gpt2-xl_alm_distill.sh"
    # "scripts/distill/qwen2.5-7b_to_opt-2.7b_distill.sh"
    "scripts/distill/qwen2.5-7b_to_opt-2.7b_alm_distill.sh"
    # "scripts/distill/mistral-7b_to_tinyllama_distill.sh"
    "scripts/distill/mistral-7b_to_tinyllama_alm_distill.sh"
)

mkdir -p outputs/run_logs

for sh in "${PAIRS[@]}"; do
    name=$(basename "$sh" .sh)
    log="outputs/run_logs/${name}.log"
    echo "=========================================================="
    echo "[$(date '+%F %T')] START $name  (log: $log)"
    echo "=========================================================="

    # tee writes to terminal AND file. PIPESTATUS[0] captures bash's exit code,
    # not tee's, so we know whether the training actually succeeded.
    bash "$sh" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%F %T')] DONE  $name"
    else
        echo "[$(date '+%F %T')] FAIL  $name (exit=$rc, see $log)"
    fi
done

echo "All pairs finished."
