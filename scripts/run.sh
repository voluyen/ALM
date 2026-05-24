#!/bin/bash
# Run all cross-tokenizer distillation pairs sequentially.
# Always invoke from repo root: `bash scripts/run.sh`.
# Each pair trains independently; failure of one does not stop the rest.
# Output is streamed to terminal AND mirrored to outputs/run_logs/<name>.log.

set -u
set -o pipefail
export PYTHONUNBUFFERED=1   # avoid Python stdout/tqdm buffering when piped through tee

PAIRS=(
    "scripts/distill/gpt2_120M_distill.sh"
    "scripts/distill/qwen1.5-1.8b_to_gpt2-medium_distill.sh"
    "scripts/distill/qwen2.5-7b_to_gpt2-xl_distill.sh"
    "scripts/distill/qwen2.5-7b_to_opt-2.7b_distill.sh"
    "scripts/distill/mistral-7b_to_tinyllama_distill.sh"
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
