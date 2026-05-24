#!/bin/bash
# Bootstrap a new machine for the PyTorch ALM distillation path.
# Run from repo root: `bash scripts/install.sh`
#
# Optional env vars:
#   CUDA_VERSION   torch CUDA wheel suffix (default: cu121). Use cu118 / cu124 / cpu.
#   PYTHON         python interpreter (default: python3). Must be <= 3.10.
#   SKIP_TORCH=1   skip torch install (already installed).

set -e

PYTHON="${PYTHON:-python3}"
CUDA_VERSION="${CUDA_VERSION:-cu121}"

echo "[1/5] Python version check (PyTorch path supports 3.10-3.12)"
$PYTHON --version
$PYTHON -c "import sys; assert (3, 10) <= sys.version_info < (3, 13), 'Python 3.10-3.12 required, got %s' % sys.version"

echo "[2/5] Upgrade pip + base tools"
$PYTHON -m pip install --upgrade pip setuptools wheel

if [ "${SKIP_TORCH:-0}" = "0" ]; then
    echo "[3/5] Install PyTorch (CUDA=$CUDA_VERSION)"
    if [ "$CUDA_VERSION" = "cpu" ]; then
        $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    else
        $PYTHON -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"
    fi
else
    echo "[3/5] Skipping torch install (SKIP_TORCH=1)"
fi

echo "[4/5] Install PyTorch path deps"
$PYTHON -m pip install \
    "transformers==4.46.0" \
    "tokenizers==0.20.3" \
    "datasets==3.2.0" \
    "peft>=0.11.0" \
    "scipy==1.14.1" \
    "numpy==1.26.4" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "pyyaml" \
    "tqdm" \
    "spacy>=3.7,<4"

$PYTHON -m spacy download en_core_web_sm

echo "[5/5] Install tokenkit (editable, no JAX extras)"
$PYTHON -m pip install -e tokenkit-main/ --no-deps

echo ""
echo "=========================================================="
echo "Install done."
echo ""
echo "Verify with:"
echo "  $PYTHON -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'"
echo "  $PYTHON -c 'from tokenkit.byteify import load_byteify_tokenizer; print(\"tokenkit OK\")'"
echo ""
echo "Next:"
echo "  1. python3 precompute_spans.py --input data/dolly_train.jsonl --output data/dolly_train_with_spans.jsonl"
echo "  2. bash scripts/run.sh    # run all distillation pairs"
echo "=========================================================="
