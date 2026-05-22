#!/usr/bin/env bash
set -e

ACCELERATOR=${1:-gpu}  # usage: bash install.sh [gpu|tpu]  (default: gpu)
CUDA_VERSION=${2:-cuda12}  # e.g. cuda11, cuda12 (GPU only)

echo "==> Installing PyTorch + JAX for $ACCELERATOR..."
if [ "$ACCELERATOR" = "tpu" ]; then
    pip install torch "jax[tpu]==0.5.0" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
    pip install torch "jax[$CUDA_VERSION]==0.5.0"
fi

echo "==> Installing lm-evaluation-harness..."
pip install git+https://github.com/bminixhofer/lm-evaluation-harness

echo "==> Installing requirements..."
pip install -r requirements.txt

echo "==> Installing tokenkit (editable)..."
pip install -e .

echo "==> Downloading spaCy model en_core_web_sm..."
python -m spacy download en_core_web_sm

echo "==> Done."
