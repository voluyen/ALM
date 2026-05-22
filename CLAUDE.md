# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tokenkit` is a JAX/Flax-based toolkit for transferring LLM models and knowledge across tokenizers. Core methods:
- **ALM** (Approximate Likelihood Matching) — cross-tokenizer distillation
- **ZeTT/FVT** (Fast Vocabulary Transfer) — zero-shot tokenizer transfer
- **Byteification** — converting subword tokenizers to byte-level tokenizers
- **Token-level ensembling** — combining models with different tokenizers

**Requires Python ≤ 3.10.** JAX is the compute backend; PyTorch is used only for data loading (no accelerator support needed for torch).

## Installation

```bash
# TPU
pip install torch jax[tpu]==0.5.0 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# GPU (substitute cuda12 for your CUDA version)
pip install torch jax[cuda12]==0.5.0

pip install git+https://github.com/bminixhofer/lm-evaluation-harness
pip install -r requirements.txt
pip install -e .
```

## Common Commands

```bash
# Cross-tokenizer distillation (main training script)
python3 scripts/cross_tokenizer_distill.py \
    --config=configs/cross_tokenizer_distill.yaml \
    --overrides losses=[sft,alm_unconstrained] name=my_run

# Zero-shot tokenizer transfer
python3 scripts/zett.py --config=configs/zett.yaml --overrides ...

# Evaluation (single model)
python3 scripts/eval.py model.pretrained_model_name_or_path='...' eval.tasks=[mmlu]

# Evaluation (token-level ensemble, uses configs/models/*.yaml)
python3 scripts/eval_lockstep.py models=llama_qwen eval.tasks=[mmlu]

# Run the quickstart example
bash examples/llama3_to_byte_tokenizer_gpu.sh

# Linting
black tokenkit/ scripts/
isort tokenkit/ scripts/

# Tests
pytest
```

## Config System

Scripts use a custom Hydra-like config loader (`tokenkit/parse_args.py`). Pass a base YAML with `--config=configs/....yaml` and override individual keys with `--overrides key=value`. Nested keys use dot notation (`optimizer.learning_rate=1e-4`). Lists use bracket notation (`losses=[sft,alm_unconstrained]`).

## Tokenizer Specification Format

Tokenizers are specified as colon-separated specs:
```
model/name:source=ModelKind[:conversion=method][:target=OtherModelKind]
```
- `source=` — model family for byte mapping: `Llama3`, `Gemma2`, `Qwen2`
- `conversion=byte` — convert to byte-level tokenizer
- `conversion=prebyteified` — model already trained on byte tokenizer
- `target=` — for zero-shot transfer, specify target tokenizer's model kind

Example: `google/gemma-2-2b-it:source=Gemma2:conversion=byte`

## Architecture

```
tokenkit/
├── byteify.py              # Tokenizer byteification logic
├── align.py                # Token alignment between two tokenizers (alignment matrices)
├── model_kinds.py          # Per-model-family tokenizer handling (Llama3, Gemma2, Qwen2, etc.)
├── constants.py            # CHARS_TO_BYTES / BYTES_TO_CHARS mappings
├── parse_args.py           # Dataclass-based argument parsing with YAML support
├── utils.py                # General utilities (tqdm, large negative numbers, etc.)
├── baseline_utils.py       # KL divergence implementations for baseline methods
├── hf/                     # Custom Flax model implementations for TPU
│   ├── modelling_flax_tpu_*.py   # Flax forward passes (Gemma2, Gemma3, Llama)
│   └── modelling_tpu_*.py        # PyTorch versions for compatibility
├── models/
│   ├── hypernet/           # Hypernetwork for embedding prediction (transformer/linear/identity)
│   ├── lora.py             # LoRA adapter implementation
│   ├── param.py            # Parameter manipulation (get_num_layers, etc.)
│   └── sharding.py         # JAX device sharding utilities
├── training/
│   ├── losses.py           # ALM loss, DSKD, ULD, MinED, SFT losses
│   ├── collators/          # Data collators (TokenizerAlignerCollator, plain, sampler)
│   ├── checkpoint.py       # Checkpoint save/restore (local + GCS)
│   ├── lr.py               # Learning rate schedules
│   ├── opt.py              # Optimizer construction
│   └── multitask.py        # Multi-loss aggregation (approx_gradmag_preserve_mag, etc.)
└── eval/
    └── generate.py         # JAX-based autoregressive generation for lm-eval harness
```

`scripts/` contains the main entry points; `configs/` holds default YAML configs; `examples/` has runnable shell scripts.

## Key Design Patterns

**Alignment matrices**: The core of ALM is alignment matrices (`alignment_matrix_a`, `alignment_matrix_b`) that map token chunks between teacher and student tokenizers. These are boolean sparse matrices computed by `TokenizerAlignerCollator` and used in `losses.py`.

**Batch structure**: Batches contain parallel `_original` (teacher) and `_new` (student) tensors — e.g., `input_ids_original`, `input_ids_new`, `loss_mask_original`, `loss_mask_new`.

**JAX sharding**: Multi-device training uses `NamedSharding` with `P("data", None, "model")` partition specs. The mesh is attached to the model config object (`config.mesh`).

**Hypernet**: When doing subword-to-subword transfer, a small hypernetwork (`tokenkit/models/hypernet/`) predicts new embeddings from old ones. For byte tokenizers, `hypernet.architecture=identity` is used instead.

**`train_model_mode`**: Controls what gets trained — `"lora"` (LoRA adapters + embeddings) or `"full"` (all parameters).

## Linting Standards

- `black` with `line-length=88`
- `isort` with `profile=black`
