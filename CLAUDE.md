# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for **Cross-Tokenizer Distillation** using **Approximate Likelihood Matching (ALM)**. The goal is to transfer knowledge from a teacher LLM (with one tokenizer) to a student LLM (with a different tokenizer), enabling models to be retokenized without full retraining.

The codebase contains two parallel implementations:
- **JAX/Flax** (`jax_cross_tokenizer_distill.py`) — primary production implementation, uses the `tokenkit-main/` library
- **PyTorch** (`pytorch_cross_tokenizer_distill.py`) — experimental port for GPU environments

## Repository Layout

```
alm/
├── jax_cross_tokenizer_distill.py      # JAX training entrypoint (production)
├── pytorch_cross_tokenizer_distill.py  # PyTorch training entrypoint (experimental)
├── pytorch_tokenizer_aligner.py        # TokenizerAlignerCollator for PyTorch dataloader
├── pytorch_span_utils.py               # Span tensor helpers for MTA loss
├── evaluator.py                        # ROUGE/generation evaluation utilities
├── run_eval.py                         # Evaluation entrypoint (argparse-based)
├── precompute_spans.py                 # spaCy-based span extractor (run once before MTA training)
├── configs/                            # YAML training configs (Hydra/OmegaConf)
│   └── *.yaml
├── scripts/                            # Shell launchers (always run from repo root)
│   ├── distill/*.sh                    # Training launchers per pair
│   ├── eval/*.sh                       # Eval launchers
│   └── run.sh                          # End-to-end pipeline
├── docs/                               # Project documentation (pytorch-distill-guide.md etc.)
├── plans/                              # Brainstorm reports & implementation plans
├── data/                               # Training data (dolly_train.jsonl, valid.jsonl, *_with_spans.jsonl)
├── vocab_alignment/                    # ⚠ LEGACY/UNUSED — code does not load these mappings.
│   └── {pair}/                         # Real tokenizer-pair data lives in artifacts/tokenizer_data/ (bias matrices)
└── tokenkit-main/                      # The tokenkit library (installed via pip -e .)
    ├── tokenkit/
    │   ├── align.py                    # Core token alignment logic
    │   ├── byteify.py                  # ByteifyTokenizer wrapper
    │   ├── parse_args.py               # Hydra/dataclass arg parsing
    │   ├── utils.py                    # Space masks, tqdm, misc helpers
    │   ├── training/
    │   │   ├── losses.py               # ALM + SFT + baseline losses (JAX)
    │   │   ├── collators/              # Data collators for JAX training
    │   │   ├── checkpoint.py
    │   │   └── opt.py                  # Optimizer construction
    │   └── models/
    │       ├── lora.py
    │       └── sharding.py             # JAX device sharding
    ├── scripts/
    │   ├── cross_tokenizer_distill.py  # tokenkit CLI entrypoint
    │   └── eval.py / eval_lockstep.py  # LM harness eval
    └── examples/                       # Reference shell scripts
```

## Environment Setup

Requires Python ≤ 3.10. Two separate dependency trees:

**JAX path (primary):**
```bash
pip install torch jax[tpu]==0.5.0 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# or for GPU:
pip install torch jax[cuda12]==0.5.0
pip install git+https://github.com/bminixhofer/lm-evaluation-harness
pip install -r tokenkit-main/requirements.txt
pip install -e tokenkit-main/
```

**PyTorch path (experimental, GPU only):**
```bash
pip install torch transformers datasets peft scipy
pip install -e tokenkit-main/  # still needs tokenkit for align/byteify
```

## Key Commands

### Training

**PyTorch distillation** (single GPU — always run from repo root):
```bash
# Legacy GPT2 pairs
bash scripts/distill/qwen1.5-1.8b_to_gpt2_distill.sh
bash scripts/distill/gpt2_1.5B_distill.sh

# New cross-tokenizer pairs (see docs/pytorch-distill-guide.md)
bash scripts/distill/qwen1.5-1.8b_to_gpt2-medium_distill.sh
bash scripts/distill/qwen2.5-7b_to_gpt2-xl_distill.sh
bash scripts/distill/qwen2.5-7b_to_opt-2.7b_distill.sh
bash scripts/distill/mistral-7b_to_tinyllama_distill.sh

# Direct invocation with config overrides:
python3 pytorch_cross_tokenizer_distill.py \
    --config=configs/qwen1.5-1.8b_to_gpt2_distill.yaml \
    --overrides \
    losses=[sft,alm_unconstrained] \
    steps=7200 \
    name=my_experiment
```

**JAX distillation** (TPU/multi-GPU via tokenkit CLI):
```bash
python3 jax_cross_tokenizer_distill.py --config=<config.yaml> --overrides key=value
# or via tokenkit scripts directly:
python3 tokenkit-main/scripts/cross_tokenizer_distill.py ...
```

### Evaluation

```bash
# Evaluate a checkpoint with ROUGE on Dolly
bash scripts/eval/eval_gpt2_0.1B.sh

# Direct invocation:
python run_eval.py \
    --model_path outputs/gpt2_120M_distill_v2/7140 \
    --tokenizer openai-community/gpt2 \
    --student_device cuda:1 \
    --val_batch_size 64 \
    --output_dir ./eval_outputs/

# LM harness eval (JAX, via tokenkit):
python3 tokenkit-main/scripts/eval.py \
    model.pretrained_model_name_or_path='<path>' \
    model.tokenizer_name='<byteify-spec>' \
    eval.tasks=[mmlu]
```

## Architecture: How ALM Works

ALM aligns teacher and student token sequences through a **chunk alignment matrix**. The core insight is that different tokenizers split the same text differently, so we compute alignment matrices that map teacher token positions to student token positions.

**Data flow:**
1. `tokenkit.byteify.load_byteify_tokenizer(spec)` — loads a tokenizer with byteification (normalized token representation)
2. `tokenkit.align.get_alignment_indices(tokens_teacher, tokens_student, ...)` — computes sparse alignment matrices
3. `TokenizerAlignerCollator` (PyTorch) or `tokenkit.training.collators` (JAX) — builds batches with `alignment_matrix_a` (student→chunks) and `alignment_matrix_b` (teacher→chunks)
4. `compute_alm_loss(chunk_kind, args, loss_args)` — computes the ALM loss using aligned log-probabilities

**Loss modes** (set via `losses:` in config):
- `sft` — standard cross-entropy on student tokens
- `alm_unconstrained` — ALM loss without debiasing constraints
- `alm_latents` — hidden state alignment loss

**Key ALM hyperparameters:**
- `binarization_temp` (default 100.0) — sharpens binary cross-entropy
- `alm_mode` — `merge_by_space_prob+append_space` (debiasing, recommended)
- `tokenizer_pair_bias_threshold` (default 0.1) — threshold for merging chunks

## Config System

Configs are YAML files parsed via Hydra (`hydra-core`). Override fields at the command line with `--overrides key=value` or `key=[val1,val2]`. The `CrossTokenizerDistillArgs` dataclass (in both `pytorch_cross_tokenizer_distill.py` and `jax_cross_tokenizer_distill.py`) is the authoritative schema — all YAML keys must match its fields.

**Tokenizer name format** (`byteify spec`):
```
<hf-model-id>:source=<ModelKind>[:conversion=<type>]
# e.g. openai-community/gpt2:source=GPT2
#      google/gemma-2-2b-it:source=Gemma2:conversion=byte
```

Supported `source=` values: `GPT2`, `Qwen2`, `Llama`, `Gemma2`, `Gemma3`, etc. (see `tokenkit/model_kinds.py`).

## GPU Assignment

In the PyTorch implementation, teacher and student models are hardcoded to separate GPUs:
- Teacher → `cuda:1`
- Student → `cuda:0`

Control which GPUs are visible via `CUDA_VISIBLE_DEVICES` (set in `*.sh` scripts via `GPUS=(...)` array).

## Data Format

Training data is JSONL with fields `prompt` and `output` (Dolly format). The `loss_mask_mode` config controls which part of the sequence the loss is computed on:
- `dolly` — loss only after `### Response:\n`
- `openmath2` — loss only after `<|start_header_id|>assistant<|end_header_id|>\n\n`
- `null` — loss over all tokens
