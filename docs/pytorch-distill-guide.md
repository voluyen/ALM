# PyTorch ALM Distillation Guide

Entry point: [`pytorch_cross_tokenizer_distill.py`](../pytorch_cross_tokenizer_distill.py).

Configs parsed by Hydra/OmegaConf. CLI override syntax:
`python pytorch_cross_tokenizer_distill.py --config=<file>.yaml --overrides key=value key=[v1,v2]`.

---

## Available pair scripts

| # | Pair | YAML | Launcher | Mode | Epochs | LR |
|---|---|---|---|---|---|---|
| 1 | Qwen1.5-1.8B → GPT2-medium | [`configs/qwen1.5-1.8b_to_gpt2-medium_distill.yaml`](../configs/qwen1.5-1.8b_to_gpt2-medium_distill.yaml) | [`.sh`](../scripts/distill/qwen1.5-1.8b_to_gpt2-medium_distill.sh) | full FT | 20 | 5e-4 |
| 2 | Qwen2.5-7B → GPT2-XL | [`configs/qwen2.5-7b_to_gpt2-xl_distill.yaml`](../configs/qwen2.5-7b_to_gpt2-xl_distill.yaml) | [`.sh`](../scripts/distill/qwen2.5-7b_to_gpt2-xl_distill.sh) | LoRA | 15 | 1e-3 |
| 3 | Qwen2.5-7B → OPT-2.7B | [`configs/qwen2.5-7b_to_opt-2.7b_distill.yaml`](../configs/qwen2.5-7b_to_opt-2.7b_distill.yaml) | [`.sh`](../scripts/distill/qwen2.5-7b_to_opt-2.7b_distill.sh) | LoRA | 15 | 1e-3 |
| 4 | Mistral-7B → TinyLlama-1.1B | [`configs/mistral-7b_to_tinyllama_distill.yaml`](../configs/mistral-7b_to_tinyllama_distill.yaml) | [`.sh`](../scripts/distill/mistral-7b_to_tinyllama_distill.sh) | LoRA | 15 | 1e-3 |

Legacy pairs: [`scripts/distill/gpt2_1.5B_distill.sh`](../scripts/distill/gpt2_1.5B_distill.sh), [`gpt2_120M_alm_mta_distill.sh`](../scripts/distill/gpt2_120M_alm_mta_distill.sh).
Renamed: `gpt2_120M_distill.sh` → [`qwen1.5-1.8b_to_gpt2_distill.sh`](../scripts/distill/qwen1.5-1.8b_to_gpt2_distill.sh) (pair 0).

**Always run scripts from the repo root** (`d:/MLResearch/alm/`) so relative paths (`configs/`, `data/`, `outputs/`) resolve.

All 4 new pairs share:
- `batch_size=8`, `max_length=256` (teacher/student)
- Losses `[sft, alm_unconstrained, mta]` weights `[1.0, 1.0, 2.0]`
- MTA enabled; hypernet `identity`
- Single GPU (`cuda:0`) — teacher + student colocated
- LoRA hardcoded `r=256, α=8, dropout=0.1` ([pytorch_cross_tokenizer_distill.py:647-649](../pytorch_cross_tokenizer_distill.py#L647-L649))

---

## New: epoch-based training

PyTorch loop is step-based. To train for an exact number of dataset passes, set:

```yaml
epochs: 15
steps: 0          # overridden at runtime
warmup_steps: 0   # auto = 10% of total if left at 0
```

Logic ([pytorch_cross_tokenizer_distill.py](../pytorch_cross_tokenizer_distill.py) after dataloader is built):

```python
if args.epochs and args.epochs > 0:
    steps_per_epoch = len(train_dataloader)
    args.steps = steps_per_epoch * args.epochs
    if not args.warmup_steps:
        args.warmup_steps = max(1, int(0.1 * args.steps))
```

Falls back to `steps`/`warmup_steps` from config if `epochs=0` (default).

---

## OPTModelKind

Added to [`tokenkit/model_kinds.py`](../tokenkit-main/tokenkit/model_kinds.py) for facebook/opt-* family.
HF specials: `bos_token='</s>'` (yes, OPT uses `</s>` for BOS), `eos_token='</s>'`, `pad_token='<pad>'`, `unk_token='<unk>'`.
Byteify spec: `facebook/opt-XXX:source=OPT`.

OPT has no chat template — keep `use_chat_template=false`.

---

## Loss weight knobs

| Knob | Where | Maps to |
|---|---|---|
| `--w-span-loss <X>` | Original spec | `loss_weights: [..., X]` (last entry = MTA) |
| `PROJECTOR_LR/LR` | Original spec | `optimizer.param_groups[0].lr_scale` |
| `LORA_R/ALPHA/DROPOUT` | Original spec | Hardcoded in code (256/8/0.1) |

If you need to change LoRA hyperparams, edit [`pytorch_cross_tokenizer_distill.py:647-649`](../pytorch_cross_tokenizer_distill.py#L647-L649) directly — they are not currently exposed via config.

---

## Memory notes (single GPU)

Teacher + student colocated on `cuda:0` (bf16):

| Pair | Teacher | Student | Approx VRAM |
|---|---|---|---|
| 1 | 1.8B (~3.6GB) | 355M full FT (~1.4GB fp32 + Adam ~5GB) | ~12GB |
| 2 | 7B (~14GB) | 1.5B LoRA (~3GB bf16 + tiny adapter) | ~22GB |
| 3 | 7B (~14GB) | 2.7B LoRA (~5.4GB bf16 + tiny adapter) | ~24–26GB ⚠ tight |
| 4 | 7B (~14GB) | 1.1B LoRA (~2.2GB bf16) | ~19GB |

If OOM on pair 3: lower `max_teacher_length`/`max_student_length` (e.g. 128) or enable `gradient_checkpointing=true`.

Pair 1 (`train_model_mode=full`) loads student in fp32 — see [pytorch_cross_tokenizer_distill.py:654-658](../pytorch_cross_tokenizer_distill.py#L654-L658). Adam states inflate memory ~5×.

---

## Checkpoint disk usage

Training auto-saves at every epoch boundary ([pytorch_cross_tokenizer_distill.py:926](../pytorch_cross_tokenizer_distill.py#L926)). Pair 1 full-FT × 20 epochs ≈ **~14GB on disk**. LoRA pairs save only adapter + embeddings (~50MB × 15 = ~750MB).

---

## Spans precomputation

`losses=[..., mta]` requires `data/dolly_train_with_spans.jsonl`. Generate once via:

```bash
python precompute_spans.py --input data/dolly_train.jsonl --output data/dolly_train_with_spans.jsonl
```

Spans are extracted via spaCy noun-chunks + verb-phrases on raw text — **tokenizer-independent**, so one file works for all teacher/student pairs.

---

## Clarification: `vocab_alignment/` folders

The `vocab_alignment/{pair}/` folders in this repo are **legacy/unused**. No code path loads them.

The real tokenizer-pair data path used by the training collator is `tokenizer_pair_data_path` ([config field](../pytorch_cross_tokenizer_distill.py)) → expects:
- `bias1_matrix.npz`, `bias2_matrix.npz`
- `teacher_counts.json`, `student_counts.json`

Generated by [`tokenkit-main/scripts/compute_tokenizer_info.py`](../tokenkit-main/scripts/compute_tokenizer_info.py).

**Bias matrices are only required when loss name contains `"unbiased"`** ([pytorch_cross_tokenizer_distill.py:686](../pytorch_cross_tokenizer_distill.py#L686)):
```python
require_bias_matrices=any("unbiased" in x for x in args.losses)
```

For `alm_unconstrained` + `mta` + `sft`, the path can be missing/invalid — collator silently sets matrices to `None`.

---

## Running

From repo root:

```bash
bash scripts/distill/qwen1.5-1.8b_to_gpt2-medium_distill.sh   # pair 1
bash scripts/distill/qwen2.5-7b_to_gpt2-xl_distill.sh         # pair 2
bash scripts/distill/qwen2.5-7b_to_opt-2.7b_distill.sh        # pair 3
bash scripts/distill/mistral-7b_to_tinyllama_distill.sh       # pair 4
```

Outputs land in `outputs/<name>/<step>/`. Evaluate via [`run_eval.py`](../run_eval.py) or `scripts/eval/*.sh` scripts.

---

## Unresolved / TODO

1. **`OPTModelKind` byteify compat** — not yet runtime-tested. If pair 3 errors on tokenizer init, the `replacements`/`special_tokens` may need adjustment (e.g. OPT prepends `</s>` automatically, may conflict with byteify's BOS handling).
2. **`tokenizer_pair_data_path`** in new YAMLs points to non-existent paths (e.g. `artifacts/tokenizer_data/qwen2.5_to_gpt2-xl`). Code tolerates this but logs warnings. Set to `null` if cleaner output desired.
3. **CLAUDE.md** lists `vocab_alignment/` as part of repo layout — could be updated to mark as legacy.
4. **LoRA hyperparams hardcoded** — exposing `model_lora_dropout`, `model_lora_target_modules` via config would help future cross-arch experiments.
