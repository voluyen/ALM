---
title: MTA Loss Integration into ALM (PyTorch)
description: ''
status: pending
priority: P2
branch: ''
tags: []
blockedBy: []
blocks: []
created: '2026-05-23T16:18:32.672Z'
createdBy: 'ck:plan'
source: skill
---

# MTA Loss Integration into ALM (PyTorch)

## Overview

Tích hợp MTA (Multi-Teacher Alignment) span-level relational similarity loss vào PyTorch cross-tokenizer distillation pipeline, dùng song song với `alm_unbiased`. Mục tiêu: cải thiện ROUGE-L Dolly +0.5 điểm vs baseline thuần ALM, không touch JAX path.

**Brainstorm:** [brainstorm-260523-2312-mta-alm-integration.md](../reports/brainstorm-260523-2312-mta-alm-integration.md)
**Source doc (có bugs cần fix):** [../../mta-loss-integration.md](../../mta-loss-integration.md)

## Key Decisions (locked)

| Item | Value |
|---|---|
| Code path | PyTorch only (`pytorch_cross_tokenizer_distill.py`) |
| MTA def | Multi-Teacher Alignment — span relational cosine-sim MSE |
| Layer mapping | student `[3,6,9,12]` ↔ teacher `[6,12,18,24]`, `split=[0,1,4,4]` |
| Loss combo | `1.0 * alm_unbiased + 2.0 * mta` |
| Span precompute | Offline script → `data/dolly_train_with_spans.jsonl` |
| Projectors | `Linear(768→2048) × 4`, trained jointly (param_group lr_scale=2) |
| Mask for aggregation | `attention_mask` (not loss_mask) |
| `entropy_weight` default | `false` |
| Offset mapping source | `batch['offset_mapping_new/original']` precomputed in collator |

## Bug fixes vs original doc

1. `loss_args.distiller` không tồn tại → dùng `loss_args.tokenizer_new`
2. PyTorch projectors trong JAX loop → no gradient → chỉ PyTorch path
3. spaCy online → precompute offline
4. Missing args: thêm `mta_mode`, `teacher_layer_mapping`, `student_layer_mapping`, `split_layer_mapping`, `w_span_loss`, `entropy_weight`, `wo_span_weight` vào `CrossTokenizerDistillArgs`

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Precompute Spans Script](./phase-01-precompute-spans-script.md) | Completed |
| 2 | [Span Utils Module](./phase-02-span-utils-module.md) | Completed |
| 3 | [Update Collator](./phase-03-update-collator.md) | Completed |
| 4 | [Update Train Script](./phase-04-update-train-script.md) | Completed |
| 5 | [Config and Launch](./phase-05-config-and-launch.md) | Completed |
| 6 | [Smoke Test and Validation](./phase-06-smoke-test-and-validation.md) | Pending |

## Dependencies

<!-- Cross-plan dependencies -->
