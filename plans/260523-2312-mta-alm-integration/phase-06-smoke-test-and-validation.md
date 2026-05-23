---
phase: 6
title: "Smoke Test and Validation"
status: pending
priority: P1
effort: "3h"
dependencies: [5]
---

# Phase 6: Smoke Test and Validation

## Overview
Smoke test 50 steps → verify training stable, loss giảm, không NaN/OOM. Sau đó chạy full 5k steps + eval ROUGE trên Dolly valid để compare với baseline `alm_unbiased`-only.

## Requirements
- **Functional:** Training run đến hết steps, save checkpoint, eval pass.
- **Non-functional:** Wall-time/step không vượt baseline quá 30%; VRAM không OOM trên config 2-GPU hiện tại.

## Architecture

```
Smoke test (50 steps, batch=4, eval_interval=disabled):
  → verify no NaN, no OOM, both losses logged
  → wall-time benchmark vs baseline

Full run (5000 steps, batch=16):
  → train MTA + ALM
  → save checkpoint outputs/gpt2_120M_alm_mta_v1/5000

Eval:
  → run_eval.py với checkpoint mới + baseline checkpoint
  → compare ROUGE-L Dolly valid
  → (optional) MMLU/ARC-Easy via lm-eval-harness
```

## Related Code Files

- **Run:** `gpt2_120M_alm_mta_distill.sh`
- **Eval:** `run_eval.py` + `eval_gpt2_0.1B.sh` (adapt path)
- **Baseline checkpoint:** `outputs/gpt2_120M_distill_v2/<step>` (assume existing)

## Implementation Steps

### 6.1 Smoke test

1. Tạo `gpt2_120M_alm_mta_smoke.yaml`: copy config Phase 5, set `steps: 50`, `batch_size: 4`, `eval_interval: 99999`, `save_interval: 99999`.
2. Chạy:
   ```bash
   python3 pytorch_cross_tokenizer_distill.py \
       --config=gpt2_120M_alm_mta_smoke.yaml \
       --overrides name=smoke
   ```
3. Verify trong log:
   - [ ] Cả `loss/alm_unbiased` và `loss/mta` xuất hiện, finite
   - [ ] `loss/mta` không stuck ở 0 hoặc constant
   - [ ] Wall-time/step in ra; compare với baseline (~30% tăng max acceptable)
   - [ ] VRAM peak in ra (qua `torch.cuda.max_memory_allocated`)
4. Nếu fail: debug → fix → re-run.

### 6.2 Full training

1. Chạy `bash gpt2_120M_alm_mta_distill.sh`
2. Monitor wandb/tensorboard nếu có; check loss curves giảm monotonic.
3. Hoàn thành 5k steps → checkpoint tại `outputs/gpt2_120M_alm_mta_v1/5000`.

### 6.3 Evaluation

1. Adapt `eval_gpt2_0.1B.sh` cho checkpoint mới:
   ```bash
   python run_eval.py \
       --model_path outputs/gpt2_120M_alm_mta_v1/5000 \
       --tokenizer openai-community/gpt2 \
       --student_device cuda:1 \
       --val_batch_size 64 \
       --output_dir ./eval_outputs/gpt2_120M_alm_mta_v1/
   ```
2. Lặp với baseline checkpoint để so sánh.
3. Ghi kết quả vào `plans/reports/eval-260523-mta-alm-integration.md`:
   - ROUGE-L: baseline vs MTA
   - Optional: MMLU, ARC-Easy
   - Wall-time/step delta
   - VRAM delta

## Success Criteria

- [ ] Smoke test pass: 50 steps, no NaN/OOM, cả 2 loss đều log
- [ ] Wall-time/step tăng ≤ 30% vs baseline
- [ ] VRAM peak tăng ≤ 500MB vs baseline
- [ ] Full training 5k steps complete
- [ ] **ROUGE-L Dolly valid ≥ baseline + 0.5 điểm** (success criteria chính)
- [ ] MMLU không regression (±1%)
- [ ] Eval report viết xong tại `plans/reports/`

## Risk Assessment

| Risk | Mitigation |
|---|---|
| MTA loss explode → gradient NaN | Clip gradient (`max_grad_norm=1.0` đã có); log loss/grad per-100 steps; nếu NaN, giảm `w_span_loss` từ 2.0 → 1.0 |
| Loss/mta = 0 do spans empty toàn batch | Phase 1 verify đã có spans; nếu vẫn 0, check field naming consistency từ collator |
| OOM khi `output_hidden_states=True` | Giảm batch 16 → 8; hoặc bật `gradient_checkpointing_enable()` cho teacher |
| ROUGE không cải thiện hoặc tệ hơn | Ablation: thử `w_span_loss ∈ {0.5, 1.0}`; thử `entropy_weight=true`; thử thay `split_layer_mapping=[0, 2, 4, 4]` (2 word + 2 span) |
| Eval script không tương thích checkpoint mới (LoRA merged?) | Kiểm tra logic merge LoRA trong `run_eval.py`; nếu fail, save merged checkpoint sau training |

## Decision: Go/No-Go

Sau eval, quyết định:
- ROUGE +0.5+ → **Ship**: merge MTA vào main config, update docs/development-roadmap.md
- ROUGE 0 đến +0.5 → **Iterate**: ablation `w_span_loss`, `entropy_weight`, layer mapping
- ROUGE âm → **Investigate**: review span quality, projector training, layer mapping. Nếu sau 2 ablation runs không tốt hơn → drop MTA, document lessons learned trong `/ck:journal`.
