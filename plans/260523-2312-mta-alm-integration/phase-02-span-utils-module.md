---
phase: 2
title: Span Utils Module
status: completed
priority: P1
effort: 3h
dependencies:
  - 1
---

# Phase 2: Span Utils Module

## Overview
Tạo `pytorch_span_utils.py` chứa toàn bộ logic MTA loss (token weighting, span aggregation, relational cosine-sim MSE). Port từ `mta-loss-integration.md` Bước 1, **fix bugs** đã identify trong brainstorm.

## Requirements
- **Functional:** Expose 4 hàm: `compute_token_weights`, `aggregate_spans_for_model`, `compute_hidden_span_loss`, `compute_overall_span_loss`.
- **Non-functional:** Pure PyTorch (không spaCy import — chỉ inference path). Type hints. Handle edge cases (empty spans, all-pad batch).

## Architecture

```
hidden_states [B, L, D]  +  layer_weights [B, L]  +  attention_mask [B, L]
              +  offset_mapping [B, L, 2]  +  spans_char_offsets list[list[(s,e)]]
              ↓
         aggregate_spans_for_model
              ↓
         span_repr [B, MaxSpans, D]  +  weight_sum [B, MaxSpans]  +  valid_mask [B, MaxSpans]
              ↓ (×2: student + teacher)
         projector(s_repr) → [B, MaxSpans, D_teacher]
              ↓
         relational cosine-sim MSE  →  scalar loss
```

## Related Code Files

- **Create:** `pytorch_span_utils.py` (~180 LOC, project root)
- **Reference:** `mta-loss-integration.md` lines 59-241 (port + fix)

## Implementation Steps

1. Copy 4 hàm chính từ doc lines 66-241:
   - `compute_token_weights(hidden_state, attention_mask)` → giữ nguyên
   - `aggregate_spans_for_model(hidden_states, layer_weights, attention_mask, offsets_mapping, spans_offsets, entropy_weights=None)` → giữ nguyên logic, đổi `spans_offsets` arg accept `list[list[list[int]]]` (jsonl format) hoặc `list[list[tuple]]`
   - `compute_hidden_span_loss(projector, s_span_repr, t_span_repr, valid_span_mask, w_sum, use_span_weight=True)` → giữ nguyên
   - `get_span_loss(...)` → giữ nguyên (helper)
   - `compute_overall_span_loss(projectors, s_att_mask, t_att_mask, s_logits, t_logits, s_hidden_states, t_hidden_states, s_offsets_mapping, t_offsets_mapping, spans_offsets, words_offsets, args)` → giữ nguyên
2. **Fix bug**: arg signature của `compute_overall_span_loss` cần `args.entropy_weight`, `args.wo_span_weight`, `args.teacher_layer_mapping`, `args.student_layer_mapping`, `args.split_layer_mapping`. Document dependency này ở docstring.
3. **Handle edge cases:**
   - `spans_offsets` empty cho 1 sample → `aggregate_spans_for_model` return `None`
   - `valid_s.size(0) == 0` → return `torch.tensor(0.0)`
   - `weight_sum` near-0 → clamp `min=1e-5`
4. Bỏ `filter_overlapping_spans` và `get_spans_offsets` (chuyển sang `precompute_spans.py` Phase 1 — đã làm offline)
5. Compile check: `python -c "import pytorch_span_utils"`

## Success Criteria

- [ ] `pytorch_span_utils.py` import được, không syntax error
- [ ] Smoke unit test: dummy hidden `[2, 10, 768]`, dummy offsets, 3 spans/sample → output shape khớp, không NaN
- [ ] Empty spans case → return None gracefully, không crash
- [ ] Function signatures match docstrings

## Risk Assessment

| Risk | Mitigation |
|---|---|
| `torch.bmm` không hỗ trợ bf16 trên một số GPU cũ | Cast hidden về fp32 trước bmm, cast back sau (đã có trong doc gốc qua `to(hidden_states.dtype)`) |
| Memory spike với batch lớn × max_spans lớn | Track `[B × MaxSpans × D]` tensor sizes, log nếu MaxSpans > 50 |
| `pad_sequence` fail nếu spans_offsets có 0 phần tử ở 1 sample | Code đã handle: tạo empty tensor `(0, 2)` shape |
