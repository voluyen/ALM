---
phase: 4
title: Update Train Script
status: completed
priority: P1
effort: 4h
dependencies:
  - 2
  - 3
---

# Phase 4: Update Train Script

## Overview
Tích hợp MTA vào `pytorch_cross_tokenizer_distill.py`: thêm args, init projectors, dispatch loss, bật `output_hidden_states`, đưa projectors vào optimizer.

## Requirements
- **Functional:**
  - `losses=[alm_unbiased, mta]` → cả 2 loss tính, sum với weights config
  - `output_hidden_states=True` cho cả student và teacher khi MTA active
  - `mta_projector_list` (4 Linear) được train (param group lr_scale=2)
- **Non-functional:** Backward-compat — khi `mta_mode=false` hoặc `mta` không trong losses list, không có overhead.

## Architecture

Modifications:

```
CrossTokenizerDistillArgs:
  + mta_mode: bool = False
  + teacher_layer_mapping: list[int] = field(default_factory=lambda: [6,12,18,24])
  + student_layer_mapping: list[int] = field(default_factory=lambda: [3,6,9,12])
  + split_layer_mapping: list[int] = field(default_factory=lambda: [0,1,4,4])
  + w_span_loss: float = 2.0
  + entropy_weight: bool = False
  + wo_span_weight: bool = False

LossArgs:
  + mta_projector_list: Any = None

main():
  ...
  + mta_projector_list = None
  + if "mta" in args.losses:
  +   assert args.mta_mode, "must set mta_mode=true"
  +   student_hidden = new_model.config.hidden_size
  +   teacher_hidden = teacher_model.config.hidden_size
  +   n_pairs = len(args.teacher_layer_mapping)
  +   mta_projector_list = nn.ModuleList([
  +     nn.Linear(student_hidden, teacher_hidden) for _ in range(n_pairs)
  +   ]).to(student_device)
  ...
  + need_hidden_states = any(loss in {"alm_latents", "mta"} for loss in args.losses)
  ...
  - student_out = new_model(..., output_hidden_states=False)
  + student_out = new_model(..., output_hidden_states=need_hidden_states)
  - teacher_out = teacher_model(..., output_hidden_states=False)
  + teacher_out = teacher_model(..., output_hidden_states=need_hidden_states)
  ...
  + optimizer = torch.optim.AdamW([
  +   {"params": [p for n, p in new_model.named_parameters() if "projector" not in n], "lr": base_lr},
  +   {"params": mta_projector_list.parameters() if mta_projector_list else [], "lr": base_lr * 2},
  + ], lr=base_lr)

compute_mta_loss(args, loss_args):
  from pytorch_span_utils import compute_overall_span_loss
  return compute_overall_span_loss(
    projectors=loss_args.mta_projector_list,
    s_att_mask=loss_args.batch['attention_mask_new'],
    t_att_mask=loss_args.batch['attention_mask_original'],
    s_logits=loss_args.student_logits,
    t_logits=loss_args.teacher_logits,
    s_hidden_states=loss_args.student_out.hidden_states,
    t_hidden_states=loss_args.teacher_out.hidden_states,
    s_offsets_mapping=loss_args.batch['offset_mapping_new'],
    t_offsets_mapping=loss_args.batch['offset_mapping_original'],
    spans_offsets=loss_args.batch['spans_char_offsets'],
    words_offsets=loss_args.batch['words_char_offsets'],
    args=args,
  )

dispatch loop:
  ...
  elif loss == "mta":
    current_loss = compute_mta_loss(args, loss_args)
  ...
```

## Related Code Files

- **Modify:** `pytorch_cross_tokenizer_distill.py` (~50 LOC added)
- **Read:** `pytorch_span_utils.py` (created in Phase 2)

## Implementation Steps

1. Add 7 fields vào `CrossTokenizerDistillArgs` dataclass (xem Architecture).
2. Add `mta_projector_list: Any = None` vào `LossArgs` dataclass.
3. Trong `main()` sau khi load `new_model` + `teacher_model`:
   - Init `mta_projector_list` nếu `"mta" in args.losses` (xem code block trên)
   - Move to `student_device` (consistent với student)
4. Update `need_hidden_states` flag (line ~837 trong JAX scripts, tìm equivalent ở PyTorch).
5. Forward call: thêm `output_hidden_states=need_hidden_states` cho cả 2 model.
6. Thêm `mta_projector_list=mta_projector_list` vào `LossArgs(...)` construction.
7. Refactor optimizer: param group `projector_*` lr_scale=2 (theo pattern existing config yaml line 56-58).
   - Implement helper `build_param_groups(model, projector_modules, base_lr)`.
8. Define `compute_mta_loss` ngay trong file (hoặc import từ span_utils nếu thuần wrapper).
9. Add `elif loss == "mta": current_loss = compute_mta_loss(args, loss_args)` ở dispatch loop (sau `elif loss.startswith("alm"):`).
10. Compile check: `python -c "import pytorch_cross_tokenizer_distill"` (hoặc syntax check qua `python -m py_compile`).

## Success Criteria

- [ ] File compile, không syntax error
- [ ] Dry-run với `losses=[alm_unbiased]` only → không break baseline
- [ ] Dry-run với `losses=[alm_unbiased, mta]` → cả 2 loss được compute, log ra scalar
- [ ] `mta_projector_list.parameters()` xuất hiện trong optimizer state
- [ ] `student_out.hidden_states is not None` khi MTA active

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Multi-GPU: projectors trên student device, teacher hidden trên teacher device | Forward teacher hidden through `.to(student_device)` trong `compute_mta_loss` trước khi cosine-sim |
| Optimizer state lớn thêm (projector params + AdamW moments) | ~6M params × 2 moments × 4 bytes ≈ 50MB. Negligible. |
| LoRA wraps model — `model.parameters()` không match plain name | Test với dry-run: print param groups + counts. Filter pattern qua name regex thay vì module identity. |
| `loss_args.batch` field name mismatch ('attention_mask_new' vs 'attention_mask') | Đọc collator output keys trong Phase 3, đảm bảo consistent |
| Teacher model có `model.config.num_hidden_layers != 24` (e.g. nếu user đổi teacher) | Validate `max(teacher_layer_mapping) <= teacher_config.num_hidden_layers` ở init, raise clear error |
