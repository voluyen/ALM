---
phase: 3
title: Update Collator
status: completed
priority: P1
effort: 2h
dependencies:
  - 1
---

# Phase 3: Update Collator

## Overview
Sửa `pytorch_tokenizer_aligner.py` (`TokenizerAlignerCollator`) để pass-through precomputed spans và emit offset mappings cho cả student + teacher tokenizer. Tránh re-tokenize trong loss compute.

## Requirements
- **Functional:** Batch output có thêm 5 field: `texts`, `offset_mapping_new`, `offset_mapping_original`, `spans_char_offsets`, `words_char_offsets`.
- **Non-functional:** Không break các field cũ (`input_ids_new`, `input_ids_original`, `attention_mask_*`, `alignment_matrix_*`, `loss_mask_*`). Backward-compat khi field spans không có trong dataset (gracefully skip MTA).

## Architecture

Input dataset row (sau khi load `dolly_train_with_spans.jsonl`):
```python
{
  "prompt": str,
  "output": str,
  "spans_char_offsets": list[list[int]],
  "words_char_offsets": list[list[int]],
}
```

Collator output batch — thêm:
```python
{
  ...existing fields...,
  "texts": list[str],                            # B strings, raw concat
  "offset_mapping_new": Tensor[B, S_len, 2],     # student tokenizer offsets
  "offset_mapping_original": Tensor[B, T_len, 2],# teacher tokenizer offsets
  "spans_char_offsets": list[list[list[int]]],   # B × N_spans × 2 (variable N per sample)
  "words_char_offsets": list[list[list[int]]],   # B × N_words × 2
}
```

## Related Code Files

- **Modify:** `pytorch_tokenizer_aligner.py` (~25 LOC added)
- **Read context:** dataset loading trong `pytorch_cross_tokenizer_distill.py` (where dataset is loaded as `datasets.Dataset.from_json`)

## Implementation Steps

1. Đọc `pytorch_tokenizer_aligner.py` để hiểu current `__call__` flow.
2. Trong `__call__`:
   - Build `texts = [row['prompt'] + "\n" + row['output'] for row in batch]` (consistent với precompute_spans.py concat strategy)
   - Khi gọi tokenizer student và teacher để tạo `input_ids_new` / `input_ids_original`, thêm `return_offsets_mapping=True` và lưu lại `offset_mapping`. Make sure same tokenization call (no duplicate) — chỉ thêm flag.
   - Pass-through `spans_char_offsets`, `words_char_offsets` from each row (default `[]` nếu field thiếu)
3. Convert offset_mapping về `torch.Tensor` shape `[B, L, 2]` (đã pad theo input_ids).
4. Verify: print batch keys + shapes trong dry-run loop (sau khi load 1 batch).

## Success Criteria

- [ ] Batch có 5 field mới với shape đúng
- [ ] `offset_mapping_new.shape == input_ids_new.shape + (2,)`
- [ ] `offset_mapping_original.shape == input_ids_original.shape + (2,)`
- [ ] `spans_char_offsets` len == batch_size
- [ ] Không break existing fields (alignment matrices vẫn đúng shape)
- [ ] Dataset không có field spans (e.g. legacy data) → collator vẫn chạy, spans field = `[[] for _ in batch]`

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Slow/fast tokenizer mismatch: `return_offsets_mapping` chỉ work với fast tokenizer | Assert `tokenizer.is_fast == True` ở init; raise rõ ràng nếu không |
| Text concat strategy khác precompute → offsets misalign | DRY: define helper `build_text(row)` dùng chung cả precompute_spans.py và collator. Hoặc document rõ format ở cả 2 nơi. |
| Padding offset_mapping với `(0, 0)` có thể nhầm với valid token tại offset 0 | Combine với `attention_mask` để filter pad — `aggregate_spans_for_model` đã `& attention_mask` |
| Loss mask path (chỉ tính loss trên response) không khớp với spans (toàn câu) | Đúng design — MTA dùng `attention_mask`, không phải `loss_mask`. Document trong code comment |
