---
phase: 1
title: Precompute Spans Script
status: completed
priority: P1
effort: 2h
dependencies: []
---

# Phase 1: Precompute Spans Script

## Overview
Tạo script offline dùng spaCy để trích noun chunks + verb phrases từ `data/dolly_train.jsonl` → ghi `data/dolly_train_with_spans.jsonl`. Tránh chạy spaCy online trong DataLoader (CPU bottleneck, conflict với num_workers=16).

## Requirements
- **Functional:** Đọc dolly_train.jsonl → spaCy pipe trên text (concat prompt + output) → trích spans + words với char offsets → ghi file mới giữ nguyên các field cũ.
- **Non-functional:** Idempotent (chạy lại nếu file output đã tồn tại thì skip hoặc overwrite có cờ). Progress bar. Chạy được trên CPU thuần.

## Architecture

Input row (jsonl):
```json
{"prompt": "...", "output": "..."}
```

Output row (jsonl):
```json
{
  "prompt": "...",
  "output": "...",
  "spans_char_offsets": [[start, end], ...],   // noun chunks + verb phrases filtered overlap
  "words_char_offsets": [[start, end], ...]    // word-level offsets
}
```

**Concat strategy:** spans được tính trên `text = prompt + "\n" + output` (text hoàn chỉnh mà tokenizer sẽ thấy). Char offset của spans tính theo `text` này.

## Related Code Files

- **Create:** `precompute_spans.py` (~80 LOC, project root)
- **Read:** `data/dolly_train.jsonl`
- **Write:** `data/dolly_train_with_spans.jsonl`

## Implementation Steps

1. Cài deps: `pip install spacy && python -m spacy download en_core_web_sm`
2. Tạo `precompute_spans.py`:
   - argparse: `--input`, `--output`, `--text-field` (default concat prompt+output)
   - Load `nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])`
   - Tạo `Matcher` với pattern verb phrase:
     ```python
     [{"POS": "AUX", "OP": "*"}, {"POS": "ADV", "OP": "*"},
      {"POS": "VERB", "OP": "+"}, {"POS": "ADV", "OP": "*"}]
     ```
   - Iterate jsonl, build `text = row['prompt'] + "\n" + row['output']`
   - `nlp.pipe(texts, n_process=4, batch_size=64)`
   - Per doc: collect noun_chunks + matcher hits → port `filter_overlapping_spans` từ doc gốc (lines 244-267)
   - Output spans_char_offsets, words_char_offsets dưới dạng `list[list[int]]` (jsonl-friendly)
3. Verify trên 100 examples: in 5 cặp (text, spans, words) đầu tiên để eyeball check.
4. Chạy full → đo time + size output.

## Success Criteria

- [ ] `data/dolly_train_with_spans.jsonl` tồn tại, số lines == input
- [ ] Mỗi row có 2 field mới: `spans_char_offsets`, `words_char_offsets` (list of `[start, end]` int pairs)
- [ ] Sample inspection: spans cover noun chunks rõ ràng (e.g. "the dog", "machine learning")
- [ ] Spans không chồng chéo (filter pass)
- [ ] Char offsets trong range `[0, len(text))`
- [ ] Wall-time < 5 phút trên dolly_train (~15k examples)

## Risk Assessment

| Risk | Mitigation |
|---|---|
| `en_core_web_sm` không trích được verb phrase phức tạp | Verb phrase đơn giản đủ dùng cho relational sim. Có thể upgrade `_md` sau. |
| spaCy crash trên text rất dài | Set `nlp.max_length = 2_000_000`; skip + log nếu vẫn fail |
| n_process=4 hang trên Windows | Fallback `n_process=1` nếu phát hiện Windows |
| File output ghi đè vô tình | Add `--overwrite` flag, default raise nếu file đã tồn tại |
