# Brainstorm — Tích hợp MTA Loss vào ALM (PyTorch)

**Date:** 2026-05-23
**Path:** PyTorch only (`pytorch_cross_tokenizer_distill.py`)
**Goal:** Train GPT2-120M student ← Qwen1.5-1.8B teacher với `losses=[alm_unbiased, mta]`

---

## 1. Problem Statement

- Cross-tokenizer distillation hiện chỉ có ALM (token-level distribution alignment qua chunk matrices).
- Thêm MTA (Multi-Teacher Alignment): span-level relational similarity loss để khớp **cấu trúc tương quan** giữa các spans (noun chunks + verb phrases) trong hidden space của 2 model.
- Hidden state matching trực tiếp khó vì tokenizer khác nhau → MTA aggregate hidden theo char-spans (tokenizer-agnostic).

---

## 2. Approaches đã evaluate

| Approach | Verdict |
|---|---|
| **A. Theo doc `mta-loss-integration.md` nguyên bản** | ❌ Bug: `loss_args.distiller` không tồn tại; mix PyTorch projectors vào JAX loop → no grad; spaCy online + n_process=4 conflict DataLoader |
| **B. Port sang JAX/Flax đầy đủ** | ❌ Workload lớn (rewrite bmm, projectors thành Flax Linear, đăng ký TrainState), không cần thiết khi PyTorch path đã chạy được |
| **C. PyTorch path, precompute spans offline, fix bugs (chosen)** | ✅ Minimal, fast iteration, không đụng JAX |

---

## 3. Final Design

### 3.1 Files

| File | Status | LOC | Mục đích |
|---|---|---|---|
| `precompute_spans.py` | **MỚI** | ~80 | spaCy extract spans/words offline → `data/dolly_train_with_spans.jsonl` |
| `pytorch_span_utils.py` | **MỚI** | ~180 | `compute_token_weights`, `aggregate_spans_for_model`, `compute_hidden_span_loss`, `get_span_loss`, `compute_overall_span_loss` |
| `pytorch_cross_tokenizer_distill.py` | **SỬA** | +50 | Add `compute_mta_loss`, projector init, dispatch `elif loss == "mta"`, optimizer include projectors, set `output_hidden_states=True` |
| `pytorch_tokenizer_aligner.py` | **SỬA** | +25 | Truyền `spans_char_offsets`, `words_char_offsets`, `offset_mapping_new`, `offset_mapping_original`, `texts` qua batch |
| `gpt2_120M_alm_mta_distill.yaml` | **MỚI** | — | Config với `losses=[alm_unbiased, mta]` |
| `gpt2_120M_alm_mta_distill.sh` | **MỚI** | — | Launch script |

### 3.2 Bug fixes vs doc gốc

| Bug | Fix |
|---|---|
| `loss_args.distiller.student_tokenizer` không tồn tại | Dùng `loss_args.tokenizer_new` (đã có trong `LossArgs`) |
| PyTorch projectors trong JAX loop → no gradient | Chỉ làm PyTorch path |
| Re-tokenize trong loss → chậm + risk mismatch sau truncation | Collator gọi tokenizer với `return_offsets_mapping=True`, lưu vào `batch['offset_mapping_new/original']`. Loss chỉ đọc, không re-tokenize |
| spaCy online | Precompute offline |
| Args missing | Thêm vào `CrossTokenizerDistillArgs` |

### 3.3 Data flow

```
[Offline] precompute_spans.py
  data/dolly_train.jsonl → spaCy → data/dolly_train_with_spans.jsonl
  Fields added: spans_char_offsets [[s,e],...], words_char_offsets [[s,e],...]

[Train] Dataset row → Collator:
  ├─ tokenize student   → input_ids_new, offset_mapping_new
  ├─ tokenize teacher   → input_ids_original, offset_mapping_original
  ├─ keep raw text      → texts (cho debug + downstream)
  └─ pass-through       → spans_char_offsets, words_char_offsets

[Forward] both models with output_hidden_states=True

[Loss] compute_mta_loss(args, loss_args):
  for (s_idx, t_idx, projector) in layer_pairs:
    s_hidden = student_out.hidden_states[s_idx]
    t_hidden = teacher_out.hidden_states[t_idx]
    s_w = compute_token_weights(s_hidden, attention_mask_new)
    t_w = compute_token_weights(t_hidden, attention_mask_original)
    s_repr = aggregate_spans_for_model(s_hidden, s_w, attention_mask_new,
                                      offset_mapping_new, spans_char_offsets)
    t_repr = aggregate_spans_for_model(t_hidden, t_w, attention_mask_original,
                                      offset_mapping_original, spans_char_offsets)
    project(s_repr) → align dim (768 → 2048)
    relational cosine-sim MSE giữa S_sim và T_sim
  total = (word_loss + span_loss) / len(layer_pairs)

total_loss = 1.0 * alm_unbiased + w_span_loss * mta_loss
```

### 3.4 Chốt từ Q&A

| Question | Decision |
|---|---|
| Layer mapping | student `[3,6,9,12]` ↔ teacher `[6,12,18,24]`, `split=[0,1,4,4]` (1 word + 3 span projectors) |
| Loss combo | Sum cố định: `1.0 * alm_unbiased + 2.0 * mta` (tune sau) |
| Span precompute | Script riêng `precompute_spans.py` → file mới `dolly_train_with_spans.jsonl` |
| Projector training | Train cùng student, param group riêng `lr_scale=2` (theo pattern `projector_*` hiện tại) |
| Mask cho aggregation | `attention_mask` (không phải `loss_mask`) — representation alignment cần context đầy đủ |
| `entropy_weight` default | `False` |
| Offset mapping cho loss | Dùng `batch['offset_mapping_new/original']` precomputed trong collator, **không** re-tokenize |

### 3.5 Config (gpt2_120M_alm_mta_distill.yaml)

```yaml
losses: [alm_unbiased, mta]
loss_weights: [1.0, 2.0]

mta_mode: true
teacher_layer_mapping: [6, 12, 18, 24]
student_layer_mapping: [3, 6, 9, 12]
split_layer_mapping: [0, 1, 4, 4]
w_span_loss: 2.0
entropy_weight: false
wo_span_weight: false

data:
  path: data/dolly_train_with_spans.jsonl
```

### 3.6 Projector params

- 4 × `Linear(768, 2048)` ≈ **6.3M params**
- Add to optimizer; param_group `projector_*` lr_scale=2 (theo config hiện tại line 57)

---

## 4. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| VRAM tăng do `output_hidden_states=True` (12+24 layers × seq × hidden) | Batch 16 × seq 512 → ~200MB extra. OK với A100. Nếu OOM: giảm batch hoặc bật gradient_checkpointing |
| Span char offsets vượt khỏi range sau truncation | `aggregate_spans_for_model` filter bằng condition `(offsets_start+1 >= span_start) & (offsets_end <= span_end)` — span ngoài range tự động có 0 tokens, weight_sum=0, được clamp |
| Empty spans batch | `aggregate_spans_for_model` return `None` → skip layer pair |
| spaCy `en_core_web_sm` accuracy thấp cho verb phrases | Tunable: upgrade `en_core_web_md` sau nếu cần |
| Param group regex `projector_*` không match `mta_projector_list` | Đặt tên module attribute là `mta_projector_list` — pattern hiện tại `.*projector_.*` sẽ match |

---

## 5. Success Metrics

- ROUGE-L Dolly valid: MTA-enabled ≥ `alm_unbiased` baseline + 0.5 điểm
- MMLU/ARC-Easy không regression vs baseline (±1%)
- `loss/mta` giảm monotonic, không NaN
- Wall-time/step tăng ≤ 30% so với `alm_unbiased` thuần

---

## 6. Next Steps

1. Implement theo plan phases (đề nghị `/ck:plan`):
   - Phase 1: `precompute_spans.py` + verify trên 100 examples
   - Phase 2: `pytorch_span_utils.py` (port từ doc + fix)
   - Phase 3: Modify `pytorch_tokenizer_aligner.py`
   - Phase 4: Modify `pytorch_cross_tokenizer_distill.py`
   - Phase 5: Config + launch script
   - Phase 6: Smoke test (50 steps) → full run

---

## 7. Unresolved Questions

- **`loss_mask` interaction**: MTA dùng `attention_mask` cho aggregation (chốt). Nhưng nếu spans rơi vào phần prompt (không phải response), có nên filter? → Đề xuất KHÔNG filter — representation alignment có lợi từ context prompt.
- **w_span_loss=2.0 vs alm=1.0**: chưa có grid search. Đề xuất chạy ablation `w_span_loss ∈ {0.5, 1.0, 2.0}` sau khi pipeline ổn.
- **Token weight detach**: doc gốc detach `attn_weights` — confirm intent là KHÔNG cho gradient flow qua self-attention weighting. Giữ như doc.
