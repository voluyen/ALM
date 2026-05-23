# Tích hợp MTA Loss (Multi-Teacher Alignment)

> Tài liệu mô tả chi tiết cách thêm MTA loss vào pipeline cross-tokenizer distillation của tokenkit.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Phụ thuộc cần cài](#2-phụ-thuộc-cần-cài)
3. [Bước 1 — Tạo `span_utils.py`](#3-bước-1--tạo-span_utilspy)
4. [Bước 2 — Cập nhật `losses.py`](#4-bước-2--cập-nhật-lossespy)
5. [Bước 3 — Cập nhật `cross_tokenizer_distill.py`](#5-bước-3--cập-nhật-cross_tokenizer_distillpy)
6. [Bước 4 — Cấu hình YAML](#6-bước-4--cấu-hình-yaml)
7. [Kiến trúc & luồng dữ liệu](#7-kiến-trúc--luồng-dữ-liệu)
8. [Tham số quan trọng](#8-tham-số-quan-trọng)
9. [Câu hỏi chưa giải quyết](#9-câu-hỏi-chưa-giải-quyết)

---

## 1. Tổng quan

**MTA (Multi-Teacher Alignment)** là một loss phụ trợ cho cross-tokenizer distillation. Thay vì khớp trực tiếp hidden states token-by-token (khó khi student/teacher có tokenizer khác nhau), MTA:

1. Trích các **span ngữ nghĩa** (noun chunks + verb phrases) từ văn bản gốc bằng spaCy.
2. **Aggregate** hidden states của từng span thành một vector đại diện.
3. Tính **relational similarity loss**: MSE giữa ma trận tương đồng cosine của student spans và teacher spans — buộc student học *cấu trúc tương quan* giữa các span, không cần khớp từng chiều hidden.

```
Student hidden states  ──► aggregate theo spans ──► span_repr_S
                                                        │
                                                   projector (Linear)
                                                        │
                                                   cos-sim matrix ──┐
                                                                     ├──► MSE loss
Teacher hidden states  ──► aggregate theo spans ──► span_repr_T ───►│
                                                   cos-sim matrix ──┘
```

---

## 2. Phụ thuộc cần cài

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Không cần thêm gì vào `requirements.txt` nếu PyTorch đã có sẵn (span_utils dùng `torch`, không phải JAX).

---

## 3. Bước 1 — Tạo `span_utils.py`

**Vị trí:** `tokenkit/training/span_utils.py` *(file mới)*

File này chứa toàn bộ logic tính MTA loss, chạy trên CPU/GPU bằng PyTorch (tách biệt với phần JAX).

```python
import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def compute_token_weights(hidden_state, attention_mask):
    """
    Tính token importance weights bằng self-attention (không có diagonal).
    hidden_state: [B, L, D]  attention_mask: [B, L]
    Returns: [B, L] weights, detached.
    """
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
    return attn_weights.mean(dim=1).detach()   # [B, L]


def aggregate_spans_for_model(hidden_states, layer_weights, attention_mask,
                               offsets_mapping, spans_offsets, entropy_weights=None):
    """
    Tổng hợp hidden states của mỗi span thành một vector đại diện.
    Dùng weighted average theo layer_weights.
    Returns: (span_repr, weight_sum, ent_weight, valid_span_mask)
    """
    device = hidden_states.device
    B_size, SeqLen, D = hidden_states.shape

    span_tensors = [
        torch.tensor(s, dtype=torch.long, device=device) if len(s) > 0
        else torch.empty((0, 2), dtype=torch.long, device=device)
        for s in spans_offsets
    ]
    padded_spans = pad_sequence(span_tensors, batch_first=True, padding_value=0)

    if padded_spans.numel() == 0 or padded_spans.size(1) == 0:
        return None, None, None, None

    max_spans = padded_spans.size(1)
    padded_span_starts = padded_spans[:, :, 0]
    padded_span_ends   = padded_spans[:, :, 1]

    lengths = torch.tensor([len(s) for s in spans_offsets], device=device)
    col_indices = torch.arange(max_spans, device=device).unsqueeze(0)
    valid_span_mask = col_indices < lengths.unsqueeze(1)          # [B, max_spans]

    current_offsets = offsets_mapping[:, :SeqLen, :]
    offsets_start = current_offsets[..., 0].unsqueeze(2)          # [B, L, 1]
    offsets_end   = current_offsets[..., 1].unsqueeze(2)

    span_starts_exp = padded_span_starts.unsqueeze(1)             # [B, 1, max_spans]
    span_ends_exp   = padded_span_ends.unsqueeze(1)

    # token_in_span_map[b, l, s] = True nếu token l thuộc span s
    token_in_span_map = (offsets_start + 1 >= span_starts_exp) & (offsets_end <= span_ends_exp)
    token_in_span_map = token_in_span_map & attention_mask.unsqueeze(2).bool()

    A = token_in_span_map.transpose(1, 2).to(hidden_states.dtype)  # [B, max_spans, L]

    weighted_hidden = hidden_states * layer_weights.unsqueeze(-1)
    span_sum  = torch.bmm(A, weighted_hidden)                      # [B, max_spans, D]
    weight_sum = torch.bmm(A, layer_weights.unsqueeze(-1)).squeeze(-1)  # [B, max_spans]

    if entropy_weights is not None:
        ent_weight_sum  = torch.bmm(A, entropy_weights.unsqueeze(-1)).squeeze(-1)
        span_lengths    = A.sum(dim=-1).clamp(min=1e-5)
        final_ent_weight = ent_weight_sum / span_lengths
    else:
        final_ent_weight = weight_sum

    span_repr = span_sum / weight_sum.unsqueeze(-1).clamp(min=1e-5)
    return span_repr, weight_sum, final_ent_weight, valid_span_mask


def compute_hidden_span_loss(projector, s_span_repr, t_span_repr,
                              valid_span_mask, w_sum, use_span_weight=True):
    """
    Relational MSE loss: so sánh ma trận cosine-similarity của student và teacher spans.
    Chỉ tính trên các cặp span hợp lệ trong cùng một sample.
    """
    device = s_span_repr.device
    B_size, Max_Spans = valid_span_mask.shape

    s_span_proj = projector(s_span_repr)                      # [B, Max_Spans, D_teacher]

    valid_s = s_span_proj[valid_span_mask]                    # [N_valid, D]
    valid_t = t_span_repr[valid_span_mask]
    valid_w = w_sum[valid_span_mask]

    if valid_s.size(0) == 0:
        return torch.tensor(0.0, device=device)

    batch_indices  = torch.arange(B_size, device=device).unsqueeze(1).expand(-1, Max_Spans)
    valid_batch_ids = batch_indices[valid_span_mask]

    S_norm = F.normalize(valid_s, p=2, dim=-1)
    T_norm = F.normalize(valid_t, p=2, dim=-1)

    S_sim = S_norm @ S_norm.T
    T_sim = T_norm @ T_norm.T

    same_batch = (valid_batch_ids.unsqueeze(1) == valid_batch_ids.unsqueeze(0))
    not_self   = ~torch.eye(valid_s.size(0), dtype=torch.bool, device=device)
    mask       = same_batch & not_self

    loss_flat = F.mse_loss(S_sim[mask], T_sim[mask], reduction='none')

    if use_span_weight:
        pair_w = (valid_w.unsqueeze(1) * valid_w.unsqueeze(0))[mask]
        return (loss_flat * pair_w).sum() / pair_w.sum().clamp(min=1e-5)
    return loss_flat.mean() if loss_flat.numel() > 0 else torch.tensor(0.0, device=device)


def get_span_loss(projectors, s_att_mask, t_att_mask, s_hidden_states, t_hidden_states,
                  s_offsets_mapping, t_offsets_mapping, spans_offsets,
                  teacher_layer_mapping, student_layer_mapping,
                  w_t_entropy=None, use_span_weight=True):
    """Tính loss qua từng cặp layer (student_idx, teacher_idx)."""
    final_loss = 0.0
    for s_idx, t_idx, proj in zip(student_layer_mapping, teacher_layer_mapping, projectors):
        s_hidden = s_hidden_states[s_idx]
        t_hidden = t_hidden_states[t_idx]

        s_weights = compute_token_weights(s_hidden, s_att_mask)
        t_weights = compute_token_weights(t_hidden, t_att_mask)

        s_repr, _, _, valid_mask = aggregate_spans_for_model(
            s_hidden, s_weights, s_att_mask, s_offsets_mapping, spans_offsets)
        t_repr, t_w, t_ent_w, _ = aggregate_spans_for_model(
            t_hidden, t_weights, t_att_mask, t_offsets_mapping, spans_offsets, w_t_entropy)

        if s_repr is None or t_repr is None:
            continue

        w_sum = t_ent_w if w_t_entropy is not None else t_w
        final_loss += compute_hidden_span_loss(proj, s_repr, t_repr, valid_mask, w_sum, use_span_weight)

    return final_loss


def compute_overall_span_loss(projectors, s_att_mask, t_att_mask, s_logits, t_logits,
                               s_hidden_states, t_hidden_states,
                               s_offsets_mapping, t_offsets_mapping,
                               spans_offsets, words_offsets, args):
    """
    Entry point: tính cả word-level và span-level loss, chia đều theo số layer.
    split_layer_mapping = [0, word_end, span_end, total]
      - layers [0 : word_end]   → word-level projectors
      - layers [word_end : span_end] → span-level projectors
    """
    w_t_entropy = None
    if args.entropy_weight:
        t_probs    = torch.softmax(t_logits.float().detach(), dim=-1)
        t_entropy  = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=-1)
        w_t_entropy = 1 - t_entropy / math.log(t_logits.size(-1))

    use_span_weight = not getattr(args, 'wo_span_weight', False)
    lo, mid, hi = args.split_layer_mapping[0], args.split_layer_mapping[1], args.split_layer_mapping[2]

    word_loss = get_span_loss(
        projectors[lo:mid], s_att_mask, t_att_mask, s_hidden_states, t_hidden_states,
        s_offsets_mapping, t_offsets_mapping, words_offsets,
        args.teacher_layer_mapping[lo:mid], args.student_layer_mapping[lo:mid],
        w_t_entropy, use_span_weight)

    span_loss = get_span_loss(
        projectors[mid:hi], s_att_mask, t_att_mask, s_hidden_states, t_hidden_states,
        s_offsets_mapping, t_offsets_mapping, spans_offsets,
        args.teacher_layer_mapping[mid:hi], args.student_layer_mapping[mid:hi],
        w_t_entropy, use_span_weight)

    return (word_loss + span_loss) / len(args.student_layer_mapping)


def filter_overlapping_spans(spans):
    """Lọc span chồng chéo; trả về (filtered_spans, word_offsets)."""
    sorted_spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    filtered, words = [], []
    if not sorted_spans:
        return filtered, words

    current = sorted_spans[0]
    for nxt in sorted_spans[1:]:
        if nxt[1] <= current[1]:
            continue
        filtered.append((current[0], current[1]))
        p = current[2]
        n = len(p)
        words.extend([(p[i-1].idx, p[i].idx) for i in range(1, n)])
        words.append((p[n-1].idx, p[n-1].idx + len(p[n-1])))
        current = nxt

    filtered.append((current[0], current[1]))
    p = current[2]
    n = len(p)
    words.extend([(p[i-1].idx, p[i].idx) for i in range(1, n)])
    words.append((p[n-1].idx, p[n-1].idx + len(p[n-1])))
    return filtered, words


def get_spans_offsets(texts, nlp, matcher):
    """
    Dùng spaCy để trích noun chunks và verb phrases.
    Returns: (spans_offsets, words_offsets) — mỗi phần tử là list[(start_char, end_char)]
    """
    spans, words = [], []
    for doc in nlp.pipe(texts, disable=["ner", "lemmatizer"], n_process=4):
        spans_with_offsets = []
        for _, start, end in matcher(doc):
            vp = doc[start:end]
            spans_with_offsets.append((vp.start_char, vp.end_char, vp))
        spans_with_offsets.extend([(nc.start_char, nc.end_char, nc) for nc in doc.noun_chunks])
        unique_spans, unique_words = filter_overlapping_spans(spans_with_offsets)
        spans.append(unique_spans)
        words.append(unique_words)
    return spans, words
```

---

## 4. Bước 2 — Cập nhật `losses.py`

**Vị trí:** `tokenkit/training/losses.py`

### 4a. Thêm import ở đầu file

```python
# thêm vào sau dòng "from tokenkit.models import param"
from .span_utils import get_spans_offsets, compute_overall_span_loss
```

### 4b. Thêm fields vào `LossArgs`

```python
@dataclass
class LossArgs:
    # ... các field hiện có ...
    mta_projector_list: Any = None   # ← thêm dòng này
    nlp: Any = None                  # ← thêm dòng này
    matcher: Any = None              # ← thêm dòng này
```

### 4c. Thêm hàm `compute_span_loss` ở cuối file

```python
def compute_span_loss(args, loss_args):
    """
    Wrapper tính MTA span loss.
    Tokenize lại text để lấy offset mapping cho cả student và teacher,
    sau đó gọi compute_overall_span_loss từ span_utils.
    """
    s_tokenizer = loss_args.distiller.student_tokenizer
    t_tokenizer = loss_args.tokenizer_teacher
    input_texts = s_tokenizer.batch_decode(
        loss_args.batch['input_ids_new'], skip_special_tokens=True)

    device = loss_args.batch['input_ids_new'].device
    s_seq_len = loss_args.batch['input_ids_new'].shape[1]
    t_seq_len = loss_args.batch['input_ids_original'].shape[1]

    s_offsets_mapping = s_tokenizer(
        input_texts, return_offsets_mapping=True,
        max_length=s_seq_len, padding='max_length',
        truncation=True, add_special_tokens=False,
        return_tensors='pt')['offset_mapping'].to(device)

    t_offsets_mapping = t_tokenizer(
        input_texts, return_offsets_mapping=True,
        max_length=t_seq_len, padding='max_length',
        truncation=True, add_special_tokens=False,
        return_tensors='pt')['offset_mapping'].to(device)

    spans_offsets, words_offsets = get_spans_offsets(
        input_texts, loss_args.nlp, loss_args.matcher)

    return compute_overall_span_loss(
        loss_args.mta_projector_list,
        loss_args.batch['attention_mask_new'],
        loss_args.batch['attention_mask_original'],
        loss_args.student_logits,
        loss_args.teacher_logits,
        loss_args.student_out.hidden_states,
        loss_args.teacher_out.hidden_states,
        s_offsets_mapping,
        t_offsets_mapping,
        spans_offsets,
        words_offsets,
        args,
    )
```

---

## 5. Bước 3 — Cập nhật `cross_tokenizer_distill.py`

**Vị trí:** `scripts/cross_tokenizer_distill.py`

### 5a. Thêm imports

```python
# thêm vào phần imports ở đầu file
import spacy
from spacy.matcher import Matcher
```

### 5b. Thêm config args vào `Args` dataclass

```python
@dataclass
class Args:
    # ... các field hiện có ...

    # ── MTA config ──────────────────────────────────────────────────
    # Bật/tắt MTA span distillation loss
    mta_mode: bool = False
    # Indices các layer của teacher dùng để tính MTA (theo thứ tự tương ứng với student)
    teacher_layer_mapping: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # Indices các layer của student dùng để tính MTA
    student_layer_mapping: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # Phân nhóm projectors: [start, word_end, span_end, total]
    # [0:word_end] → word-level projectors; [word_end:span_end] → span-level projectors
    split_layer_mapping: list[int] = field(default_factory=lambda: [0, 1, 4, 4])
    # Trọng số của MTA loss trong tổng loss
    w_span_loss: float = 2.0
```

### 5c. Khởi tạo spaCy + projectors (sau khi load mined_mapping)

```python
# Tìm đoạn: "mined_mapping = mined_distances = None"
# Thêm ngay phía dưới:

nlp = matcher = mta_projector_list = None
if args.mta_mode:
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    matcher.add("VERB_PHRASE", [[
        {"POS": "AUX",  "OP": "*"},
        {"POS": "ADV",  "OP": "*"},
        {"POS": "VERB", "OP": "+"},
        {"POS": "ADV",  "OP": "*"},
    ]])

    student_hidden_size = student_config.hidden_size
    teacher_hidden_size = teacher_config.hidden_size
    mta_projector_list = torch.nn.ModuleList([
        torch.nn.Linear(student_hidden_size, teacher_hidden_size)
        for _ in range(len(args.teacher_layer_mapping))
    ])
```

### 5d. Bật `output_hidden_states` khi cần

```python
# Tìm dòng:
need_hidden_states = len([loss for loss in args.losses if loss in {"alm_latents", "baseline_dskd"}]) > 0

# Sửa thành:
need_hidden_states = len([loss for loss in args.losses if loss in {"alm_latents", "baseline_dskd", "mta"}]) > 0
```

### 5e. Pass MTA objects vào `LossArgs`

```python
loss_args = losses.LossArgs(
    # ... các field hiện có ...
    mta_projector_list=mta_projector_list,   # ← thêm
    nlp=nlp,                                  # ← thêm
    matcher=matcher,                          # ← thêm
)
```

### 5f. Dispatch loss `"mta"` trong training loop

```python
for loss_idx, loss in enumerate(args.losses):
    if loss == "sft":
        current_loss = losses.compute_sft_loss(args, loss_args)
    elif loss.startswith("alm"):
        ...
    elif loss == "mta":                                          # ← thêm block này
        current_loss = losses.compute_span_loss(args, loss_args)
    else:
        raise ValueError(f"Invalid loss: {loss}")
```

---

## 6. Bước 4 — Cấu hình YAML

Thêm vào file config (ví dụ `configs/gpt2_1.5B_cross_tokenizer_distill.yaml`):

```yaml
# Kích hoạt MTA
losses: [alm_unbiased, mta]
mta_mode: true

# Layer mapping — chọn các layer trung gian của mỗi model
# (điều chỉnh theo số layer thực tế của model đang dùng)
teacher_layer_mapping: [6, 12, 18, 24]   # GPT2-XL có 48 layer
student_layer_mapping: [2, 4, 6, 8]      # GPT2-small có 12 layer

# split_layer_mapping: [start, word_end, span_end, total]
# Ý nghĩa:
#   layers [0:1]  → 1 projector cho word-level loss
#   layers [1:4]  → 3 projectors cho span-level loss
split_layer_mapping: [0, 1, 4, 4]

# Trọng số của MTA trong tổng loss
w_span_loss: 2.0
```

> **Lưu ý:** `split_layer_mapping[2]` phải bằng `len(teacher_layer_mapping)`.

---

## 7. Kiến trúc & luồng dữ liệu

```
Input texts (batch)
    │
    ├─► spaCy (noun chunks + verb phrases) ──► spans_offsets, words_offsets
    │
    ├─► Student tokenizer (offset mapping) ──► s_offsets_mapping [B, S_len, 2]
    └─► Teacher tokenizer (offset mapping) ──► t_offsets_mapping [B, T_len, 2]

Student hidden_states[layer_idx]  [B, S_len, D_s]
    │
    ├─► compute_token_weights ──► s_weights [B, S_len]
    └─► aggregate_spans_for_model ──► s_span_repr [B, num_spans, D_s]
                                           │
                                      Linear projector
                                           │
                                      s_span_proj [B, num_spans, D_t]
                                           │
                                      cosine-sim matrix ──┐
                                                           ├──► MSE ──► loss
Teacher hidden_states[layer_idx]  [B, T_len, D_t]         │
    │                                                      │
    ├─► compute_token_weights ──► t_weights [B, T_len]     │
    └─► aggregate_spans_for_model ──► t_span_repr ────────►│
                                      cosine-sim matrix ──┘
```

---

## 8. Tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `mta_mode` | `false` | Bật/tắt MTA loss |
| `teacher_layer_mapping` | `[0,0,0,0]` | Indices layer teacher (1 index per projector group) |
| `student_layer_mapping` | `[0,0,0,0]` | Indices layer student tương ứng |
| `split_layer_mapping` | `[0,1,4,4]` | Biên phân tách word/span projector group |
| `w_span_loss` | `2.0` | Weight của MTA loss |
| `entropy_weight` | — | Nếu `True`, dùng entropy token teacher làm weight khi aggregate spans |
| `wo_span_weight` | — | Nếu `True`, tắt weighted loss (dùng mean thay vì weighted sum) |

### Cách chọn layer mapping

```
teacher_layer_mapping[i] ↔ student_layer_mapping[i]  (1-1 correspondence)

Ví dụ — Teacher 48 layer, Student 12 layer, 4 layer pairs:
  teacher: [12, 24, 36, 48]   (mỗi 1/4 mạng)
  student: [ 3,  6,  9, 12]   (mỗi 1/4 mạng)
  split_layer_mapping: [0, 1, 4, 4]
    → layer pair 0       → word-level projector
    → layer pairs 1,2,3  → span-level projectors
```

---

## 9. Câu hỏi chưa giải quyết

- **`loss_args.distiller`**: `compute_span_loss` đang truy cập `loss_args.distiller.student_tokenizer` — cần xác nhận field `distiller` có tồn tại trong `LossArgs` hay cần thay bằng `loss_args.tokenizer_new`.
- **Gradient qua projectors**: `mta_projector_list` là `torch.nn.ModuleList` (PyTorch), trong khi training loop dùng JAX — cần kiểm tra projectors có thực sự được tối ưu không, hay chỉ được dùng inference-only.
- **`entropy_weight` flag**: Tham số `args.entropy_weight` chưa được khai báo trong `Args` dataclass — cần thêm nếu muốn dùng.
- **Multi-GPU/TPU**: spaCy `n_process=4` trong `get_spans_offsets` có thể conflict với data parallel trên TPU — nên set `n_process=1` khi train trên TPU pod.
