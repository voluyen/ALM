"""
MTA (Multi-Teacher Alignment) span-level relational similarity loss for
cross-tokenizer distillation. Pure PyTorch -- no spaCy import (spans are
precomputed offline by precompute_spans.py).

Public API:
    compute_token_weights(hidden, attention_mask)
    aggregate_spans_for_model(hidden, layer_weights, attention_mask,
                              offsets_mapping, spans_offsets,
                              entropy_weights=None)
    compute_hidden_span_loss(projector, s_repr, t_repr, valid_mask,
                             w_sum, use_span_weight=True)
    compute_overall_span_loss(projectors, s_att_mask, t_att_mask,
                              s_logits, t_logits,
                              s_hidden_states, t_hidden_states,
                              s_offsets_mapping, t_offsets_mapping,
                              spans_offsets, words_offsets, args)

`args` is expected to expose:
    entropy_weight        : bool
    wo_span_weight        : bool
    teacher_layer_mapping : list[int]
    student_layer_mapping : list[int]
    split_layer_mapping   : list[int]  [start, word_end, span_end, total]
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Token weights (per-token importance derived from self-attention, no diagonal)
# ---------------------------------------------------------------------------

def compute_token_weights(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute self-attention based token importance weights, averaged across
    attention positions, with the diagonal removed.

    Returns:
        weights [B, L] -- detached (no gradient).
    """
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    q = hidden_state / std
    k = hidden_state / std
    scores = torch.matmul(q, k.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    diag = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag.unsqueeze(0), float("-inf"))

    attn = F.softmax(scores, dim=-1)
    attn = attn * mask
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    return attn.mean(dim=1).detach()  # [B, L]


# ---------------------------------------------------------------------------
# Span aggregation: weighted-average of token hidden states inside each span
# ---------------------------------------------------------------------------

def _spans_to_padded(
    spans_offsets: Sequence[Sequence[Sequence[int]]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert variable-length spans per sample into a padded `[B, MaxSpans, 2]` long tensor + valid mask."""
    span_tensors = [
        torch.tensor(s, dtype=torch.long, device=device) if len(s) > 0
        else torch.empty((0, 2), dtype=torch.long, device=device)
        for s in spans_offsets
    ]
    padded = pad_sequence(span_tensors, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(s) for s in spans_offsets], device=device)
    max_spans = padded.size(1) if padded.numel() else 0
    col = torch.arange(max_spans, device=device).unsqueeze(0)
    valid_mask = col < lengths.unsqueeze(1)  # [B, MaxSpans]
    return padded, valid_mask


def aggregate_spans_for_model(
    hidden_states: torch.Tensor,
    layer_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    offsets_mapping: torch.Tensor,
    spans_offsets: Sequence[Sequence[Sequence[int]]],
    entropy_weights: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Aggregate token hidden states into per-span representations.
    Tokens are included in a span iff their character offset_mapping is fully
    inside the span's char range AND the token is not pad.

    Returns (span_repr, weight_sum, ent_weight, valid_span_mask) or all-None
    when no spans exist in the entire batch.
    """
    device = hidden_states.device
    B, seq_len, D = hidden_states.shape

    padded_spans, valid_span_mask = _spans_to_padded(spans_offsets, device)
    if padded_spans.numel() == 0 or padded_spans.size(1) == 0:
        return None, None, None, None

    span_starts = padded_spans[:, :, 0]        # [B, MaxSpans]
    span_ends   = padded_spans[:, :, 1]        # [B, MaxSpans]

    cur_offsets = offsets_mapping[:, :seq_len, :]
    off_start = cur_offsets[..., 0].unsqueeze(2)   # [B, L, 1]
    off_end   = cur_offsets[..., 1].unsqueeze(2)

    span_starts_exp = span_starts.unsqueeze(1)     # [B, 1, MaxSpans]
    span_ends_exp   = span_ends.unsqueeze(1)

    # Token l belongs to span s iff offsets fully within [span_start, span_end].
    # The +1 in start matches the original ALM doc's loose tokenizer boundary.
    token_in_span = (off_start + 1 >= span_starts_exp) & (off_end <= span_ends_exp)
    token_in_span = token_in_span & attention_mask.unsqueeze(2).bool()

    A = token_in_span.transpose(1, 2).to(hidden_states.dtype)   # [B, MaxSpans, L]

    weighted_hidden = hidden_states * layer_weights.unsqueeze(-1)
    span_sum = torch.bmm(A, weighted_hidden)                    # [B, MaxSpans, D]
    weight_sum = torch.bmm(A, layer_weights.unsqueeze(-1)).squeeze(-1)  # [B, MaxSpans]

    if entropy_weights is not None:
        ent_sum = torch.bmm(A, entropy_weights.unsqueeze(-1)).squeeze(-1)
        span_lens = A.sum(dim=-1).clamp(min=1e-5)
        ent_weight = ent_sum / span_lens
    else:
        ent_weight = weight_sum

    span_repr = span_sum / weight_sum.unsqueeze(-1).clamp(min=1e-5)
    return span_repr, weight_sum, ent_weight, valid_span_mask


# ---------------------------------------------------------------------------
# Relational similarity loss between student & teacher span representations
# ---------------------------------------------------------------------------

def compute_hidden_span_loss(
    projector: torch.nn.Module,
    s_span_repr: torch.Tensor,
    t_span_repr: torch.Tensor,
    valid_span_mask: torch.Tensor,
    w_sum: torch.Tensor,
    use_span_weight: bool = True,
) -> torch.Tensor:
    """
    MSE between within-batch cosine-similarity matrices of student vs teacher
    spans. Only same-sample, non-diagonal pairs contribute. Optionally weighted
    by the product of teacher span weights.
    """
    device = s_span_repr.device
    B, MaxSpans = valid_span_mask.shape

    s_proj = projector(s_span_repr)              # [B, MaxSpans, D_teacher]

    valid_s = s_proj[valid_span_mask]             # [N, D]
    valid_t = t_span_repr[valid_span_mask]
    valid_w = w_sum[valid_span_mask]

    N = valid_s.size(0)
    if N == 0:
        return torch.tensor(0.0, device=device, dtype=s_span_repr.dtype)

    batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, MaxSpans)
    valid_batch_ids = batch_ids[valid_span_mask]

    s_n = F.normalize(valid_s, p=2, dim=-1)
    t_n = F.normalize(valid_t, p=2, dim=-1)

    s_sim = s_n @ s_n.T
    t_sim = t_n @ t_n.T

    same_batch = (valid_batch_ids.unsqueeze(1) == valid_batch_ids.unsqueeze(0))
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)
    mask = same_batch & not_self

    if not mask.any():
        return torch.tensor(0.0, device=device, dtype=s_span_repr.dtype)

    diff = F.mse_loss(s_sim[mask], t_sim[mask], reduction="none")

    if use_span_weight:
        pair_w = (valid_w.unsqueeze(1) * valid_w.unsqueeze(0))[mask]
        return (diff * pair_w).sum() / pair_w.sum().clamp(min=1e-5)
    return diff.mean()


# ---------------------------------------------------------------------------
# Higher-level orchestration: iterate over layer pairs / projectors
# ---------------------------------------------------------------------------

def _get_span_loss(
    projectors: List[torch.nn.Module],
    s_att_mask: torch.Tensor,
    t_att_mask: torch.Tensor,
    s_hidden_states: Sequence[torch.Tensor],
    t_hidden_states: Sequence[torch.Tensor],
    s_offsets: torch.Tensor,
    t_offsets: torch.Tensor,
    spans_offsets: Sequence[Sequence[Sequence[int]]],
    teacher_layer_mapping: Sequence[int],
    student_layer_mapping: Sequence[int],
    w_t_entropy: Optional[torch.Tensor],
    use_span_weight: bool,
) -> torch.Tensor:
    final = None
    for s_idx, t_idx, proj in zip(student_layer_mapping, teacher_layer_mapping, projectors):
        s_h = s_hidden_states[s_idx]
        t_h = t_hidden_states[t_idx]
        # Keep teacher hidden on student's device so projector + downstream ops align
        if t_h.device != s_h.device:
            t_h = t_h.to(s_h.device)
        s_w = compute_token_weights(s_h, s_att_mask)
        t_w = compute_token_weights(t_h, t_att_mask.to(t_h.device))

        s_repr, _, _, valid_mask = aggregate_spans_for_model(
            s_h, s_w, s_att_mask, s_offsets, spans_offsets
        )
        t_repr, t_ws, t_ent_w, _ = aggregate_spans_for_model(
            t_h, t_w, t_att_mask.to(t_h.device), t_offsets.to(t_h.device),
            spans_offsets, w_t_entropy,
        )
        if s_repr is None or t_repr is None:
            continue

        w_sum = t_ent_w if w_t_entropy is not None else t_ws
        layer_loss = compute_hidden_span_loss(
            proj, s_repr, t_repr, valid_mask, w_sum, use_span_weight
        )
        final = layer_loss if final is None else final + layer_loss

    if final is None:
        # No layers produced spans -> return device/dtype-safe zero
        ref = s_hidden_states[student_layer_mapping[0]] if student_layer_mapping else s_att_mask
        return torch.tensor(0.0, device=ref.device, dtype=torch.float32)
    return final


def compute_overall_span_loss(
    projectors: torch.nn.ModuleList,
    s_att_mask: torch.Tensor,
    t_att_mask: torch.Tensor,
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    s_hidden_states: Sequence[torch.Tensor],
    t_hidden_states: Sequence[torch.Tensor],
    s_offsets_mapping: torch.Tensor,
    t_offsets_mapping: torch.Tensor,
    spans_offsets: Sequence[Sequence[Sequence[int]]],
    words_offsets: Sequence[Sequence[Sequence[int]]],
    args,
) -> torch.Tensor:
    """
    Combined word-level + span-level MTA loss, averaged across all projector
    pairs. `args.split_layer_mapping = [lo, mid, hi, total]` selects which
    projectors are word-level vs span-level.
    """
    w_t_entropy = None
    if getattr(args, "entropy_weight", False):
        t_probs = torch.softmax(t_logits.float().detach(), dim=-1)
        t_entropy = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=-1)
        w_t_entropy = 1 - t_entropy / math.log(t_logits.size(-1))

    use_span_weight = not getattr(args, "wo_span_weight", False)
    lo, mid, hi, _ = args.split_layer_mapping[0], args.split_layer_mapping[1], \
                     args.split_layer_mapping[2], args.split_layer_mapping[3]

    projector_list = list(projectors)

    word_loss = _get_span_loss(
        projector_list[lo:mid], s_att_mask, t_att_mask,
        s_hidden_states, t_hidden_states,
        s_offsets_mapping, t_offsets_mapping, words_offsets,
        args.teacher_layer_mapping[lo:mid], args.student_layer_mapping[lo:mid],
        w_t_entropy, use_span_weight,
    )
    span_loss = _get_span_loss(
        projector_list[mid:hi], s_att_mask, t_att_mask,
        s_hidden_states, t_hidden_states,
        s_offsets_mapping, t_offsets_mapping, spans_offsets,
        args.teacher_layer_mapping[mid:hi], args.student_layer_mapping[mid:hi],
        w_t_entropy, use_span_weight,
    )
    n_pairs = max(len(args.student_layer_mapping), 1)
    return (word_loss + span_loss) / n_pairs
