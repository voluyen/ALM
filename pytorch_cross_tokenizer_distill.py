"""
Run cross-tokenizer distillation.
"""

import logging
import os
import shutil
import math
from pathlib import Path
from pprint import pformat
from typing import Any
from dataclasses import dataclass, asdict, field
import yaml

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from tokenkit import data, parse_args, utils
from tokenkit.byteify import load_byteify_tokenizer
from pytorch_tokenizer_aligner import TokenizerAlignerCollator
from pytorch_span_utils import compute_overall_span_loss
from tokenkit.utils import tqdm
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

seed = 42

logger = logging.getLogger(__name__)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

@dataclass
class LossArgs:
    projector_latents: Any
    batch: Any
    teacher_out: Any
    student_out: Any
    tokenizer_teacher: Any
    tokenizer_new: Any
    teacher_probs: Any
    teacher_logprobs: Any
    teacher_logits: Any
    student_probs: Any
    student_logprobs: Any
    student_logits: Any
    scalar_report: Any
    space_mask_teacher: Any
    space_mask_new: Any
    # MTA: list of Linear projectors aligning student hidden dim -> teacher hidden dim.
    mta_projector_list: Any = None

@dataclass
class CrossTokenizerDistillArgs:
    # list of losses to use, e.g. "[sft,alm_unconstrained]" to use SFT and cross-tokenizer distillation via ALM.
    losses: list[str]
    # number of steps to train for.
    steps: int
    # number of steps to warmup the learning rate linearly.
    warmup_steps: int
    # name of the experiment, used for logging and saving checkpoints.
    name: str
    # output directory for checkpoints and logs. CAREFUL: will be deleted if it exists.
    output: str
    # number of CPU workers to use for e.g. data loading.
    num_workers: int
    # interval to log training metrics at.
    log_interval: int
    # interval to sync logged training metrics to the host.
    # `sync_interval` is separate from `log_interval` since we might sometimes want to log very frequently, but not move the tensors to the host every time.
    sync_interval: int
    # interval to evaluate the model.
    eval_interval: int
    # interval to save the model checkpoint.
    save_interval: int
    # name of the target tokenizer to transfer to as a byteify spec (see https://github.com/bminixhofer/tokenkit/blob/main/docs/byteification.md).
    target_tokenizer_name: str
    # training data specification
    data: dict[str, Any]
    # hypernet configuration
    hypernet: parse_args.HypernetArgs
    # optimizer configuration
    optimizer: dict[str, Any]
    # eval configuration, e.g. which LM harness likelihood scoring/generation tasks to run.
    eval: parse_args.EvalArgs
    # student model configuration (pretrained weights path + tokenizer)
    student: parse_args.ModelArgs
    # teacher model configuration (pretrained weights path + tokenizer)
    teacher: parse_args.ModelArgs | None = None
    # # baseline configuration, e.g. for MinED and DSKD. Likely only necessary to replicate experiments from a paper.
    # baseline: None
    # lowest-precision dtype. some parameters (e.g. trainable) will always be kept in fp32.
    dtype: str = "bfloat16"
    # debug mode: run on CPU, disable optimizations.
    debug: bool = False
    # seed for e.g. randomly initialized parameters, data order.
    seed: int = 1234
    # maximum length (in tokens) of the teacher inputs.
    max_teacher_length: int = 512
    # maximum length (in tokens) of the student inputs.
    max_student_length: int = 512
    # multiple to pad to along the embedding (vocabulary) dimension.
    pad_to_multiple_of: int = 64
    # whether to eval after the first training step (useful for debugging).
    eval_at_step_zero: bool = False
    # whether to save after the first training step (useful for debugging).
    save_at_step_zero: bool = False
    # whether to skip LM harness evaluation (useful for debugging).
    skip_lm_eval: bool = False
    # output embedding mode: "preserve" to keep as is, "untie" to train input/output embeddings separately even if they were tied originally.
    output_embeddings_mode: str = "preserve"
    # whether the data is in chat template format and should be decoded as such. usually true.
    use_chat_template: bool = True
    # chat template mode, see `tokenkit.utils.preprocess_prompt` for details.
    chat_template_mode: str = "direct_encode"
    # loss mask mode (only partially supported): None to compute the loss over all input tokens, "dolly" or "openmath2" for corpus-specific prompt masking.
    loss_mask_mode: str = "dolly"
    # whether to use gradient checkpointing to save memory at the cost of compute. not implemeted for all models.
    gradient_checkpointing: bool = False
    # whether to analyze training step cost (FLOPS and memory) then exit, or to run the training loop.
    do_cost_analysis: bool = False
    # whether to run the training loop in dry-run mode, i.e. without actually training the model, only iterating over the data (useful for debugging).
    dry_run: bool = False
    # FSDP data parallelism axis size
    n_data_parallel: int = 1
    # FSDP model parallelism axis size
    n_model_parallel: int = 8
    # loss weights. CAREFUL: does not have an effect for `approx_gradmag*` losses at the moment since the magnitude balancing cancels out the weights.
    loss_weights: list[float] | None = None
    # loss schedules, e.g. ["linear", "cosine", "constant"] to use a linear warmup schedule for the first loss, cosine for the second, and constant for the third.
    loss_schedules: list[str] | None = None
    # how to aggregate the losses. `None` uses a simple arithmetic sum of (weighted) losses. `approx_gradmag_preserve_mag` uses GradMag (see https://arxiv.org/pdf/2503.20083).
    multitask_aggregation_fn: str | None = None
    # temperature to calculate the ALM loss with.
    binarization_temp: float = 100.0
    # DEPRECATED distillation chunk sizes. should always be one.
    distill_chunk_sizes: list[int] = field(default_factory=lambda: [1])
    # ALM loss distance function, e.g. "binary_ce" for binary cross-entropy.
    alm_diff_fn: str = "binary_ce"
    # ALM loss numerator. usually keep "chunk_count".
    distill_main_path_numerator: str = "chunk_count"
    # ALM loss denominator. usually keep "chunk_count".
    distill_main_path_denominator: str = "chunk_count"
    # model training mode: "lora" to only train LoRA adapters, "full" to train the full model instead.
    train_model_mode: str = "lora"
    # LoRA rank.
    model_lora_rank: int = 64
    # LoRA alpha scaling factor.
    model_lora_alpha: int = 64
    # whether to train or freeze the input embeddings. If a hypernet is used, and train_embeddings=False, the embeddings will still be
    # updated (through the hypernet predictions), but not trained directly.
    train_embeddings: bool = True
    # tokens to add to the target tokenizer.
    tokens_to_add: list[str] | None = None
    # which latents to align, e.g. "last_hidden_state" for the last hidden state of the model.
    latents_to_align: str = "last_hidden_state"
    # loss function to use for latent alignment.
    latents_normalization: str = "l2_channelwise"
    # whether to use a naive or a more complex chunking strategy for the latents, usually "naive".
    latents_chunks: str = "naive"
    # whether to project the latents, necessary if using the latents loss in a non-self-distillation setting.
    # CAREFUL: we have not observed it helping in this setting, it is probably better to disable the latent loss when not self-distilling.
    latents_do_project: bool = False
    # ALM loss mode, "append_space" means debiasing, should usually be added (see https://arxiv.org/abs/2503.20083).
    # "merge_by_space_prob" means joining chunks such that the endings have a debiasing probability above the threshold, should usually be added.
    alm_mode: str = "merge_by_space_prob+append_space"
    # which bytes to assume to not cross token boundaries for debiasing.
    space_mask_mode: str = "space+tab+newline+special"
    # path to tokenizer data directory, needed for e.g. the MinED baseline, but not necessary for the default ALM setting.
    tokenizer_pair_data_path: str | None = None
    # chunk threshold, used for `merge_by_space_prob` chunk combination.
    tokenizer_pair_bias_threshold: float = 0.1
    # whether to expand the input IDs for conversion to the byte-level (see "Adjustments for Transfer to Bytes" section in https://arxiv.org/abs/2503.20083).
    expand_input_ids: bool = False
    # GCP bucket to export the model checkpoints to. Only the name without the prefix, e.g. "my-bucket".
    export_to_gcs_bucket: str | None = None
    # Dataset to use for perplexity evaluation. LM harness evaluation is usually more informative.
    ppl_eval_data: dict[str, Any] | None = None

    # Device placement (single string per model, e.g. "cuda:0", "cuda:1", "cpu").
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"

    # ── MTA (Multi-Teacher Alignment) span loss config ────────────────────────
    # Master switch: must be true if "mta" is in `losses`.
    mta_mode: bool = False
    # Teacher hidden-layer indices used for each projector pair.
    teacher_layer_mapping: list[int] = field(default_factory=lambda: [6, 12, 18, 24])
    # Student hidden-layer indices used for each projector pair.
    student_layer_mapping: list[int] = field(default_factory=lambda: [3, 6, 9, 12])
    # Slice points [start, word_end, span_end, total] dividing projectors into
    # word-level vs span-level groups.
    split_layer_mapping: list[int] = field(default_factory=lambda: [0, 1, 4, 4])
    # Standalone weight (legacy from MTA paper); main scaling should go through `loss_weights`.
    w_span_loss: float = 2.0
    # Use teacher entropy as aggregation weight (lower entropy ⇒ higher weight).
    entropy_weight: bool = False
    # If True, MSE is unweighted (mean over pairs) instead of teacher-weight weighted.
    wo_span_weight: bool = False


def cross_entropy(
    logits,
    labels,
    attention_mask,
    logits_already_shifted=False,
    logit_mask=None,
    denom=None,
):
    # 1. Shifting (Giữ nguyên logic của bạn)
    if not logits_already_shifted:
        shift_logits = logits[..., :-1, :].contiguous() # [B, L-1, V]
        shift_labels = labels[..., 1:].contiguous()     # [B, L-1]
        shift_attention_mask = attention_mask[..., 1:].contiguous() # [B, L-1]
    else:
        shift_logits = logits
        shift_labels = labels
        shift_attention_mask = attention_mask

    # 2. Logit Masking (Giữ nguyên)
    if logit_mask is not None:
        shift_logits = shift_logits + logit_mask.view(1, 1, -1)

    # 3. Tính Loss bằng F.cross_entropy (Tiết kiệm memory nhất)
    # Flatten logits về [N, V] và labels về [N] để đưa vào hàm loss
    vocab_size = shift_logits.size(-1)
    
    # reduction='none' để lấy loss từng token, sau đó mới nhân mask
    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size), 
        shift_labels.view(-1), 
        reduction='none'
    ) # Kết quả shape: [B * (L-1)]

    # Reshape lại về [B, L-1] để áp mask
    token_losses = token_losses.view(shift_labels.size())

    # 4. Masking & Reduction
    # Chuyển mask sang float
    mask_float = shift_attention_mask.float()
    
    # Chỉ tính loss trên các token thật (mask = 1)
    masked_loss = token_losses * mask_float
    
    # Tính tổng loss rồi chia cho tổng số token thực (Safe division)
    if denom is None:
        denom = mask_float.sum() + 1e-5 # Cộng epsilon để tránh chia cho 0
        
    loss = masked_loss.sum() / denom

    return loss

def get_last_index_per_column(matrix):
    matrix = matrix.bool()
    matrix_last_only = matrix.clone()
    matrix_last_only[:, :-1] = matrix[:, :-1] & (~matrix[:, 1:])
    last_only_index = matrix_last_only.long().argmax(dim=-2) 
    mask = matrix_last_only.any(dim=-2)
    return last_only_index, mask

def compute_mta_loss(args, loss_args):
    """MTA (Multi-Teacher Alignment) span relational-similarity loss.

    Relies on:
      - `loss_args.student_out.hidden_states` / `teacher_out.hidden_states`
        (set `need_hidden_states` accordingly).
      - `loss_args.batch['offset_mapping_new' | 'offset_mapping_original']`
        emitted by `TokenizerAlignerCollator`.
      - `loss_args.batch['spans_char_offsets' | 'words_char_offsets']`
        precomputed by `precompute_spans.py`.
      - `loss_args.mta_projector_list` -- ModuleList of `Linear(D_student, D_teacher)`.
    """
    return compute_overall_span_loss(
        projectors=loss_args.mta_projector_list,
        s_att_mask=loss_args.batch["attention_mask_new"],
        t_att_mask=loss_args.batch["attention_mask_original"],
        s_logits=loss_args.student_logits,
        t_logits=loss_args.teacher_logits,
        s_hidden_states=loss_args.student_out.hidden_states,
        t_hidden_states=loss_args.teacher_out.hidden_states,
        s_offsets_mapping=loss_args.batch["offset_mapping_new"],
        t_offsets_mapping=loss_args.batch["offset_mapping_original"],
        spans_offsets=loss_args.batch["spans_char_offsets"],
        words_offsets=loss_args.batch["words_char_offsets"],
        args=args,
    )


def compute_alm_latents_loss(args, loss_args, epsilon=1e-5):
    if args.latents_chunks == "naive":
        alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
            loss_args.batch["alignment_matrix_b_unconstrained"]
        )
        alignment_matrix_a_last_only_index, mask = get_last_index_per_column(
            loss_args.batch["alignment_matrix_a_unconstrained"]
        )
    elif args.latents_chunks == "space":
        alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
            loss_args.batch["alignment_matrix_b_space"]
        )
        alignment_matrix_a_last_only_index, mask = get_last_index_per_column(
            loss_args.batch["alignment_matrix_a_space"]
        )

    if "last_hidden_state" in args.latents_to_align:
        layer_indices = [(-1, -1)]
    else:
        layer_indices = []

    hidden_state_latent_loss = 0.0

    for teacher_idx, student_idx in layer_indices:
        t_hidden = loss_args.teacher_out.hidden_states[teacher_idx]
        s_hidden = loss_args.student_out.hidden_states[student_idx]
        
        # Teacher Gather
        t_dim = t_hidden.size(-1)
        t_idx_expanded = alignment_matrix_b_last_only_index.unsqueeze(-1).expand(-1, -1, t_dim)
        t_aligned_last_hidden_state = torch.gather(t_hidden, 1, t_idx_expanded)

        # Student Gather
        s_dim = s_hidden.size(-1)
        s_idx_expanded = alignment_matrix_a_last_only_index.unsqueeze(-1).expand(-1, -1, s_dim)
        s_aligned_last_hidden_state = torch.gather(s_hidden, 1, s_idx_expanded)

        if args.latents_do_project:
            s_aligned_last_hidden_state = loss_args.projector_latents(s_aligned_last_hidden_state)

        elementwise_layer_latent_loss = torch.square(
            s_aligned_last_hidden_state - t_aligned_last_hidden_state
        )

        # Masking
        # mask: [Batch, Target_Len] -> unsqueeze thành [Batch, Target_Len, 1]
        mask_unsqueezed = mask.unsqueeze(-1).float()
        layer_latent_loss = elementwise_layer_latent_loss * mask_unsqueezed

        # JAX: square(t * mask).mean([0, 1]) -> Mean over Batch and Seq dims
        denom_numerator = torch.square(t_aligned_last_hidden_state * mask_unsqueezed).mean(dim=(0, 1), keepdim=True)
        denom_denominator = mask.float().mean() # Mean scalar
        
        normalization_factor = (denom_numerator / denom_denominator) + epsilon
        
        layer_latent_loss = layer_latent_loss / normalization_factor
        layer_latent_loss = layer_latent_loss.mean()
        
        hidden_state_latent_loss += layer_latent_loss / len(layer_indices)

    loss = hidden_state_latent_loss
    return loss

def get_large_negative_number(dtype: torch.dtype):
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                   torch.uint8):
        dtype_max = torch.iinfo(dtype).max
    else:
        raise ValueError("Unsupported dtype for inputs.")

    return torch.tensor(-0.7 * dtype_max, dtype=dtype)

def log1mexp(x):
    """Computes log(1 - exp(x)) in a numerically stable way for x < 0."""
    # For x < log(0.5), use log1p(-exp(x)) directly
    # For x >= log(0.5), use log(-expm1(x)) to avoid precision issues
    log_half = -torch.log(torch.tensor(2, device=x.device))
    return torch.where(x < log_half, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))

def compute_alm_loss(chunk_kind, args, loss_args: LossArgs, epsilon=1e-5):
    device = loss_args.batch["input_ids_original"].device
    original_shift_labels = loss_args.batch["input_ids_original"][..., 1:]

    if chunk_kind == "unconstrained":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_unconstrained"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_unconstrained"]
    else:
        raise ValueError(f"Unknown chunk kind: {chunk_kind}")

    def binary_ce(log_y_true, log_y_pred):
        log_y_true = (log_y_true.to(torch.float32) / args.binarization_temp) - epsilon
        log_y_pred = (log_y_pred.to(torch.float32) / args.binarization_temp) - epsilon

        return -(
            torch.exp(log_y_true) * log_y_pred
            + (-torch.expm1(log_y_true) * log1mexp(log_y_pred))
        )

    diff_fn = binary_ce

    alignment_matrix_a = alignment_matrix_a * loss_args.batch["loss_mask_new"][:, :, None]
    alignment_matrix_b = alignment_matrix_b * loss_args.batch["loss_mask_original"][:, :, None]

    alignment_matrix_b_last_only_index, _ = get_last_index_per_column(alignment_matrix_b)
    alignment_matrix_a_last_only_index, mask = get_last_index_per_column(alignment_matrix_a)
    
    teacher_main_path_logprobs = torch.take_along_dim(
            loss_args.teacher_logprobs[:, :-1],
            original_shift_labels[..., None],
            dim=-1,
        ).squeeze(-1)
    t_aligned_main_logp = torch.clamp(
        (teacher_main_path_logprobs[:, None] @ alignment_matrix_b[:, 1:].float()).squeeze(1),
        max=0.0
    )

    if "eos_as_space" in args.alm_mode:
        t_space_logp = loss_args.teacher_logprobs[
            :, :, loss_args.tokenizer_teacher.eos_token_id
        ]
    else:
        t_space_logp = torch.clamp(
            torch.log(
                torch.matmul(loss_args.teacher_probs, loss_args.space_mask_teacher.float())
            ),
            max=0.0
        )
    t_aligned_space_logp = torch.take_along_dim(
        t_space_logp, alignment_matrix_b_last_only_index, dim=-1
    )

    new_shift_labels = loss_args.batch["input_ids_new"][..., 1:]
    student_main_path_logprobs = torch.take_along_dim(
            loss_args.student_logprobs[:, :-1],
            new_shift_labels[..., None],
            dim=-1,
        ).squeeze(-1)

    s_aligned_main_logp = torch.clamp(
        (student_main_path_logprobs[:, None] @ alignment_matrix_a[:, 1:].float()).squeeze(1),
        max=0.0
    )

    if "eos_as_space" in args.alm_mode:
        s_space_logp = loss_args.student_logprobs[
            :, :, loss_args.tokenizer_new.eos_token_id
        ]
    else:
        vocab_size = loss_args.student_probs.shape[-1]

        s_space_logp = torch.clamp(
            torch.log(torch.matmul(loss_args.student_probs, loss_args.space_mask_new.float()[:vocab_size])),
            max=0.0
        )

    s_aligned_space_logp = torch.take_along_dim(
        s_space_logp, alignment_matrix_a_last_only_index, dim=-1
    )

    aligned_count = alignment_matrix_b[:, 1:].sum(-2)
    global_aligned_count = alignment_matrix_b[:, 1:].sum(-2)


    if "merge_by_space_prob" in args.alm_mode:
        batch_size = t_aligned_space_logp.shape[0]
        chunk_count = t_aligned_space_logp.shape[-1]

        t_aligned_space_chunk_mask = (
            torch.exp(t_aligned_space_logp) > args.tokenizer_pair_bias_threshold
        )
        cumsum_mask = torch.cumsum(t_aligned_space_chunk_mask.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        chunk_merging_indices = cumsum_mask.max(dim=-1, keepdim=True).values - cumsum_mask
        chunk_merging_values = (aligned_count > 0).float()
        chunk_merging_matrix = torch.zeros(
            (batch_size * chunk_count, chunk_count), 
            dtype=alignment_matrix_a.dtype, 
            device=device
        )
        row_indices = torch.arange(batch_size * chunk_count, device=device)
        col_indices = chunk_merging_indices.reshape(-1).long() # index cột
        chunk_merging_matrix[row_indices, col_indices] = chunk_merging_values.reshape(-1).to(chunk_merging_matrix.dtype)
        chunk_merging_matrix = chunk_merging_matrix.reshape(batch_size, chunk_count, chunk_count)
        chunk_merging_matrix_last_only_index, _ = get_last_index_per_column(
            chunk_merging_matrix
        )

        t_aligned_main_logp = torch.matmul(t_aligned_main_logp[:, None], chunk_merging_matrix.float()).squeeze(1)
        s_aligned_main_logp = torch.matmul(s_aligned_main_logp[:, None], chunk_merging_matrix.float()).squeeze(1)

        t_aligned_space_logp = torch.take_along_dim(
            t_aligned_space_logp, chunk_merging_matrix_last_only_index, dim=-1
        )
        s_aligned_space_logp = torch.take_along_dim(
            s_aligned_space_logp, chunk_merging_matrix_last_only_index, dim=-1
        )

        global_aligned_count = aligned_count = torch.matmul(aligned_count[:, None].float(), chunk_merging_matrix.float()).squeeze(1)


    valid_mask = (aligned_count > 0).float()
    s_aligned_main_logp = s_aligned_main_logp * valid_mask
    t_aligned_space_logp = t_aligned_space_logp * valid_mask
    s_aligned_space_logp = s_aligned_space_logp * valid_mask

    all_aligned_s_logps = []
    all_aligned_t_logps = []
    all_aligned_counts = []
    global_all_aligned_counts = []

    batch_size = loss_args.batch["input_ids_original"].shape[0]
    global_batch_size = loss_args.batch["input_ids_original"].shape[0]

    for size in args.distill_chunk_sizes:
        size_s_logp = s_aligned_main_logp.view(batch_size, -1, size).sum(-1)
        size_t_logp = t_aligned_main_logp.view(batch_size, -1, size).sum(-1)
        size_count = aligned_count.view(batch_size, -1, size)
        global_size_count = global_aligned_count.view(global_batch_size, -1, size)

        # if "append_space" in args.alm_mode:
        #     mask = (size_count > 0)
        #     reversed_mask = torch.flip(mask, dims=[-1])
        #     reversed_cumsum = torch.cumsum(reversed_mask.long(), dim=-1)
        #     last_position_in_chunk = torch.flip(reversed_cumsum == 1, dims=[-1])

        #     size_s_logp = size_s_logp + (
        #         s_aligned_space_logp.view(batch_size, -1, size)
        #         * last_position_in_chunk.float()
        #     ).sum(-1)
        #     size_t_logp = size_t_logp + (
        #         t_aligned_space_logp.view(batch_size, -1, size)
        #         * last_position_in_chunk.float()
        #     ).sum(-1)

        all_aligned_s_logps.append(size_s_logp)
        all_aligned_t_logps.append(size_t_logp)
        all_aligned_counts.append(size_count.sum(-1))
        global_all_aligned_counts.append(global_size_count.sum(-1))

    s_full_aligned_main_logp = torch.cat(all_aligned_s_logps, -1)
    t_full_aligned_main_logp = torch.cat(all_aligned_t_logps, -1)
    full_aligned_counts = torch.cat(all_aligned_counts, -1)
    global_full_aligned_counts = torch.cat(global_all_aligned_counts, -1)

    
    t_full_aligned_main_logp = torch.where(
        full_aligned_counts > 0,
        t_full_aligned_main_logp,
        get_large_negative_number(torch.float16),
    )
    s_full_aligned_main_logp = torch.where(
        full_aligned_counts > 0,
        s_full_aligned_main_logp,
        get_large_negative_number(torch.float16),
    )

    if args.distill_main_path_numerator == "token_count":
        numerator = full_aligned_counts
    elif args.distill_main_path_numerator == "chunk_count":
        numerator = full_aligned_counts > 0

    if args.distill_main_path_denominator == "token_count":
        denominator = global_full_aligned_counts.mean()
    elif args.distill_main_path_denominator == "chunk_count":
        denominator = (global_full_aligned_counts > 0).float().mean()

    # elementwise_loss = (
    #     diff_fn(t_full_aligned_main_logp, s_full_aligned_main_logp)
    #     * numerator
    #     / denominator
    # )
    elementwise_loss = (
        (diff_fn(t_full_aligned_main_logp, s_full_aligned_main_logp) * numerator).mean() 
        / numerator.float().mean()
    )

    if torch.isnan(elementwise_loss.mean()):
        print(loss_args.batch["alignment_matrix_b_unconstrained"].sum())
        print(loss_args.batch["alignment_matrix_b_unconstrained"][:, 1:].sum())
        print(chunk_merging_matrix.float().mean())
        print(t_aligned_main_logp.mean())
        print(s_aligned_main_logp.mean())
        print(student_main_path_logprobs.mean())
        print(alignment_matrix_a[:, 1:].float().mean())
        print(t_full_aligned_main_logp.mean())
        print(s_full_aligned_main_logp.mean())
        print(diff_fn(t_full_aligned_main_logp, s_full_aligned_main_logp) * numerator)
        print(denominator)
        print((global_full_aligned_counts > 0).float().sum(-1))
        print(numerator.mean())

    distill_main_path_loss = elementwise_loss.mean() / len(args.distill_chunk_sizes)

    return distill_main_path_loss


def main(args: CrossTokenizerDistillArgs):
    logger.info(pformat(args))

    output_dir = Path(args.output)
    # clear previous output dir
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "args.yaml", "w") as f:
        yaml.dump(asdict(args), f)

    # prepare dataset
    dataset = data.get_dataset(**args.data, seed=args.seed)

    teacher_model_kwargs = (
        asdict(args.teacher) if args.teacher is not None else asdict(args.student)
    )
    teacher_tokenizer_name = teacher_model_kwargs.pop("tokenizer_name")
    target_tokenizer_name = args.target_tokenizer_name

    tokenizer_teacher = load_byteify_tokenizer(teacher_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(target_tokenizer_name)

    # tokenizer_teacher = AutoTokenizer.from_pretrained(teacher_tokenizer_name.split(":")[0])
    # tokenizer_teacher.pad_token = tokenizer_teacher.eos_token

    # # target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name.split(":")[0])
    # target_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    # target_tokenizer.pad_token = target_tokenizer.eos_token

    if args.tokens_to_add is not None:
        logger.info("Adding tokens: %s", args.tokens_to_add)
        target_tokenizer.add_tokens(args.tokens_to_add)
       

    logger.info("GPU layout: student=%s teacher=%s",
                args.student_device, args.teacher_device)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.teacher_device,
    )

    if args.train_model_mode == 'lora':
        new_model = AutoModelForCausalLM.from_pretrained(
            args.student.pretrained_model_name_or_path,
            device_map=args.student_device,
            torch_dtype=torch.bfloat16,
        )
        if target_tokenizer_name.split("=")[1] == 'GPT2':
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]
        lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=256,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=target_modules
            )
        new_model = get_peft_model(new_model, lora_config).to(new_model.device)
        new_model.print_trainable_parameters()
    else:
        new_model = AutoModelForCausalLM.from_pretrained(
            args.student.pretrained_model_name_or_path,
            device_map=args.student_device,
        )

    device = new_model.device

    space_mask_teacher = torch.from_numpy(utils.get_space_mask(tokenizer_teacher, args.space_mask_mode)).to(device)
    if space_mask_teacher.size(0) < teacher_model.config.vocab_size:
        padding_len = teacher_model.config.vocab_size - space_mask_teacher.size(0)
        space_mask_teacher = F.pad(space_mask_teacher, (0, padding_len), value=0)
        
    space_mask_new = torch.from_numpy(utils.get_space_mask(target_tokenizer, args.space_mask_mode)).to(device)
    if space_mask_new.size(0) < new_model.config.vocab_size:
        padding_len = new_model.config.vocab_size - space_mask_new.size(0)
        space_mask_new = F.pad(space_mask_new, (0, padding_len), value=0)

    
    expand_input_ids_dict = None

    collator = TokenizerAlignerCollator(
        tokenizer_teacher,
        target_tokenizer,
        max_teacher_length=args.max_teacher_length,
        max_student_length=args.max_student_length,
        use_chat_template=args.use_chat_template,
        chat_template_mode=args.chat_template_mode,
        expand_input_ids_dict=expand_input_ids_dict,
        loss_mask_mode=args.loss_mask_mode,
        tokenizer_pair_data_path=args.tokenizer_pair_data_path,
        tokenizer_pair_bias_threshold=args.tokenizer_pair_bias_threshold,
        require_bias_matrices=any("unbiased" in x for x in args.losses),
    )

    train_dataloader = DataLoader(
        dataset.get_torch_dataset(),
        batch_size=1,  # batched internally
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=True
    )


    def train_step(batch):
        scalar_report = {}  # extra logging

        need_teacher = len([loss for loss in args.losses if loss != "sft"]) > 0
        # MTA also requires hidden states from both models.
        need_hidden_states = any(
            loss in {"alm_latents", "baseline_dskd", "mta"} for loss in args.losses
        )

        teacher_out = None
        teacher_probs = None
        teacher_logprobs = None
        teacher_logits = None

        if need_teacher:
            with torch.no_grad():
                teacher_out = teacher_model(
                    input_ids=batch["input_ids_original"].to(teacher_model.device),
                    attention_mask=batch["attention_mask_original"].to(teacher_model.device),
                    output_hidden_states=need_hidden_states
                )
            teacher_logits = teacher_out.logits.to(device)

            # Log Softmax & Clipping
            teacher_logprobs = torch.clamp(
                F.log_softmax(teacher_logits, dim=-1), max=0.0
            )
            teacher_probs = torch.exp(teacher_logprobs)
        else:
            teacher_out = teacher_logits = teacher_logprobs = teacher_probs = None

        student_out = new_model(
            input_ids=batch["input_ids_new"],
            attention_mask=batch["attention_mask_new"],
            output_hidden_states=need_hidden_states
        )
        student_logits = student_out.logits

        if torch.isnan(student_logits).any():
            print(batch["input_ids_new"].max(), batch["input_ids_new"].min())
            print(new_model.config.vocab_size)
            print('student_logits nan')

        student_logprobs = torch.clamp(
            F.log_softmax(student_logits, dim=-1), max=0.0
        )
        student_probs = torch.exp(student_logprobs)


        loss_args = LossArgs(
            projector_latents=projector_latents,
            batch=batch,
            teacher_out=teacher_out,
            student_out=student_out,
            tokenizer_teacher=tokenizer_teacher,
            tokenizer_new=target_tokenizer,
            teacher_probs=teacher_probs,
            teacher_logprobs=teacher_logprobs,
            teacher_logits=teacher_logits,
            student_probs=student_probs,
            student_logprobs=student_logprobs,
            student_logits=student_logits,
            scalar_report=scalar_report,
            space_mask_teacher=space_mask_teacher,
            space_mask_new=space_mask_new,
            mta_projector_list=mta_projector_list,
        )

        total_loss = 0.0

        for loss_idx, loss in enumerate(args.losses):
            if loss == "sft":
                labels = batch["input_ids_new"].clone().detach()
                labels.masked_fill_(batch["loss_mask_new"] == 0, -100)
                current_loss = new_model.loss_function(
                    loss_args.student_logits,
                    labels,
                    new_model.config.vocab_size
                )
            elif loss == "alm_latents":
                current_loss = compute_alm_latents_loss(args, loss_args)
            elif loss.startswith("alm"):
                kind = loss[len("alm_") :]
                if len(kind) == 0:
                    kind = "unbiased"
                current_loss = 3 * compute_alm_loss(
                    chunk_kind=kind,
                    args=args,
                    loss_args=loss_args,
                )
            elif loss == "mta":
                current_loss = compute_mta_loss(args, loss_args)
            else:
                raise ValueError(f"Invalid loss: {loss}")

            weight = args.loss_weights[loss_idx] if args.loss_weights else 1.0
    
            if args.loss_schedules:
                schedule_type = args.loss_schedules[loss_idx]
                progress = step / args.steps
                
                if schedule_type == "cosine":
                    weight = weight * (1 + math.cos(math.pi * progress)) / 2
                elif schedule_type == "reverse_cosine":
                    weight = weight * (1 - math.cos(math.pi * progress)) / 2
                elif schedule_type == "linear":
                    weight = weight * progress
                elif schedule_type == "constant":
                    pass
                else:
                    raise ValueError(f"Invalid schedule: {schedule_type}")

            scalar_report[f"loss/{loss}"] = current_loss
            # scalar_report[f"loss/{loss}_weight"] = weight

            total_loss += weight * current_loss
 
        return total_loss, scalar_report

 
    diter = iter(train_dataloader)
    # first_batch = next(iter(train_dataloader))

    # utils.print_example_alignments(
    #     first_batch["alignment_matrix_b_unconstrained"][0],
    #     first_batch["alignment_matrix_a_unconstrained"][0],
    #     tokenizer_teacher.convert_ids_to_tokens(first_batch["input_ids_original"][0]),
    #     target_tokenizer.convert_ids_to_tokens(first_batch["input_ids_new"][0]),
    # )

    print("train start")

    new_model.train()
    
    optimizer = torch.optim.AdamW(new_model.parameters(), 
                                  lr=args.optimizer['learning_rate'], 
                                #   betas=(args.optimizer['b1'], args.optimizer['b2']), 
                                #   eps=args.optimizer['eps'], 
                                #   weight_decay=args.optimizer['weight_decay']
                                  )
    

    projector_latents = None
    mta_projector_list = None

    # Helper: read hidden size in a way that works for both GPT2 and Qwen/Llama configs.
    def _hidden_size(cfg):
        return getattr(cfg, "n_embd", None) or cfg.hidden_size

    if teacher_model is not None and args.latents_do_project:
        teacher_hidden_size = _hidden_size(teacher_model.config)
        student_hidden_size = _hidden_size(new_model.config)

        projector_latents = torch.nn.Linear(student_hidden_size, teacher_hidden_size)
        projector_latents = projector_latents.to(device)

        optimizer.add_param_group(
            {"params": projector_latents.parameters(), "lr": 2 * args.optimizer['learning_rate']}
        )

    if "mta" in args.losses:
        assert args.mta_mode, "Set `mta_mode: true` in config when using 'mta' loss."
        assert teacher_model is not None, "MTA requires a teacher model."
        assert len(args.teacher_layer_mapping) == len(args.student_layer_mapping), \
            "teacher_layer_mapping and student_layer_mapping must have equal length"
        assert args.split_layer_mapping[2] == len(args.teacher_layer_mapping), \
            "split_layer_mapping[2] must equal len(teacher_layer_mapping)"

        teacher_hidden_size = _hidden_size(teacher_model.config)
        student_hidden_size = _hidden_size(new_model.config)
        n_pairs = len(args.teacher_layer_mapping)
        # Each projector aligns one student layer to its paired teacher layer's dim.
        mta_projector_list = torch.nn.ModuleList([
            torch.nn.Linear(student_hidden_size, teacher_hidden_size)
            for _ in range(n_pairs)
        ]).to(device)

        optimizer.add_param_group(
            {"params": mta_projector_list.parameters(),
             "lr": 2 * args.optimizer['learning_rate']}
        )
        logger.info(
            "MTA projectors: %d × Linear(%d, %d) on %s",
            n_pairs, student_hidden_size, teacher_hidden_size, device
        )


    num_training_steps = args.steps
    num_warmup_steps = args.warmup_steps

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scaler = GradScaler()

    total_loss = 0.0
    total_sft_loss = 0.0


    for step in tqdm(range(args.steps)):
        try:
            batch = next(diter)
        except StopIteration:
            new_model.save_pretrained(args.output + f"/{step}", state_dict=new_model.state_dict())
            diter = iter(train_dataloader)
            batch = next(diter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if args.dry_run:
            continue      

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, step_metrics = train_step(batch)

        scaler.scale(loss).backward()

        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(new_model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)          

        total_loss += loss.item()
        total_sft_loss += step_metrics['loss/sft'].item()

        # train_metrics.append(step_metrics)
        if (step + 1) % args.log_interval == 0:
            step_metrics['loss_avg'] = total_loss / args.log_interval
            step_metrics['sft_loss_avg'] = total_sft_loss / args.log_interval
            total_loss = 0.0
            total_sft_loss = 0.0
            step_metrics['lr'] = scheduler.get_last_lr()[0]
            print(step_metrics)
                    
        if (step + 1) % args.eval_interval == 0 or (
            step == 0 and args.eval_at_step_zero
        ):
            # TODO: probably extract into eval function doing everything here
            pass


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "100"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = (
        True  # careful about this, required for lm_eval
    )

    main(parse_args.parse_args(CrossTokenizerDistillArgs))
