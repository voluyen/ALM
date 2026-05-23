import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from tokenkit import align, utils


class TokenizerAlignerCollator:
    def __init__(
        self,
        tokenizer_original,
        tokenizer_new,
        max_teacher_length,
        max_student_length,
        use_chat_template=False,
        chat_template_mode="direct_encode",
        expand_input_ids_dict=None,
        loss_mask_mode=None,
        tokenizer_pair_data_path=None,
        tokenizer_pair_bias_threshold=0.0,
        require_bias_matrices=False,
    ):
        self.tokenizer_original = tokenizer_original
        self.tokenizer_original_vocab = tokenizer_original.get_vocab()
        self.tokenizer_new = tokenizer_new
        self.max_teacher_length = max_teacher_length
        self.max_student_length = max_student_length
        self.use_chat_template = use_chat_template
        self.chat_template_mode = chat_template_mode
        self.expand_input_ids_dict = expand_input_ids_dict

        if loss_mask_mode is None:
            loss_mask_string = None
        elif loss_mask_mode == "dolly":
            loss_mask_string = "### Response:\n"
        elif loss_mask_mode == "openmath2":
            loss_mask_string = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise ValueError(f"Unknown loss mask mode: {loss_mask_mode}")

        self.loss_mask_tokens_original = (
            self.tokenizer_original.encode(loss_mask_string, add_special_tokens=False)
            if loss_mask_string is not None
            else None
        )
        self.loss_mask_tokens_new = (
            self.tokenizer_new.encode(loss_mask_string, add_special_tokens=False)
            if loss_mask_string is not None
            else None
        )

        bias1_matrix_path = Path(tokenizer_pair_data_path) / "bias1_matrix.npz"
        bias2_matrix_path = Path(tokenizer_pair_data_path) / "bias2_matrix.npz"
        teacher_token_counts_path = (
            Path(tokenizer_pair_data_path) / "teacher_counts.json"
        )
        student_token_counts_path = (
            Path(tokenizer_pair_data_path) / "student_counts.json"
        )

        if bias1_matrix_path.exists():
            self.tokenizer_pair_bias1_matrix = sparse.load_npz(
                bias1_matrix_path
            ).todok()
        else:
            self.tokenizer_pair_bias1_matrix = None
        if bias2_matrix_path.exists():
            self.tokenizer_pair_bias2_matrix = sparse.load_npz(
                bias2_matrix_path
            ).todok()
        else:
            self.tokenizer_pair_bias2_matrix = None
        if teacher_token_counts_path.exists():
            self.teacher_token_probs = utils.compute_unigram_probabilities(
                tokenizer_original, json.load(open(teacher_token_counts_path))
            )
        else:
            self.teacher_token_probs = None
        if student_token_counts_path.exists():
            self.student_token_probs = utils.compute_unigram_probabilities(
                tokenizer_new, json.load(open(student_token_counts_path))
            )
        else:
            self.student_token_probs = None

        if require_bias_matrices and (
            self.tokenizer_pair_bias1_matrix is None
            or self.tokenizer_pair_bias2_matrix is None
        ):
            raise ValueError(
                "Bias matrices are required but not found in the given path."
            )

        self.tokenizer_pair_bias_threshold = tokenizer_pair_bias_threshold

        self.prefix_map_original = self._compute_prefix_map(tokenizer_original)
        self.prefix_map_new = self._compute_prefix_map(tokenizer_new)

    def _compute_loss_mask(self, input_ids, attention_mask, loss_mask_tokens):
        loss_mask = attention_mask.astype(bool)
        input_lengths = attention_mask.sum(axis=1)
        if loss_mask_tokens is not None:
            for i in range(len(input_ids)):
                if input_lengths[i] < len(loss_mask[i]):
                    loss_mask[i, input_lengths[i]] = True
                for j in range(len(input_ids[i])):
                    if input_ids[i][j] != loss_mask_tokens[0]:
                        continue

                    if (
                        input_ids[i][j : j + len(loss_mask_tokens)].tolist()
                        == loss_mask_tokens
                    ):
                        loss_mask[i, : j + len(loss_mask_tokens)] = False

        return loss_mask

    def _compute_prefix_map(self, tokenizer):
        prefix_map = {}

        for token in tokenizer.get_vocab().keys():
            for i in range(1, len(token) + 1):
                if token[:i] in prefix_map:
                    prefix_map[token[:i]].append(token)
                else:
                    prefix_map[token[:i]] = [token]

        return prefix_map

    def _encode_with_chat_template(self, texts, tokenizer, max_length):
        input_ids = np.full(
            (len(texts), max_length), fill_value=tokenizer.pad_token_id, dtype=np.int32
        )
        attention_mask = np.zeros((len(texts), max_length), dtype=np.int32)

        for i in range(len(texts)):
            current_input_ids, _ = utils.encode_prompt(
                utils.preprocess_prompt(texts[i], self.chat_template_mode),
                tokenizer,
                max_length=max_length,
            )
            input_ids[i, : len(current_input_ids)] = current_input_ids
            attention_mask[i, : len(current_input_ids)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __call__(self, examples):
        # batched internally
        examples = examples[0]

        texts = examples["text"]

        # Pass-through precomputed MTA spans (default empty if dataset lacks them).
        spans_char_offsets = examples.get(
            "spans_char_offsets", [[] for _ in texts]
        )
        words_char_offsets = examples.get(
            "words_char_offsets", [[] for _ in texts]
        )

        # NOTE: offset_mapping requires fast tokenizers (Rust-backed).
        # MTA loss uses these offsets to map char spans -> token indices.
        encoding_original = self.tokenizer_original(
            texts,
            max_length=self.max_teacher_length,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_offsets_mapping=True,
        )
        encoding_new = self.tokenizer_new(
            texts,
            max_length=self.max_student_length,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_offsets_mapping=True,
        )

        input_ids_original = encoding_original["input_ids"]
        attention_mask_original = encoding_original["attention_mask"]
        offset_mapping_original = encoding_original["offset_mapping"]
        input_ids_new = encoding_new["input_ids"]
        attention_mask_new = encoding_new["attention_mask"]
        offset_mapping_new = encoding_new["offset_mapping"]

        # alignment_matrix_a = None
        # alignment_matrix_b = None
        # alignment_matrix_a_space = None
        # alignment_matrix_b_space = None
        # alignment_matrix_a_unbiased = None
        # alignment_matrix_b_unbiased = None

        (
            alignment_matrix_a,
            alignment_matrix_b,
        ) = align.get_unconstrained_alignments(
            input_ids_original,
            input_ids_new,
            attention_mask_original,
            attention_mask_new,
            tokenizer_teacher=self.tokenizer_original,
            tokenizer_student=self.tokenizer_new,
        )

        (
            alignment_matrix_a_space,
            alignment_matrix_b_space,
        ) = align.get_space_alignments(
            input_ids_original,
            input_ids_new,
            attention_mask_original,
            attention_mask_new,
            tokenizer_teacher=self.tokenizer_original,
            tokenizer_student=self.tokenizer_new,
        )

        if (
            self.tokenizer_pair_bias1_matrix is not None
            and self.tokenizer_pair_bias2_matrix is not None
        ):
            (
                alignment_matrix_a_unbiased,
                alignment_matrix_b_unbiased,
            ) = align.get_unbiased_alignments(
                input_ids_original,
                input_ids_new,
                attention_mask_original,
                attention_mask_new,
                tokenizer_teacher=self.tokenizer_original,
                tokenizer_student=self.tokenizer_new,
                pair_data=(
                    self.tokenizer_pair_bias1_matrix,
                    self.tokenizer_pair_bias2_matrix,
                    self.teacher_token_probs,
                    self.student_token_probs,
                ),
                bias_threshold=self.tokenizer_pair_bias_threshold,
            )
        else:
            alignment_matrix_a_unbiased = np.full_like(
                alignment_matrix_a, fill_value=np.nan
            )
            alignment_matrix_b_unbiased = np.full_like(
                alignment_matrix_b, fill_value=np.nan
            )

        occuring_tokens_mask_original = np.zeros(
            len(self.tokenizer_original), dtype=bool
        )
        occuring_tokens_mask_new = np.zeros(len(self.tokenizer_new), dtype=bool)

        occuring_tokens_mask_original[input_ids_original.flatten()] = True
        occuring_tokens_mask_new[input_ids_new.flatten()] = True

        loss_mask_original = self._compute_loss_mask(
            input_ids_original, attention_mask_original, self.loss_mask_tokens_original
        )
        loss_mask_new = self._compute_loss_mask(
            input_ids_new, attention_mask_new, self.loss_mask_tokens_new
        )

        batch = {
            "input_ids_new": input_ids_new,
            "attention_mask_new": attention_mask_new,
            "occuring_tokens_mask_new": occuring_tokens_mask_new,
            "input_ids_original": input_ids_original,
            "attention_mask_original": attention_mask_original,
            "occuring_tokens_mask_original": occuring_tokens_mask_original,
            "alignment_matrix_a_unconstrained": alignment_matrix_a,
            "alignment_matrix_b_unconstrained": alignment_matrix_b,
            "alignment_matrix_a_space": alignment_matrix_a_space,
            "alignment_matrix_b_space": alignment_matrix_b_space,
            "alignment_matrix_a_unbiased": alignment_matrix_a_unbiased,
            "alignment_matrix_b_unbiased": alignment_matrix_b_unbiased,
            "loss_mask_original": loss_mask_original,
            "loss_mask_new": loss_mask_new,
            # ── MTA fields ─────────────────────────────────────────────────
            "offset_mapping_new": offset_mapping_new,
            "offset_mapping_original": offset_mapping_original,
            "spans_char_offsets": spans_char_offsets,    # list[list[[s,e]]]
            "words_char_offsets": words_char_offsets,    # list[list[[s,e]]]
            "texts": texts,                               # raw strings (debug + downstream)
        }

        batch = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}

        if self.expand_input_ids_dict is not None:
            batch["expanded_input_ids_new"] = utils.np_expand_input_ids(
                input_ids_new,
                self.expand_input_ids_dict,
            )

        return batch

    