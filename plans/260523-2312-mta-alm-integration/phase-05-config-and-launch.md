---
phase: 5
title: Config and Launch
status: completed
priority: P2
effort: 1h
dependencies:
  - 4
---

# Phase 5: Config and Launch

## Overview
Tạo YAML config và shell launch script cho experiment `gpt2_120M + alm_unbiased + mta`. Giữ baseline config cũ không thay đổi để dễ A/B compare.

## Requirements
- **Functional:** `bash gpt2_120M_alm_mta_distill.sh` chạy được training với MTA enabled.
- **Non-functional:** Config self-documenting, comment các MTA params. Output dir khác baseline để không ghi đè.

## Related Code Files

- **Create:** `gpt2_120M_alm_mta_distill.yaml` (project root)
- **Create:** `gpt2_120M_alm_mta_distill.sh` (project root)
- **Reference:** `gpt2_120M_cross_tokenizer_distill.yaml`, `gpt2_120M_distill.sh`

## Implementation Steps

### 5.1 Create `gpt2_120M_alm_mta_distill.yaml`

Copy từ `gpt2_120M_cross_tokenizer_distill.yaml`, thay đổi:

```yaml
steps: 5_000
warmup_steps: 1_000
name: "gpt2_120M_alm_mta_v1"
output: "outputs/gpt2_120M_alm_mta_v1"
num_workers: 16
log_interval: 50
sync_interval: 100
eval_interval: 5000
save_interval: 5000

# MTA combined with alm_unbiased
losses: [alm_unbiased, mta]
loss_weights: [1.0, 2.0]

target_tokenizer_name: openai-community/gpt2:source=GPT2
tokens_to_add: []

train_model_mode: "lora"
model_lora_rank: 64
model_lora_alpha: 64
train_embeddings: true

# ALM hyperparams
binarization_temp: 100.0
alm_diff_fn: "binary_ce"
alm_mode: "space_merge+append_space"
tokenizer_pair_data_path: "artifacts/tokenizer_data/math_llama3_to_gemma2"
tokenizer_pair_bias_threshold: 0.1

# MTA hyperparams ──────────────────────────────────────
mta_mode: true
teacher_layer_mapping: [6, 12, 18, 24]
student_layer_mapping: [3, 6, 9, 12]
split_layer_mapping: [0, 1, 4, 4]
w_span_loss: 2.0
entropy_weight: false
wo_span_weight: false
# ──────────────────────────────────────────────────────

student:
  pretrained_model_name_or_path: "openai-community/gpt2"
  tokenizer_name: "openai-community/gpt2:source=GPT2"

teacher:
  pretrained_model_name_or_path: "VoCuc/Qwen1.5_1.8B_SFT"
  tokenizer_name: "VoCuc/Qwen1.5_1.8B_SFT:source=Qwen2"

data:
  path: data/dolly_train_with_spans.jsonl    # ← precomputed file from Phase 1
  batch_size: 16
  num_workers: 16
  kind: "jsonl"
  lang_code: en

hypernet:
  architecture: transformer
  num_layers: 1
  residual: true
  residual_alpha: 1
  use_attention: false

optimizer:
  type: adamw
  weight_decay: 0.01
  b1: 0.9
  b2: 0.95
  eps: 1.e-8
  grad_acc_steps: null
  learning_rate: 1.e-5
  max_grad_norm: 1.0
  param_groups:
    - pattern: .*(projector_query|projector_s2t|projector_t2s|projector_latents|loss_weights|mta_projector).*
      lr_scale: 2

eval:
  tasks: [arc_easy,arc_challenge,piqa,hellaswag,boolq,arithmetic,mmlu]
  lengths: [128, 256, 512, 1024, 2048]
  tokens_per_batch: 8192
  add_bos: true
  chat_template_mode: surround_instruct
  confirm_run_unsafe_code: true
```

### 5.2 Create `gpt2_120M_alm_mta_distill.sh`

Copy từ `gpt2_120M_distill.sh`, thay config:

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
GPUS=(0 1)

python3 pytorch_cross_tokenizer_distill.py \
    --config=gpt2_120M_alm_mta_distill.yaml \
    --overrides \
    name=gpt2_120M_alm_mta_v1
```

## Success Criteria

- [ ] YAML parse được bằng `python -c "import yaml; yaml.safe_load(open('gpt2_120M_alm_mta_distill.yaml'))"`
- [ ] Shell script `chmod +x` và `bash -n` pass
- [ ] Config có đầy đủ field MTA mới khớp với `CrossTokenizerDistillArgs`
- [ ] Output dir `outputs/gpt2_120M_alm_mta_v1` khác baseline

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Param group pattern miss `mta_projector_list` | Tên module phải có substring `projector` để match `.*projector.*` (đã dùng `mta_projector_list` trong Phase 4) |
| `loss_weights` mismatch length losses | Length cả 2 list = 2, document rõ trong YAML comment |
| Path tokenizer_pair_data_path không tồn tại (artifact missing) | Verify trước launch: `ls artifacts/tokenizer_data/math_llama3_to_gemma2` |
