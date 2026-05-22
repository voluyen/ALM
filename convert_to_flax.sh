# convert teacher Qwen sang Flax, lưu local
python3 scripts/push_flax_version_to_hub.py \
    --model_name_or_path VoCuc/Qwen1.5_1.8B_SFT_Dolly \
    --model_class Llama \
    --extra_args '{"attention_bias": true, "max_length": 8192}' \
    --use_cpu \
    --output_dir ./models/qwen1.5-1.8b-sft-dolly-flax \
    --tmp_path ./tmp/pt_staging
