#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

# Infomaniak-AI/vllm-translategemma-4b-it
# Infomaniak-AI/vllm-translategemma-12b-it
# Infomaniak-AI/vllm-translategemma-27b-it
MODEL_ID='Infomaniak-AI/vllm-translategemma-27b-it'

echo "Using model: ${MODEL_ID}"

uv run python scripts/run_translate_dataset_translategemma.py \
  --dataset_path minhnguyent546/virex-bench-datasets \
  --dataset_name logical-reasoning \
  --dataset_split test \
  --output_dir ./data/logical-reasoning/translated \
  --seed 42 \
  --model "$MODEL_ID" \
  --use_bfloat16 \
  --max_model_len 2048 \
  --max_new_tokens 1024 \
  --gpu_memory_utilization 0.9 \
  --tensor_parallel_size 1 \
  --source_lang en \
  --target_lang vi
