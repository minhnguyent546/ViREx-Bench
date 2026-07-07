#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SAFETENSORS_FAST_GPU=1

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TVM_FFI_GPU_BACKEND=cuda

# Optional: enable schedule overlapping (experimental, may not be stable)
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

PORT=${PORT:-8124}
MODEL_ID='microsoft/Phi-4-mini-reasoning'
CHUNKED_PREFILL_SIZE=8192
CONTEXT_LENGTH=32768
MEM_FRACTION_STATIC=0.875
TP="${TP:-1}"
DP="${DP:-1}"

echo "Using model: $MODEL_ID"

echo "PORT=${PORT}, TP=${TP}, DP=${DP}, CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}, CONTEXT_LENGTH=${CONTEXT_LENGTH}, MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"

# python -m sglang.compile_deep_gemm
uv run --no-sync sglang serve \
  --model-path "$MODEL_ID" \
  --port "$PORT" \
  --tp-size "$TP" \
  --dp-size "$DP" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --max-running-requests 32 \
  --cuda-graph-max-bs 32 \
  --enable-tokenizer-batch-encode \
  --enable-mixed-chunk \
  --grammar-backend xgrammar \
  --enable-flashinfer-allreduce-fusion
