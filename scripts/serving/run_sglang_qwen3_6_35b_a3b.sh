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
MODEL_ID='Qwen/Qwen3.6-35B-A3B'
CHUNKED_PREFILL_SIZE=8192
CONTEXT_LENGTH=32768
MEM_FRACTION_STATIC=0.825
SPEC_DECODING_METHOD=${SPEC_DECODING_METHOD:-MTP}
TP="${TP:-1}"
DP="${DP:-1}"

echo "Using model: $MODEL_ID"
echo "Spec decoding method: $SPEC_DECODING_METHOD"

echo "PORT=${PORT}, TP=${TP}, DP=${DP}, CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}, CONTEXT_LENGTH=${CONTEXT_LENGTH}, MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"

if [ "$SPEC_DECODING_METHOD" = "MTP" ]; then
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
  echo "Overriding SGLANG_ENABLE_OVERLAP_PLAN_STREAM to 0 as Spec MTP do not support this flag yet"
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
elif [ "$SPEC_DECODING_METHOD" = "DFLASH" ]; then
  SPEC_ARGS=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash
    --speculative-num-draft-tokens 8
    --attention-backend fa3
    --speculative-draft-attention-backend fa4
  )
elif [ "$SPEC_DECODING_METHOD" = "OFF" ]; then
  # No speculative decoding
  SPEC_ARGS=()
else
  echo "Unknown SPEC_DECODING_METHOD: $SPEC_DECODING_METHOD" >&2
  exit 1
fi

echo "SPEC_ARGS: ${SPEC_ARGS[*]}"

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
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --grammar-backend xgrammar \
  --enable-flashinfer-allreduce-fusion \
  --mamba-scheduler-strategy extra_buffer \
  "${SPEC_ARGS[@]}"
