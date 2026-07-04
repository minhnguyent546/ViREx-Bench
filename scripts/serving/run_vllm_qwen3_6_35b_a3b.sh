#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SAFETENSORS_FAST_GPU=1

# vLLM runner v2, DFLASH is not supported
# export VLLM_USE_V2_MODEL_RUNNER=1

PORT=${PORT:-8124}
MODEL_ID='Qwen/Qwen3.6-35B-A3B'
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.825
SPEC_DECODING_METHOD=${SPEC_DECODING_METHOD:-MTP}
TP="${TP:-1}"
DP="${DP:-1}"

if [[ $SPEC_DECODING_METHOD == "DFLASH" ]]; then
  SPECULATIVE_CONFIG='{"method": "dflash", "model": "z-lab/Qwen3.6-35B-A3B-DFlash", "num_speculative_tokens": 7}'
elif [[ $SPEC_DECODING_METHOD == "MTP" ]]; then
  SPECULATIVE_CONFIG='{"method": "mtp", "num_speculative_tokens": 2}'
elif [[ $SPEC_DECODING_METHOD == "OFF" ]]; then
  SPECULATIVE_CONFIG=''
else
  echo "Invalid SPEC_DECODING_METHOD = $SPEC_DECODING_METHOD. Expected one of dflash or mtp" >&2
  exit 1
fi

echo "Using model: $MODEL_ID"
echo "Using SPEC_DECODING_METHOD=${SPEC_DECODING_METHOD} with SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG}"

echo "PORT=${PORT}, TP=${TP}, DP=${DP}, MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}, MAX_MODEL_LEN=${MAX_MODEL_LEN}, GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}, MAX_NUM_SEQS=${MAX_NUM_SEQS}"

uv run --no-sync vllm serve "$MODEL_ID" \
  --port "$PORT" \
  --trust-remote-code \
  --async-scheduling \
  -tp "$TP" \
  -dp "$DP" \
  --enable-auto-tool-choice \
  --language-model-only \
  --distributed-executor-backend mp \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-config "$SPECULATIVE_CONFIG"
