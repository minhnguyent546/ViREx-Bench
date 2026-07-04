#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SAFETENSORS_FAST_GPU=1

# vLLM runner v2, DFLASH is not supported
# export VLLM_USE_V2_MODEL_RUNNER=1

PORT=${PORT:-8124}
MODEL_ID='google/gemma-4-E2B-it'
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.875
TP="${TP:-1}"
DP="${DP:-1}"

echo "Using model: $MODEL_ID"

echo "PORT=${PORT}, TP=${TP}, DP=${DP}, MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}, MAX_MODEL_LEN=${MAX_MODEL_LEN}, GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}, MAX_NUM_SEQS=${MAX_NUM_SEQS}"

uv run --no-sync vllm serve "$MODEL_ID" \
  --port "$PORT" \
  --trust-remote-code \
  --async-scheduling \
  -tp "$TP" \
  -dp "$DP" \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --enable-auto-tool-choice \
  --language-model-only \
  --distributed-executor-backend mp \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --language-model-only \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --speculative-config "$SPECULATIVE_CONFIG"
