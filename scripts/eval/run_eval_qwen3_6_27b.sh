#!/usr/bin/env bash

set -euo pipefail

# using DeepSeek API
export VIREX_BENCH_JUDGE_API_KEY="$DEEPSEEK_API_KEY"
export VIREX_BENCH_JUDGE_MODEL='deepseek/deepseek-v4-flash'

RESULTS_DIR='./final_results/seed-0'

# STRATEGY="${STRATEGY:-cot}"
NUM_THREADS="${NUM_THREADS:-32}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
API_BASE="${API_BASE:-https://lm.minhnguyent546.io.vn/v1}"
NUM_EXAMPLES=''

if [[ -n "$NUM_EXAMPLES" ]]; then
  EXTRA_PARAMS+=(--max-examples "$NUM_EXAMPLES")
fi

BASE_EXTRA_PARAMS=("${EXTRA_PARAMS[@]}")

for STRATEGY in direct cot cot-sc3 cot-sc5 tot-beam tot-dfs tot-mcts cr pot_z3; do
  EXTRA_PARAMS=("${BASE_EXTRA_PARAMS[@]}")
  DECODING_STRATEGY=single-pass
  if [[ "$STRATEGY" == "cot-sc3" ]]; then
    STRATEGY='cot'
    DECODING_STRATEGY='self-consistency'
    EXTRA_PARAMS+=(--self-consistency-num-samples 3)
  elif [[ "$STRATEGY" == "cot-sc5" ]]; then
    STRATEGY='cot'
    DECODING_STRATEGY='self-consistency'
    EXTRA_PARAMS+=(--self-consistency-num-samples 5)
  fi

  echo ">> Config: STRATEGY=${STRATEGY}, DECODING_STRATEGY=${DECODING_STRATEGY}, NUM_THREADS=${NUM_THREADS}, NUM_EXAMPLES=${NUM_EXAMPLES:-'all'}"
  echo ">> Extra params: ${EXTRA_PARAMS[*]}"

  uv run --no-sync vb run \
    --model Qwen/Qwen3.6-27B \
    --task vietnamese-logical-reasoning \
    --strategy "$STRATEGY" \
    --decoding "$DECODING_STRATEGY" \
    --backend hosted_vllm \
    --api-base "$API_BASE" \
    --api-key '<empty>' \
    --num-threads "$NUM_THREADS" \
    --model-kwargs '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,"presence_penalty": 1.5,"repetition_penalty": 1.0,"chat_template_kwargs": {"enable_thinking": false}}' \
    --output-dir "$RESULTS_DIR" \
    --log-level "$LOG_LEVEL" \
    "${EXTRA_PARAMS[@]}"

done
