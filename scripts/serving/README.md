# Serving scripts

These serving scripts are intended to be run with `vLLM v0.23.0` or `SGLang v0.5.14`.

**Requirements:**

- Linux-based OS with `glibc>=2.31`, `gcc/g++>=12.0.0`
- CUDA toolkit 13.0
- NVIDIA GPUs with sm>=8.0 (e.g., A100, A6000, H100, B200, etc)

## Serving model

- With `SGLang v0.5.14`:
```bash
# at the project root
uv sync --group sglang-v0-5-14

# verify installation
uv run --no-sync sglang version

# serving model
CUDA_VISIBLE_DEVICES=0 SPEC_DECODING_METHOD=DFLASH bash scripts/serving/run_sglang_qwen3_5_4b.sh

# serving larger model
CUDA_VISIBLE_DEVICES=0,1 SPEC_DECODING_METHOD=DFLASH DP=2 bash scripts/serving/run_sglang_qwen3_6_35b_a3b.sh
```

- With `vLLM v0.23.0`:
```bash
# at the project root
uv sync --group vllm-v0-23

# verify installation
uv run --no-sync vllm --version

# serving model
CUDA_VISIBLE_DEVICES=0 SPEC_DECODING_METHOD=DFLASH bash scripts/serving/run_vllm_qwen3_5_4b.sh

# serving larger model
CUDA_VISIBLE_DEVICES=0,1 SPEC_DECODING_METHOD=DFLASH DP=2 bash scripts/serving/run_vllm_qwen3_6_35b_a3b.sh
```
