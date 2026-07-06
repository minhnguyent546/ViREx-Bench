<h1 align="center" style="border-bottom: none;">
  <img src="https://github.com/minhnguyent546/ViREx-Bench/blob/main/docs/assets/virex_bench_logo.png?raw=true" alt="ViREX-Bench" width="64" style="vertical-align: middle;"/> ViREx-Bench
</h1>

<h3 align="center" style="border-bottom: none;">A framework for inference-time scaling on Vietnamese reasoning tasks</h3>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#example-of-usage">Example of Usage</a> ·
  <a href="#development">Development</a>
</p>

<p align="center">
    <a href="https://github.com/minhnguyent546/ViREx-Bench/releases">
        <img alt="ViREX-Bench latest release" src="https://img.shields.io/github/v/release/minhnguyent546/ViREx-Bench.svg">
    </a>
    <a href="https://github.com/minhnguyent546/ViREx-Bench/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/minhnguyent546/ViREx-Bench">
    </a>
</p>

---

> [!WARNING]
> Work in progress: dataset format, evaluation protocol, and supported reasoning methods are **under active development**.

## Overview

`ViREx-Bench` is a framework for inference-time scaling on Vietnamese reasoning tasks. The project is designed to benchmark different prompting strategies under consistent evaluation settings and measure how much each strategy improves over a direct-answer baseline.

## Installation

You can install ViREx-Bench using [`uv`](https://docs.astral.sh/uv/):

```bash
uv add virex-bench
```

or using `pip`:
```bash
pip install virex-bench
```

Verify the installation by running:
```bash
uv run virex-bench --version

# or simply
uv run vb --version
```

## Example of Usage

ViREx-Bench evaluates a model on a Vietnamese reasoning task using a prompting strategy and a decoding method. Models are served behind any OpenAI-compatible endpoint (e.g. vLLM, SGLang).

> The example below evaluates a model on `vietnamese-logical-reasoning`, which is scored with an LLM-as-a-judge. The default judge model is `deepseek/deepseek-v4-flash` (you can override this via setting `VIREX_BENCH_JUDGE_MODEL`, e.g. `export VIREX_BENCH_JUDGE_MODEL=deepseek/deepseek-v4-pro`). To use the judge, export `VIREX_BENCH_JUDGE_API_KEY` with a valid API key.

### Via the CLI

```bash
# Evaluate Qwen3.5-4B with Tree-of-Thought (with Beam search) on the vietnamese-logical-reasoning task
uv run vb run \
  --model Qwen/Qwen3.5-4B \
  --task vietnamese-logical-reasoning \
  --strategy tot-beam \
  --backend hosted_vllm \
  --api-base https://your-endpoint/v1 \
  --api-key "$OPENAI_API_KEY" \
  --num-threads 32 \
  --output-dir results
```

### Via Python

```python
import virex_bench as vb

# Init the model
lm = vb.get_model(
    "Qwen/Qwen3.5-4B",
    backend="hosted_vllm",
    api_base="https://your-endpoint/v1",
    api_key="<empty>",
)

# Get the task
task = vb.get_task("vietnamese-logical-reasoning")

# Select a strategy and decoding method
# Here we will use Tree-of-Thought with Beam search (tot-beam) as the strategy and single-pass decoding
strategy = task.get_strategy("tot-beam")
decoding = vb.get_decoding(name="single-pass", strategy=strategy)

report = vb.evaluate(
    task=task,
    lm=lm,
    strategy=strategy,
    decoding=decoding,
    num_threads=32,
    # max_examples=10,  # Uncomment this line for quick testing with a small number of examples
)
vb.save_report(report, "./results")

print(f"{report.metric}={report.score:.4f} over {report.num_evaluated_examples} examples")
```

## Development

Clone the repository and sync the environment with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/minhnguyent546/ViREx-Bench.git
cd ViREx-Bench
uv sync  # creates the venv and installs the package with dev dependencies
```

Quality checks — prefix commands with `uv run` so they use the project environment:

```bash
uv run --no-sync ruff check             # lint
uv run --no-sync ruff format            # format
uv run --no-sync pytest                 # test suite
uv run --no-sync ruff check path/to/file.py   # scope lint/format to a single file
uv run --no-sync ruff format path/to/file.py
```

### Serving models locally

ViREx-Bench talks to models through any OpenAI-compatible endpoint. To serve a model
locally with **vLLM** or **SGLang**, install the corresponding dependency group and use
the provided launchers — see [`scripts/serving/README.md`](scripts/serving/README.md) for
hardware requirements and full instructions.

```bash
uv sync --group vllm-v0-23        # or: uv sync --group sglang-v0-5-14
```

## License

This repository's source code and associated datasets are licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Citing

If you find `ViREx-Bench` useful in your research, please consider citing:
```bibtex
@misc{nguyen2026virexbench,
    author={Minh-Thien Nguyen},
    title={{ViREx-Bench}: A Framework for Inference-Time Scaling on {Vietnamese} Reasoning Tasks},
    year={2026},
    howpublished={\url{https://github.com/minhnguyent546/ViREx-Bench}}
}
```
