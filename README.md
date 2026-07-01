<h1 align="center" style="border-bottom: none;">
  <img src="https://github.com/minhnguyent546/ViREx-Bench/blob/main/docs/assets/virex_bench_logo.png?raw=true" alt="ViREX-Bench" width="64" style="vertical-align: middle;"/> ViREX-Bench
</h1>

<h3 align="center" style="border-bottom: none;">A framework for inference-time scaling on Vietnamese reasoning tasks</h3>

<p align="center">
    <a href="https://github.com/minhnguyent546/ViREx-Bench/releases">
        <img alt="ViREX-Bench latest release" src="https://img.shields.io/github/v/release/minhnguyent546/ViREx-Bench.svg">
    </a>
    <a href="https://github.com/minhnguyent546/ViREx-Bench/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/minhnguyent546/ViREx-Bench">
    </a>
</p>

---

`ViREx-Bench` is a framework for inference-time scaling on Vietnamese reasoning tasks. The project is designed to benchmark different strategies under consistent evaluation settings and measure how much each strategy improves over a direct-answer baseline.

The framework aims to compare approaches such as Chain-of-Thought, Tree-of-Thought, Monte Carlo Tree-of-Thought, Program-of-Thought, and symbolic reasoning with tools such as Z3.

> Work in progress: dataset format, evaluation protocol, and supported reasoning methods are under active development.

## Installation

You can install ViREx-Bench using `pip`:

```bash
pip install virex-bench
```

or if you are using `uv`:

```bash
uv add virex-bench
```

Verify the installation by running:
```bash
uv run virex-bench --version

# or simply
uv run vb --version
```

## Example of Usage

ViREx-Bench evaluates a model on a Vietnamese reasoning task using a prompting strategy and a decoding method. Models are served behind any OpenAI-compatible endpoint (e.g. vLLM, SGLang).

> The example below evaluates a model on `vietnamese-logical-reasoning`, which is scored with an LLM-as-a-judge. The default judge model is `deepseek-v4-flash` (you can override this via setting `VIREX_BENCH_JUDGE_MODEL`, e.g. `export VIREX_BENCH_JUDGE_MODEL=deepseek-v4-pro`). To use the judge, export `VIREX_BENCH_JUDGE_API_KEY` with a valid API key.

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
