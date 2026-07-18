<h1 align="center" style="border-bottom: none;">
  <img src="https://github.com/minhnguyent546/ViREx-Bench/blob/main/docs/assets/virex_bench_logo.png?raw=true" alt="ViREX-Bench" width="64" style="vertical-align: middle;"/> ViREx-Bench
</h1>

<h3 align="center" style="border-bottom: none;">A framework for inference-time scaling on Vietnamese reasoning tasks</h3>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#example-of-usage">Example of Usage</a> ·
  <a href="#strategies--decoding-reference">Strategies &amp; Decoding Reference</a> ·
  <a href="#development">Development</a>
</p>

<p align="center">
    <a href="https://doi.org/10.5281/zenodo.21247474">
        <img alt="Zenodo DOI" src="https://zenodo.org/badge/1272222849.svg">
    </a>
    <a href="https://github.com/minhnguyent546/ViREx-Bench/releases">
        <img alt="ViREX-Bench latest release" src="https://img.shields.io/github/v/release/minhnguyent546/ViREx-Bench.svg?color=green">
    </a>
    <a href="https://github.com/minhnguyent546/ViREx-Bench/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/minhnguyent546/ViREx-Bench?color=blue">
    </a>
    <a href="https://pepy.tech/projects/virex-bench">
        <img alt="PyPI downloads" src="https://static.pepy.tech/personalized-badge/virex-bench?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
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

The example below evaluates a model on `vietnamese-logical-reasoning`, which is scored with an LLM-as-a-judge. Before you run, configure the judge via one of:

```bash
# DeepSeek API (default model is deepseek/deepseek-v4-flash)
export VIREX_BENCH_JUDGE_API_KEY=...
export VIREX_BENCH_JUDGE_MODEL=deepseek/deepseek-v4-flash

# or self-hosted deepseek-v4-{flash,pro} behind an OpenAI-compatible endpoint
export VIREX_BENCH_JUDGE_API_KEY=...
export VIREX_BENCH_JUDGE_MODEL=hosted_vllm/deepseek-v4-flash
export VIREX_BENCH_JUDGE_BASE_URL=http://localhost:<PORT>/v1
```

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

## Strategies & Decoding Reference

Every ViREx-Bench run is a **task × model × strategy × decoding** combination. You pick a prompting strategy with `--strategy` and (optionally) a decoding strategy with `--decoding`. Other settings (search depth, sample counts, temperatures, …) have sensible defaults and can be overridden via environment variables — see [`docs/ENVIRONMENT_VARIABLES.md`](docs/ENVIRONMENT_VARIABLES.md). If you omit `--decoding`, the single-pass baseline is used.

> **See what's available:**
> ```bash
> uv run vb strategies --list   # prompting strategies (--strategy)
> uv run vb decodings --list    # decoding strategies (--decoding)
> uv run vb tasks --list        # tasks / datasets (--task)
> ```

### Prompting strategies (`--strategy`)

Inference-time scaling methods. Each wraps a `dspy.Module`; cost rises roughly with the number of internal LM calls per example, so `direct` is cheapest and the `tot-*` searches are the most expensive. Tuning knobs are optional environment variables with sensible defaults — see [`docs/ENVIRONMENT_VARIABLES.md`](docs/ENVIRONMENT_VARIABLES.md) for the full list.

| Name | `--strategy` value | What it does |
|---|---|---|
| **Direct** | `direct` | Baseline — ask for the answer directly, no intermediate reasoning. Cheapest option and the reference every other strategy is measured against. |
| **Chain-of-Thought** | `cot` | Ask the model to write a step-by-step rationale, then the answer — still a single generation. |
| **Cumulative Reasoning** | `cr` | Build a short working memory of intermediate claims. The model proposes a new claim from the premises (and claims found so far); a verifier labels it *entailed*, *contradicted*, or *undetermined* and stores it; the loop repeats until enough claims are collected (or too many proposals fail). A final solver call reads that accumulated context and commits the answer. Linear accumulation — no search tree. |
| **Tree of Thoughts** | `tot` | Explore many partial reasoning paths instead of one chain. At each step the model proposes several next thoughts, an evaluator scores how promising each path looks (1–10), and a search algorithm decides which paths to expand. After search, a final call turns the best path into a well-formed answer. Bare `tot` picks the algorithm from `VIREX_BENCH_TOT_SEARCH_ALGORITHM` (default `beam`); the three composite names below pin it explicitly. |
| &nbsp;&nbsp;↳ with beam search | `tot-beam` | Breadth-first ToT — expand a layer of paths, keep only the top-scoring survivors, and repeat by depth. Can stop early once a path looks good enough. |
| &nbsp;&nbsp;↳ with DFS | `tot-dfs` | Depth-first ToT — push one path as deep as it looks promising, backtrack on weak branches (value pruning), and stop early on a strong path. |
| &nbsp;&nbsp;↳ with MCTS | `tot-mcts` | Monte-Carlo ToT — over many short simulations, select paths with UCB1/UCT (balance high scores against less-explored branches), then take the best path found. |
| **Program-of-Thought + Z3** | `pot_z3` | Ask the model to translate the premises into an executable Python program that uses the Z3 theorem prover, run it in a sandboxed subprocess, and (on crash/timeout) regenerate the program with the error fed back. A final call maps the solver output to the task's Vietnamese answer format. |

> The `tot-beam`, `tot-dfs`, and `tot-mcts` names are just `tot` with the search algorithm pinned — use them when you want a specific search without setting an env var.

> Sampling temperatures for ToT / CR / PoT roles default to inheriting the model's `--model-kwargs` profile; set the corresponding `*_TEMPERATURE` env var to override.

### Decoding strategies

Answer aggregation methods (`--decoding`), which wrap a prompting strategy and control how many candidates are sampled and merged.

| Name | `--decoding` | What it does | Compatible strategies |
|---|---|---|---|
| **Single-pass** | `single-pass` | Baseline — a single candidate from the wrapped strategy. | all |
| **Self-consistency** | `self-consistency` | Sample N paths in parallel, majority-vote the normalized answers, then optionally ask an LLM aggregator to synthesize a clean final answer (vote winner is the fallback if aggregation fails; disable with `VIREX_BENCH_SC_USE_AGGREGATOR=0`). | `direct`, `cot` |
| **Self-certainty** | `self-certainty` | Sample N candidates, score each by how confident the model was (mean token log-probability), then weight the vote so higher-certainty samples count more (Borda vote; power 0 = plain majority vote). Without log-probs, falls back to plain majority vote. | `direct`, `cot` |

> Self-consistency and self-certainty require `temperature > 0` on the model under test for sample diversity. Self-certainty additionally needs an endpoint that returns token `logprobs` (otherwise it falls back to plain majority vote). Sample counts and related knobs: CLI flags `--self-consistency-num-samples`, `--self-certainty-num-samples`, `--self-certainty-borda-power`, or the matching env vars in [`docs/ENVIRONMENT_VARIABLES.md`](docs/ENVIRONMENT_VARIABLES.md).

### Combining strategy × decoding

A few representative recipes (common endpoint flags `--backend`, `--api-base`, `--api-key`, `--task`, `--output-dir` are omitted for brevity — see the [Example of Usage](#example-of-usage)):

```bash
# Chain-of-Thought + self-consistency (5 samples)
uv run vb run --model Qwen/Qwen3.5-4B --strategy cot \
  --decoding self-consistency --self-consistency-num-samples 5

# Tree-of-Thoughts with MCTS search, single-pass decoding
uv run vb run --model Qwen/Qwen3.5-4B --strategy tot-mcts --decoding single-pass

# Direct baseline + self-certainty (8 samples, Borda power 0.5)
uv run vb run --model Qwen/Qwen3.5-4B --strategy direct \
  --decoding self-certainty --self-certainty-num-samples 8 --self-certainty-borda-power 0.5

# Program-of-Thought (Z3) + single-pass
uv run vb run --model Qwen/Qwen3.5-4B --strategy pot_z3 --decoding single-pass
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

> [!IMPORTANT]
> These groups pull custom-built wheels (compiled against **glibc 2.31** so they run on
> older Linux distros) that only target **Python 3.12, Linux, x86_64**. On anything else,
> the sync still succeeds but quietly installs nothing from the group. So if a serving
> import fails right after syncing, check `python --version` first — you likely need a
> 3.12 environment (`uv venv --python 3.12`). The benchmark itself works fine on Python
> 3.12–3.14.

## License

This repository's source code and associated datasets are licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Citing

If you find `ViREx-Bench` useful in your research, please consider citing:
```bibtex
@misc{nguyen2026virexbench,
    author={Minh-Thien Nguyen},
    title={{ViREx-Bench}: A Framework for Inference-Time Scaling on {Vietnamese} Reasoning Tasks},
    year={2026},
    doi={10.5281/zenodo.21247474},
    howpublished={\url{https://github.com/minhnguyent546/ViREx-Bench}}
}
```
