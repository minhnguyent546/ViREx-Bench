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

## Strategies & Decoding Reference

Every ViREx-Bench run is a **task × model × strategy × decoding** combination. You pick a prompting strategy with `--strategy` and (optionally) a decoding strategy with `--decoding` — everything else is a tuning knob with a sensible default. If you omit `--decoding`, the single-pass baseline is used.

> **See what's available at any time:**
> ```bash
> uv run vb strategies --list   # prompting strategies (--strategy)
> uv run vb decodings --list    # decoding strategies (--decoding)
> uv run vb tasks --list        # tasks / datasets (--task)
> ```

### Prompting strategies (`--strategy`)

Inference-time scaling methods. Each wraps a `dspy.Module`; cost rises roughly with the number of internal LM calls per example, so `direct` is cheapest and the `tot-*` searches are the most expensive. All knobs are optional environment variables — the defaults give reasonable behavior out of the box.

| Name | `--strategy` value | What it does | Key knobs (env vars) |
|---|---|---|---|
| **Direct** | `direct` | Baseline — ask for the answer directly, no intermediate reasoning. Cheapest option and the reference every other strategy is measured against. | — |
| **Chain-of-Thought** | `cot` | Reason step by step, then answer (single call). | — |
| **Cumulative Reasoning** | `cr` | Propose → verify propositions into entailed / contradicted / undetermined buckets, then feed the accumulated context to a solver. | `VIREX_BENCH_CR_TARGET_PROPOSITIONS`, `VIREX_BENCH_CR_VERIFIER_MODE` (`single` / `multi`), `VIREX_BENCH_CR_N_PROPOSE_SAMPLES` |
| **Tree of Thoughts** | `tot` | Search over a tree of reasoning steps, then commit an answer. Bare `tot` picks its search algorithm from `VIREX_BENCH_TOT_SEARCH_ALGORITHM` (default `beam`); the three composite names below pin it explicitly. | `VIREX_BENCH_TOT_MAX_DEPTH`, `VIREX_BENCH_TOT_BRANCHING_FACTOR`, `VIREX_BENCH_TOT_EVAL_SAMPLES` |
| &nbsp;&nbsp;↳ with beam search | `tot-beam` | ToT's BFS — keep the top `VIREX_BENCH_TOT_BEAM_WIDTH` paths per layer; early-stop at `VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD`. | `VIREX_BENCH_TOT_BEAM_WIDTH`, `VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD` |
| &nbsp;&nbsp;↳ with DFS | `tot-dfs` | Depth-first with value pruning (`VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD`) and stop-on-success (`VIREX_BENCH_TOT_DFS_STOP_THRESHOLD`). | `VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD`, `VIREX_BENCH_TOT_DFS_STOP_THRESHOLD`, `VIREX_BENCH_TOT_DFS_MAX_ITERATIONS` |
| &nbsp;&nbsp;↳ with MCTS | `tot-mcts` | Monte-Carlo Tree Search with UCT exploration (`VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT`). | `VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT`, `VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS`, `VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD` |
| **Program-of-Thought + Z3** | `pot_z3` | Generate a Z3 program from the premises, execute it in a sandboxed interpreter, regenerate on failure, then commit a Vietnamese answer from the solver output. | `VIREX_BENCH_POT_MAX_ITERS`, `VIREX_BENCH_POT_EXECUTION_TIMEOUT`, `VIREX_BENCH_POT_FALLBACK_ON_ERROR` |

> The `tot-beam`, `tot-dfs`, and `tot-mcts` names are just `tot` with the search algorithm pinned — use them when you want a specific search without setting an env var.

> Sampling temperatures for ToT / CR / PoT roles default to inheriting the model's `--model-kwargs` profile; set the corresponding `*_TEMPERATURE` env var to override. The full list of knobs lives in [`docs/ENVIRONMENT_VARIABLES.md`](docs/ENVIRONMENT_VARIABLES.md).

### Decoding strategies

Answer aggregation methods (`--decoding`), which wrap a prompting strategy and control how many candidates are sampled and merged.

| Name | `--decoding` | What it does | Compatible strategies | Key knobs |
|---|---|---|---|---|
| **Single-pass** | `single-pass` | Baseline — a single candidate from the wrapped strategy. | all | — |
| **Self-consistency** | `self-consistency` | Sample N paths in parallel, deterministic majority-vote, then an LLM aggregator synthesizes a clean answer (vote winner is the fallback; disable with `VIREX_BENCH_SC_USE_AGGREGATOR=0`). | `direct`, `cot` | `--self-consistency-num-samples N`, `VIREX_BENCH_SC_NUM_SAMPLES`, `VIREX_BENCH_SC_USE_AGGREGATOR` |
| **Self-certainty** | `self-certainty` | Sample N candidates, score each by mean token log-prob, then Borda-vote (power 0 = plain majority vote). | `direct`, `cot` | `--self-certainty-num-samples N`, `--self-certainty-borda-power P`, `VIREX_BENCH_SELFC_*` |

> Self-consistency and self-certainty require `temperature > 0` on the model under test for sample diversity. Self-certainty additionally needs an endpoint that returns token `logprobs` (otherwise it falls back to plain majority vote).

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
