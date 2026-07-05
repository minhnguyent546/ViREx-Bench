# AGENTS.md

## Project

**ViREx-Bench** (Vietnamese Reasoning Exploration Benchmark) is a lightweight framework
for exploring **multi-step reasoning of LLMs for Vietnamese**.

The project has two expected outputs:

1. **A Vietnamese logical-reasoning dataset** — each row contains a set of *premises* and a
   *question* (with a gold answer), framing logical-reasoning problems in Vietnamese.
2. **An evaluation framework** that measures how much each inference-time reasoning strategy
   improves over a direct-answer baseline on that dataset.

The framework targets two orthogonal research axes (planned scope listed; see each
registry for what is currently implemented):

- **Prompting / inference-time scaling** (`strategies/`): Chain-of-Thought (CoT),
  Tree-of-Thought (ToT), Self-Consistency, Program-of-Thought with symbolic reasoning
  (Z3 solver), and Monte-Carlo Tree-of-Thought (MCToT). *Implemented: direct baseline,
  CoT, ToT (beam + DFS search).*
- **Decoding** (`decoding/`): the baseline (single candidate, or multiple candidates
  aggregated by majority vote), Best-of-N with a verifier, and speculative decoding to
  speed up inference. *Implemented: single-pass baseline, self-consistency (N-sample
  majority vote + LLM aggregation).*

**DSPy** is the main framework for implementing reasoning strategies. The CLI design is
modeled on [`mteb`](https://github.com/embeddings-benchmark/mteb) — a CLI is sufficient
(no web/leaderboard UI or serving layer).

## Commands

Always prefix Python/tool commands with `uv run --no-sync`, omit the `--no-sync` flag only when you actually needed to avoid syncing packages everytime (which take times).

```bash
# Lint
uv run --no-sync ruff check

# Format
uv run --no-sync ruff format

# Run the benchmark CLI (task × model × strategy × decoding)
# Models are served behind an OpenAI-compatible endpoint (no local loading).
uv run virex-bench run \
    --model Qwen/Qwen3.5 \
    --task vietnamese-logical-reasoning \
    --strategy cot \
    --backend openai \
    --api-base http://localhost:8000/v1 \
    --output-dir results

# `vb` is a shorter alias for `virex-bench`
uv run vb run --model Qwen/Qwen3.5 --strategy cot

# Inspect available tasks / strategies
uv run virex-bench tasks --list
uv run virex-bench strategies --list
```

### File-scoped fast feedback

```bash
uv run --no-sync ruff check path/to/file.py
uv run --no-sync ruff format path/to/file.py
```

## Project Structure

The package is organized around four axes — **task** (dataset), **model** (backend),
**strategy** (prompting), and **decoding** (aggregation) — wired together by the
evaluation orchestrator. Layout is inspired by `mteb` (task × model registries + a thin
CLI), with two extra axes for reasoning and decoding.

```
virex_bench/
├── cli/            # argparse CLI: `virex-bench run | tasks | strategies`
│   └── build_cli.py #  build_parser / main + subcommand handlers
├── envs.py         # centralized, lazily-read environment variables (vLLM-style)
├── logger.py       # init_logger(__name__) — shared logging setup
├── types/          # pydantic data models + type aliases (mirrors mteb/types)
│   ├── __init__.py #   re-exports all public types
│   ├── _task.py    #   ReasoningExample, DatasetConfig, TaskMetadata
│   ├── _result.py  #   TaskResult, EvaluationReport, JudgeOutcome
│   └── _metric.py  #   ReasoningMetric callable alias
├── tasks/          # reasoning datasets + registry
│   ├── base.py     #   ReasoningTask base class
│   ├── registry.py #   get_task / list_tasks
│   └── vietnamese_logical_reasoning.py
├── models/         # LM layer — OpenAI-compatible endpoints via dspy.LM
│   ├── __init__.py #   get_model
│   ├── base.py     #   BaseLM(dspy.LM) wrapper
│   └── backends.py #   load_backend: build a BaseLM for an OpenAI-compatible endpoint
├── strategies/     # PROMPTING / inference-time scaling (dspy.Module subclasses)
│   ├── base.py     #   ReasoningStrategy base class
│   ├── registry.py #   get_strategy / list_strategies (registered: direct, cot, tot)
│   ├── direct.py   #   baseline: direct answer
│   ├── cot.py      #   Chain-of-Thought
│   ├── modules.py  #   reusable dspy modules: ChainOfThought, DualTask2ChainOfThought, ThinkingCaptureLM
│   └── tot/        #   Tree-of-Thoughts strategy package (Yao et al., 2023)
│       ├── strategy.py # ToTStrategy — propose/evaluate/aggregate over a thought tree
│       ├── common.py  # shared signatures + helpers (proposer/evaluator, scoring, dedupe)
│       └── search/     # pluggable search algorithms (beam, DFS; MCTS planned)
│           ├── __init__.py # search registry: build_search / list_search
│           ├── base.py     # SearchConfig, SearchResult, ThoughtSearch ABC
│           ├── beam.py     # BeamSearch (ToT's BFS)
│           └── dfs.py      # DFSSearch (backtracking + value pruning)
├── decoding/       # DECODING / answer aggregation
│   ├── base.py     #   DecodingStrategy base + SinglePass (single-candidate baseline)
│   ├── registry.py #   get_decoding / list_decoding
│   └── self_consistency.py # SelfConsistency: N-sample majority vote + LLM aggregation
├── evaluation/     # orchestration + scoring + LLM-as-a-judge
│   ├── evaluate.py #   task × model × strategy loop + report serialization (save_report)
│   ├── metrics.py  #   accuracy metrics + METRIC_REGISTRY / get_metric / judge_example
│   └── judge.py    #   LLMJudge base, LogicalReasoningJudge, JUDGE_REGISTRY / build_judge
└── __main__.py     # `python -m virex_bench` entry point -> cli.main

data/               # datasets (managed externally — do not edit by hand)
notebooks/          # dataset-prep notebooks (e.g. eval-round → HF dataset conversion)
scripts/            # dataset translation helpers + model-serving launchers (serving/)
tests/              # pytest suite — mirrors the package layout (cli / evaluation / strategies [+ tot])
results/            # benchmark run outputs
```

## Development and Code Style

### Development Guide

- **Python 3.12** — use modern syntax (`str | None`, `list[int]`, etc.)
- **Git**: Do not try to use `git checkout <file>` to discard or revert changes as you might cause loose uncommitted work.
- **Formatter**: Ruff (`uv run --no-sync ruff format`) — double quotes, 4-space indent, LF endings, line length 99
- **Linter**: Ruff (`uv run --no-sync ruff check`) — rules: B, C, E, F, I, W, RUF013, UP006
- **Type checker**: Pyright in strict mode. Use `# pyright: ignore` to suppress false positive or workaround some limitation in the type checker. DO NOT silence legitimate type errors.
- **Testing**: pytest (`uv run --no-sync pytest`)
- **Imports**: sorted by Ruff (isort rules), grouped: stdlib → third-party → local
- **DSPy**: Reasoning strategies are implemented as `dspy.Module` subclasses under `strategies/`. Follow the existing patterns there for signatures and modules. Write signature docstrings and field descriptions in **English** (the instruction language), even for Vietnamese tasks — the target models are multilingual, and English instructions keep the prompt contract consistent across tasks. The task data (premises/questions/answers) stays in its native language.
- **Grepping**: Prefer using `rg` (ripgrep) over `grep` for speed and better defaults.

### Code Style

- **Pydantic**: Use `pydantic.BaseModel` for data models (examples, configs, results, reports) — not `dataclasses`. Serialize with `model_dump()` / `model_dump_json()`.
- **Logging**: Use the shared logger — `from virex_bench.logger import init_logger` then `logger = init_logger(__name__)` at module top. Prefer f-strings in logging calls over `%`-style logger formatting. Do not call `print()` for diagnostics or configure `logging` directly (CLI user-facing output via `print` is fine).
- **Env vars**: All environment variables go through `virex_bench/envs.py` — never read `os.environ` directly elsewhere. Add a typed declaration in the `TYPE_CHECKING` block and a lazy `lambda` entry in `environment_variables`, then access via `from virex_bench import envs` / `envs.VAR_NAME`. Access is lazy (read at use time, not import time), so set vars before first access.
- **CLI**: `argparse`-based subcommands (`run`, `tasks`, `strategies`), modeled on `mteb`. Both `virex-bench` and the short alias `vb` map to `cli:main`. Keep command handlers thin — delegate to `evaluation/`, `tasks/`, `models/`.
- **File path:** Use raw `str` for file paths and `os.path` for path manipulations. Avoid `pathlib.Path` to keep things simple and consistent across the codebase.
- **Variable names:** Use verbose, self-documenting names. Avoid short abbreviations (e.g., `ri`, `ip`, `ua`, `ui`, `el`, `t`) — prefer `request_metadata`, `client_ip`, `client_user_agent`, `elapsed`, etc. Single-letter names are only acceptable as loop indices or in very narrow local scopes (e.g., list comprehensions).
- **None check:** Use `if x is None` and `if x is not None` for checking if a variable is `None`. Avoid truthy/falsy checks like `if x` or `if not x` when the variable can have valid falsy values (e.g., empty string, zero, empty list).
- **`type[T]` hints:** Use `type[T]` when a function/field expects a *class* (to instantiate or inspect later), not an instance — e.g. `signature: type[dspy.Signature]` and registries like `dict[str, type[ReasoningStrategy]]`. Pass the class itself (`DirectStrategy`), not an instance (`DirectStrategy()`).
- Do not create small helper methods that are referenced only once.

## Code review

When asked for reviewing changes, focus on:
- Potential bugs introduced by the change
- Check for behavior mismatches for refactoring changes
- Code quality and readability (style, structure, naming, etc should match the existing codebase and be clear)
- Ignore directories and files starting with `_` (e.g., `_testing_results/`, `_test_format.py`) as they are testing or local files and not part of the main codebase.

## Boundaries

- **Never commit** `.env` files or API keys/secrets
- **Do not edit** files under `data/` directly — datasets are managed externally
- **Do not modify** generated/cached directories: `.ruff_cache/`, `.cache/`, `_testing_results/`
- **Do not touch** `uv.lock` by hand
