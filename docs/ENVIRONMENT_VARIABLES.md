# Environment Variables

ViREx-Bench is configured in part via environment variables. Set them before launching
the CLI for the values to take effect. Unless noted otherwise, unset variables fall back
to the listed default.

Table of Contents
===

 * [Logging](#logging)
 * [Paths](#paths)
 * [Model Endpoint (OpenAI-compatible)](#model-endpoint-openai-compatible)
 * [HuggingFace Hub](#huggingface-hub)
 * [LLM-as-a-Judge](#llm-as-a-judge)
 * [LM Retry (transient connection / HTTP errors)](#lm-retry-transient-connection--http-errors)
 * [Prompting Strategies (--strategy)](#prompting-strategies---strategy)
    * [Tree-of-Thoughts (ToT) — General](#tree-of-thoughts-tot--general)
    * [Tree-of-Thoughts (ToT) — Per-Variant Knobs](#tree-of-thoughts-tot--per-variant-knobs)
    * [Cumulative Reasoning (CR)](#cumulative-reasoning-cr)
    * [Program-of-Thought (PoT / Z3)](#program-of-thought-pot--z3)
 * [Decoding Strategies (--decoding)](#decoding-strategies---decoding)
    * [Self-Consistency Decoding](#self-consistency-decoding)
    * [Self-Certainty Decoding](#self-certainty-decoding)


## Logging

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_LOG_LEVEL` | Logging verbosity for the `virex_bench` logger. | `INFO` | Matched case-insensitively, then upper-cased. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `VIREX_BENCH_LOG_COLOR` | ANSI color in log output. | `auto` | `auto` = colorize only on a TTY; `1` = always; `0` = never. Matched case-insensitively. |
| `NO_COLOR` | Standard cross-tool opt-out ([no-color.org](https://no-color.org)). | _unset_ (`False`) | Any **non-empty** value disables colored output, overriding `VIREX_BENCH_LOG_COLOR`. |

## Paths

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_OUTPUT_DIR` | Default output directory when `--output-dir` is unset. | `results` | |

## Model Endpoint (OpenAI-compatible)

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `OPENAI_BASE_URL` | Base URL of the OpenAI-compatible endpoint (e.g. `http://localhost:8000/v1`). | _unset_ (`None`) | |
| `OPENAI_API_KEY` | API key for the OpenAI-compatible endpoint. | _unset_ (`None`) | |

## HuggingFace Hub

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `HF_TOKEN` | HuggingFace Hub access token. | _unset_ (`None`) | Required for gated/private task datasets. |

## LLM-as-a-Judge

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_JUDGE_API_KEY` | API key for the judge model provider. | _unset_ (`None`) | Used by the LLM-as-a-judge metric. |
| `VIREX_BENCH_JUDGE_BASE_URL` | Base URL of the OpenAI-compatible endpoint serving a self-hosted judge model. | _unset_ (`None`) | Required when `VIREX_BENCH_JUDGE_MODEL` starts with `hosted_vllm/` (raises otherwise). |
| `VIREX_BENCH_JUDGE_MODEL` | DeepSeek model used by the LLM-as-a-judge metric. | `deepseek/deepseek-v4-flash` | Matched case-insensitively. Choices: `deepseek/deepseek-v4-pro`, `deepseek/deepseek-v4-flash`, `opencode-go/deepseek-v4-pro`, `opencode-go/deepseek-v4-flash`, `hosted_vllm/deepseek-v4-pro`, `hosted_vllm/deepseek-v4-flash`. |

## LM Retry (transient connection / HTTP errors)

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_LM_MAX_RETRIES` | Max attempts (incl. the first call) for transient LM errors (timeouts, resets, 429/500/503). | `3` | |
| `VIREX_BENCH_LM_RETRY_MIN_WAIT` | Initial wait between LM retries (seconds). | `1.0` | |
| `VIREX_BENCH_LM_RETRY_MAX_WAIT` | Capped max wait between LM retries (seconds). | `30.0` | |
| `VIREX_BENCH_LM_RETRY_JITTER` | Max random jitter added to each LM retry wait (seconds). | `1.0` | |

## Prompting Strategies (`--strategy`)

Knobs for the inference-time reasoning strategies selected via `--strategy`.

### Tree-of-Thoughts (ToT) — General

Search a tree of reasoning steps, then commit an answer. Knobs shared by all search variants (`tot`, `tot-beam`, `tot-dfs`, `tot-mcts`).

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_TOT_MAX_DEPTH` | Number of reasoning steps explored (tree depth). | `10` | Tuned for the dataset (chains can reach 10+ steps); lower per-run for shorter-chain subsets. |
| `VIREX_BENCH_TOT_BRANCHING_FACTOR` | Candidate next-thoughts requested per node per step (proposer branching factor). | `3` | |
| `VIREX_BENCH_TOT_BEAM_WIDTH` | Survivors kept per tree layer after evaluation (beam width). | `3` | |
| `VIREX_BENCH_TOT_EVAL_SAMPLES` | Independent evaluator votes averaged to score each candidate path. | `3` | |
| `VIREX_BENCH_TOT_PROPOSE_TEMPERATURE` | Proposer sampling temperature. | _unset_ (`None`) | `None` inherits the LM's `--model-kwargs` profile; set higher (e.g. `0.7`) for more diverse thoughts. |
| `VIREX_BENCH_TOT_EVALUATE_TEMPERATURE` | Evaluator sampling temperature. | _unset_ (`None`) | `None` inherits the LM's profile; set to `0.0` for deterministic scoring. |
| `VIREX_BENCH_TOT_SEARCH_ALGORITHM` | Search algorithm for the bare `tot` strategy. | `beam` | CLI composites like `tot-beam` override this. Matched case-insensitively. Choices: `beam`, `dfs`, `mcts`. |
| `VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD` | Fuzzy dedupe threshold for proposed thoughts. | `0.9` | Higher is more conservative. |

### Tree-of-Thoughts (ToT) — Per-Variant Knobs

Knobs specific to one search algorithm; semantics differ by variant (resolved by `_build_search_config`).

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD` | Beam: stop the search once the best path scores >= this. | _unset_ (`None`) | `_build_search_config` applies `9.0` when unset. `0` disables the threshold. DFS reuses this config slot for value-pruning via `TOT_DFS_PRUNE_THRESHOLD` (default `3.0`, different semantics); for DFS stop-on-success, see `TOT_DFS_STOP_THRESHOLD`. |
| `VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD` | DFS: value-pruning threshold (`v_th`). Children scored below this are evaluated & counted but NOT expanded. | _unset_ (`None`) | `_build_search_config` applies `3.0` when unset. `0` disables pruning. Shares beam's `early_stop_threshold` slot but is pruning, not stop-on-success; the beam/DFS stop-on-success knobs are `TOT_BEAM_EARLY_STOP_THRESHOLD` and `TOT_DFS_STOP_THRESHOLD`. |
| `VIREX_BENCH_TOT_DFS_STOP_THRESHOLD` | DFS: stop-on-success threshold (analogous to beam's). | _unset_ (`None`) | `_build_search_config` applies `9.0` when unset. `0` disables it. |
| `VIREX_BENCH_TOT_DFS_MAX_ITERATIONS` | DFS: hard cap on node expansions (one proposer call each). | _unset_ (`None`) | Unset/empty lets `DFSSearch` auto-derive `max_depth * branching_factor`. Beam ignores this. |
| `VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT` | MCTS: UCT exploration constant (`c`). | `1.414` | |
| `VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS` | MCTS: hard cap on iterations. | _unset_ (`None`) | `MCTSSearch` applies a 30-iteration default when unset. Ignored by beam/DFS. |
| `VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD` | MCTS: stop-on-success threshold (analogous to beam/DFS). | _unset_ (`None`) | `_build_search_config` applies `9.0` when unset. `0` disables it. |

### Cumulative Reasoning (CR)

Propose/verify loop that accumulates entailed propositions before a solver commits the answer.

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_CR_TARGET_PROPOSITIONS` | Cap on accepted propositions accumulated before the Solver runs. | `7` | Default covers ~96% of gold reasoning chains (chains top out at 10). Lower for cheaper runs (`3` -> ~71%, `5` -> ~87%); raise toward `10` for max fidelity. |
| `VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS` | Cap on rejected/duplicate proposals before the loop gives up. | `6` | Guards a proposer that keeps emitting junk on a hard example. |
| `VIREX_BENCH_CR_VERIFIER_MODE` | Verifier gate config. | `multi` | `multi` = meaningfulness pre-filter then a validity check; `single` = validity check only. Matched case-insensitively. |
| `VIREX_BENCH_CR_PROPOSE_TEMPERATURE` | Proposer sampling temperature. | _unset_ (`None`) | `None` inherits the LM's `--model-kwargs` profile (no role-specific temperature is forced). |
| `VIREX_BENCH_CR_VERIFY_TEMPERATURE` | Verifier sampling temperature. | _unset_ (`None`) | `None` inherits the LM's profile; set to a float to override (e.g. `0.5` for more deterministic verdicts). |
| `VIREX_BENCH_CR_N_PROPOSE_SAMPLES` | Number of candidate propositions sampled per propose call. | `3` | `> 1` gives the loop multiple diverse candidates (deduped, verified in order), combating proposer diversity exhaustion. Mirrors ToT's branching factor. |
| `VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD` | Fuzzy dedupe threshold for proposed propositions. | `0.9` | Higher is more conservative. |

### Program-of-Thought (PoT / Z3)

Generates a Z3 program from the premises, executes it sandboxed, and regenerates on failure.

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_POT_MAX_ITERS` | Code-regeneration attempts allowed on execution failure. Each retry feeds the previous code + error back to the LM. | `3` | |
| `VIREX_BENCH_POT_EXECUTION_TIMEOUT` | Wall-clock timeout for the sandboxed Z3-program subprocess (seconds). | `15.0` | Guards against pathological/quantified formulas hanging the solver; typical tasks decide in well under a second. |
| `VIREX_BENCH_POT_GENERATE_TEMPERATURE` | Sampling temperature for the initial code-generation call. | _unset_ (`None`) | `None` inherits the LM's `--model-kwargs` profile. |
| `VIREX_BENCH_POT_REGENERATE_TEMPERATURE` | Sampling temperature for the regeneration call (fed previous code + error). | _unset_ (`None`) | `None` inherits the LM's profile. |
| `VIREX_BENCH_POT_FALLBACK_ON_ERROR` | Behavior after exhausting `MAX_ITERS` regenerations on an unrecoverable execution failure. | `1` (`True`) | `True` = commit an answer by reasoning over the premises directly (no solver result); `False` (or `0`) = raise `RuntimeError`. Truthy values: `1`, `true`, `yes`, `on` (case-insensitive). |

## Decoding Strategies (`--decoding`)

Knobs for the answer-aggregation methods selected via `--decoding`.

### Self-Consistency Decoding

Samples N reasoning paths and merges them via majority vote + optional LLM aggregation. `direct`/`cot` only.

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_SC_NUM_SAMPLES` | Number of independent reasoning paths sampled per example. | `5` | |
| `VIREX_BENCH_SC_MAX_WORKERS` | Maximum parallel workers for the path thread pool. | `5` | |
| `VIREX_BENCH_SC_SOLVE_TIMEOUT` | Per-path wall-clock timeout (seconds). | `360` | |
| `VIREX_BENCH_SC_AGGREGATE_TIMEOUT` | Aggregator LM call timeout (seconds). | `120` | |
| `VIREX_BENCH_SC_USE_AGGREGATOR` | Whether to use the LLM aggregator. | `1` (`True`) | `False` (or `0`) = deterministic majority vote only. Truthy values: `1`, `true`, `yes`, `on` (case-insensitive). |

### Self-Certainty Decoding

Samples N candidates and Borda-votes by mean token log-prob. `direct`/`cot` only; needs `temperature > 0` and an endpoint that returns `logprobs`.

| Variable | Description | Default | Note |
| --- | --- | --- | --- |
| `VIREX_BENCH_SELFC_NUM_SAMPLES` | Number of independent candidates sampled per example before Borda-voting. | `5` | Must be `>= 1`. |
| `VIREX_BENCH_SELFC_MAX_WORKERS` | Maximum parallel workers for the path thread pool. | `5` | Must be `>= 1`. |
| `VIREX_BENCH_SELFC_SOLVE_TIMEOUT` | Per-path wall-clock timeout (seconds). | `360` | |
| `VIREX_BENCH_SELFC_BORDA_POWER` | Borda voting exponent. Candidates are ranked by certainty (mean token log-prob) descending; each casts a weighted vote of `(N - rank + 1) ** borda_power` for its normalized answer. | `0.5` | Must be `>= 0`. `0` collapses to plain majority vote; larger values push toward pure certainty-based selection. Falls back to uniform weights (majority vote) if the endpoint returns no log-probs. |
