"""Centralized environment-variable access for ViREx-Bench.

All environment variables are read lazily (at access time, not import time) through a
module-level `__getattr__`, following the pattern used by vLLM's `vllm/envs.py`. Read
them as attributes, e.g.:

    from virex_bench import envs

    level = envs.VIREX_BENCH_LOG_LEVEL

Lazy access matters because tests (and code that sets variables programmatically before
use) see the current value of `os.environ` rather than whatever it was at import time.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

# Type-only declarations so Pyright/IDEs see properly typed module attributes even
# though the values are produced dynamically by `__getattr__`. This block is never
# executed at runtime (`TYPE_CHECKING` is always False).
if TYPE_CHECKING:
    # Logging verbosity for the `virex_bench` logger (matched case-insensitively).
    VIREX_BENCH_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Controls ANSI color in log output: "auto" colors only when the stream is a TTY,
    # "1" always colors, "0" never colors.
    VIREX_BENCH_LOG_COLOR: Literal["auto", "0", "1"] = "auto"

    # Standard cross-tool opt-out (https://no-color.org): when set to any non-empty
    # value, disables colored output regardless of VIREX_BENCH_LOG_COLOR.
    NO_COLOR: bool = False

    # Default directory benchmark results are written into when `--output-dir` is unset.
    VIREX_BENCH_OUTPUT_DIR: str = "results"

    # Base URL of the OpenAI-compatible endpoint (e.g. "http://localhost:8000/v1").
    OPENAI_BASE_URL: str | None = None

    # API key for the OpenAI-compatible endpoint.
    OPENAI_API_KEY: str | None = None

    # Access token for the HuggingFace Hub (used to load task datasets). Optional for
    # public datasets; required for gated or private ones.
    HF_TOKEN: str | None = None

    # API key for the judge model provider, used by the LLM-as-a-judge metric.
    VIREX_BENCH_JUDGE_API_KEY: str | None = None

    # DeepSeek model used by the LLM-as-a-judge metric.
    VIREX_BENCH_JUDGE_MODEL: Literal["deepseek-v4-pro", "deepseek-v4-flash"] = "deepseek-v4-flash"

    # Maximum number of attempts (including the first call) for transient LM
    # connection errors (timeouts, connection resets, HTTP 429/500/503).
    VIREX_BENCH_LM_MAX_RETRIES: int = 3
    # Initial wait between LM retries, in seconds.
    VIREX_BENCH_LM_RETRY_MIN_WAIT: float = 1.0
    # Maximum (capped) wait between LM retries, in seconds.
    VIREX_BENCH_LM_RETRY_MAX_WAIT: float = 30.0
    # Maximum random jitter added to each LM retry wait, in seconds.
    VIREX_BENCH_LM_RETRY_JITTER: float = 1.0

    # --- Tree-of-Thoughts (ToT) strategy tuning ---
    # All ToT knobs are optional; when unset the defaults below are used. Tune a
    # run with e.g. `VIREX_BENCH_TOT_MAX_DEPTH=4 uv run vb run --strategy tot ...`.

    # Number of reasoning steps (tree depth) explored before committing an answer.
    # Tuned for the dataset, whose chains can reach 10+ steps; lower per-run for
    # shorter-chain subsets. Affects beam (D*B*(1+b) LM calls) and DFS alike.
    VIREX_BENCH_TOT_MAX_DEPTH: int = 10
    # Candidate next-thoughts requested per node per step (proposer branching factor).
    VIREX_BENCH_TOT_BRANCHING_FACTOR: int = 3
    # Survivors kept per tree layer after evaluation (beam width).
    VIREX_BENCH_TOT_BEAM_WIDTH: int = 3
    # Independent evaluator votes averaged to score each candidate path.
    VIREX_BENCH_TOT_EVAL_SAMPLES: int = 3
    # Sampling temperature for the proposer (higher -> more diverse thoughts).
    VIREX_BENCH_TOT_PROPOSE_TEMPERATURE: float = 0.7
    # Sampling temperature for the evaluator (0.0 -> deterministic scoring).
    VIREX_BENCH_TOT_EVALUATE_TEMPERATURE: float = 0.0
    # Search algorithm for the bare `tot` strategy (CLI composite names like
    # `tot-beam` override this). Choices: beam, dfs, mcts.
    VIREX_BENCH_TOT_SEARCH_ALGORITHM: str = "beam"
    # Fuzzy dedupe threshold for proposed thoughts. Higher is more conservative.
    VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD: float = 0.9
    # --- Per-variant knobs (semantics differ by algorithm; resolved by
    # _build_search_config in strategies/tot/strategy.py). ---
    # Beam: stop the WHOLE search once the best path scores >= this (stop-on-success).
    # The env returns None when unset/empty; _build_search_config applies the 9.0
    # default (the evaluator reserves 9-10 for paths that have already derived the
    # answer). Setting this to "0" disables the threshold. DFS uses
    # TOT_DFS_PRUNE_THRESHOLD instead.
    VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD: float | None = None
    # DFS: ToT's value-pruning threshold (v_th). Children scored below this are
    # evaluated & counted but NOT expanded. The env returns None when unset/empty;
    # _build_search_config applies the 3.0 default (the evaluator reserves 1-3 for
    # "dead end" paths, so this prunes only true dead-ends while letting "on track
    # but incomplete" steps survive). Setting this to "0" disables
    # pruning. Beam uses TOT_BEAM_EARLY_STOP_THRESHOLD instead.
    VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD: float | None = None
    # DFS: stop-on-success threshold. When the best path's evaluator score reaches
    # this, the entire search stops immediately -- analogous to beam's
    # TOT_BEAM_EARLY_STOP_THRESHOLD. The env returns None when unset/empty;
    # _build_search_config applies the 9.0 default. Setting this to "0" disables
    # the stop-on-success.
    VIREX_BENCH_TOT_DFS_STOP_THRESHOLD: float | None = None
    # DFS: hard cap on node expansions (one proposer call each). Unset/empty lets
    # DFSSearch auto-derive max_depth * branching_factor. Beam ignores this.
    VIREX_BENCH_TOT_DFS_MAX_ITERATIONS: int | None = None
    # MCTS: UCT exploration constant (c). Forward-declared.
    VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT: float = 1.414
    # MCTS: hard cap on iterations. The env returns None when unset/empty;
    # MCTSSearch then applies the 30-iteration default (empirically tuned).
    # Ignored by beam/DFS.
    VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS: int | None = None
    # MCTS: stop-on-success threshold (analogous to beam/DFS). When the max raw
    # evaluator score seen during the search reaches it, MCTS halts early. The
    # env returns None when unset/empty; _build_search_config applies the 9.0
    # default (the evaluator reserves 9-10 for paths that have already derived
    # the answer). Setting this to "0" disables the stop-on-success.
    VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD: float | None = None

    # --- Cumulative Reasoning (CR) strategy ---
    # CR accumulates verified propositions, then solves. All knobs are optional;
    # when unset the defaults below are used. Tune a run with e.g.
    # `VIREX_BENCH_CR_TARGET_PROPOSITIONS=5 uv run vb run --strategy cr ...`.

    # Cap on accepted propositions accumulated before the Solver runs. Default 7
    # covers ~96% of gold reasoning chains in the dataset (the `premises_used`
    # histogram tops out at 10). Lower for cheaper experiments (3 -> ~71%,
    # 5 -> ~87%); raise toward 10 for max fidelity. See the strategy plan for
    # the full coverage/cost table.
    VIREX_BENCH_CR_TARGET_PROPOSITIONS: int = 7
    # Cap on rejected/duplicate proposals before the loop gives up and solves
    # with whatever it accumulated. Guards against a proposer that keeps emitting
    # junk on a hard example.
    VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS: int = 6
    # Verifier gate configuration. "multi" (default) runs a meaningfulness
    # pre-filter then a validity check -- more robust, one extra LLM call per
    # proposal. "single" runs only the validity check.
    VIREX_BENCH_CR_VERIFIER_MODE: Literal["single", "multi"] = "multi"
    # Sampling temperature for the proposer. Unset/empty -> inherit the LM's
    # `--model-kwargs` profile (the default: modern cards like Qwen3's publish a
    # tuned temp+penalty system and forbid greedy decoding, so we do not force a
    # role-specific temperature). Proposer diversity comes from the per-call `n`
    # rather than a cranked temperature. Set this to override temperature only.
    # Proposer sampling temperature. None -> inherit the LM's `--model-kwargs`
    # profile (we do not force a role-specific temperature).
    VIREX_BENCH_CR_PROPOSE_TEMPERATURE: float | None = None
    # Verifier sampling temperature. None (default) inherits the LM's profile;
    # set to a float to override (e.g. 0.5 for more deterministic verdicts).
    VIREX_BENCH_CR_VERIFY_TEMPERATURE: float | None = None
    # Number of candidate propositions sampled per propose call. > 1 gives the
    # loop multiple diverse candidates to try (deduped, verified in order),
    # combating proposer diversity exhaustion. Mirrors ToT's branching factor.
    VIREX_BENCH_CR_N_PROPOSE_SAMPLES: int = 3
    # Fuzzy dedupe threshold for proposed propositions. Higher is more conservative.
    VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD: float = 0.9

    # --- Self-consistency decoding ---
    # Number of independent reasoning paths sampled per example.
    VIREX_BENCH_SC_NUM_SAMPLES: int = 5
    # Maximum parallel workers for the path thread pool.
    VIREX_BENCH_SC_MAX_WORKERS: int = 5
    # Per-path wall-clock timeout in seconds.
    VIREX_BENCH_SC_SOLVE_TIMEOUT: int = 360
    # Aggregator LM call timeout in seconds.
    VIREX_BENCH_SC_AGGREGATE_TIMEOUT: int = 120
    # Whether to use the LLM aggregator (False = deterministic vote only).
    VIREX_BENCH_SC_USE_AGGREGATOR: bool = True


def get_bool(env_name: str, default: str) -> bool:
    """Parse a boolean env var. Truthy values: 1, true, yes, on (case-insensitive)."""
    return os.environ.get(env_name, default).strip().lower() in ("1", "true", "yes", "on")


def env_with_choices(
    env_name: str,
    default: str,
    choices: list[str],
    case_sensitive: bool = True,
) -> Callable[[], str]:
    """Build a lazy accessor that validates an env var against allowed choices.

    The returned callable is suitable as a value in ``environment_variables``. On a
    match the original (un-normalized) value is returned; an invalid choice raises
    ``ValueError``.

    Args:
        env_name: Name of the environment variable.
        default: Value returned when the variable is unset.
        choices: Allowed string values.
        case_sensitive: When ``False``, the value is matched against the choices
            case-insensitively.
    """

    def _get_validated_env() -> str:
        value = os.environ.get(env_name)
        if value is None:
            return default
        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in choices]
        else:
            check_value = value
            check_choices = choices
        if check_value not in check_choices:
            raise ValueError(f"Invalid value {value!r} for {env_name}. Valid options: {choices}.")
        return value

    return _get_validated_env


def maybe_convert_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def maybe_convert_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return bool(int(value))


# Single source of truth. Each value is a zero-argument callable that reads `os.environ`
# when invoked, keeping access lazy. Group with comments as the set grows.
environment_variables: dict[str, Callable[[], Any]] = {
    # --- Logging ---
    # Output is upper-cased so it can be passed straight to `logging.setLevel`, which
    # only accepts canonical upper-case level names.
    "VIREX_BENCH_LOG_LEVEL": lambda: env_with_choices(
        "VIREX_BENCH_LOG_LEVEL",
        "INFO",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        case_sensitive=False,
    )().upper(),
    "VIREX_BENCH_LOG_COLOR": env_with_choices(
        "VIREX_BENCH_LOG_COLOR",
        "auto",
        ["auto", "0", "1"],
        case_sensitive=False,
    ),
    "NO_COLOR": lambda: len(os.environ.get("NO_COLOR", "")) > 0,
    # --- Paths ---
    "VIREX_BENCH_OUTPUT_DIR": lambda: os.environ.get("VIREX_BENCH_OUTPUT_DIR", "results"),
    # --- Model endpoint (OpenAI-compatible) ---
    "OPENAI_BASE_URL": lambda: os.environ.get("OPENAI_BASE_URL", None),
    "OPENAI_API_KEY": lambda: os.environ.get("OPENAI_API_KEY", None),
    # --- HuggingFace Hub (task datasets) ---
    "HF_TOKEN": lambda: os.environ.get("HF_TOKEN", None),
    # --- LLM-as-a-judge ---
    "VIREX_BENCH_JUDGE_API_KEY": lambda: os.environ.get("VIREX_BENCH_JUDGE_API_KEY", None),
    "VIREX_BENCH_JUDGE_MODEL": env_with_choices(
        "VIREX_BENCH_JUDGE_MODEL",
        "deepseek-v4-flash",
        ["deepseek-v4-pro", "deepseek-v4-flash"],
        case_sensitive=False,
    ),
    # --- LM retry (transient connection / HTTP errors) ---
    "VIREX_BENCH_LM_MAX_RETRIES": lambda: int(os.environ.get("VIREX_BENCH_LM_MAX_RETRIES", "3")),
    "VIREX_BENCH_LM_RETRY_MIN_WAIT": lambda: float(
        os.environ.get("VIREX_BENCH_LM_RETRY_MIN_WAIT", "1.0")
    ),
    "VIREX_BENCH_LM_RETRY_MAX_WAIT": lambda: float(
        os.environ.get("VIREX_BENCH_LM_RETRY_MAX_WAIT", "30.0")
    ),
    "VIREX_BENCH_LM_RETRY_JITTER": lambda: float(
        os.environ.get("VIREX_BENCH_LM_RETRY_JITTER", "1.0")
    ),
    # --- Tree-of-Thoughts (ToT) strategy ---
    "VIREX_BENCH_TOT_MAX_DEPTH": lambda: int(os.environ.get("VIREX_BENCH_TOT_MAX_DEPTH", "10")),
    "VIREX_BENCH_TOT_BRANCHING_FACTOR": lambda: int(
        os.environ.get("VIREX_BENCH_TOT_BRANCHING_FACTOR", "3")
    ),
    "VIREX_BENCH_TOT_BEAM_WIDTH": lambda: int(os.environ.get("VIREX_BENCH_TOT_BEAM_WIDTH", "3")),
    "VIREX_BENCH_TOT_EVAL_SAMPLES": lambda: int(
        os.environ.get("VIREX_BENCH_TOT_EVAL_SAMPLES", "3")
    ),
    "VIREX_BENCH_TOT_PROPOSE_TEMPERATURE": lambda: float(
        os.environ.get("VIREX_BENCH_TOT_PROPOSE_TEMPERATURE", "0.7")
    ),
    "VIREX_BENCH_TOT_EVALUATE_TEMPERATURE": lambda: float(
        os.environ.get("VIREX_BENCH_TOT_EVALUATE_TEMPERATURE", "0.0")
    ),
    "VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD", None)
    ),
    "VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD", None)
    ),
    "VIREX_BENCH_TOT_DFS_STOP_THRESHOLD": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_TOT_DFS_STOP_THRESHOLD", None)
    ),
    "VIREX_BENCH_TOT_SEARCH_ALGORITHM": lambda: env_with_choices(
        "VIREX_BENCH_TOT_SEARCH_ALGORITHM",
        "beam",
        ["beam", "dfs", "mcts"],
        case_sensitive=False,
    )().lower(),
    "VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD": lambda: float(
        os.environ.get("VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD", "0.9")
    ),
    "VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT": lambda: float(
        os.environ.get("VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT", "1.414")
    ),
    "VIREX_BENCH_TOT_DFS_MAX_ITERATIONS": lambda: maybe_convert_int(
        os.environ.get("VIREX_BENCH_TOT_DFS_MAX_ITERATIONS", None)
    ),
    "VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS": lambda: maybe_convert_int(
        os.environ.get("VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS", None)
    ),
    "VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD", None)
    ),
    # --- Cumulative Reasoning (CR) strategy ---
    "VIREX_BENCH_CR_TARGET_PROPOSITIONS": lambda: int(
        os.environ.get("VIREX_BENCH_CR_TARGET_PROPOSITIONS", "7")
    ),
    "VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS": lambda: int(
        os.environ.get("VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS", "6")
    ),
    "VIREX_BENCH_CR_VERIFIER_MODE": lambda: env_with_choices(
        "VIREX_BENCH_CR_VERIFIER_MODE",
        "multi",
        ["single", "multi"],
        case_sensitive=False,
    )().lower(),
    "VIREX_BENCH_CR_PROPOSE_TEMPERATURE": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_CR_PROPOSE_TEMPERATURE", None)
    ),
    "VIREX_BENCH_CR_VERIFY_TEMPERATURE": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_CR_VERIFY_TEMPERATURE", None)
    ),
    "VIREX_BENCH_CR_N_PROPOSE_SAMPLES": lambda: int(
        os.environ.get("VIREX_BENCH_CR_N_PROPOSE_SAMPLES", "3")
    ),
    "VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD": lambda: float(
        os.environ.get("VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD", "0.9")
    ),
    # --- Self-consistency decoding ---
    "VIREX_BENCH_SC_NUM_SAMPLES": lambda: int(os.environ.get("VIREX_BENCH_SC_NUM_SAMPLES", "5")),
    "VIREX_BENCH_SC_MAX_WORKERS": lambda: int(os.environ.get("VIREX_BENCH_SC_MAX_WORKERS", "5")),
    "VIREX_BENCH_SC_SOLVE_TIMEOUT": lambda: int(
        os.environ.get("VIREX_BENCH_SC_SOLVE_TIMEOUT", "360")
    ),
    "VIREX_BENCH_SC_AGGREGATE_TIMEOUT": lambda: int(
        os.environ.get("VIREX_BENCH_SC_AGGREGATE_TIMEOUT", "120")
    ),
    "VIREX_BENCH_SC_USE_AGGREGATOR": lambda: get_bool("VIREX_BENCH_SC_USE_AGGREGATOR", "1"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)


def is_set(env_name: str) -> bool:
    """Return whether an environment variable is explicitly set in the environment."""
    return env_name in os.environ
