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

# Type-only declarations so static checkers see typed attributes; the values are
# produced dynamically by `__getattr__` (this block never runs at runtime).
if TYPE_CHECKING:
    # Logging verbosity for the `virex_bench` logger (matched case-insensitively).
    VIREX_BENCH_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ANSI color in log output: "auto" (TTY only), "1" always, "0" never.
    VIREX_BENCH_LOG_COLOR: Literal["auto", "0", "1"] = "auto"

    # Standard cross-tool opt-out (https://no-color.org); any non-empty value
    # disables colored output regardless of VIREX_BENCH_LOG_COLOR.
    NO_COLOR: bool = False

    # Default output directory when `--output-dir` is unset.
    VIREX_BENCH_OUTPUT_DIR: str = "results"

    # Base URL of the OpenAI-compatible endpoint (e.g. "http://localhost:8000/v1").
    OPENAI_BASE_URL: str | None = None

    # API key for the OpenAI-compatible endpoint.
    OPENAI_API_KEY: str | None = None

    # HuggingFace Hub access token (required for gated/private task datasets).
    HF_TOKEN: str | None = None

    # API key for the judge model provider, used by the LLM-as-a-judge metric.
    VIREX_BENCH_JUDGE_API_KEY: str | None = None

    # Base URL of the OpenAI-compatible endpoint serving a self-hosted judge model
    # (required when VIREX_BENCH_JUDGE_MODEL starts with "hosted_vllm/").
    VIREX_BENCH_JUDGE_BASE_URL: str | None = None
    VIREX_BENCH_JUDGE_MODEL: Literal[
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "opencode-go/deepseek-v4-pro",
        "opencode-go/deepseek-v4-flash",
        "hosted_vllm/deepseek-v4-pro",
        "hosted_vllm/deepseek-v4-flash",
    ] = "deepseek/deepseek-v4-flash"

    # Max attempts (incl. the first call) for transient LM errors (timeouts, resets, 429/500/503).
    VIREX_BENCH_LM_MAX_RETRIES: int = 3
    # Initial wait between LM retries (seconds).
    VIREX_BENCH_LM_RETRY_MIN_WAIT: float = 1.0
    # Capped max wait between LM retries (seconds).
    VIREX_BENCH_LM_RETRY_MAX_WAIT: float = 30.0
    # Max random jitter added to each LM retry wait (seconds).
    VIREX_BENCH_LM_RETRY_JITTER: float = 1.0

    # --- Tree-of-Thoughts (ToT) tuning ---

    # Number of reasoning steps explored (tree depth). Tuned for the dataset
    # (chains can reach 10+ steps); lower per-run for shorter-chain subsets.
    VIREX_BENCH_TOT_MAX_DEPTH: int = 10
    # Candidate next-thoughts requested per node per step (proposer branching factor).
    VIREX_BENCH_TOT_BRANCHING_FACTOR: int = 3
    # Survivors kept per tree layer after evaluation (beam width).
    VIREX_BENCH_TOT_BEAM_WIDTH: int = 3
    # Independent evaluator votes averaged to score each candidate path.
    VIREX_BENCH_TOT_EVAL_SAMPLES: int = 3
    # Proposer sampling temperature. None (default) -> inherit the LM's
    # `--model-kwargs` profile; set higher (e.g. 0.7) for more diverse thoughts.
    VIREX_BENCH_TOT_PROPOSE_TEMPERATURE: float | None = None
    # Evaluator sampling temperature. None (default) -> inherit the LM's
    # `--model-kwargs` profile; set to 0.0 for deterministic scoring.
    VIREX_BENCH_TOT_EVALUATE_TEMPERATURE: float | None = None
    # Search algorithm for the bare `tot` strategy. CLI composites like `tot-beam`
    # override this. Choices: beam, dfs, mcts.
    VIREX_BENCH_TOT_SEARCH_ALGORITHM: str = "beam"
    # Fuzzy dedupe threshold for proposed thoughts. Higher is more conservative.
    VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD: float = 0.9
    # --- Per-variant knobs (semantics differ by algorithm; resolved by
    # _build_search_config in strategies/tot/strategy.py). ---

    # Beam: stop the search once the best path scores >= this. None when unset;
    # _build_search_config applies the 9.0 default. "0" disables the threshold.
    # DFS uses TOT_DFS_PRUNE_THRESHOLD instead.
    VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD: float | None = None
    # DFS: value-pruning threshold (v_th). Children scored below this are
    # evaluated & counted but NOT expanded. None when unset; _build_search_config
    # applies the 3.0 default. "0" disables pruning. Beam uses
    # TOT_BEAM_EARLY_STOP_THRESHOLD instead.
    VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD: float | None = None
    # DFS: stop-on-success threshold (analogous to beam's). None when unset;
    # _build_search_config applies the 9.0 default. "0" disables it.
    VIREX_BENCH_TOT_DFS_STOP_THRESHOLD: float | None = None
    # DFS: hard cap on node expansions (one proposer call each). Unset/empty
    # lets DFSSearch auto-derive max_depth * branching_factor. Beam ignores this.
    VIREX_BENCH_TOT_DFS_MAX_ITERATIONS: int | None = None
    # MCTS: UCT exploration constant (c). Forward-declared.
    VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT: float = 1.414
    # MCTS: hard cap on iterations. None when unset; MCTSSearch applies the
    # 30-iteration default. Ignored by beam/DFS.
    VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS: int | None = None
    # MCTS: stop-on-success threshold (analogous to beam/DFS). None when unset;
    # _build_search_config applies the 9.0 default. "0" disables it.
    VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD: float | None = None

    # --- Cumulative Reasoning (CR) strategy ---

    # Cap on accepted propositions accumulated before the Solver runs. Default 7
    # covers ~96% of gold reasoning chains in the dataset (chains top out at 10).
    # Lower for cheaper runs (3 -> ~71%, 5 -> ~87%); raise toward 10 for max fidelity.
    VIREX_BENCH_CR_TARGET_PROPOSITIONS: int = 7
    # Cap on rejected/duplicate proposals before the loop gives up. Guards a
    # proposer that keeps emitting junk on a hard example.
    VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS: int = 6
    # Verifier gate config. "multi" (default) = meaningfulness pre-filter then a
    # validity check; "single" = validity check only.
    VIREX_BENCH_CR_VERIFIER_MODE: Literal["single", "multi"] = "multi"
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

    # --- Program-of-Thought + Z3 (pot_z3) strategy ---

    # Number of code-regeneration attempts on execution failure (feed
    # previous_code + error back to the LM). >= 1.
    VIREX_BENCH_POT_MAX_ITERS: int = 3
    # Wall-clock timeout for the Z3 subprocess (seconds). These tasks decide in
    # well under a second; the ceiling guards pathological quantified formulas or
    VIREX_BENCH_POT_EXECUTION_TIMEOUT: float = 15.0
    # Code-generation sampling temperature. None -> inherit the LM's profile.
    VIREX_BENCH_POT_GENERATE_TEMPERATURE: float | None = None
    # Regeneration sampling temperature. None -> inherit the LM's profile.
    VIREX_BENCH_POT_REGENERATE_TEMPERATURE: float | None = None
    # On unrecoverable execution failure: True lets the commit LM fall back to
    # plain reasoning over the premises; False raises.
    VIREX_BENCH_POT_FALLBACK_ON_ERROR: bool = True

    # --- Self-consistency decoding ---
    # Number of independent reasoning paths sampled per example.
    VIREX_BENCH_SC_NUM_SAMPLES: int = 5
    # Maximum parallel workers for the path thread pool.
    VIREX_BENCH_SC_MAX_WORKERS: int = 5
    # Per-path wall-clock timeout (seconds).
    VIREX_BENCH_SC_SOLVE_TIMEOUT: int = 360
    # Aggregator LM call timeout (seconds).
    VIREX_BENCH_SC_AGGREGATE_TIMEOUT: int = 120
    # Whether to use the LLM aggregator (False = deterministic vote only).
    VIREX_BENCH_SC_USE_AGGREGATOR: bool = True

    # --- Self-certainty decoding ---
    # Number of independent reasoning paths sampled per example.
    VIREX_BENCH_SELFC_NUM_SAMPLES: int = 5
    # Maximum parallel workers for the path thread pool.
    VIREX_BENCH_SELFC_MAX_WORKERS: int = 5
    # Per-path wall-clock timeout (seconds).
    VIREX_BENCH_SELFC_SOLVE_TIMEOUT: int = 360
    # Borda voting exponent. 0 -> plain majority vote (self-consistency);
    # larger -> pure certainty selection. The paper's code default is 0.5.
    VIREX_BENCH_SELFC_BORDA_POWER: float = 0.5


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


# Single source of truth. Each value is a zero-arg callable that reads `os.environ`
# at invocation time, keeping access lazy.
environment_variables: dict[str, Callable[[], Any]] = {
    # --- Logging ---
    # Upper-cased so it can be passed straight to `logging.setLevel`.
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
    "VIREX_BENCH_JUDGE_BASE_URL": lambda: os.environ.get("VIREX_BENCH_JUDGE_BASE_URL", None),
    "VIREX_BENCH_JUDGE_MODEL": env_with_choices(
        "VIREX_BENCH_JUDGE_MODEL",
        "deepseek/deepseek-v4-flash",
        [
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "opencode-go/deepseek-v4-pro",
            "opencode-go/deepseek-v4-flash",
            "hosted_vllm/deepseek-v4-pro",
            "hosted_vllm/deepseek-v4-flash",
        ],
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
    "VIREX_BENCH_TOT_PROPOSE_TEMPERATURE": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_TOT_PROPOSE_TEMPERATURE", None)
    ),
    "VIREX_BENCH_TOT_EVALUATE_TEMPERATURE": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_TOT_EVALUATE_TEMPERATURE", None)
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
    # --- Program-of-Thought + Z3 (pot_z3) strategy ---
    "VIREX_BENCH_POT_MAX_ITERS": lambda: int(os.environ.get("VIREX_BENCH_POT_MAX_ITERS", "3")),
    "VIREX_BENCH_POT_EXECUTION_TIMEOUT": lambda: float(
        os.environ.get("VIREX_BENCH_POT_EXECUTION_TIMEOUT", "15.0")
    ),
    "VIREX_BENCH_POT_GENERATE_TEMPERATURE": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_POT_GENERATE_TEMPERATURE", None)
    ),
    "VIREX_BENCH_POT_REGENERATE_TEMPERATURE": lambda: maybe_convert_float(
        os.environ.get("VIREX_BENCH_POT_REGENERATE_TEMPERATURE", None)
    ),
    "VIREX_BENCH_POT_FALLBACK_ON_ERROR": lambda: get_bool(
        "VIREX_BENCH_POT_FALLBACK_ON_ERROR", "1"
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
    # --- Self-certainty decoding ---
    "VIREX_BENCH_SELFC_NUM_SAMPLES": lambda: int(
        os.environ.get("VIREX_BENCH_SELFC_NUM_SAMPLES", "5")
    ),
    "VIREX_BENCH_SELFC_MAX_WORKERS": lambda: int(
        os.environ.get("VIREX_BENCH_SELFC_MAX_WORKERS", "5")
    ),
    "VIREX_BENCH_SELFC_SOLVE_TIMEOUT": lambda: int(
        os.environ.get("VIREX_BENCH_SELFC_SOLVE_TIMEOUT", "360")
    ),
    "VIREX_BENCH_SELFC_BORDA_POWER": lambda: float(
        os.environ.get("VIREX_BENCH_SELFC_BORDA_POWER", "0.5")
    ),
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
