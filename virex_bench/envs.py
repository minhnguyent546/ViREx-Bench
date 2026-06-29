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

    # API key for the DeepSeek API, used by the LLM-as-a-judge metric.
    DEEPSEEK_API_KEY: str | None = None

    # DeepSeek model used by the LLM-as-a-judge metric.
    VIREX_BENCH_JUDGE_MODEL: Literal["deepseek-v4-pro", "deepseek-v4-flash"] = "deepseek-v4-pro"

    # Maximum number of attempts (including the first call) for transient LM
    # connection errors (timeouts, connection resets, HTTP 429/500/503).
    VIREX_BENCH_LM_MAX_RETRIES: int = 3
    # Initial wait between LM retries, in seconds.
    VIREX_BENCH_LM_RETRY_MIN_WAIT: float = 1.0
    # Maximum (capped) wait between LM retries, in seconds.
    VIREX_BENCH_LM_RETRY_MAX_WAIT: float = 30.0
    # Maximum random jitter added to each LM retry wait, in seconds.
    VIREX_BENCH_LM_RETRY_JITTER: float = 1.0

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
    "OPENAI_BASE_URL": lambda: os.environ.get("OPENAI_BASE_URL"),
    "OPENAI_API_KEY": lambda: os.environ.get("OPENAI_API_KEY"),
    # --- HuggingFace Hub (task datasets) ---
    "HF_TOKEN": lambda: os.environ.get("HF_TOKEN"),
    # --- LLM-as-a-judge (DeepSeek) ---
    "DEEPSEEK_API_KEY": lambda: os.environ.get("DEEPSEEK_API_KEY"),
    "VIREX_BENCH_JUDGE_MODEL": env_with_choices(
        "VIREX_BENCH_JUDGE_MODEL",
        "deepseek-v4-pro",
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
