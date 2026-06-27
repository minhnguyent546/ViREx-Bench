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
