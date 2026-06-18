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
from typing import TYPE_CHECKING, Any

# Type-only declarations so Pyright/IDEs see properly typed module attributes even
# though the values are produced dynamically by `__getattr__`. This block is never
# executed at runtime (`TYPE_CHECKING` is always False).
if TYPE_CHECKING:
    # Logging verbosity for the `virex_bench` logger (e.g. "DEBUG", "INFO", "WARNING").
    VIREX_BENCH_LOG_LEVEL: str = "INFO"

    # Default directory benchmark results are written into when `--output-dir` is unset.
    VIREX_BENCH_OUTPUT_DIR: str = "results"


def get_bool(env_name: str, default: str) -> bool:
    """Parse a boolean env var. Truthy values: 1, true, yes, on (case-insensitive)."""
    return os.environ.get(env_name, default).strip().lower() in ("1", "true", "yes", "on")


# Single source of truth. Each value is a zero-argument callable that reads `os.environ`
# when invoked, keeping access lazy. Group with comments as the set grows.
environment_variables: dict[str, Callable[[], Any]] = {
    # --- Logging ---
    "VIREX_BENCH_LOG_LEVEL": lambda: os.environ.get("VIREX_BENCH_LOG_LEVEL", "INFO").upper(),
    # --- Paths ---
    "VIREX_BENCH_OUTPUT_DIR": lambda: os.environ.get("VIREX_BENCH_OUTPUT_DIR", "results"),
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
