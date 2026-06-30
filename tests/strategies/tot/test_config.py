"""Tests for per-variant env-var resolution in ``_build_search_config``.

Verifies the targeted split: the shared operation knobs are read once, while the
semantically-overloaded threshold and the budget cap are resolved per
``search_algorithm`` (beam = stop-on-success + no cap; DFS = v_th pruning + DFS cap).
Defaults and env-var overrides are both exercised.
"""

import dspy
import pytest

from virex_bench.strategies.registry import get_strategy
from virex_bench.strategies.tot.strategy import ToTStrategy

_VARIANT_ENVS = [
    "VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD",
    "VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD",
    "VIREX_BENCH_TOT_DFS_MAX_ITERATIONS",
    "VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS",
]


class _TestSignature(dspy.Signature):
    """Minimal signature for strategy construction in tests."""

    premises: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _clear_variant_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VARIANT_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_beam_default_early_stop_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beam ships with stop-on-success disabled by default (preserves Phase 1 behavior)."""
    _clear_variant_envs(monkeypatch)
    strategy = get_strategy("tot-beam", _TestSignature)
    assert isinstance(strategy, ToTStrategy)
    assert strategy.config.early_stop_threshold is None
    # beam has a fixed budget; max_iterations is unused.
    assert strategy.config.max_iterations is None


def test_dfs_default_prune_threshold_is_five(monkeypatch: pytest.MonkeyPatch) -> None:
    """DFS defaults to a v_th pruning threshold of 5.0 (midpoint of the 1-10 band)."""
    _clear_variant_envs(monkeypatch)
    strategy = get_strategy("tot-dfs", _TestSignature)
    assert isinstance(strategy, ToTStrategy)
    assert strategy.config.early_stop_threshold == 5.0
    # Unset -> None here; DFSSearch auto-derives max_depth * branching_factor.
    assert strategy.config.max_iterations is None


def test_dfs_prune_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD`` overrides the 5.0 default."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD", "7.0")
    strategy = get_strategy("tot-dfs", _TestSignature)
    assert strategy.config.early_stop_threshold == 7.0


def test_dfs_max_iterations_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_DFS_MAX_ITERATIONS`` feeds the DFS expansion cap."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_MAX_ITERATIONS", "50")
    strategy = get_strategy("tot-dfs", _TestSignature)
    assert strategy.config.max_iterations == 50


def test_beam_early_stop_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD`` feeds beam's stop-on-success."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD", "9.0")
    strategy = get_strategy("tot-beam", _TestSignature)
    assert strategy.config.early_stop_threshold == 9.0


def test_dfs_threshold_does_not_leak_into_beam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the DFS prune threshold must not change beam's (separate) threshold."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD", "7.0")
    strategy = get_strategy("tot-beam", _TestSignature)
    assert strategy.config.early_stop_threshold is None  # beam reads its own var


def test_beam_threshold_does_not_leak_into_dfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting beam's stop threshold must not change DFS's pruning threshold."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD", "9.0")
    strategy = get_strategy("tot-dfs", _TestSignature)
    assert strategy.config.early_stop_threshold == 5.0  # DFS default, unaffected
