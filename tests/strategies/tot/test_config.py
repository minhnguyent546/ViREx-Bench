"""Tests for per-variant env-var resolution in ``_build_search_config``.

Verifies the targeted split: the shared operation knobs are read once, while the
semantically-overloaded threshold and the budget cap are resolved per
``search_algorithm`` (beam = stop-on-success + no cap; DFS = v_th pruning + DFS cap).
Defaults and env-var overrides are both exercised.
"""

import dspy
import pytest

from virex_bench.strategies.registry import get_strategy
from virex_bench.strategies.tot.search import SearchConfig
from virex_bench.strategies.tot.strategy import ToTStrategy

_VARIANT_ENVS = [
    "VIREX_BENCH_TOT_BEAM_WIDTH",
    "VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD",
    "VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD",
    "VIREX_BENCH_TOT_DFS_STOP_THRESHOLD",
    "VIREX_BENCH_TOT_DFS_MAX_ITERATIONS",
    "VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT",
    "VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS",
    "VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD",
]


class _TestSignature(dspy.Signature):
    """Minimal signature for strategy construction in tests."""

    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _clear_variant_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VARIANT_ENVS:
        monkeypatch.delenv(name, raising=False)


def _get_tot_config(strategy_name: str) -> SearchConfig:
    strategy = get_strategy(strategy_name, _TestSignature)
    assert isinstance(strategy, ToTStrategy)
    return strategy.config


def test_beam_default_early_stop_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beam ships with a stop-on-success threshold of 9.0 by default."""
    _clear_variant_envs(monkeypatch)
    config = _get_tot_config("tot-beam")
    assert config.early_stop_threshold == 9.0
    # beam has a fixed budget; max_iterations is unused.
    assert config.max_iterations is None


def test_dfs_default_prune_threshold_is_three(monkeypatch: pytest.MonkeyPatch) -> None:
    """DFS defaults to a v_th pruning threshold of 3.0 (the 'dead end' boundary)."""
    _clear_variant_envs(monkeypatch)
    config = _get_tot_config("tot-dfs")
    assert config.early_stop_threshold == 3.0
    # Unset -> None here; DFSSearch auto-derives max_depth * branching_factor.
    assert config.max_iterations is None


def test_dfs_default_stop_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """DFS defaults to a stop-on-success threshold of 9.0 (analogous to beam)."""
    _clear_variant_envs(monkeypatch)
    config = _get_tot_config("tot-dfs")
    assert config.success_threshold == 9.0


def test_dfs_stop_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_DFS_STOP_THRESHOLD`` overrides the 9.0 default."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_STOP_THRESHOLD", "8.0")
    config = _get_tot_config("tot-dfs")
    assert config.success_threshold == 8.0


def test_dfs_prune_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD`` overrides the 3.0 default."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD", "7.0")
    config = _get_tot_config("tot-dfs")
    assert config.early_stop_threshold == 7.0


def test_dfs_max_iterations_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_DFS_MAX_ITERATIONS`` feeds the DFS expansion cap."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_MAX_ITERATIONS", "50")
    config = _get_tot_config("tot-dfs")
    assert config.max_iterations == 50


def test_dfs_ignores_other_algorithm_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """DFS construction should not parse beam/MCTS-only knobs."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_BEAM_WIDTH", "not-an-int")
    monkeypatch.setenv("VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT", "not-a-float")
    config = _get_tot_config("tot-dfs")
    assert config.beam_width is None
    assert config.exploration_constant is None


def test_beam_early_stop_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD`` feeds beam's stop-on-success."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD", "9.0")
    config = _get_tot_config("tot-beam")
    assert config.early_stop_threshold == 9.0


def test_beam_ignores_other_algorithm_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beam construction should not parse DFS/MCTS-only knobs."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_MAX_ITERATIONS", "not-an-int")
    monkeypatch.setenv("VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT", "not-a-float")
    config = _get_tot_config("tot-beam")
    assert config.max_iterations is None
    assert config.exploration_constant is None
    assert config.success_threshold is None


def test_dfs_threshold_does_not_leak_into_beam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the DFS prune threshold must not change beam's (separate) threshold."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD", "7.0")
    config = _get_tot_config("tot-beam")
    assert config.early_stop_threshold == 9.0  # beam default, unaffected by DFS env


def test_mcts_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCTS defaults: exploration constant sqrt(2), stop-on-success 9.0, cap 30."""
    _clear_variant_envs(monkeypatch)
    config = _get_tot_config("tot-mcts")
    assert config.exploration_constant == 1.414
    assert config.success_threshold == 9.0
    assert config.max_iterations is None  # MCTSSearch resolves the default (30).
    assert config.early_stop_threshold is None  # MCTS does not use v_th pruning.


def test_mcts_stop_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD`` overrides the 9.0 default."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD", "8.0")
    config = _get_tot_config("tot-mcts")
    assert config.success_threshold == 8.0


def test_mcts_max_iterations_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS`` feeds the MCTS iteration cap."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS", "20")
    config = _get_tot_config("tot-mcts")
    assert config.max_iterations == 20


def test_mcts_exploration_constant_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT`` overrides the sqrt(2) default."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT", "2.0")
    config = _get_tot_config("tot-mcts")
    assert config.exploration_constant == 2.0


def test_mcts_ignores_other_algorithm_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCTS construction should not parse beam/DFS-only knobs."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_BEAM_WIDTH", "not-an-int")
    monkeypatch.setenv("VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD", "7.0")
    config = _get_tot_config("tot-mcts")
    assert config.early_stop_threshold is None
    assert config.beam_width is None


def test_beam_threshold_does_not_leak_into_dfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting beam's stop threshold must not change DFS's pruning threshold."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD", "9.0")
    config = _get_tot_config("tot-dfs")
    assert config.early_stop_threshold == 3.0  # DFS default, unaffected


def test_mcts_threshold_does_not_leak_into_dfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the MCTS stop threshold must not change DFS's pruning threshold."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD", "8.0")
    config = _get_tot_config("tot-dfs")
    assert config.early_stop_threshold == 3.0  # DFS default, unaffected by MCTS env
    assert config.success_threshold == 9.0  # DFS stop threshold, unaffected


@pytest.mark.parametrize(
    ("env_name", "strategy_name", "threshold_attr"),
    [
        ("VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD", "tot-mcts", "success_threshold"),
        ("VIREX_BENCH_TOT_DFS_STOP_THRESHOLD", "tot-dfs", "success_threshold"),
        (
            "VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD",
            "tot-dfs",
            "early_stop_threshold",
        ),
        (
            "VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD",
            "tot-beam",
            "early_stop_threshold",
        ),
    ],
)
def test_threshold_disable_sentinel_zero(
    monkeypatch: pytest.MonkeyPatch, env_name: str, strategy_name: str, threshold_attr: str
) -> None:
    """Setting a threshold to ``"0"`` disables it (returns ``None``).

    ToT thresholds live on the evaluator's 1-10 scale, so 0 is never a useful
    real value -- it is the explicit disable sentinel. ``maybe_convert_float``
    parses ``"0"`` to ``0.0``; ``_resolve_threshold`` then maps ``0.0`` to
    ``None`` so consumers keep their existing ``if threshold is not None`` guard.
    """
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv(env_name, "0")
    config = _get_tot_config(strategy_name)
    assert getattr(config, threshold_attr) is None


@pytest.mark.parametrize(
    ("env_name", "strategy_name", "threshold_attr", "expected_default"),
    [
        ("VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD", "tot-mcts", "success_threshold", 9.0),
        ("VIREX_BENCH_TOT_DFS_STOP_THRESHOLD", "tot-dfs", "success_threshold", 9.0),
        (
            "VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD",
            "tot-dfs",
            "early_stop_threshold",
            3.0,
        ),
        (
            "VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD",
            "tot-beam",
            "early_stop_threshold",
            9.0,
        ),
    ],
)
def test_threshold_empty_string_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    strategy_name: str,
    threshold_attr: str,
    expected_default: float | None,
) -> None:
    """An empty value is indistinguishable from unset -- both fall back to the
    algorithm default via ``_resolve_threshold``. Use ``"0"`` to explicitly
    disable a threshold."""
    _clear_variant_envs(monkeypatch)
    monkeypatch.setenv(env_name, "")
    config = _get_tot_config(strategy_name)
    assert getattr(config, threshold_attr) == expected_default
