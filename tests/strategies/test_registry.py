"""Tests for the variant-aware strategy registry.

Covers ``parse_strategy_name``, ``get_strategy`` with composite names,
``list_strategies`` variant discovery, and error paths for invalid variants.
"""

from collections.abc import Mapping
from typing import Any

import dspy
import pytest

from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.registry import (
    get_strategy,
    list_strategies,
    parse_strategy_name,
)
from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import DatasetConfig, ReasoningExample, TaskMetadata


class _TestSignature(dspy.Signature):
    """Minimal signature for strategy construction in tests."""

    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _TaskToTSignature(dspy.Signature):
    """Task-specific ToT signature for helper equivalence tests."""

    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _FakeTask(ReasoningTask):
    metadata = TaskMetadata(
        name="fake-task",
        description="Fake task for strategy construction tests",
        dataset=DatasetConfig(path="fake-dataset"),
        main_metric="exact_match",
    )
    signatures = {"default": _TestSignature, "tot": _TaskToTSignature}
    rationale_fields = {"tot": dspy.OutputField(desc="Task-specific ToT reasoning field")}

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        raise NotImplementedError


def test_parse_strategy_name_resolves_composite() -> None:
    assert parse_strategy_name("tot-beam") == ("tot", "beam")
    assert parse_strategy_name("tot-dfs") == ("tot", "dfs")
    assert parse_strategy_name("tot-mcts") == ("tot", "mcts")


def test_parse_strategy_name_passthrough_bare() -> None:
    assert parse_strategy_name("tot") == ("tot", None)
    assert parse_strategy_name("cot") == ("cot", None)
    assert parse_strategy_name("direct") == ("direct", None)


def test_list_strategies_includes_variants() -> None:
    names = list_strategies()
    assert "cot" in names
    assert "cr" in names
    assert "direct" in names
    assert "pot_z3" in names
    assert "tot" in names
    assert "tot-beam" in names
    assert "tot-dfs" in names
    assert "tot-mcts" in names


def test_get_strategy_tot_beam_yields_correct_name_and_algorithm() -> None:
    strategy = get_strategy("tot-beam", _TestSignature)
    assert strategy.name == "tot-beam"
    assert strategy.search_algorithm == "beam"  # type: ignore[attr-defined]


def test_task_get_strategy_resolves_composite_strategy() -> None:
    task = _FakeTask()
    strategy = task.get_strategy("tot-beam")
    old_path_strategy = get_strategy(
        "tot-beam",
        task.get_signature("tot"),
        rationale_field=task.get_rationale_field("tot"),
    )

    assert strategy.name == "tot-beam"
    assert strategy.search_algorithm == "beam"  # type: ignore[attr-defined]
    assert strategy.signature is old_path_strategy.signature
    assert strategy.rationale_field is old_path_strategy.rationale_field


def test_get_strategy_tot_dfs_yields_correct_name_and_algorithm() -> None:
    strategy = get_strategy("tot-dfs", _TestSignature)
    assert strategy.name == "tot-dfs"
    assert strategy.search_algorithm == "dfs"  # type: ignore[attr-defined]


def test_get_strategy_tot_mcts_yields_correct_name_and_algorithm() -> None:
    strategy = get_strategy("tot-mcts", _TestSignature)
    assert strategy.name == "tot-mcts"
    assert strategy.search_algorithm == "mcts"  # type: ignore[attr-defined]


def test_get_strategy_pot_z3_constructs_without_variant() -> None:
    strategy = get_strategy("pot_z3", _TestSignature)
    assert strategy.name == "pot_z3"
    assert strategy.accepts_variant is False


def test_get_strategy_pot_z3_with_variant_raises_no_variant_axis() -> None:
    with pytest.raises(ValueError, match="no variant axis"):
        get_strategy("pot_z3-foo", _TestSignature)


def test_get_strategy_bare_tot_uses_env_default_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIREX_BENCH_TOT_SEARCH_ALGORITHM", raising=False)
    strategy = get_strategy("tot", _TestSignature)
    assert strategy.name == "tot"
    assert strategy.search_algorithm == "beam"  # type: ignore[attr-defined]


def test_get_strategy_bare_tot_uses_env_algorithm_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIREX_BENCH_TOT_SEARCH_ALGORITHM", "DFS")
    strategy = get_strategy("tot", _TestSignature)
    assert strategy.name == "tot"
    assert strategy.search_algorithm == "dfs"  # type: ignore[attr-defined]


def test_get_strategy_cot_beam_raises_no_variant_axis() -> None:
    with pytest.raises(ValueError, match="no variant axis"):
        get_strategy("cot-beam", _TestSignature)


def test_get_strategy_tot_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        get_strategy("tot-unknown", _TestSignature)


def test_get_strategy_unknown_base_raises() -> None:
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("nonexistent", _TestSignature)


def test_direct_and_cot_construct_without_variant() -> None:
    direct = get_strategy("direct", _TestSignature)
    assert isinstance(direct, ReasoningStrategy)
    assert direct.name == "direct"

    cot = get_strategy("cot", _TestSignature)
    assert isinstance(cot, ReasoningStrategy)
    assert cot.name == "cot"
