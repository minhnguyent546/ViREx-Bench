"""Tests for ``ReasoningTask`` base-class helpers and dataset loading."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dspy
import pytest

from virex_bench.strategies.direct import DirectStrategy
from virex_bench.tasks import base as task_base
from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import DatasetConfig, ReasoningExample, TaskMetadata


class _TestSignature(dspy.Signature):
    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _FakeTask(ReasoningTask):
    metadata = TaskMetadata(
        name="fake-task",
        description="Fake task for base-class tests",
        dataset=DatasetConfig(path="fake-dataset", split="test"),
        main_metric="exact_match",
    )
    signatures = {"default": _TestSignature, "cot": _TestSignature}

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        # Deliberately leaves ``category`` unset so ``load_examples`` fills it.
        return ReasoningExample(
            example_id=str(row.get("id", "")),
            premises=list(row["premises"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
        )


def test_get_signature_returns_named_or_falls_back_to_default() -> None:
    task = _FakeTask()
    assert task.get_signature("cot") is _TestSignature
    assert task.get_signature("unknown") is _TestSignature


def test_get_rationale_field_falls_back_to_default_then_none() -> None:
    task = _FakeTask()
    task.rationale_fields = {"default": dspy.OutputField(desc="d")}
    field = task.get_rationale_field("cot")
    assert field is not None
    # No entry for "direct" -> falls back to "default".
    assert task.get_rationale_field("default") is field
    # With an empty rationale_fields map, the fallback is None.
    task.rationale_fields = {}
    assert task.get_rationale_field("cot") is None


def test_example_to_inputs_collapses_premises_to_numbered_block() -> None:
    task = _FakeTask()
    inputs = task.example_to_inputs(
        ReasoningExample(
            example_id="x",
            premises=["alpha", "beta"],
            question="Which is first?",
            answer="alpha",
        )
    )
    assert inputs["premises"] == "Premise 1. alpha\nPremise 2. beta"
    assert inputs["question"] == "Which is first?"


def test_recorded_inputs_copies_model_inputs() -> None:
    task = _FakeTask()
    model_inputs: dict[str, object] = {"premises": "p", "question": "q"}
    recorded = task.recorded_inputs(
        ReasoningExample(example_id="x", premises=["p"], question="q", answer="a"),
        model_inputs,
    )
    assert recorded == model_inputs
    assert recorded is not model_inputs


def test_compute_score_defaults_to_answer_score_unchanged() -> None:
    task = _FakeTask()
    example = ReasoningExample(example_id="x", premises=["p"], question="q", answer="a")
    prediction = dspy.Prediction(answer="a")
    components = task.compute_score(example, prediction, answer_score=0.75)
    assert components.score == 0.75
    assert components.llm_judge_score is None
    assert components.premises_f1 is None


def test_name_property_returns_metadata_name() -> None:
    assert _FakeTask().name == "fake-task"


def test_get_strategy_builds_registered_strategy_with_task_signature() -> None:
    task = _FakeTask()
    strategy = task.get_strategy("direct")
    assert isinstance(strategy, DirectStrategy)
    assert strategy.signature is _TestSignature


def test_row_to_example_base_raises_not_implemented() -> None:
    base_task = ReasoningTask()
    with pytest.raises(NotImplementedError):
        base_task._row_to_example({})  # pyright: ignore[reportPrivateUsage]


def _fake_rows() -> list[dict[str, Any]]:
    return [
        {"id": "r1", "premises": ["p1"], "question": "q1", "answer": "a1", "category": "mcq"},
        {"id": "r2", "premises": ["p2"], "question": "q2", "answer": "a2", "category": None},
    ]


def _passthrough_load_dataset(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    return _fake_rows()


def test_load_examples_fills_category_from_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load_dataset(path: str, name: str | None, **kwargs: Any) -> list[dict[str, Any]]:
        captured["path"] = path
        captured["name"] = name
        captured.update(kwargs)
        return _fake_rows()

    monkeypatch.setattr(task_base.datasets, "load_dataset", fake_load_dataset)
    examples = _FakeTask().load_examples()

    assert len(examples) == 2
    assert examples[0].category == "mcq"
    # A None column value leaves the category as None.
    assert examples[1].category is None
    assert captured["path"] == "fake-dataset"
    assert captured["split"] == "test"


def test_load_examples_skips_category_fill_when_column_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoColumnTask(_FakeTask):
        category_column = None

    monkeypatch.setattr(task_base.datasets, "load_dataset", _passthrough_load_dataset)
    examples = _NoColumnTask().load_examples()

    # With category_column disabled, the row's category never overrides.
    assert all(ex.category is None for ex in examples)


def test_load_examples_preserves_row_set_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PreSetTask(_FakeTask):
        def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
            example = super()._row_to_example(row)
            example.category = "preset"
            return example

    monkeypatch.setattr(task_base.datasets, "load_dataset", _passthrough_load_dataset)
    examples = _PreSetTask().load_examples()

    # A category already set in _row_to_example is not overwritten by the column.
    assert all(ex.category == "preset" for ex in examples)
