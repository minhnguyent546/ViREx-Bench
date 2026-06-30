import importlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import DatasetConfig, ReasoningExample, TaskMetadata, TaskResult

evaluate_module = importlib.import_module("virex_bench.evaluation.evaluate")


class _FakeTask(ReasoningTask):
    metadata = TaskMetadata(
        name="fake-task",
        description="Fake task for evaluator tests",
        dataset=DatasetConfig(path="fake-dataset"),
        main_metric="exact_match",
    )
    signatures = {}

    def __init__(self, examples: list[ReasoningExample]) -> None:
        self.examples = examples

    def load_examples(self) -> list[ReasoningExample]:
        return self.examples

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        raise NotImplementedError


class _FakeDecoding:
    name = "single-pass"
    display_name = "single-pass"
    config: dict[str, object] = {}

    def configure_from_task(self, task: ReasoningTask) -> None:
        return None


def test_evaluate_reports_original_and_evaluated_example_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = [
        ReasoningExample(
            example_id=f"example-{example_index}",
            premises=[],
            question="question",
            answer="answer",
        )
        for example_index in range(5)
    ]

    def fake_process_example(
        example: ReasoningExample,
        **_kwargs: object,
    ) -> TaskResult:
        return TaskResult(
            example_id=example.example_id,
            inputs={},
            predicted=example.answer,
            gold=example.answer,
            score=1.0,
        )

    monkeypatch.setattr(evaluate_module, "_process_example", fake_process_example)
    monkeypatch.setattr(evaluate_module, "get_metric", lambda _metric_name: lambda *_args: 1.0)
    monkeypatch.setattr(evaluate_module.dspy, "configure", lambda **_kwargs: None)

    report = evaluate_module.evaluate(
        _FakeTask(examples),
        SimpleNamespace(kwargs={}),
        SimpleNamespace(name="direct", report_config={}),
        model_name="test-model",
        backend="openai",
        num_threads=1,
        decoding=_FakeDecoding(),
        max_examples=2,
    )

    assert report.num_examples == 5
    assert report.max_examples == 2
    assert report.num_evaluated_examples == 2
    assert len(report.results) == 2
