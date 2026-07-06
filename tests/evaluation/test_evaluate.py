import importlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import dspy
import pytest

from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import (
    DatasetConfig,
    EvaluationReport,
    ReasoningExample,
    ReasoningMetric,
    TaskMetadata,
    TaskResult,
)

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

    def fake_get_metric(_metric_name: str) -> ReasoningMetric:
        def metric(_example: ReasoningExample, _prediction: dspy.Prediction) -> float:
            return 1.0

        return metric

    def fake_configure(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(evaluate_module, "get_metric", fake_get_metric)
    monkeypatch.setattr(evaluate_module.dspy, "configure", fake_configure)

    report = evaluate_module.evaluate(
        task=_FakeTask(examples),
        lm=SimpleNamespace(model="test-model", kwargs={}),
        strategy=SimpleNamespace(name="direct", report_config={}),
        num_threads=1,
        decoding=_FakeDecoding(),
        max_examples=2,
    )

    assert report.num_examples == 5
    assert report.max_examples == 2
    assert report.num_evaluated_examples == 2
    assert len(report.results) == 2


def _result(
    score: float = 1.0,
    category: str | None = None,
    extra: dict[str, object] | None = None,
    token_usage: dict[str, int] | None = None,
    llm_judge_score: float | None = None,
    premises_f1: float | None = None,
) -> TaskResult:
    return TaskResult(
        example_id="ex",
        inputs={},
        predicted="pred",
        gold="gold",
        score=score,
        category=category,
        extra=extra if extra is not None else {},
        token_usage=token_usage,
        llm_judge_score=llm_judge_score,
        premises_f1=premises_f1,
    )


def test_aggregate_scores_sums_and_breaks_by_category_skipping_none() -> None:
    results = [
        _result(score=1.0, category="mcq"),
        _result(score=0.0, category="mcq"),
        _result(score=1.0, category="yes_no_uncertain"),
        _result(score=0.5, category=None),
        _result(score=0.0, extra={"error": "ValueError: boom"}),
    ]
    total_score, num_failed, category_scores = evaluate_module._aggregate_scores(results)
    assert total_score == 2.5
    assert num_failed == 1
    assert category_scores["mcq"].score == 0.5
    assert category_scores["mcq"].num_examples == 2
    assert category_scores["yes_no_uncertain"].score == 1.0
    assert category_scores["yes_no_uncertain"].num_examples == 1
    # The None-category result is excluded from the per-category breakdown.
    assert set(category_scores) == {"mcq", "yes_no_uncertain"}


def test_aggregate_search_stats_returns_none_when_no_stats() -> None:
    results = [_result(), _result(extra={"search_stats": "not a dict"})]
    assert evaluate_module._aggregate_search_stats(results) is None


def test_aggregate_search_stats_means_numeric_keys_over_examples() -> None:
    results = [
        _result(extra={"search_stats": {"nodes_visited": 4, "propose_calls": 2}}),
        _result(extra={"search_stats": {"nodes_visited": 6, "propose_calls": 4}}),
    ]
    stats = evaluate_module._aggregate_search_stats(results)
    assert stats is not None
    assert stats["nodes_visited"] == 5.0
    assert stats["propose_calls"] == 3.0


def test_aggregate_search_stats_discards_non_numeric_keys() -> None:
    """Non-numeric keys (e.g. ``algorithm``) are excluded from the aggregate —
    only numbers and bools pass through. Uses ToT-style keys here since PoT
    now reports via ``pot_info`` / ``pot_stats`` (see ``test_aggregate_pot_stats_*``)
    rather than ``search_stats``."""
    results = [
        _result(
            extra={
                "search_stats": {
                    "algorithm": "beam",
                    "nodes_visited": 4,
                    "propose_calls": 2,
                    "evaluate_calls": 4,
                    "depth_reached": 2,
                    "best_score": 9.0,
                }
            }
        ),
        _result(
            extra={
                "search_stats": {
                    "algorithm": "beam",
                    "nodes_visited": 6,
                    "propose_calls": 4,
                    "evaluate_calls": 8,
                    "depth_reached": 3,
                    "best_score": 7.0,
                }
            }
        ),
    ]
    stats = evaluate_module._aggregate_search_stats(results)
    assert stats is not None
    assert stats["nodes_visited"] == 5.0
    assert stats["propose_calls"] == 3.0
    assert stats["evaluate_calls"] == 6.0
    assert stats["depth_reached"] == 2.5
    assert stats["best_score"] == 8.0
    # Non-numeric keys (algorithm) are excluded.
    assert "algorithm" not in stats


def test_aggregate_pot_stats_returns_none_when_no_pot_info() -> None:
    results = [_result(), _result(extra={"pot_info": "not a dict"})]
    assert evaluate_module._aggregate_pot_stats(results) is None


def test_aggregate_pot_stats_means_numeric_keys_over_examples() -> None:
    """PoT-Z3 pipeline telemetry (generate/regenerate/execute counts,
    execution_success) is aggregated from per-example ``pot_info`` dicts.
    Artifact keys (generated_code, execution_output, execution_error) are
    non-numeric and skipped. ``execution_success`` (bool) averages into a
    success rate."""
    results = [
        _result(
            extra={
                "pot_info": {
                    "generated_code": "code1",
                    "execution_output": "{'answer': 'Có'}",
                    "execution_error": None,
                    "execution_success": True,
                    "generate_calls": 1,
                    "regenerate_calls": 0,
                    "execute_calls": 1,
                    "total_llm_calls": 2,
                }
            }
        ),
        _result(
            extra={
                "pot_info": {
                    "generated_code": "code2",
                    "execution_output": None,
                    "execution_error": "RuntimeError: boom",
                    "execution_success": False,
                    "generate_calls": 1,
                    "regenerate_calls": 2,
                    "execute_calls": 3,
                    "total_llm_calls": 4,
                }
            }
        ),
    ]
    stats = evaluate_module._aggregate_pot_stats(results)
    assert stats is not None
    assert stats["generate_calls"] == 1.0
    assert stats["regenerate_calls"] == 1.0
    assert stats["execute_calls"] == 2.0
    assert stats["total_llm_calls"] == 3.0
    # bool True/False averages to 0.5 — the execution success rate.
    assert stats["execution_success"] == 0.5
    # Non-numeric artifact keys are excluded.
    assert "generated_code" not in stats
    assert "execution_output" not in stats
    assert "execution_error" not in stats


def test_aggregate_pot_stats_averages_only_over_examples_with_pot_info() -> None:
    """Examples without ``pot_info`` are skipped from both numerator and
    denominator — the mean is over the examples that carried it."""
    results = [
        _result(extra={"pot_info": {"execute_calls": 1, "execution_success": True}}),
        _result(),  # no pot_info -> excluded
        _result(extra={"pot_info": "not a dict"}),  # non-dict -> excluded
        _result(extra={"pot_info": {"execute_calls": 3, "execution_success": False}}),
    ]
    stats = evaluate_module._aggregate_pot_stats(results)
    assert stats is not None
    # Two valid examples: (1+3)/2 = 2.0, not /4.
    assert stats["execute_calls"] == 2.0
    assert stats["execution_success"] == 0.5


def test_aggregate_search_stats_averages_only_over_examples_with_stats() -> None:
    """Examples without ``search_stats`` are skipped from both the numerator and
    the denominator — the mean is over the examples that carried stats, not all
    results."""
    results = [
        _result(extra={"search_stats": {"nodes_visited": 4, "propose_calls": 2}}),
        _result(),  # no search_stats -> excluded
        _result(extra={"search_stats": "not a dict"}),  # non-dict -> excluded
        _result(extra={"search_stats": {"nodes_visited": 8, "propose_calls": 4}}),
    ]
    stats = evaluate_module._aggregate_search_stats(results)
    assert stats is not None
    # Two valid examples: (4+8)/2 = 6.0, not (4+8)/4 = 3.0.
    assert stats["nodes_visited"] == 6.0
    assert stats["propose_calls"] == 3.0


def test_aggregate_decoding_stats_returns_none_when_no_stats() -> None:
    results = [_result(), _result(extra={"decoding_stats": "not a dict"})]
    assert evaluate_module._aggregate_decoding_stats(results) is None


def test_aggregate_decoding_stats_means_self_consistency_telemetry() -> None:
    """Self-consistency ``decoding_stats`` (confidence, vote counts,
    aggregator_applied flag) is averaged from per-example dicts. The bool
    ``aggregator_applied`` averages into the rate the aggregator genuinely
    contributed (vs. falling back to the vote)."""
    results = [
        _result(
            extra={
                "decoding_stats": {
                    "confidence": 1.0,
                    "vote_count": 5,
                    "total_samples": 5,
                    "requested_samples": 5,
                    "aggregator_applied": True,
                }
            }
        ),
        _result(
            extra={
                "decoding_stats": {
                    "confidence": 0.6,
                    "vote_count": 3,
                    "total_samples": 5,
                    "requested_samples": 5,
                    "aggregator_applied": False,
                }
            }
        ),
        _result(),  # no decoding_stats -> excluded
    ]
    stats = evaluate_module._aggregate_decoding_stats(results)
    assert stats is not None
    assert stats["confidence"] == 0.8
    assert stats["vote_count"] == 4.0
    assert stats["total_samples"] == 5.0
    assert stats["requested_samples"] == 5.0
    # bool True/False averages to 0.5 — the aggregator application rate.
    assert stats["aggregator_applied"] == 0.5


def test_aggregate_score_components_returns_none_when_absent() -> None:
    results = [_result(), _result()]
    assert evaluate_module._aggregate_score_components(results) is None


def test_aggregate_score_components_decomposes_blended_score() -> None:
    """The headline ``score`` blends answer + premises_f1 for the
    logical-reasoning task; ``score_components`` decomposes it by averaging each
    component independently over the examples that carried a non-null value."""
    results = [
        _result(score=0.75, llm_judge_score=1.0, premises_f1=0.5),
        _result(score=0.6, llm_judge_score=0.8, premises_f1=0.4),
        # An example without gold premise supervision carries no premises_f1;
        # it still counts toward the llm_judge_score mean.
        _result(score=0.9, llm_judge_score=0.9, premises_f1=None),
    ]
    components = evaluate_module._aggregate_score_components(results)
    assert components is not None
    # llm_judge_score averaged over all 3: (1.0 + 0.8 + 0.9) / 3.
    assert components["llm_judge_score"] == pytest.approx((1.0 + 0.8 + 0.9) / 3)
    # premises_f1 averaged over the 2 that carried it: (0.5 + 0.4) / 2.
    assert components["premises_f1"] == pytest.approx(0.45)


class _FakeUsageTracker:
    """Stand-in for dspy.UsageTracker returning a canned totals dict."""

    def __init__(self, totals: dict[str, dict[str, int]]) -> None:
        self._totals = totals

    def get_total_tokens(self) -> dict[str, dict[str, int]]:
        return self._totals


def test_summarize_usage_returns_none_when_empty() -> None:
    assert evaluate_module._summarize_usage(_FakeUsageTracker({})) is None


def test_summarize_usage_sums_tokens_across_models() -> None:
    tracker = _FakeUsageTracker(
        {
            "model-a": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model-b": {"prompt_tokens": 20, "completion_tokens": 0, "total_tokens": 20},
        }
    )
    summary = evaluate_module._summarize_usage(tracker)
    assert summary == {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}


def test_summarize_usage_recomputes_total_when_zero() -> None:
    tracker = _FakeUsageTracker(
        {"model-a": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 0}}
    )
    summary = evaluate_module._summarize_usage(tracker)
    assert summary is not None
    assert summary["total_tokens"] == 20


def test_aggregate_token_usage_returns_none_when_untracked() -> None:
    mean, total = evaluate_module._aggregate_token_usage([_result(), _result()])
    assert mean is None
    assert total is None


def test_aggregate_token_usage_returns_mean_and_total() -> None:
    results = [
        _result(token_usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}),
        _result(token_usage={"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}),
    ]
    mean, total = evaluate_module._aggregate_token_usage(results)
    assert mean is not None
    assert total is not None
    assert mean["prompt_tokens"] == 20.0
    assert mean["total_tokens"] == 25.0
    assert total["completion_tokens"] == 10


def test_save_report_writes_json_and_omits_absent_optional_fields(
    tmp_path: str,
) -> None:
    report = EvaluationReport(
        task="org/task",
        dataset=DatasetConfig(path="ds"),
        model="org/model",
        strategy="direct",
        decoding="single-pass",
        metric="exact_match",
        judge=None,
        score=0.5,
        num_examples=4,
    )
    output_path = evaluate_module.save_report(report, tmp_path)
    assert output_path.endswith(".json")
    # Slashes in task/model are translated into nested directories.
    assert "org--task" in output_path
    assert "org--model" in output_path
    with open(output_path, encoding="utf-8") as file:
        content = file.read()
    assert '"search_stats"' not in content
    assert '"pot_stats"' not in content
    assert '"decoding_stats"' not in content
    assert '"score_components"' not in content
    assert '"token_usage"' not in content
    assert '"judge"' not in content


class _StubDecoding:
    """Decoding stub returning a canned prediction (or raising)."""

    name = "stub"
    display_name = "stub"
    config: dict[str, object] = {}

    def configure_from_task(self, task: ReasoningTask) -> None:
        return None

    def __init__(self, prediction: dspy.Prediction) -> None:
        self._prediction = prediction

    def __call__(self, **_inputs: object) -> dspy.Prediction:
        return self._prediction


class _RaisingDecoding:
    name = "raising"
    display_name = "raising"
    config: dict[str, object] = {}

    def configure_from_task(self, task: ReasoningTask) -> None:
        return None

    def __call__(self, **_inputs: object) -> dspy.Prediction:
        raise RuntimeError("strategy exploded")


def _make_task() -> ReasoningTask:
    return _FakeTask(
        [
            ReasoningExample(
                example_id="ex-1",
                premises=["p1"],
                question="q",
                answer="gold",
            )
        ]
    )


def _always_one(_example: ReasoningExample, _prediction: dspy.Prediction) -> float:
    return 1.0


def _match_gold(_example: ReasoningExample, prediction: dspy.Prediction) -> float:
    return 1.0 if prediction.answer == "gold" else 0.0


def test_process_example_error_path_records_zero_score_and_error() -> None:
    result = evaluate_module._process_example(
        _make_task().load_examples()[0],
        task=_make_task(),
        decoding_strategy=_RaisingDecoding(),
        judge_module=None,
        metric_func=_always_one,
    )
    assert result.score == 0.0
    assert "error" in result.extra
    assert "RuntimeError" in str(result.extra["error"])


def test_process_example_metric_path_records_score() -> None:
    example = _make_task().load_examples()[0]
    result = evaluate_module._process_example(
        example,
        task=_make_task(),
        decoding_strategy=_StubDecoding(dspy.Prediction(answer="gold", reasoning="r")),
        judge_module=None,
        metric_func=_match_gold,
    )
    assert result.score == 1.0
    assert result.predicted == "gold"
    assert result.extra["reasoning"] == "r"


class _StubJudge:
    def __call__(self, **_kwargs: object) -> dspy.Prediction:
        return dspy.Prediction(verdict="YES", error_type="correct", feedback="ok", method="llm")


def test_process_example_judge_path_records_judge_result() -> None:
    example = _make_task().load_examples()[0]
    result = evaluate_module._process_example(
        example,
        task=_make_task(),
        decoding_strategy=_StubDecoding(dspy.Prediction(answer="gold")),
        judge_module=_StubJudge(),
        metric_func=None,
    )
    assert result.score == 1.0
    assert result.extra["judge_result"]["verdict"] == "YES"
    assert result.extra["judge_result"]["method"] == "llm"
