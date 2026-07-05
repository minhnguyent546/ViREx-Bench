import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import cast

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.usage_tracker import UsageTracker
from tqdm.auto import tqdm

from virex_bench import envs
from virex_bench.decoding import DecodingStrategy, SinglePass
from virex_bench.evaluation.judge import LLMJudge, build_judge
from virex_bench.evaluation.metrics import (
    get_metric,
    judge_example,
)
from virex_bench.logger import init_logger, log_query_context
from virex_bench.models import BaseLM
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import (
    CategoryScore,
    EvaluationReport,
    ReasoningExample,
    ReasoningMetric,
    TaskResult,
)

logger = init_logger(__name__)


def _process_example(
    example: ReasoningExample,
    *,
    task: ReasoningTask,
    decoding_strategy: DecodingStrategy,
    judge_module: LLMJudge | None = None,
    metric_func: ReasoningMetric | None = None,
) -> TaskResult:
    if (judge_module is None) == (metric_func is None):
        raise ValueError("Exactly one of judge_module or metric_func must be provided")

    # Tag this example's log lines with its id so concurrent workers stay attributable.
    with log_query_context(example.example_id):
        inputs = task.example_to_inputs(example)
        recorded_inputs = task.recorded_inputs(example, inputs)
        try:
            # Scope a usage tracker to the STRATEGY call only -- the judge runs after
            # the `with` exits so its tokens are excluded. dspy.settings is thread-local,
            # so each worker thread's tracker stays isolated to its own example.
            with dspy.track_usage() as usage_tracker:
                prediction = decoding_strategy(**inputs)
            token_usage = _summarize_usage(usage_tracker)
            predicted = str(prediction.answer)
            extra: dict[str, object] = {}
            reasoning = getattr(prediction, "reasoning", None)
            if reasoning is not None:
                extra["reasoning"] = reasoning
            search_stats = getattr(prediction, "search_stats", None)
            if search_stats is not None:
                extra["search_stats"] = search_stats
            if judge_module is not None:
                judgement = judge_example(
                    example=example, prediction=prediction, judge_module=judge_module
                )
                extra["judge_result"] = {
                    "verdict": judgement.verdict,
                    "error_type": judgement.error_type,
                    "feedback": judgement.feedback,
                    "method": judgement.method,
                }
                answer_score = judgement.score
            else:
                assert metric_func is not None
                answer_score = float(metric_func(example, prediction))
            score_components = task.compute_score(example, prediction, answer_score)
        except Exception as error:
            logger.warning(f"Example {example.example_id} failed: {error!r}")
            return TaskResult(
                example_id=example.example_id,
                inputs=recorded_inputs,
                predicted="",
                gold=example.answer,
                score=0.0,
                category=example.category,
                extra={"error": f"{type(error).__name__}: {error}"},
            )
        return TaskResult(
            example_id=example.example_id,
            inputs=recorded_inputs,
            predicted=predicted,
            gold=example.answer,
            score=score_components.score,
            llm_judge_score=score_components.llm_judge_score,
            premises_f1=score_components.premises_f1,
            predicted_premises_used=score_components.predicted_premises_used,
            category=example.category,
            token_usage=token_usage,
            extra=extra,
        )


def _aggregate_scores(
    results: list[TaskResult],
) -> tuple[float, int, dict[str, CategoryScore]]:
    """Compute total score, failure count, and per-category score breakdowns."""
    total_score = sum(result.score for result in results)
    num_failed = sum(1 for result in results if "error" in result.extra)
    category_totals: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    for result in results:
        if result.category is None:
            continue
        category_totals[result.category] = category_totals.get(result.category, 0.0) + result.score
        category_counts[result.category] = category_counts.get(result.category, 0) + 1
    category_scores = {
        category: CategoryScore(
            score=category_totals[category] / category_counts[category],
            num_examples=category_counts[category],
        )
        for category in category_totals
    }
    return total_score, num_failed, category_scores


def _aggregate_search_stats(results: list[TaskResult]) -> dict[str, float] | None:
    """Mean of the per-example ``search_stats`` (nodes/calls/depth/score).

    Returns ``None`` when no example carried search stats -- i.e. for non
    search-based strategies such as direct and cot, in which case the field is
    omitted from the report. Search strategies (tot-beam, tot-dfs, ...) attach
    ``search_stats`` on every successful prediction.
    """
    numeric_keys = (
        "nodes_visited",
        "propose_calls",
        "evaluate_calls",
        "depth_reached",
        "best_score",
        "total_llm_calls",
        "total_completions",
    )
    sums: dict[str, float] = dict.fromkeys(numeric_keys, 0.0)
    num_search_examples = 0
    for result in results:
        stats_raw = result.extra.get("search_stats")
        if not isinstance(stats_raw, dict):
            continue
        stats = cast("dict[str, object]", stats_raw)
        num_search_examples += 1
        for key in numeric_keys:
            value = stats.get(key)
            if isinstance(value, (int, float)):
                sums[key] += value
    if num_search_examples == 0:
        return None
    return {key: total / num_search_examples for key, total in sums.items()}


def _summarize_usage(usage_tracker: UsageTracker) -> dict[str, int] | None:
    """Flatten a per-example ``UsageTracker`` into prompt/completion/total tokens.

    Returns ``None`` when the tracker captured no usage entry (e.g. the backend
    did not report a ``usage`` block), so the field stays absent for that record.
    Covers all models the example touched (typically just the strategy LM).
    """
    totals = usage_tracker.get_total_tokens()
    if not totals:
        return None
    prompt_tokens = sum(int(usage.get("prompt_tokens", 0) or 0) for usage in totals.values())
    completion_tokens = sum(
        int(usage.get("completion_tokens", 0) or 0) for usage in totals.values()
    )
    total_tokens = sum(int(usage.get("total_tokens", 0) or 0) for usage in totals.values())
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _aggregate_token_usage(
    results: list[TaskResult],
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    """Mean (per example) and total (whole run) token usage over tracked records.

    Returns ``(None, None)`` when no record carried token usage.
    """
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    sums: dict[str, int] = dict.fromkeys(keys, 0)
    num_tracked_examples = 0
    for result in results:
        usage = result.token_usage
        if usage is None:
            continue
        num_tracked_examples += 1
        for key in keys:
            sums[key] += int(usage.get(key, 0) or 0)
    if num_tracked_examples == 0:
        return None, None
    mean = {key: sums[key] / num_tracked_examples for key in keys}
    total = {key: sums[key] for key in keys}
    return mean, total


def evaluate(
    task: ReasoningTask,
    lm: BaseLM,
    strategy: ReasoningStrategy,
    num_threads: int = 8,
    decoding: DecodingStrategy | None = None,
    max_examples: int | None = None,
) -> EvaluationReport:
    """Run `strategy` with `lm` over every example in `task` and score with the task's metric.

    The scoring metric is a property of the task (``task.metadata.main_metric``)
    and is resolved against the metric registry. The reported score is the mean
    per-example metric score.

    When `max_examples` is set, only the first N loaded examples are evaluated.
    Examples are evaluated concurrently across `num_threads` worker threads. A
    failure on a single example (e.g. an API, parsing, or metric error) is logged
    and recorded with a score of ``0.0`` instead of aborting the whole run.
    """
    loaded_examples = task.load_examples()
    num_examples = len(loaded_examples)
    examples = loaded_examples
    if max_examples is not None:
        examples = loaded_examples[:max_examples]
    metric_name = task.metadata.main_metric
    judge_name = task.metadata.judge
    judge_module: LLMJudge | None = None
    judge_model_name: str | None = None
    metric_func: ReasoningMetric | None = None

    if judge_name is not None:
        judge_module = build_judge(judge_name)
        judge_model_name = judge_module.judge_lm.model
    else:
        metric_func = get_metric(metric_name)

    assert (judge_module is not None) ^ (metric_func is not None)

    decoding_strategy = decoding if decoding is not None else SinglePass(strategy)
    decoding_strategy.configure_from_task(task)

    logger.info(
        f"Evaluating task={task.name} model={lm.model} "
        f"strategy={strategy.name} decoding={decoding_strategy.display_name} "
        f"metric={metric_name} "
        + (f"judge={judge_name} [{judge_model_name}] " if judge_name is not None else "")
        + (f"max_examples={max_examples} " if max_examples is not None else "")
        + f"on {len(examples)}/{num_examples} examples (num_threads={num_threads})"
    )

    _process_example_fn = partial(
        _process_example,
        task=task,
        decoding_strategy=decoding_strategy,
        judge_module=judge_module,
        metric_func=metric_func,
    )

    progress_desc = f"Eval [{task.name}/{strategy.name}/{decoding_strategy.display_name}]"
    dspy.configure(lm=lm, adapter=ChatAdapter(use_json_adapter_fallback=False))
    if num_threads > 1:
        executor = ThreadPoolExecutor(max_workers=num_threads)
        result_iter: Iterable[TaskResult] = executor.map(_process_example_fn, examples)
    else:
        executor = None
        result_iter = (_process_example_fn(example) for example in examples)

    results: list[TaskResult] = []
    running_score = 0.0
    num_failed = 0
    progress_bar = tqdm(total=len(examples), desc=progress_desc, unit="example")
    start_time = time.perf_counter()
    try:
        for result in result_iter:
            results.append(result)
            running_score += result.score
            if "error" in result.extra:
                num_failed += 1
            seen = len(results)
            progress_bar.set_postfix_str(
                f"{metric_name}={running_score / seen:.4f}, failed={num_failed}"
            )
            progress_bar.update(1)
    finally:
        progress_bar.close()
        if executor is not None:
            executor.shutdown(wait=True)
    total_time = time.perf_counter() - start_time

    total_score, num_failed, category_scores = _aggregate_scores(results)
    search_stats = _aggregate_search_stats(results)
    mean_token_usage, total_token_usage = _aggregate_token_usage(results)
    score = total_score / len(results) if results else 0.0
    logger.info(
        f"Done: {metric_name}={score:.4f} ({total_score:.4f}/{len(results)})"
        + (f", {num_failed} failed" if num_failed else "")
        + f", total_time={total_time:.2f}s"
    )
    if category_scores:
        breakdown = ", ".join(
            f"{category}={entry.score:.4f} ({entry.num_examples})"
            for category, entry in sorted(category_scores.items())
        )
        logger.info(f"Per-category {metric_name}: {breakdown}")
    if search_stats is not None:
        stats_breakdown = ", ".join(f"{key}={value:.2f}" for key, value in search_stats.items())
        logger.info(f"Search stats (mean): {stats_breakdown}")
    if mean_token_usage is not None and total_token_usage is not None:
        logger.info(
            f"Token usage (mean per example): "
            f"prompt={mean_token_usage['prompt_tokens']:.0f}, "
            f"completion={mean_token_usage['completion_tokens']:.0f}, "
            f"total={mean_token_usage['total_tokens']:.0f} | "
            f"(run total) total={total_token_usage['total_tokens']}"
        )

    model_kwargs = {
        key: value
        for key, value in cast("dict[str, object]", lm.kwargs).items()
        if key != "api_key"
    }
    model_cache = getattr(lm, "cache", None)
    if model_cache is not None:
        model_kwargs["cache"] = model_cache
    model_num_retries = getattr(lm, "num_retries", None)
    if model_num_retries is not None:
        model_kwargs["num_retries"] = model_num_retries
    lm_retry_min_wait = max(0.0, envs.VIREX_BENCH_LM_RETRY_MIN_WAIT)
    lm_retry_kwargs: dict[str, object] = {
        "max_retries": max(1, envs.VIREX_BENCH_LM_MAX_RETRIES),
        "min_wait": lm_retry_min_wait,
        "max_wait": max(lm_retry_min_wait, envs.VIREX_BENCH_LM_RETRY_MAX_WAIT),
        "jitter": max(0.0, envs.VIREX_BENCH_LM_RETRY_JITTER),
    }

    judge_kwargs: dict[str, object] = {}
    if judge_module is not None:
        judge_kwargs = {
            key: value
            for key, value in cast("dict[str, object]", judge_module.judge_lm.kwargs).items()
            if key != "api_key"
        }
        judge_kwargs["max_attempts"] = judge_module.max_attempts
        judge_kwargs["retry_wait_seconds"] = judge_module.retry_wait_seconds
        judge_cache = getattr(judge_module.judge_lm, "cache", None)
        if judge_cache is not None:
            judge_kwargs["cache"] = judge_cache

    return EvaluationReport(
        task=task.name,
        dataset=task.metadata.dataset,
        model=lm.model,
        model_kwargs=model_kwargs,
        lm_retry_kwargs=lm_retry_kwargs,
        strategy=strategy.name,
        strategy_kwargs=strategy.report_config,
        decoding=decoding_strategy.name,
        decoding_kwargs=decoding_strategy.config,
        metric=metric_name,
        judge=judge_name,
        judge_model=judge_model_name,
        judge_kwargs=judge_kwargs,
        score=score,
        num_examples=num_examples,
        max_examples=max_examples,
        num_evaluated_examples=len(results),
        num_threads=num_threads,
        num_failed=num_failed,
        total_time=total_time,
        category_scores=category_scores,
        search_stats=search_stats,
        token_usage=mean_token_usage,
        total_token_usage=total_token_usage,
        results=results,
    )


def save_report(report: EvaluationReport, output_dir: str) -> str:
    dataset_name = report.task.replace("/", "--")
    model_name = report.model.replace("/", "--")
    nested_dir = os.path.join(
        output_dir, dataset_name, model_name, report.strategy, report.decoding
    )
    os.makedirs(nested_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path = os.path.join(nested_dir, f"results-{timestamp}.json")
    exclude: set[str] = {"judge", "judge_model", "judge_kwargs"} if report.judge is None else set()
    if report.search_stats is None:
        exclude.add("search_stats")
    if report.token_usage is None:
        exclude.update(("token_usage", "total_token_usage"))
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(report.model_dump_json(indent=2, exclude=exclude))
    logger.info(f"Saved results to {output_path}")

    return output_path
