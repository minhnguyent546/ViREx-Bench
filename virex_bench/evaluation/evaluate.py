import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import dspy
from tqdm.auto import tqdm

from virex_bench.evaluation.judge import LLMJudge, build_judge
from virex_bench.evaluation.metrics import get_metric, judge_example
from virex_bench.logger import init_logger
from virex_bench.models import BaseLM
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import EvaluationReport, ReasoningExample, ReasoningMetric, TaskResult

logger = init_logger(__name__)


def evaluate(
    task: ReasoningTask,
    lm: BaseLM,
    strategy: ReasoningStrategy,
    model_name: str,
    backend: str,
    num_threads: int = 8,
) -> EvaluationReport:
    """Run `strategy` with `lm` over every example in `task` and score with the task's metric.

    The scoring metric is a property of the task (``task.metadata.main_metric``)
    and is resolved against the metric registry. The reported score is the mean
    per-example metric score.

    Examples are evaluated concurrently across `num_threads` worker threads. A
    failure on a single example (e.g. an API, parsing, or metric error) is logged
    and recorded with a score of ``0.0`` instead of aborting the whole run.
    """
    examples = task.load_examples()
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

    if judge_module is None:
        assert metric_func is not None

    logger.info(
        f"Evaluating task={task.name} model={model_name} "
        f"strategy={strategy.name} metric={metric_name} "
        + (f"judge={judge_name} [{judge_model_name}] " if judge_name is not None else "")
        + f"on {len(examples)} examples (num_threads={num_threads})"
    )

    def process_example(example: ReasoningExample) -> TaskResult:
        try:
            inputs = task.example_to_inputs(example)
            prediction = strategy(**inputs)
            predicted = str(prediction.answer)
            extra: dict[str, object] = {}
            reasoning = getattr(prediction, "reasoning", None)
            if reasoning is not None:
                extra["reasoning"] = reasoning
            if judge_module is not None:
                outcome = judge_example(
                    example=example, prediction=prediction, judge_module=judge_module
                )
                extra["judge_result"] = {
                    "verdict": outcome.verdict,
                    "error_type": outcome.error_type,
                    "feedback": outcome.feedback,
                }
                score = outcome.score
            else:
                assert metric_func is not None
                score = float(metric_func(example, prediction))
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Example {example.example_id} failed: {error!r}")
            return TaskResult(
                example_id=example.example_id,
                predicted="",
                gold=example.answer,
                score=0.0,
                extra={"error": f"{type(error).__name__}: {error}"},
            )
        return TaskResult(
            example_id=example.example_id,
            predicted=predicted,
            gold=example.answer,
            score=score,
            extra=extra,
        )

    progress_desc = f"Eval [{task.name}/{strategy.name}]"
    dspy.configure(lm=lm)
    if num_threads > 1:
        executor = ThreadPoolExecutor(max_workers=num_threads)
        result_iter: Iterable[TaskResult] = executor.map(process_example, examples)
    else:
        executor = None
        result_iter = (process_example(example) for example in examples)

    results: list[TaskResult] = []
    running_score = 0.0
    progress_bar = tqdm(total=len(examples), desc=progress_desc, unit="example")
    try:
        for result in result_iter:
            results.append(result)
            running_score += result.score
            seen = len(results)
            progress_bar.set_postfix_str(f"{metric_name}={running_score / seen:.4f}")
            progress_bar.update(1)
    finally:
        progress_bar.close()
        if executor is not None:
            executor.shutdown(wait=True)

    total_score = running_score
    num_failed = sum(1 for result in results if "error" in result.extra)
    score = total_score / len(results) if results else 0.0
    logger.info(
        f"Done: {metric_name}={score:.4f} ({total_score:.4f}/{len(results)})"
        + (f", {num_failed} failed" if num_failed else "")
    )

    return EvaluationReport(
        task=task.name,
        model=model_name,
        backend=backend,
        strategy=strategy.name,
        metric=metric_name,
        judge=judge_name,
        judge_model=judge_model_name,
        score=score,
        num_examples=len(results),
        results=results,
    )


def save_report(report: EvaluationReport, output_dir: str) -> str:
    dataset_name = report.task.replace("/", "--")
    model_name = report.model.replace("/", "--")
    nested_dir = os.path.join(output_dir, dataset_name, model_name, report.strategy)
    os.makedirs(nested_dir, exist_ok=True)
    output_path = os.path.join(nested_dir, "results.json")
    exclude: set[str] = {"judge", "judge_model"} if report.judge is None else set()
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(report.model_dump_json(indent=2, exclude=exclude))
    logger.info(f"Saved results to {output_path}")

    return output_path
