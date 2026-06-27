import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import dspy
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from virex_bench.logger import init_logger
from virex_bench.models import BaseLM
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningExample, ReasoningTask, TaskResult

logger = init_logger(__name__)


def _normalize(answer: str) -> str:
    return answer.strip().lower().rstrip(".")


class EvaluationReport(BaseModel):
    task: str
    model: str
    backend: str
    strategy: str
    accuracy: float
    num_examples: int
    results: list[TaskResult] = Field(default_factory=list)


def evaluate(
    task: ReasoningTask,
    lm: BaseLM,
    strategy: ReasoningStrategy,
    model_name: str,
    backend: str,
    num_threads: int = 4,
) -> EvaluationReport:
    """Run `strategy` with `lm` over every example in `task` and score accuracy.

    Examples are evaluated concurrently across `num_threads` worker threads. A
    failure on a single example (e.g. an API or parsing error) is logged and
    recorded as an incorrect result instead of aborting the whole run.
    """
    examples = task.load_examples()
    logger.info(
        f"Evaluating task={task.name} model={model_name} "
        f"strategy={strategy.name} on {len(examples)} examples (num_threads={num_threads})"
    )

    def process_example(example: ReasoningExample) -> TaskResult:
        try:
            inputs = task.example_to_inputs(example)
            prediction = strategy(**inputs)
            predicted = str(prediction.answer)
            is_correct = _normalize(predicted) == _normalize(example.answer)
            extra: dict[str, object] = {}
            reasoning = getattr(prediction, "reasoning", None)
            if reasoning is not None:
                extra["reasoning"] = reasoning
        except Exception as error:
            logger.warning(f"Example {example.example_id} failed: {error!r}")
            return TaskResult(
                example_id=example.example_id,
                predicted="",
                gold=example.answer,
                is_correct=False,
                extra={"error": f"{type(error).__name__}: {error}"},
            )
        return TaskResult(
            example_id=example.example_id,
            predicted=predicted,
            gold=example.answer,
            is_correct=is_correct,
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
    num_running_correct = 0
    progress_bar = tqdm(total=len(examples), desc=progress_desc, unit="example")
    try:
        for result in result_iter:
            results.append(result)
            num_running_correct += result.is_correct
            seen = len(results)
            progress_bar.set_postfix_str(
                f"acc={num_running_correct / seen:.4f} ({num_running_correct}/{seen})"
            )
            progress_bar.update(1)
    finally:
        progress_bar.close()
        if executor is not None:
            executor.shutdown(wait=True)

    num_correct = num_running_correct
    num_failed = sum(1 for result in results if "error" in result.extra)
    accuracy = num_correct / len(results) if results else 0.0
    logger.info(
        f"Done: accuracy={accuracy:.4f} ({num_correct}/{len(results)})"
        + (f", {num_failed} failed" if num_failed else "")
    )
    return EvaluationReport(
        task=task.name,
        model=model_name,
        backend=backend,
        strategy=strategy.name,
        accuracy=accuracy,
        num_examples=len(results),
        results=results,
    )


def save_report(report: EvaluationReport, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{report.task}__{report.strategy}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(report.model_dump_json(indent=2))
    logger.info(f"Saved results to {output_path}")
    return output_path
