import os

from pydantic import BaseModel, Field

from virex_bench.logger import init_logger
from virex_bench.models.base import LanguageModel
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningTask, TaskResult

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
    model: LanguageModel,
    strategy: ReasoningStrategy,
    backend: str,
) -> EvaluationReport:
    """Run `strategy` with `model` over every example in `task` and score accuracy."""
    examples = task.load_examples()
    logger.info(
        f"Evaluating task={task.name} model={model.name} "
        f"strategy={strategy.name} on {len(examples)} examples"
    )
    results: list[TaskResult] = []
    for example in examples:
        predicted = strategy.run(model, example)
        is_correct = _normalize(predicted) == _normalize(example.answer)
        results.append(
            TaskResult(
                example_id=example.example_id,
                predicted=predicted,
                gold=example.answer,
                is_correct=is_correct,
            )
        )

    num_correct = sum(1 for result in results if result.is_correct)
    accuracy = num_correct / len(results) if results else 0.0
    logger.info(f"Done: accuracy={accuracy:.4f} ({num_correct}/{len(results)})")
    return EvaluationReport(
        task=task.name,
        model=model.name,
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
