from pydantic import BaseModel

from virex_bench.types._task import DatasetConfig


class TaskResult(BaseModel):
    """Per-example prediction record produced during evaluation."""

    example_id: str
    inputs: dict[str, object] = {}
    predicted: str
    gold: str
    score: float
    llm_judge_score: float | None = None
    premises_f1: float | None = None
    predicted_premises_used: list[int] | None = None
    category: str | None = None
    extra: dict[str, object] = {}


class ScoreComponents(BaseModel):
    """Final score plus optional auxiliary components a task may report per example.

    ``score`` is the value the evaluator aggregates. The remaining fields are
    recorded on :class:`TaskResult` for inspection and are populated only by tasks
    that blend extra signals into the score (e.g. premise-selection F1); tasks
    without such signals leave them ``None``.
    """

    score: float
    llm_judge_score: float | None = None
    premises_f1: float | None = None
    predicted_premises_used: list[int] | None = None


class CategoryScore(BaseModel):
    """Mean score and supporting count for a single example category."""

    score: float
    num_examples: int


class EvaluationReport(BaseModel):
    task: str
    dataset: DatasetConfig
    model: str
    backend: str
    model_kwargs: dict[str, object] = {}
    strategy: str
    metric: str
    judge: str | None = None
    judge_model: str | None = None
    score: float
    num_examples: int
    num_failed: int = 0
    total_time: float = 0.0
    category_scores: dict[str, CategoryScore] = {}
    results: list[TaskResult] = []


class JudgeOutcome(BaseModel):
    """Full outcome of judging a single example with an LLM-as-a-judge."""

    verdict: str
    error_type: str
    feedback: str
    score: float
