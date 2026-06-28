from pydantic import BaseModel


class TaskResult(BaseModel):
    """Per-example prediction record produced during evaluation."""

    example_id: str
    inputs: dict[str, object] = {}
    predicted: str
    gold: str
    score: float
    category: str | None = None
    extra: dict[str, object] = {}


class CategoryScore(BaseModel):
    """Mean score and supporting count for a single example category."""

    score: float
    num_examples: int


class EvaluationReport(BaseModel):
    task: str
    model: str
    backend: str
    strategy: str
    metric: str
    judge: str | None = None
    judge_model: str | None = None
    score: float
    num_examples: int
    category_scores: dict[str, CategoryScore] = {}
    results: list[TaskResult] = []


class JudgeOutcome(BaseModel):
    """Full outcome of judging a single example with an LLM-as-a-judge."""

    verdict: str
    error_type: str
    feedback: str
    score: float
