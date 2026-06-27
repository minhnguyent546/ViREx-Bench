from pydantic import BaseModel


class TaskResult(BaseModel):
    """Per-example prediction record produced during evaluation."""

    example_id: str
    inputs: dict[str, object] = {}
    predicted: str
    gold: str
    score: float
    extra: dict[str, object] = {}


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
    results: list[TaskResult] = []


class JudgeOutcome(BaseModel):
    """Full outcome of judging a single example with an LLM-as-a-judge."""

    verdict: str
    error_type: str
    feedback: str
    score: float
