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
    # LM token cost of the STRATEGY only (judge/extraction excluded), per example.
    token_usage: dict[str, int] | None = None
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
    model_kwargs: dict[str, object] = {}
    lm_retry_kwargs: dict[str, object] = {}
    strategy: str
    strategy_kwargs: dict[str, object] = {}
    decoding: str = "single-pass"
    decoding_kwargs: dict[str, object] = {}
    metric: str
    judge: str | None = None
    judge_model: str | None = None
    judge_kwargs: dict[str, object] = {}
    score: float
    num_examples: int
    max_examples: int | None = None
    num_evaluated_examples: int | None = None
    num_threads: int = 8
    num_failed: int = 0
    total_time: float = 0.0
    category_scores: dict[str, CategoryScore] = {}
    # Decomposes the headline ``score`` into its components (e.g.
    # ``llm_judge_score``, ``premises_f1``); logical-reasoning blends them as
    # ``0.5 * answer + 0.5 * premises_f1``. Mean over non-null values.
    score_components: dict[str, float] | None = None
    search_stats: dict[str, float] | None = None
    # PoT-Z3 pipeline telemetry (generate/execute counts, execution success
    # rate), mean over examples that carried ``pot_info``.
    pot_stats: dict[str, float] | None = None
    # Self-consistency decoding telemetry (agreement confidence, vote counts),
    # mean over examples that carried ``decoding_stats``. Absent for single-pass.
    decoding_stats: dict[str, float] | None = None
    # LM token cost of the STRATEGY only (judge/extraction excluded): mean per
    # example (`token_usage`) and summed over the run (`total_token_usage`).
    token_usage: dict[str, float] | None = None
    total_token_usage: dict[str, int] | None = None
    results: list[TaskResult] = []


class JudgeOutcome(BaseModel):
    """Full outcome of judging a single example with an LLM-as-a-judge.

    ``method`` records how the verdict was produced: ``"deterministic"`` for the
    closed-answer fast path (MCQ label-set / yes-no-uncertain stance comparison,
    no judge LM call) or ``"llm"`` when the judge LM was invoked.
    """

    verdict: str
    error_type: str
    feedback: str
    score: float
    method: str = "llm"
