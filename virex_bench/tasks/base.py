from pydantic import BaseModel, Field


class ReasoningExample(BaseModel):
    """A single logical-reasoning instance: premises + a question and its gold answer."""

    premises: list[str]
    question: str
    answer: str
    example_id: str = ""


class TaskMetadata(BaseModel):
    name: str
    description: str
    language: str = "vie"


class ReasoningTask:
    """Base class for reasoning tasks. Subclasses provide metadata and load examples."""

    metadata: TaskMetadata

    def load_examples(self) -> list[ReasoningExample]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.metadata.name


class TaskResult(BaseModel):
    """Per-example prediction record produced during evaluation."""

    example_id: str
    predicted: str
    gold: str
    is_correct: bool
    extra: dict[str, object] = Field(default_factory=dict)
