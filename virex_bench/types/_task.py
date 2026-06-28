from pydantic import BaseModel


class ReasoningExample(BaseModel):
    """A single logical-reasoning instance: premises + a question and its gold answer."""

    premises: list[str]
    question: str
    answer: str
    example_id: str = ""
    category: str | None = None


class DatasetConfig(BaseModel):
    """Location of a task's dataset on the HuggingFace Hub.

    Mirrors the keyword arguments of ``datasets.load_dataset`` so that a task is
    fully described by its metadata, and the dataset can be reloaded for any run.
    """

    path: str
    name: str | None = None
    split: str = "test"
    revision: str | None = None
    num_proc: int | None = None


class TaskMetadata(BaseModel):
    name: str
    description: str
    language: str = "vie"
    dataset: DatasetConfig
    main_metric: str = "exact_match"
    judge: str | None = None
