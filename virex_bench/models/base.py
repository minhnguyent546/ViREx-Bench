from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class GenerationConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 512
    num_samples: int = 1


@runtime_checkable
class LanguageModel(Protocol):
    """Minimal interface every backend must implement."""

    name: str

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        """Return `config.num_samples` completions for `prompt`."""
        ...
