from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningExample


class CoTStrategy(ReasoningStrategy):
    """Chain-of-Thought: prompt the model to reason step by step before answering."""

    name = "cot"

    def build_prompt(self, example: ReasoningExample) -> str:
        return (
            super().build_prompt(example)
            + "\nHãy suy luận từng bước, sau đó đưa ra kết luận.\nCâu trả lời:"
        )
