from virex_bench.models.base import GenerationConfig, LanguageModel
from virex_bench.tasks.base import ReasoningExample


class ReasoningStrategy:
    """Base class for prompting / inference-time-scaling strategies.

    Real strategies will be implemented as `dspy.Module` subclasses; this toy base
    keeps the package runnable without DSPy installed. A strategy turns an example
    into a prompt, asks the model, and extracts a final answer.
    """

    name: str = "base"

    def build_prompt(self, example: ReasoningExample) -> str:
        premises = "\n".join(f"- {premise}" for premise in example.premises)
        return f"Cho các tiền đề sau:\n{premises}\n\nCâu hỏi: {example.question}\n"

    def extract_answer(self, completion: str) -> str:
        marker = "Câu trả lời:"
        if marker in completion:
            return completion.split(marker, 1)[1].strip()
        return completion.strip()

    def run(self, model: LanguageModel, example: ReasoningExample) -> str:
        prompt = self.build_prompt(example)
        completions = model.generate(prompt, GenerationConfig())
        return self.extract_answer(completions[0])
