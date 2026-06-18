from virex_bench.models.base import GenerationConfig, LanguageModel


class EchoBackend:
    """A dependency-free toy backend.

    It does not call any real model; it returns a fixed placeholder answer so the
    CLI and evaluation loop can be exercised end-to-end. Replace with vLLM / HF /
    API backends for real runs.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        completion = "Câu trả lời: Có"
        return [completion for _ in range(config.num_samples)]


def load_backend(backend: str, model: str) -> LanguageModel:
    if backend == "echo":
        return EchoBackend(name=model)
    if backend in {"vllm", "hf", "api"}:
        raise NotImplementedError(
            f"Backend {backend!r} is not implemented yet; use --backend echo for now."
        )
    raise ValueError(f"Unknown backend {backend!r}. Available: echo, vllm, hf, api.")
