from virex_bench import envs
from virex_bench.models.base import BaseLM


def load_backend(
    backend: str,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    cache: bool = False,
    **kwargs,
) -> BaseLM:
    """Build a `BaseLM` for an OpenAI-compatible endpoint.

    `backend` is the litellm provider prefix (e.g. "openai"); the model string handed
    to `dspy.LM` is f"{backend}/{model}". `api_base` and `api_key` fall back to the
    `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables when not provided.
    """
    resolved_api_base = api_base if api_base is not None else envs.OPENAI_BASE_URL
    resolved_api_key = api_key if api_key is not None else envs.OPENAI_API_KEY
    return BaseLM(
        model=f"{backend}/{model}",
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        cache=cache,
        **kwargs,
    )
