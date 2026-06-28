from typing import Any

import dspy
import litellm.exceptions
import tenacity

from virex_bench import envs
from virex_bench.logger import init_logger

logger = init_logger(__name__)

# Transient-only: connection drops/timeouts, 429, 500, 502, 503. Permanent client
# errors (400/401/403/404/422) are excluded so the retry budget isn't wasted.
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.BadGatewayError,
    litellm.exceptions.ServiceUnavailableError,
    ConnectionError,
)


def _log_retry(retry_state: tenacity.RetryCallState) -> None:
    outcome = retry_state.outcome
    next_action = retry_state.next_action
    if outcome is None or not outcome.failed or next_action is None:
        return
    exception = outcome.exception()
    message = " ".join(str(exception).split())  # collapse whitespace (e.g. gateway HTML)
    if len(message) > 1024:
        message = message[:1024] + "..."
    logger.warning(
        f"Transient LM error ({type(exception).__name__}: {message}); "
        f"retrying in {next_action.sleep:.1f}s (attempt {retry_state.attempt_number + 1})"
    )


def _build_retry_kwargs() -> dict[str, Any]:
    initial_wait = max(0.0, envs.VIREX_BENCH_LM_RETRY_MIN_WAIT)
    return {
        "stop": tenacity.stop_after_attempt(max(1, envs.VIREX_BENCH_LM_MAX_RETRIES)),
        "wait": tenacity.wait_exponential_jitter(
            initial=initial_wait,
            max=max(initial_wait, envs.VIREX_BENCH_LM_RETRY_MAX_WAIT),
            exp_base=2.0,
            jitter=max(0.0, envs.VIREX_BENCH_LM_RETRY_JITTER),
        ),
        "retry": tenacity.retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        "before_sleep": _log_retry,
        "reraise": True,
    }


class BaseLM(dspy.LM):
    """Project-wide language-model wrapper that owns the retry layer.

    litellm's built-in retry is disabled (`num_retries=0`); transient connection
    errors are retried here with env-driven exponential backoff + jitter. Sync
    (`forward`) and async (`aforward`) share identical semantics.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("num_retries", None)
        super().__init__(*args, num_retries=0, **kwargs)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        return tenacity.Retrying(**_build_retry_kwargs())(
            super().forward, prompt=prompt, messages=messages, **kwargs
        )

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        return await tenacity.AsyncRetrying(**_build_retry_kwargs())(
            super().aforward, prompt=prompt, messages=messages, **kwargs
        )
