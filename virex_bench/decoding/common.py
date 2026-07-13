import contextvars
import copy
import re
from collections.abc import Callable
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

from virex_bench.logger import init_logger
from virex_bench.models.base import BaseLM

logger = init_logger(__name__)

T = TypeVar("T")


def normalize_case_and_whitespaces(answer: str) -> str:
    """Normalize whitespace and case for answer grouping/comparison."""
    return re.sub(r"\s+", " ", str(answer).strip().lower())


def copy_lm_with_request_timeout(
    base_lm: BaseLM,
    request_timeout: int | None,
    *,
    cache: bool | None = None,
) -> BaseLM:
    """Shallow-copy an LM, optionally overriding the request timeout and cache flag.

    The ``kwargs`` dict is replaced (not mutated) so the base LM is untouched.
    """
    copied_lm = copy.copy(base_lm)
    if cache is not None:
        copied_lm.cache = cache

    lm_kwargs = dict(copied_lm.kwargs)
    if request_timeout is not None:
        lm_kwargs["timeout"] = request_timeout
    copied_lm.kwargs = lm_kwargs
    return copied_lm


def run_parallel_paths(
    *,
    num_samples: int,
    max_workers: int,
    solve_timeout: int,
    inputs: dict[str, object],
    run_single_path: Callable[[dict[str, object], int], T],
    strategy_label: str,
) -> list[T]:
    """Run ``num_samples`` paths in parallel; return successful results in index order.

    Uses ``contextvars.copy_context`` so the evaluator's query-id tag reaches worker
    threads. Failed or timed-out paths are logged and dropped from the result.
    """
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_index: dict[Future[T], int] = {
            executor.submit(
                contextvars.copy_context().run,
                run_single_path,
                inputs,
                path_index,
            ): path_index
            for path_index in range(num_samples)
        }
        done, not_done = wait(
            set(future_to_index),
            timeout=solve_timeout,
            return_when=ALL_COMPLETED,
        )

        results_by_index: dict[int, T] = {}
        for future in done:
            path_index = future_to_index[future]
            try:
                path_result = future.result()
            except Exception as error:
                logger.warning(f"{strategy_label} path {path_index} raised: {error!r}")
                path_result = None
            if path_result is not None:
                results_by_index[path_index] = path_result

        for future in not_done:
            path_index = future_to_index[future]
            was_cancelled = future.cancel()
            cancel_status = (
                "cancelled before start"
                if was_cancelled
                else "already running; it may continue until the LM request timeout"
            )
            logger.warning(
                f"{strategy_label} path {path_index} did not finish within "
                f"{solve_timeout}s ({cancel_status})."
            )

        return [
            results_by_index[path_index]
            for path_index in range(num_samples)
            if path_index in results_by_index
        ]
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
