import time
from collections.abc import Sequence
from typing import Any

import dspy

from virex_bench import envs
from virex_bench.decoding.base import DecodingStrategy
from virex_bench.decoding.common import (
    copy_lm_with_request_timeout,
    normalize_case_and_whitespaces,
    run_parallel_paths,
)
from virex_bench.logger import init_logger
from virex_bench.models.base import BaseLM, CapturingLMWrapper
from virex_bench.strategies.base import ReasoningStrategy

logger = init_logger(__name__)


def _extract_token_logprobs(logprobs: Any) -> list[float]:
    """Extract per-token log-prob values from a choice's logprobs blob.

    Handles both pydantic (litellm) and plain-dict shapes; returns ``[]`` on ``None``.
    """
    if logprobs is None:
        return []
    content = getattr(logprobs, "content", None)
    if content is None and isinstance(logprobs, dict):
        content = logprobs.get("content")
    if not content:
        return []
    values: list[float] = []
    for token in content:
        logprob = getattr(token, "logprob", None)
        if logprob is None and isinstance(token, dict):
            logprob = token.get("logprob")
        if logprob is not None:
            values.append(float(logprob))
    return values


def _mean_logprob(logprobs: Any) -> float | None:
    """Mean log-probability of generated tokens, or ``None`` if unavailable."""
    values = _extract_token_logprobs(logprobs)
    if not values:
        return None
    return sum(values) / len(values)


def _last_completion_logprobs(outputs: Sequence[Any]) -> Any:
    """Return the logprobs blob of the last completion that carries one."""
    last = None
    for output in outputs:
        if isinstance(output, dict):
            logprobs = output.get("logprobs")
            if logprobs is not None:
                last = logprobs
    return last


class SelfCertaintyLM(CapturingLMWrapper):
    """LM wrapper that injects ``logprobs=True`` and captures per-token log-probs.

    Uses mean token log-prob as the certainty signal — the strongest peakedness
    proxy a serving API offers (full-vocab KL-divergence needs local model loading).
    """

    _local_attrs = ("last_logprobs",)

    def __init__(self, base_lm: BaseLM) -> None:
        super().__init__(base_lm)
        object.__setattr__(self, "last_logprobs", None)

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        kwargs["logprobs"] = True
        result = self._base_lm(messages=messages, prompt=prompt, **kwargs)
        object.__setattr__(self, "last_logprobs", _last_completion_logprobs(result))
        return result

    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        kwargs["logprobs"] = True
        result = await self._base_lm.acall(messages=messages, prompt=prompt, **kwargs)
        object.__setattr__(self, "last_logprobs", _last_completion_logprobs(result))
        return result


class SelfCertainty(DecodingStrategy):
    """Self-certainty decoding: sample N candidates, score by mean log-prob, Borda-vote.

    The original metric (Kang et al., 2025) uses full-vocab KL-divergence, unavailable
    on API-served models. We substitute mean token log-prob as the certainty signal.

    Each path wraps the LM in :class:`SelfCertaintyLM` (``cache=False``,
    ``logprobs=True``). Candidates are ranked by certainty descending; each casts a
    Borda-weighted vote ``(N - rank + 1) ** borda_power`` for its normalized answer.
    ``borda_power = 0`` reproduces plain majority vote; degrades to it when the
    endpoint returns no log-probs. Requires ``temperature > 0`` for diversity.

    Restricted to single-call strategies (``direct``, ``cot``): the certainty signal
    is the mean log-prob of the answer-generating LM call. Multi-step strategies
    (``cr``, ``tot``, ``pot``) issue many internal calls, so the captured log-probs
    would reflect an arbitrary internal completion rather than the answer, turning the
    Borda weighting into noise.
    """

    name = "self-certainty"
    description = "Sample N candidates, score by mean log-prob, Borda vote."
    compatible_strategies = {"direct", "cot"}

    def __init__(
        self,
        strategy: ReasoningStrategy,
        *,
        num_samples: int | None = None,
        max_workers: int | None = None,
        solve_timeout: int | None = None,
        borda_power: float | None = None,
    ) -> None:
        super().__init__(strategy)
        self.num_samples = (
            num_samples if num_samples is not None else envs.VIREX_BENCH_SELFC_NUM_SAMPLES
        )
        self.max_workers = (
            max_workers if max_workers is not None else envs.VIREX_BENCH_SELFC_MAX_WORKERS
        )
        self.solve_timeout = (
            solve_timeout if solve_timeout is not None else envs.VIREX_BENCH_SELFC_SOLVE_TIMEOUT
        )
        self.borda_power = (
            borda_power if borda_power is not None else envs.VIREX_BENCH_SELFC_BORDA_POWER
        )

        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.borda_power < 0:
            raise ValueError(f"borda_power must be >= 0, got {self.borda_power}")

    @property
    def config(self) -> dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "max_workers": self.max_workers,
            "solve_timeout": self.solve_timeout,
            "borda_power": self.borda_power,
        }

    @property
    def display_name(self) -> str:
        return f"{self.name}@{self.num_samples}"

    def forward(self, **inputs: object) -> dspy.Prediction:
        start_time = time.perf_counter()

        paths = run_parallel_paths(
            num_samples=self.num_samples,
            max_workers=self.max_workers,
            solve_timeout=self.solve_timeout,
            inputs=inputs,
            run_single_path=self._run_single_path,
            strategy_label="Self-certainty",
        )

        if not paths:
            raise RuntimeError(
                f"All {self.num_samples} self-certainty paths failed to produce an answer."
            )

        results = [prediction for prediction, _certainty in paths]
        certainties = [certainty for _prediction, certainty in paths]

        _winning_answer, winner_indices, confidence, representative_index = self._borda_vote(
            results, certainties
        )
        representative = results[representative_index]

        valid_certainties = [certainty for certainty in certainties if certainty is not None]
        mean_certainty = (
            sum(valid_certainties) / len(valid_certainties) if valid_certainties else None
        )

        merged = {**representative}
        merged["decoding_stats"] = {
            "confidence": confidence,
            "winner_certainty": certainties[representative_index],
            "mean_certainty": mean_certainty,
            "vote_count": len(winner_indices),
            "total_samples": len(results),
            "requested_samples": self.num_samples,
            "borda_power": self.borda_power,
            "logprobs_available": any(certainty is not None for certainty in certainties),
        }

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"Self-certainty completed in {elapsed:.2f}s "
            f"(answer={merged.get('answer', '?')}, confidence={confidence:.2f})"
        )
        return dspy.Prediction(**merged)

    def _run_single_path(
        self,
        inputs: dict[str, object],
        path_index: int,
    ) -> tuple[dspy.Prediction, float | None]:
        base_lm = dspy.settings.lm
        assert isinstance(base_lm, BaseLM)  # set by the evaluator via dspy.configure(lm=...)
        sampling_lm = copy_lm_with_request_timeout(
            base_lm,
            self.solve_timeout,
            cache=False,
        )
        capture_lm = SelfCertaintyLM(sampling_lm)
        with dspy.context(lm=capture_lm):
            result = self.strategy(**inputs)
        certainty = _mean_logprob(capture_lm.last_logprobs)
        logger.debug(
            f"Self-certainty path {path_index} completed: "
            f"answer={getattr(result, 'answer', '?')}, certainty={certainty}"
        )
        return result, certainty

    def _borda_vote(
        self,
        results: Sequence[dspy.Prediction],
        certainties: Sequence[float | None],
    ) -> tuple[str, list[int], float, int]:
        """Borda-weighted vote: rank by certainty descending, weight ``(N - rank + 1) ** power``.

        ``borda_power = 0`` collapses to majority vote; no log-probs → uniform weights.
        Returns ``(winning_answer, winner_indices, confidence, representative_index)``.
        """
        num_results = len(results)
        safe_certainties: list[float] = [
            certainty if certainty is not None else float("-inf") for certainty in certainties
        ]
        logprobs_available = any(certainty is not None for certainty in certainties)

        if logprobs_available:
            order = sorted(
                range(num_results), key=lambda index: safe_certainties[index], reverse=True
            )
            ranks = {index: position + 1 for position, index in enumerate(order)}
            weights = [
                (num_results - ranks[index] + 1) ** self.borda_power
                for index in range(num_results)
            ]
        else:
            logger.warning(
                "Endpoint returned no token log-probabilities; falling back to "
                "plain majority voting (enable logprobs on the serving endpoint to "
                "activate self-certainty selection)."
            )
            weights = [1.0] * num_results

        answer_indices: dict[str, list[int]] = {}
        for index, result in enumerate(results):
            normalized = normalize_case_and_whitespaces(str(getattr(result, "answer", "Unknown")))
            answer_indices.setdefault(normalized, []).append(index)

        answer_weights = {
            normalized: sum(weights[index] for index in indices)
            for normalized, indices in answer_indices.items()
        }
        total_weight = sum(weights)

        # Tie-break: max() returns the first key in iteration (insertion) order,
        # so ties resolve first-seen deterministically (as in the majority vote).
        best_answer = max(answer_weights, key=lambda normalized: answer_weights[normalized])
        winner_indices = answer_indices[best_answer]
        winning_answer = str(getattr(results[winner_indices[0]], "answer", "Unknown"))
        confidence = answer_weights[best_answer] / total_weight if total_weight > 0 else 0.0

        if logprobs_available:
            representative_index = max(winner_indices, key=lambda index: safe_certainties[index])
        else:
            representative_index = winner_indices[0]

        return winning_answer, winner_indices, confidence, representative_index
