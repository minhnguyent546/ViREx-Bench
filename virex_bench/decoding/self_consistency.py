from __future__ import annotations

import contextvars
import copy
import difflib
import json
import re
import time
from collections.abc import Sequence
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any

import dspy

from virex_bench import envs
from virex_bench.decoding.base import DecodingStrategy
from virex_bench.logger import init_logger
from virex_bench.models.base import BaseLM
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.modules import ChainOfThought

if TYPE_CHECKING:
    from virex_bench.tasks.base import ReasoningTask

logger = init_logger(__name__)

_OPEN_ANSWER_TYPES = frozenset({"numeric", "open_ended"})
_FUZZY_MATCH_THRESHOLD = 0.65


class SelfConsistency(DecodingStrategy):
    """Self-consistency: sample N paths, majority-vote + LLM-aggregate the answer.

    Runs the wrapped strategy ``num_samples`` times in parallel (each path uses a shallow
    copy of the LM with ``cache=False`` so DSPy's cache produces genuinely different
    samples), then merges the candidates in two stages:

    1. **Deterministic majority vote** on the normalized ``answer`` field — selects the
       winning group and a representative prediction for all structural fields.
    2. **LLM aggregator** — synthesizes a clean ``answer`` + ``explanation`` from all
       candidates. Handles open-ended answers where semantic equivalence != string
       equality.

    The deterministic vote is always computed and used as the fallback whenever the
    aggregator fails, times out, or returns an answer matching no candidate.

    Requires the task to provide ``aggregation_signature`` — otherwise
    :meth:`configure_from_task` raises. All knobs default from the
    ``VIREX_BENCH_SC_*`` env vars; explicit kwargs override. Requires
    ``temperature > 0`` on the model under test for sample diversity.
    """

    name = "self-consistency"

    def __init__(
        self,
        strategy: ReasoningStrategy,
        *,
        num_samples: int | None = None,
        max_workers: int | None = None,
        solve_timeout: int | None = None,
        aggregate_timeout: int | None = None,
        use_aggregator: bool | None = None,
    ) -> None:
        super().__init__(strategy)
        self.num_samples = (
            num_samples if num_samples is not None else envs.VIREX_BENCH_SC_NUM_SAMPLES
        )
        self.max_workers = (
            max_workers if max_workers is not None else envs.VIREX_BENCH_SC_MAX_WORKERS
        )
        self.solve_timeout = (
            solve_timeout if solve_timeout is not None else envs.VIREX_BENCH_SC_SOLVE_TIMEOUT
        )
        self.aggregate_timeout = (
            aggregate_timeout
            if aggregate_timeout is not None
            else envs.VIREX_BENCH_SC_AGGREGATE_TIMEOUT
        )
        self.use_aggregator = (
            use_aggregator if use_aggregator is not None else envs.VIREX_BENCH_SC_USE_AGGREGATOR
        )

        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")

        self._aggregation_signature: type[dspy.Signature] | None = None
        self._aggregator: ChainOfThought | None = None
        self._aggregator_rationale_field = dspy.OutputField(
            desc=(
                "Reason step by step about which candidate answer is the majority. "
                "Count occurrences of each distinct answer value, applying semantic "
                "equivalence for open-ended answers. Identify the majority before "
                "producing the output fields."
            ),
        )

    def configure_from_task(self, task: ReasoningTask) -> None:
        if task.aggregation_signature is None:
            raise ValueError(
                f"Task '{task.name}' does not provide an aggregation_signature; "
                f"self-consistency is not supported for this task."
            )
        self._aggregation_signature = task.aggregation_signature
        self._aggregator = ChainOfThought(
            task.aggregation_signature,
            rationale_field=self._aggregator_rationale_field,
        )
        logger.debug(
            f"SelfConsistency configured with aggregator signature: "
            f"{task.aggregation_signature.__name__}"
        )

    @property
    def config(self) -> dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "max_workers": self.max_workers,
            "solve_timeout": self.solve_timeout,
            "aggregate_timeout": self.aggregate_timeout,
            "use_aggregator": self.use_aggregator,
        }

    @property
    def display_name(self) -> str:
        return f"{self.name}@{self.num_samples}"

    def forward(self, **inputs: object) -> dspy.Prediction:
        start_time = time.perf_counter()

        results = self._run_parallel_paths(inputs)

        if not results:
            raise RuntimeError(
                f"All {self.num_samples} self-consistency paths failed to produce an answer."
            )

        winning_answer, winner_indices = self._majority_vote(results)
        representative = results[winner_indices[0]]
        confidence = len(winner_indices) / len(results)

        # Validate the aggregator's output before merging it onto the representative.
        # The vote winner is always the fallback if the aggregator fails, returns
        # an empty/hallucinated answer, or diverges too far from the candidates.
        if self.use_aggregator:
            overrides = self._aggregate(inputs, results)
            if overrides is not None:
                agg_answer = str(overrides.get("answer", ""))
                agg_answer_type = str(overrides.get("answer_type", ""))
                if not agg_answer:
                    # Aggregator produced nothing usable — pure fallback.
                    logger.warning(
                        "Aggregator returned empty answer; falling back to vote winner."
                    )
                    merged = {**representative}
                elif agg_answer_type in _OPEN_ANSWER_TYPES:
                    # For open-ended / numeric answers, the aggregator may legitimately
                    # rephrase or merge candidates, so we accept it if the output is
                    # sufficiently similar (SequenceMatcher ratio >= threshold) to at
                    # least one candidate. Otherwise it's likely hallucinated.
                    best_ratio = self._best_answer_similarity(results, agg_answer)
                    if best_ratio >= _FUZZY_MATCH_THRESHOLD:
                        merged = {**representative, **overrides}
                    else:
                        logger.warning(
                            f"Aggregator answer '{agg_answer}' (best similarity "
                            f"{best_ratio:.2f}, threshold {_FUZZY_MATCH_THRESHOLD}) "
                            f"matched no candidate; falling back to vote winner "
                            f"'{winning_answer}'."
                        )
                        merged = {**representative}
                else:
                    # For closed-answer types (multiple_choice, yes_no_uncertain) the
                    # aggregator's answer must exactly match (after normalization) one
                    # of the candidates — any deviation is a hallucination.
                    matching = self._find_matching_result(results, agg_answer)
                    if matching is not None:
                        merged = {**representative, **overrides}
                    else:
                        logger.warning(
                            f"Aggregator answer '{agg_answer}' for closed type "
                            f"'{agg_answer_type}' matched no candidate; "
                            f"falling back to vote winner '{winning_answer}'."
                        )
                        merged = {**representative}
            else:
                # Aggregator call itself failed (timeout, exception, or empty result).
                merged = {**representative}
        else:
            # Aggregator disabled — use the vote winner as-is.
            merged = {**representative}

        merged["self_consistency_confidence"] = confidence
        merged["self_consistency_vote_count"] = len(winner_indices)
        merged["self_consistency_total"] = len(results)

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"Self-consistency completed in {elapsed:.2f}s "
            f"(answer={merged.get('answer', '?')}, confidence={confidence:.2f})"
        )
        return dspy.Prediction(**merged)

    def _run_parallel_paths(self, inputs: dict[str, object]) -> list[dspy.Prediction]:
        """Run ``num_samples`` paths in parallel and return successful paths in index order."""
        # Run each path in a copied context so the active query-id tag (set by
        # `log_query_context` in the evaluator) reaches the worker threads, which
        # ThreadPoolExecutor does not propagate on its own.
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            future_to_index: dict[Future[dspy.Prediction], int] = {
                executor.submit(
                    contextvars.copy_context().run,
                    self._run_single_path,
                    inputs,
                    path_index,
                ): path_index
                for path_index in range(self.num_samples)
            }
            done, not_done = wait(
                set(future_to_index),
                timeout=self.solve_timeout,
                return_when=ALL_COMPLETED,
            )

            results_by_index: dict[int, dspy.Prediction] = {}
            for future in done:
                path_index = future_to_index[future]
                try:
                    path_result = future.result()
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"Self-consistency path {path_index} raised: {error!r}")
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
                    f"Self-consistency path {path_index} did not finish within "
                    f"{self.solve_timeout}s ({cancel_status})."
                )

            return [
                results_by_index[path_index]
                for path_index in range(self.num_samples)
                if path_index in results_by_index
            ]
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_single_path(
        self,
        inputs: dict[str, object],
        path_index: int,
    ) -> dspy.Prediction:
        base_lm = dspy.settings.lm
        assert isinstance(base_lm, BaseLM)  # set by the evaluator via dspy.configure(lm=...)
        sampling_lm = self._copy_lm_with_request_timeout(
            base_lm,
            self.solve_timeout,
            cache=False,
        )
        with dspy.context(lm=sampling_lm):
            result = self.strategy(**inputs)
        logger.debug(
            f"Self-consistency path {path_index} completed: "
            f"answer={getattr(result, 'answer', '?')}"
        )
        return result

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        return re.sub(r"\s+", " ", str(answer).strip().lower())

    def _majority_vote(
        self,
        results: Sequence[dspy.Prediction],
    ) -> tuple[str, list[int]]:
        """Group results by normalized answer and pick the largest group.

        Tie-break: first-seen (deterministic).
        """
        groups: dict[str, list[int]] = {}
        for index, result in enumerate(results):
            normalized = self._normalize_answer(str(getattr(result, "answer", "Unknown")))
            groups.setdefault(normalized, []).append(index)

        best_key = max(groups, key=lambda norm: len(groups[norm]))
        winner_indices = groups[best_key]
        winning_answer = str(getattr(results[winner_indices[0]], "answer", "Unknown"))
        return winning_answer, winner_indices

    def _find_matching_result(
        self,
        results: Sequence[dspy.Prediction],
        target_answer: str,
    ) -> dspy.Prediction | None:
        """Find the first result whose answer matches *target_answer*."""
        target_norm = self._normalize_answer(target_answer)
        for result in results:
            if self._normalize_answer(str(getattr(result, "answer", ""))) == target_norm:
                return result
        return None

    def _best_answer_similarity(
        self,
        results: Sequence[dspy.Prediction],
        target_answer: str,
    ) -> float:
        """Return the highest SequenceMatcher ratio between *target_answer* and any candidate."""
        target_norm = self._normalize_answer(target_answer)
        best_ratio = 0.0
        for result in results:
            candidate_norm = self._normalize_answer(str(getattr(result, "answer", "")))
            ratio = difflib.SequenceMatcher(None, target_norm, candidate_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
        return best_ratio

    def _aggregate(
        self,
        inputs: dict[str, object],
        results: Sequence[dspy.Prediction],
    ) -> dict[str, Any] | None:
        """Call the aggregator LM to determine the majority answer and synthesize fields.

        Returns a dict of fields to override on the representative prediction, or
        ``None`` on failure / timeout so the caller can fall back to the code-based
        majority vote. The ``explanation`` output field is mapped to ``reasoning``
        (the aggregator's own CoT ``reasoning`` field is dropped).
        """
        candidates: list[dict[str, Any]] = [
            {key: getattr(result, key) for key in result.keys() if not key.startswith("_")}
            for result in results
        ]

        signature = self._aggregation_signature
        assert signature is not None  # set by configure_from_task
        aggregator = self._aggregator
        assert aggregator is not None  # set by configure_from_task
        aggregator_kwargs: dict[str, object] = {}
        for field_name in signature.input_fields:
            if field_name == "candidate_answers":
                aggregator_kwargs["candidate_answers"] = json.dumps(candidates, ensure_ascii=False)
            elif field_name in inputs:
                aggregator_kwargs[field_name] = inputs[field_name]

        try:
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future: Future[dspy.Prediction] = executor.submit(
                    contextvars.copy_context().run,
                    self._call_aggregator,
                    **aggregator_kwargs,
                )
                done, _not_done = wait(
                    {future},
                    timeout=self.aggregate_timeout,
                    return_when=ALL_COMPLETED,
                )
                if future not in done:
                    was_cancelled = future.cancel()
                    cancel_status = (
                        "cancelled before start"
                        if was_cancelled
                        else "already running; it may continue until the LM request timeout"
                    )
                    logger.warning(
                        f"Aggregator LM did not finish within {self.aggregate_timeout}s "
                        f"({cancel_status})."
                    )
                    return None
                agg_result = future.result()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            overrides = self._extract_overrides(agg_result)
            answer = str(overrides.get("answer", "")).strip()
            if not answer:
                logger.warning("Aggregator returned empty answer.")
                return None
            logger.debug(f"Aggregator returned answer='{answer}'")
            return overrides
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Aggregator LM failed: {error!r}")
            return None

    @staticmethod
    def _extract_overrides(agg_result: dspy.Prediction) -> dict[str, Any]:
        """Convert an aggregator prediction into a dict of fields to override.

        Drops ``reasoning`` (the aggregator's internal CoT) and maps
        ``explanation`` → ``reasoning``.
        """
        overrides: dict[str, Any] = {}
        for field_name in agg_result.keys():
            if field_name == "reasoning":
                continue
            value = getattr(agg_result, field_name, None)
            if value is None:
                continue
            if field_name == "explanation":
                overrides["reasoning"] = value
            else:
                overrides[field_name] = value
        return overrides

    def _call_aggregator(self, **kwargs: object) -> dspy.Prediction:
        """Call the aggregator module — extracted so it can run in a timed thread."""
        assert self._aggregator is not None  # set by configure_from_task
        base_lm = dspy.settings.lm
        assert isinstance(base_lm, BaseLM)  # set by the evaluator via dspy.configure(lm=...)
        aggregation_lm = self._copy_lm_with_request_timeout(base_lm, self.aggregate_timeout)
        with dspy.context(lm=aggregation_lm):
            return self._aggregator(**kwargs)

    @staticmethod
    def _copy_lm_with_request_timeout(
        base_lm: BaseLM,
        request_timeout: int | None,
        *,
        cache: bool | None = None,
    ) -> BaseLM:
        copied_lm = copy.copy(base_lm)
        if cache is not None:
            copied_lm.cache = cache

        lm_kwargs = dict(copied_lm.kwargs)
        if request_timeout is not None:
            lm_kwargs["timeout"] = request_timeout
        copied_lm.kwargs = lm_kwargs
        return copied_lm
