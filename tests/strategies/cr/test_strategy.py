"""Unit tests for ``CRStrategy.forward`` with fake proposer/verifier/aggregate modules.

Drives the CR accumulation loop with canned propositions/verdicts to verify:
acceptance accumulation, failure-cap bail, multi vs single verifier paths,
duplicate rejection, context-overflow graceful exit, and ``search_stats``
correctness -- without any real LM calls.

Pattern mirrors ``tests/strategies/tot/test_beam_search.py``: each fake module
returns canned outputs in sequence (one per ``__call__``) and tracks its call
count so we can assert cost accounting.
"""

from typing import Any

import dspy
import pytest
from dspy.adapters.base import AdapterParseError

from virex_bench.strategies.cr.common import (
    PropositionMeaningfulnessSignature,
    PropositionProposerSignature,
    PropositionValiditySignature,
)
from virex_bench.strategies.cr.strategy import CRStrategy
from virex_bench.strategies.registry import get_strategy


class _FakePrediction(dspy.Prediction):
    """A dspy.Prediction whose fields are set from arbitrary kwargs."""

    def __init__(self, **fields: Any) -> None:
        super().__init__()
        for key, value in fields.items():
            setattr(self, key, value)


class _FakeProposer:
    """Returns canned proposition strings in sequence, one per ``__call__``."""

    def __init__(self, propositions: list[str]) -> None:
        self._propositions = list(propositions)
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        if self._index >= len(self._propositions):
            raise AssertionError(
                f"FakeProposer exhausted: only {len(self._propositions)} canned outputs were provided"
            )
        proposition = self._propositions[self._index]
        self._index += 1
        self.call_count += 1
        return _FakePrediction(next_proposition=proposition)


class _FakeMeaningful:
    """Returns a canned is_useful verdict per call."""

    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = list(verdicts)
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        if self._index >= len(self._verdicts):
            raise AssertionError("FakeMeaningful exhausted")
        verdict = self._verdicts[self._index]
        self._index += 1
        self.call_count += 1
        return _FakePrediction(is_useful=verdict)


class _FakeValidity:
    """Returns canned (is_entailed, is_contradicted) verdicts per call."""

    def __init__(self, verdicts: list[tuple[bool, bool]]) -> None:
        self._verdicts = list(verdicts)
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        if self._index >= len(self._verdicts):
            raise AssertionError("FakeValidity exhausted")
        is_entailed, is_contradicted = self._verdicts[self._index]
        self._index += 1
        self.call_count += 1
        return _FakePrediction(is_entailed=is_entailed, is_contradicted=is_contradicted)


class _FakeAggregate:
    """Records the accumulated_context it was called with and returns a stub."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_context: str | None = None

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        self.last_context = kwargs.get("accumulated_context")
        prediction = _FakePrediction(answer="Đáp án")
        return prediction


class _OverflowProposer:
    """Raises ``ContextWindowExceededError`` on its Nth call (1-indexed)."""

    def __init__(self, overflow_on_call: int) -> None:
        self._overflow_on_call = overflow_on_call
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        if self.call_count == self._overflow_on_call:
            raise dspy.ContextWindowExceededError(message="simulated overflow")
        return _FakePrediction(next_proposition=f"proposition {self.call_count}")


class _FakeMultiProposer:
    """Returns a *batch* of canned propositions per call (multi-sample mode).

    Simulates dspy's ``.completions`` structure so ``completion_values`` picks
    up the full batch. Each call pops the next batch from the list.
    """

    def __init__(self, batches: list[list[str]]) -> None:
        self._batches = [list(batch) for batch in batches]
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        if self._index >= len(self._batches):
            raise AssertionError(
                f"FakeMultiProposer exhausted: only {len(self._batches)} batches provided"
            )
        batch = self._batches[self._index]
        self._index += 1
        self.call_count += 1
        prediction = _FakePrediction(next_proposition=batch[0])
        # Simulate dspy's n>1 completions structure (``completions`` is a
        # read-only property returning ``self._completions``).
        completions_prediction = _FakePrediction()
        completions_prediction.next_proposition = batch
        prediction._completions = completions_prediction  # type: ignore[attr-defined]
        return prediction


class _UnparseableProposer:
    """Raises ``AdapterParseError`` starting from its Nth call (1-indexed).

    When ``always`` is False (default), only the Nth call raises and the next
    call recovers (mirrors a one-off malformed response). When ``always`` is
    True, every call from the Nth onward raises (a model stuck on bare
    sentinels).
    """

    def __init__(
        self,
        unparseable_on_call: int = 1,
        recovery_proposition: str = "recovered prop",
        *,
        always: bool = False,
    ) -> None:
        self._unparseable_on_call = unparseable_on_call
        self._recovery = recovery_proposition
        self._always = always
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        should_raise = (
            self.call_count >= self._unparseable_on_call
            if self._always
            else self.call_count == self._unparseable_on_call
        )
        if should_raise:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=PropositionProposerSignature,  # pyright: ignore[reportArgumentType]
                lm_response="KHÔNG CÓ MỆNH ĐỀ MỚI",
                message=(
                    "Adapter ChatAdapter failed to parse the LM response.\n\n"
                    "LM Response: KHÔNG CÓ MỆNH ĐỀ MỚI\n\n"
                    "Expected to find output fields: [proposition]\n"
                    "Actual output fields parsed: []"
                ),
            )
        return _FakePrediction(next_proposition=self._recovery)


class _UnparseableMeaningful:
    """Raises ``AdapterParseError`` on its Nth call (1-indexed), recovering on
    subsequent calls -- mirrors the one-off malformed-response shape seen in the
    wild (e.g. the LM emitting the field label ``is_use`` instead of
    ``is_useful``). When ``always`` is True, every call from the Nth onward
    raises (a model stuck on the malformed label)."""

    def __init__(
        self,
        unparseable_on_call: int = 1,
        recovery_verdict: bool = True,
        *,
        always: bool = False,
    ) -> None:
        self._unparseable_on_call = unparseable_on_call
        self._recovery = recovery_verdict
        self._always = always
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        should_raise = (
            self.call_count >= self._unparseable_on_call
            if self._always
            else self.call_count == self._unparseable_on_call
        )
        if should_raise:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=PropositionMeaningfulnessSignature,  # pyright: ignore[reportArgumentType]
                lm_response="[[ ## is_mean ## ]]\nTrue",
                message=(
                    "Adapter ChatAdapter failed to parse the LM response.\n\n"
                    "LM Response: [[ ## is_mean ## ]]\nTrue\n\n"
                    "Expected to find output fields: [is_useful]\n"
                    "Actual output fields parsed: []"
                ),
            )
        return _FakePrediction(is_useful=self._recovery)


class _UnparseableValidity:
    """Raises ``AdapterParseError`` on its Nth call (1-indexed), recovering on
    subsequent calls. Same recovery semantics as ``_UnparseableMeaningful``."""

    def __init__(
        self,
        unparseable_on_call: int = 1,
        recovery_entailed: bool = True,
        recovery_contradicted: bool = False,
        *,
        always: bool = False,
    ) -> None:
        self._unparseable_on_call = unparseable_on_call
        self._recovery_entailed = recovery_entailed
        self._recovery_contradicted = recovery_contradicted
        self._always = always
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        should_raise = (
            self.call_count >= self._unparseable_on_call
            if self._always
            else self.call_count == self._unparseable_on_call
        )
        if should_raise:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=PropositionValiditySignature,  # pyright: ignore[reportArgumentType]
                lm_response="",
                message=(
                    "Adapter ChatAdapter failed to parse the LM response.\n\n"
                    "Expected to find output fields: [reasoning, is_contradicted, is_entailed]\n"
                    "Actual output fields parsed: [reasoning]"
                ),
            )
        return _FakePrediction(
            is_entailed=self._recovery_entailed,
            is_contradicted=self._recovery_contradicted,
        )


class _TestSignature(dspy.Signature):
    """Minimal signature for strategy construction."""

    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _make_strategy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_propositions: int,
    max_failed_attempts: int,
    verifier_mode: str,
    n_propose_samples: int = 1,
) -> CRStrategy:
    monkeypatch.setenv("VIREX_BENCH_CR_TARGET_PROPOSITIONS", str(target_propositions))
    monkeypatch.setenv("VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS", str(max_failed_attempts))
    monkeypatch.setenv("VIREX_BENCH_CR_VERIFIER_MODE", verifier_mode)
    # Default to 1 so existing tests get the legacy single-proposition path.
    monkeypatch.setenv("VIREX_BENCH_CR_N_PROPOSE_SAMPLES", str(n_propose_samples))
    strategy = get_strategy("cr", _TestSignature)
    assert isinstance(strategy, CRStrategy)
    return strategy


def _wire(
    strategy: CRStrategy,
    *,
    proposer: Any,
    meaningful: Any,
    validity: Any,
    aggregate: Any,
) -> None:
    """Replace the dspy.Predict modules with fakes."""
    strategy.propose = proposer  # type: ignore[assignment]
    strategy.verify_meaningful = meaningful  # type: ignore[assignment]
    strategy.verify_validity = validity  # type: ignore[assignment]
    strategy.aggregate = aggregate  # type: ignore[assignment]


_INPUTS: dict[str, str] = {"premises": "some premises", "question": "some question"}


def test_multi_mode_accumulates_then_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi mode: 2 propositions all pass both checks -> accumulated and solved."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=2,
        max_failed_attempts=6,
        verifier_mode="multi",
    )
    proposer = _FakeProposer(["prop A", "prop B"])
    meaningful = _FakeMeaningful([True, True])
    validity = _FakeValidity([(True, False), (True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert prediction.answer == "Đáp án"
    assert proposer.call_count == 2
    assert meaningful.call_count == 2  # multi: meaningfulness ran each time
    assert validity.call_count == 2
    assert aggregate.call_count == 1
    assert aggregate.last_context == (
        "[Mệnh đề được xác nhận 1] prop A\n[Mệnh đề được xác nhận 2] prop B"
    )
    assert prediction["reasoning"] == (
        "[Mệnh đề được xác nhận 1] prop A\n[Mệnh đề được xác nhận 2] prop B"
    )


def test_single_mode_skips_meaningfulness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single mode: only the validity check runs, never the meaningfulness one."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=6,
        verifier_mode="single",
    )
    proposer = _FakeProposer(["prop A"])
    meaningful = _FakeMeaningful([])  # must not be called
    validity = _FakeValidity([(True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert prediction.answer == "Đáp án"
    assert proposer.call_count == 1
    assert meaningful.call_count == 0
    assert validity.call_count == 1


def test_multi_mode_rejects_on_failed_meaningfulness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proposition that fails the meaningfulness pre-filter is rejected without
    a validity call -- and the loop keeps going until target is met or budget exhausted."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=2,
        verifier_mode="multi",
    )
    proposer = _FakeProposer(["filler prop", "real prop"])
    meaningful = _FakeMeaningful([False, True])  # first fails, second passes
    validity = _FakeValidity([(True, False)])  # only called for the meaningful one
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert meaningful.call_count == 2
    assert validity.call_count == 1  # skipped on the first proposal
    assert aggregate.last_context == "[Mệnh đề được xác nhận 1] real prop"
    assert prediction["reasoning"] == "[Mệnh đề được xác nhận 1] real prop"


def test_verdicts_bucketed_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every verdict (entailed, contradicted, undetermined) is accumulated into
    its bucket -- none counts as a failure. Only filler / duplicate / parse-error
    proposals increment ``failed``."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=3,
        max_failed_attempts=2,
        verifier_mode="single",
    )
    proposer = _FakeProposer(["entailed prop", "contradicted prop", "undetermined prop"])
    validity = _FakeValidity(
        [
            (True, False),  # entailed
            (False, True),  # contradicted
            (False, False),  # undetermined
        ]
    )
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    context = aggregate.last_context
    assert context is not None
    assert "[Mệnh đề được xác nhận 1] entailed prop" in context
    assert "[Mệnh đề bị bác bỏ 1] contradicted prop" in context
    assert "[Mệnh đề không xác định 1] undetermined prop" in context
    stats = prediction["search_stats"]
    assert stats["entailed_count"] == 1
    assert stats["contradicted_count"] == 1
    assert stats["undetermined_count"] == 1
    assert stats["nodes_visited"] == 3


def test_loop_bails_at_max_failed_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``max_failed_attempts`` filler proposals hit, the loop stops and
    solves with whatever it accumulated (here: nothing)."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=3,
        max_failed_attempts=2,
        verifier_mode="single",
    )
    # Both are filler -> short-circuit before verifier, counted as failed.
    proposer = _FakeProposer(["Không có mệnh đề mới", "Không có mệnh đề mới."])
    meaningful = _FakeMeaningful([])
    validity = _FakeValidity([])  # never called
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert proposer.call_count == 2
    assert validity.call_count == 0
    assert aggregate.call_count == 1
    assert aggregate.last_context == ""  # nothing accumulated
    assert prediction["reasoning"] == ""


def test_empty_proposition_short_circuits_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 'no new proposition' filler proposal is rejected without any verifier call."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=1,
        verifier_mode="multi",
    )
    proposer = _FakeProposer(["Không có mệnh đề mới."])
    meaningful = _FakeMeaningful([])  # must not be called
    validity = _FakeValidity([])  # must not be called
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert proposer.call_count == 1
    assert meaningful.call_count == 0
    assert validity.call_count == 0
    assert prediction["reasoning"] == ""


def test_duplicate_proposition_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A near-duplicate of an accepted proposition is rejected without a verifier call."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=2,
        verifier_mode="single",
    )
    # Second is a case-only variation of the first -> Jaccard = 1.0 >= threshold.
    proposer = _FakeProposer(["the cat is on the mat", "The cat is on the mat"])
    meaningful = _FakeMeaningful([])
    validity = _FakeValidity([(True, False)])  # only the first gets verified
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert validity.call_count == 1  # duplicate never reached the verifier
    assert aggregate.last_context == "[Mệnh đề được xác nhận 1] the cat is on the mat"
    assert prediction["reasoning"] == "[Mệnh đề được xác nhận 1] the cat is on the mat"


def test_context_overflow_breaks_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the LM raises ``ContextWindowExceededError`` mid-loop, the strategy
    solves with whatever it accumulated so far and stamps ``context_overflow``."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=5,
        max_failed_attempts=6,
        verifier_mode="single",
    )
    proposer = _OverflowProposer(overflow_on_call=3)
    meaningful = _FakeMeaningful([])
    validity = _FakeValidity([(True, False), (True, False)])  # first two accepted before overflow
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert proposer.call_count == 3  # the third call raised
    assert validity.call_count == 2  # two accepted before the overflow
    assert aggregate.call_count == 1
    assert prediction["search_stats"]["context_overflow"] is True
    assert "[Mệnh đề được xác nhận 1]" in prediction["reasoning"]
    assert "[Mệnh đề được xác nhận 2]" in prediction["reasoning"]


def test_search_stats_counts_are_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-case multi-mode run: every cost counter matches the expected budget."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=3,
        max_failed_attempts=6,
        verifier_mode="multi",
    )
    proposer = _FakeProposer(["p1", "p2", "p3"])
    meaningful = _FakeMeaningful([True, True, True])
    validity = _FakeValidity([(True, False), (True, False), (True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)
    stats = prediction["search_stats"]

    assert stats["algorithm"] == "cr"
    assert stats["verifier_mode"] == "multi"
    assert stats["propose_calls"] == 3
    assert stats["validity_calls"] == 3
    assert stats["meaningfulness_calls"] == 3
    assert stats["evaluate_calls"] == 6  # validity + meaningfulness
    assert stats["entailed_count"] == 3
    assert stats["contradicted_count"] == 0
    assert stats["undetermined_count"] == 0
    assert stats["nodes_visited"] == 3
    assert stats["depth_reached"] == 3
    assert stats["best_score"] == 1.0
    # 3 propose + 3 meaningful + 3 validity + 1 aggregate
    assert stats["total_llm_calls"] == 10
    assert stats["total_completions"] == 10
    assert stats["context_overflow"] is False


def test_missing_inputs_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``forward`` requires both ``premises`` and ``question``."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=1,
        verifier_mode="single",
    )
    with pytest.raises(ValueError, match="premises"):
        strategy.forward(premises="only premises")


class _ConfigCapturingProposer:
    """Records the ``config`` kwarg passed to each propose call."""

    def __init__(self, proposition: str = "captured prop") -> None:
        self._proposition = proposition
        self.captured_configs: list[Any] = []

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.captured_configs.append(kwargs.get("config"))
        return _FakePrediction(next_proposition=self._proposition)


def test_inherited_temperature_passes_empty_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When temperatures are unset (None), the per-call ``config`` is empty so
    dspy uses the LM's ``--model-kwargs`` profile verbatim -- the
    inherit-by-default behavior that respects model-card sampling systems."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=1,
        verifier_mode="single",
    )
    # Clearing envs leaves propose/verify temperatures at their None default.
    for name in ("VIREX_BENCH_CR_PROPOSE_TEMPERATURE", "VIREX_BENCH_CR_VERIFY_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    strategy.config = strategy.config.__class__(  # rebuild config with empty overrides
        target_propositions=strategy.config.target_propositions,
        max_failed_attempts=strategy.config.max_failed_attempts,
        verifier_mode=strategy.config.verifier_mode,
        propose_config={},
        verify_config={},
        n_propose_samples=strategy.config.n_propose_samples,
        dedupe_similarity_threshold=strategy.config.dedupe_similarity_threshold,
    )
    proposer = _ConfigCapturingProposer()
    validity = _FakeValidity([(True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),
        validity=validity,
        aggregate=aggregate,
    )

    strategy.forward(**_INPUTS)

    assert proposer.captured_configs == [{}]


def test_explicit_temperature_passes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a temperature is set, only that key is forced into the per-call
    config; other sampling params still inherit from --model-kwargs."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=1,
        verifier_mode="single",
    )
    strategy.config = strategy.config.__class__(
        target_propositions=strategy.config.target_propositions,
        max_failed_attempts=strategy.config.max_failed_attempts,
        verifier_mode=strategy.config.verifier_mode,
        propose_config={"temperature": 0.9},
        verify_config={},
        n_propose_samples=strategy.config.n_propose_samples,
        dedupe_similarity_threshold=strategy.config.dedupe_similarity_threshold,
    )
    proposer = _ConfigCapturingProposer()
    validity = _FakeValidity([(True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),
        validity=validity,
        aggregate=aggregate,
    )

    strategy.forward(**_INPUTS)

    assert proposer.captured_configs == [{"temperature": 0.9}]


def test_unparseable_propose_counts_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the proposer emits an unparseable response (e.g. a bare sentinel
    line with no ``proposition:`` field label, raising ``AdapterParseError``),
    the loop counts it as a failed attempt and retries -- it does NOT kill the
    example. The recovered proposition is then accepted and accumulated."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=3,
        verifier_mode="single",
    )
    proposer = _UnparseableProposer(unparseable_on_call=1, recovery_proposition="good prop")
    validity = _FakeValidity([(True, False)])  # only the recovered prop gets verified
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    # One failed parse (counted) + one successful propose (recovered).
    assert proposer.call_count == 2
    assert validity.call_count == 1
    assert aggregate.last_context == "[Mệnh đề được xác nhận 1] good prop"
    assert prediction["reasoning"] == "[Mệnh đề được xác nhận 1] good prop"


def test_unparseable_propose_respects_failure_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """If every propose call is unparseable, ``max_failed_attempts`` still caps
    the loop and the strategy solves with whatever it has (here: nothing)."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=2,
        max_failed_attempts=2,
        verifier_mode="multi",
    )
    # Raises on every call; recovery path never reached.
    proposer = _UnparseableProposer(unparseable_on_call=1, always=True)
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),  # never reached
        validity=_FakeValidity([]),  # never reached
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert proposer.call_count == 2  # exactly max_failed_attempts, then bail
    assert aggregate.last_context == ""
    assert prediction["reasoning"] == ""


def test_unparseable_meaningfulness_counts_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the meaningfulness verifier emits an unparseable response (e.g. the
    field label ``is_use`` instead of ``is_useful``, raising
    ``AdapterParseError``), the proposition is conservatively rejected and the
    loop retries -- it does NOT propagate and kill the whole example. The
    recovered proposition is then accepted and accumulated."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=3,
        verifier_mode="multi",
    )
    proposer = _FakeProposer(["prop A", "prop B"])
    meaningful = _UnparseableMeaningful(unparseable_on_call=1, recovery_verdict=True)
    validity = _FakeValidity([(True, False)])  # only the recovered prop reaches validity
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert meaningful.call_count == 2  # one failed parse + one recovered
    assert validity.call_count == 1  # skipped on the first (rejected) proposal
    assert aggregate.last_context == "[Mệnh đề được xác nhận 1] prop B"
    assert prediction["reasoning"] == "[Mệnh đề được xác nhận 1] prop B"


def test_unparseable_validity_counts_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the validity verifier emits an unparseable response, the proposition
    is rejected and the loop retries -- not propagated as a whole-example failure."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=3,
        verifier_mode="single",
    )
    proposer = _FakeProposer(["prop A", "prop B"])
    meaningful = _FakeMeaningful([])  # single mode: never called
    validity = _UnparseableValidity(
        unparseable_on_call=1, recovery_entailed=True, recovery_contradicted=False
    )
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)

    assert validity.call_count == 2  # one failed parse + one recovered
    assert aggregate.last_context == "[Mệnh đề được xác nhận 1] prop B"
    assert prediction["reasoning"] == "[Mệnh đề được xác nhận 1] prop B"


def test_unparseable_verifier_call_is_counted_in_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parse-failed verifier call still consumed an LLM request, so it must be
    counted in ``search_stats`` (meaningfulness_calls / total_llm_calls)."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=3,
        verifier_mode="multi",
    )
    proposer = _FakeProposer(["prop A", "prop B"])
    meaningful = _UnparseableMeaningful(unparseable_on_call=1, recovery_verdict=True)
    validity = _FakeValidity([(True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)
    stats = prediction["search_stats"]

    assert stats["meaningfulness_calls"] == 2  # one failed parse + one recovered
    assert stats["validity_calls"] == 1
    assert stats["propose_calls"] == 2
    # 2 propose + 2 meaningfulness + 1 validity + 1 aggregate
    assert stats["total_llm_calls"] == 6


def test_multi_sample_tries_candidates_until_one_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With n > 1, one propose call yields a batch. Candidates are tried in
    order; a meaningfulness failure doesn't waste the batch -- the next is tried."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=6,
        verifier_mode="multi",
        n_propose_samples=3,
    )
    proposer = _FakeMultiProposer([["prop A", "prop B", "prop C"]])
    meaningful = _FakeMeaningful([False, True])  # first fails, second passes
    validity = _FakeValidity([(True, False)])  # only the accepted one
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=meaningful,
        validity=validity,
        aggregate=aggregate,
    )

    strategy.forward(**_INPUTS)

    assert proposer.call_count == 1  # one batch yielded all candidates
    assert meaningful.call_count == 2  # first rejected, second passed
    assert validity.call_count == 1
    assert aggregate.last_context == "[Mệnh đề được xác nhận 1] prop B"


def test_multi_sample_all_filler_counts_as_one_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every candidate in a batch is filler, the batch counts as ONE failed
    attempt -- not N. The loop retries with a fresh batch."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=2,
        verifier_mode="single",
        n_propose_samples=3,
    )
    proposer = _FakeMultiProposer(
        [
            ["Không có mệnh đề mới", "Không có mệnh đề mới.", "nothing new"],
            ["real prop"],
        ]
    )
    validity = _FakeValidity([(True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),
        validity=validity,
        aggregate=aggregate,
    )

    strategy.forward(**_INPUTS)

    assert proposer.call_count == 2  # batch 1 all filler, batch 2 accepted
    assert validity.call_count == 1
    assert aggregate.last_context == "[Mệnh đề được xác nhận 1] real prop"


def test_multi_sample_dedupes_within_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Near-duplicate candidates within the same batch are collapsed before any
    verifier call."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=1,
        max_failed_attempts=6,
        verifier_mode="single",
        n_propose_samples=3,
    )
    # "the cat sat" and "The cat sat" are exact dups (Jaccard = 1.0).
    proposer = _FakeMultiProposer([["the cat sat", "The cat sat", "the dog ran"]])
    validity = _FakeValidity([(True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),
        validity=validity,
        aggregate=aggregate,
    )

    strategy.forward(**_INPUTS)

    assert validity.call_count == 1  # deduped to 2, first accepted
    assert "the cat sat" in (aggregate.last_context or "")


def test_multi_sample_stats_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    """total_completions accounts for n samples per propose call."""
    strategy = _make_strategy(
        monkeypatch,
        target_propositions=2,
        max_failed_attempts=6,
        verifier_mode="single",
        n_propose_samples=4,
    )
    proposer = _FakeMultiProposer(
        [
            ["prop A", "prop B", "prop C", "prop D"],
            ["prop E", "prop F", "prop G", "prop H"],
        ]
    )
    validity = _FakeValidity([(True, False), (True, False)])
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        proposer=proposer,
        meaningful=_FakeMeaningful([]),
        validity=validity,
        aggregate=aggregate,
    )

    prediction = strategy.forward(**_INPUTS)
    stats = prediction["search_stats"]

    assert stats["propose_calls"] == 2
    # 2 batches × 4 completions = 8 from proposing; 2 validity + 1 aggregate = 11
    assert stats["total_completions"] == 11
    # LM requests: 2 propose + 2 validity + 1 aggregate = 5
    assert stats["total_llm_calls"] == 5
