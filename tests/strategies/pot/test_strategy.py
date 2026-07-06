"""Unit tests for ``PoTZ3Strategy.forward`` with fake generate/regenerate/aggregate
modules and a fake interpreter.

Drives the PoT generate -> execute -> regenerate loop with canned programs and
canned interpreter outputs to verify: happy-path success, regeneration on
execution failure, regeneration on parse failure, graceful fallback when all
attempts fail, hard failure when ``fallback_on_error`` is False, the no-retry
invariant (LLM parse errors propagate, not retried), and ``pot_info``
correctness (the per-example telemetry dict that feeds the report-level
``pot_stats`` aggregate) -- without any real LM calls or subprocess spawns.

Pattern mirrors ``tests/strategies/cr/test_strategy.py``: each fake module
returns canned outputs in sequence and tracks its call count.
"""

from typing import Any

import dspy
import pytest
from dspy.adapters.base import AdapterParseError

from virex_bench.strategies.pot.common import ProgramGenerationSignature
from virex_bench.strategies.pot.strategy import PoTZ3Strategy


class _FakePrediction(dspy.Prediction):
    """A dspy.Prediction whose fields are set from arbitrary kwargs."""

    def __init__(self, **fields: Any) -> None:
        super().__init__()
        for key, value in fields.items():
            setattr(self, key, value)


class _FakeGenerate:
    """Returns canned ``generated_code`` strings in sequence, one per call."""

    def __init__(self, codes: list[str]) -> None:
        self._codes = list(codes)
        self._index = 0
        self.call_count = 0
        self.captured_configs: list[Any] = []

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        if self._index >= len(self._codes):
            raise AssertionError(
                f"FakeGenerate exhausted: only {len(self._codes)} canned outputs were provided"
            )
        self.captured_configs.append(kwargs.get("config"))
        code = self._codes[self._index]
        self._index += 1
        self.call_count += 1
        return _FakePrediction(generated_code=code)


class _FakeRegenerate:
    """Returns canned ``generated_code`` strings in sequence, tracking
    ``previous_code`` and ``error`` kwargs."""

    def __init__(self, codes: list[str]) -> None:
        self._codes = list(codes)
        self._index = 0
        self.call_count = 0
        self.captured_configs: list[Any] = []
        self.last_previous_code: str | None = None
        self.last_error: str | None = None

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        if self._index >= len(self._codes):
            raise AssertionError(
                f"FakeRegenerate exhausted: only {len(self._codes)} canned outputs were provided"
            )
        self.captured_configs.append(kwargs.get("config"))
        self.last_previous_code = kwargs.get("previous_code")
        self.last_error = kwargs.get("error")
        code = self._codes[self._index]
        self._index += 1
        self.call_count += 1
        return _FakePrediction(generated_code=code)


class _FakeAggregate:
    """Records ``solver_result`` and returns a stub prediction with task fields."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_solver_result: str | None = None

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        self.last_solver_result = kwargs.get("solver_result")
        return _FakePrediction(
            answer="Có",
            answer_type="yes_no_uncertain",
            supporting_premise_indices=[1],
            relevant_premises=["Lan là sinh viên"],
        )


class _FakeInterpreter:
    """Returns canned stdout strings or raises canned exceptions in sequence."""

    def __init__(self, outputs: list[str | Exception]) -> None:
        self._outputs = list(outputs)
        self._index = 0
        self.call_count = 0

    def execute(self, code: str) -> str:
        if self._index >= len(self._outputs):
            raise AssertionError(
                f"FakeInterpreter exhausted: only {len(self._outputs)} canned outputs were provided"
            )
        item = self._outputs[self._index]
        self._index += 1
        self.call_count += 1
        if isinstance(item, Exception):
            raise item
        return item


class _TestSignature(dspy.Signature):
    """Minimal signature for strategy construction."""

    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


_VALID_CODE = (
    "domain = create_domain('Person')\n"
    "lan = create_constant('lan', domain)\n"
    "IsStudent = create_predicate('IsStudent', domain)\n"
    "premises_list = [IsStudent(lan)]\n"
    "status = check_entailment(premises_list, IsStudent(lan))\n"
    "result = {'answer': 'Yes', 'z3_status': status, "
    "'supporting_premises': [1], 'solution': 'Lan is a student.'}\n"
    "print(result)"
)
_VALID_OUTPUT = (
    "{'answer': 'Yes', 'z3_status': 'entailed', "
    "'supporting_premises': [1], 'solution': 'Lan is a student.'}"
)
# Multiple "=" on a single line -> parse_code rejects as garbled.
_GARBLED_CODE = "answer = result = broken"
# Valid second-attempt code (for regeneration success tests).
_VALID_CODE_2 = _VALID_CODE.replace("'Yes'", "'No'")


_INPUTS: dict[str, str] = {"premises": "some premises", "question": "some question"}


def _make_strategy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_iters: int = 3,
    fallback_on_error: bool = True,
) -> PoTZ3Strategy:
    monkeypatch.setenv("VIREX_BENCH_POT_MAX_ITERS", str(max_iters))
    monkeypatch.setenv("VIREX_BENCH_POT_FALLBACK_ON_ERROR", str(fallback_on_error))
    monkeypatch.delenv("VIREX_BENCH_POT_GENERATE_TEMPERATURE", raising=False)
    monkeypatch.delenv("VIREX_BENCH_POT_REGENERATE_TEMPERATURE", raising=False)
    return PoTZ3Strategy(_TestSignature)


def _wire(
    strategy: PoTZ3Strategy,
    *,
    generate: Any,
    regenerate: Any,
    aggregate: Any,
    interpreter: Any,
) -> None:
    """Replace the dspy modules and interpreter with fakes."""
    strategy.generate = generate  # type: ignore[assignment]
    strategy.regenerate = regenerate  # type: ignore[assignment]
    strategy.aggregate = aggregate  # type: ignore[assignment]
    strategy.interpreter = interpreter  # type: ignore[assignment]


def test_happy_path_generate_execute_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best case: generate -> execute succeeds -> commit. 2 LLM calls total."""
    strategy = _make_strategy(monkeypatch)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([])
    aggregate = _FakeAggregate()
    interpreter = _FakeInterpreter([_VALID_OUTPUT])
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=aggregate,
        interpreter=interpreter,
    )

    prediction = strategy.forward(**_INPUTS)

    assert prediction.answer == "Có"
    assert generate.call_count == 1
    assert regenerate.call_count == 0
    assert aggregate.call_count == 1
    assert interpreter.call_count == 1
    solver_result = aggregate.last_solver_result
    assert solver_result is not None
    assert "Z3 Program:" in solver_result
    assert "Program Output:" in solver_result
    assert "failed" not in solver_result


def test_happy_path_pot_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """pot_info on the happy path: 1 generate, 0 regen, 1 execute, success.

    PoT is a single-pass generate-execute-regenerate pipeline, not a tree
    search, so it emits ``pot_info`` (per-example pipeline telemetry) and does
    NOT emit ``search_stats`` (a ToT-style tree-search construct).
    """
    strategy = _make_strategy(monkeypatch)
    _wire(
        strategy,
        generate=_FakeGenerate([_VALID_CODE]),
        regenerate=_FakeRegenerate([]),
        aggregate=_FakeAggregate(),
        interpreter=_FakeInterpreter([_VALID_OUTPUT]),
    )

    prediction = strategy.forward(**_INPUTS)

    # PoT does not emit search_stats -- it is a pipeline, not a tree search.
    assert "search_stats" not in prediction
    pot_info = prediction["pot_info"]

    assert pot_info["generated_code"] == _VALID_CODE
    assert pot_info["execution_output"] == _VALID_OUTPUT
    assert pot_info["execution_error"] is None
    assert pot_info["execution_success"] is True
    assert pot_info["generate_calls"] == 1
    assert pot_info["regenerate_calls"] == 0
    assert pot_info["execute_calls"] == 1
    assert pot_info["total_llm_calls"] == 2  # generate + commit


def test_regenerate_succeeds_after_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """First program crashes at execute; regenerate fixes it -> success."""
    strategy = _make_strategy(monkeypatch, max_iters=3)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([_VALID_CODE_2])
    aggregate = _FakeAggregate()
    interpreter = _FakeInterpreter(
        [RuntimeError("Error during execution:\nNameError: x"), _VALID_OUTPUT]
    )
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=aggregate,
        interpreter=interpreter,
    )

    prediction = strategy.forward(**_INPUTS)

    assert generate.call_count == 1
    assert regenerate.call_count == 1
    assert interpreter.call_count == 2
    assert aggregate.call_count == 1
    # Regenerate was called with the failed code + the error string.
    assert regenerate.last_previous_code == _VALID_CODE
    assert regenerate.last_error is not None
    assert "RuntimeError" in regenerate.last_error
    assert "NameError" in regenerate.last_error
    # solver_result reflects the SECOND (successful) program.
    solver_result = aggregate.last_solver_result
    assert solver_result is not None
    assert "Program Output:" in solver_result

    # pot_info reflects the final (successful) regenerated program.
    pot_info = prediction["pot_info"]
    assert pot_info["generated_code"] == _VALID_CODE_2
    assert pot_info["execution_output"] == _VALID_OUTPUT
    assert pot_info["execution_error"] is None
    assert pot_info["execution_success"] is True
    assert pot_info["regenerate_calls"] == 1
    assert pot_info["execute_calls"] == 2
    assert pot_info["total_llm_calls"] == 3  # generate + regen + commit


def test_regenerate_succeeds_after_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """First generated code is garbled (parse_code rejects); regenerate fixes it."""
    strategy = _make_strategy(monkeypatch, max_iters=3)
    generate = _FakeGenerate([_GARBLED_CODE])
    regenerate = _FakeRegenerate([_VALID_CODE])
    aggregate = _FakeAggregate()
    # Interpreter never called on the garbled code (parse failed first).
    interpreter = _FakeInterpreter([_VALID_OUTPUT])
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=aggregate,
        interpreter=interpreter,
    )

    prediction = strategy.forward(**_INPUTS)

    assert generate.call_count == 1
    assert regenerate.call_count == 1
    assert interpreter.call_count == 1  # only the regenerated code was executed
    pot_info = prediction["pot_info"]
    assert pot_info["regenerate_calls"] == 1
    assert pot_info["execute_calls"] == 1
    assert pot_info["execution_success"] is True
    # Regenerate received the garbled code as previous_code + a parse error.
    assert regenerate.last_previous_code == _GARBLED_CODE
    assert regenerate.last_error is not None
    assert "Code format is not correct" in regenerate.last_error


def test_regenerate_returns_garbled_code_loops_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regenerate output that itself fails parse_code keeps the loop going:
    the parse error becomes the next regenerate prompt, execute is skipped for
    the garbled attempt, and a second regenerate succeeds."""
    strategy = _make_strategy(monkeypatch, max_iters=3)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([_GARBLED_CODE, _VALID_CODE_2])
    aggregate = _FakeAggregate()
    # First execute crashes on the generated code; the garbled regen is never
    # executed (parse fails first); the second regen executes successfully.
    interpreter = _FakeInterpreter([RuntimeError("Error during execution:\nboom"), _VALID_OUTPUT])
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=aggregate,
        interpreter=interpreter,
    )

    prediction = strategy.forward(**_INPUTS)

    assert generate.call_count == 1
    assert regenerate.call_count == 2
    assert interpreter.call_count == 2  # generated code + second regen only
    # The second regenerate was prompted by the garbled first regen + its parse error.
    assert regenerate.last_previous_code == _GARBLED_CODE
    assert regenerate.last_error is not None
    assert "Code format is not correct" in regenerate.last_error
    pot_info = prediction["pot_info"]
    assert pot_info["regenerate_calls"] == 2
    assert pot_info["execute_calls"] == 2
    assert pot_info["execution_success"] is True
    assert pot_info["generated_code"] == _VALID_CODE_2


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("Execution timed out after 2.0 seconds."),
        SyntaxError("simulated invalid syntax"),
    ],
)
def test_regenerate_succeeds_after_non_runtime_execution_error(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """The regenerate loop catches SyntaxError and TimeoutError (not just
    RuntimeError) from execute and feeds the typed error back to the LM."""
    strategy = _make_strategy(monkeypatch, max_iters=3)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([_VALID_CODE_2])
    aggregate = _FakeAggregate()
    interpreter = _FakeInterpreter([exc, _VALID_OUTPUT])
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=aggregate,
        interpreter=interpreter,
    )

    prediction = strategy.forward(**_INPUTS)

    assert generate.call_count == 1
    assert regenerate.call_count == 1
    assert interpreter.call_count == 2
    assert regenerate.last_previous_code == _VALID_CODE
    assert regenerate.last_error is not None
    assert type(exc).__name__ in regenerate.last_error
    pot_info = prediction["pot_info"]
    assert pot_info["execution_success"] is True
    assert pot_info["regenerate_calls"] == 1


def test_fallback_on_error_true_commits_with_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All max_iters attempts fail; fallback_on_error=True -> commit still runs."""
    strategy = _make_strategy(monkeypatch, max_iters=2, fallback_on_error=True)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([_VALID_CODE, _VALID_CODE])
    aggregate = _FakeAggregate()
    interpreter = _FakeInterpreter(
        [
            RuntimeError("Error during execution:\nfail 1"),
            RuntimeError("Error during execution:\nfail 2"),
            RuntimeError("Error during execution:\nfail 3"),
        ]
    )
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=aggregate,
        interpreter=interpreter,
    )

    prediction = strategy.forward(**_INPUTS)

    assert regenerate.call_count == 2  # exactly max_iters
    assert interpreter.call_count == 3  # initial + 2 regens
    assert aggregate.call_count == 1  # commit ran despite failures
    solver_result = aggregate.last_solver_result
    assert solver_result is not None
    assert "Z3 Program (failed):" in solver_result
    assert "Error:" in solver_result
    assert "Reason over the premises directly." in solver_result

    # pot_info captures the failed program, the error, and no output.
    pot_info = prediction["pot_info"]
    assert pot_info["execution_success"] is False
    assert pot_info["execution_output"] is None
    assert pot_info["execution_error"] is not None
    assert "RuntimeError" in pot_info["execution_error"]
    assert pot_info["regenerate_calls"] == 2
    assert pot_info["execute_calls"] == 3
    assert pot_info["total_llm_calls"] == 4  # generate + 2 regen + commit
    assert pot_info["generated_code"] == _VALID_CODE  # last attempted code


def test_fallback_on_error_false_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """All attempts fail; fallback_on_error=False -> forward raises RuntimeError."""
    strategy = _make_strategy(monkeypatch, max_iters=1, fallback_on_error=False)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([_VALID_CODE])
    aggregate = _FakeAggregate()
    interpreter = _FakeInterpreter([RuntimeError("crash"), RuntimeError("crash")])
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=aggregate,
        interpreter=interpreter,
    )

    with pytest.raises(RuntimeError, match="PoT-Z3 solver failed"):
        strategy.forward(**_INPUTS)

    assert aggregate.call_count == 0  # commit never reached


def test_generate_adapter_parse_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """When generate raises AdapterParseError, forward propagates (no retry)."""

    class _ExplodingGenerate:
        call_count = 0

        def __call__(self, **kwargs: Any) -> Any:
            self.call_count += 1
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=ProgramGenerationSignature,  # pyright: ignore[reportArgumentType]
                lm_response="garbage",
                message="parse failed",
            )

    strategy = _make_strategy(monkeypatch)
    exploding = _ExplodingGenerate()
    _wire(
        strategy,
        generate=exploding,
        regenerate=_FakeRegenerate([]),
        aggregate=_FakeAggregate(),
        interpreter=_FakeInterpreter([]),
    )

    with pytest.raises(AdapterParseError):
        strategy.forward(**_INPUTS)

    assert exploding.call_count == 1  # not retried


def test_regenerate_adapter_parse_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """When regenerate raises AdapterParseError mid-loop, forward propagates."""

    class _ExplodingRegenerate:
        call_count = 0

        def __call__(self, **kwargs: Any) -> Any:
            self.call_count += 1
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=ProgramGenerationSignature,  # pyright: ignore[reportArgumentType]
                lm_response="garbage",
                message="parse failed",
            )

    strategy = _make_strategy(monkeypatch, max_iters=3)
    exploding_regen = _ExplodingRegenerate()
    _wire(
        strategy,
        generate=_FakeGenerate([_VALID_CODE]),
        regenerate=exploding_regen,
        aggregate=_FakeAggregate(),
        interpreter=_FakeInterpreter([RuntimeError("initial crash")]),
    )

    with pytest.raises(AdapterParseError):
        strategy.forward(**_INPUTS)

    assert exploding_regen.call_count == 1  # not retried
    # Aggregate was never reached.
    assert strategy.aggregate.call_count == 0  # type: ignore[attr-defined]


def test_generate_context_window_exceeded_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ContextWindowExceededError from generate propagates (no graceful catch)."""

    class _OverflowGenerate:
        call_count = 0

        def __call__(self, **kwargs: Any) -> Any:
            self.call_count += 1
            raise dspy.ContextWindowExceededError(message="simulated overflow")

    strategy = _make_strategy(monkeypatch)
    overflowing = _OverflowGenerate()
    _wire(
        strategy,
        generate=overflowing,
        regenerate=_FakeRegenerate([]),
        aggregate=_FakeAggregate(),
        interpreter=_FakeInterpreter([]),
    )

    with pytest.raises(dspy.ContextWindowExceededError):
        strategy.forward(**_INPUTS)


def test_missing_inputs_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """forward requires both premises and question."""
    strategy = _make_strategy(monkeypatch)
    _wire(
        strategy,
        generate=_FakeGenerate([]),
        regenerate=_FakeRegenerate([]),
        aggregate=_FakeAggregate(),
        interpreter=_FakeInterpreter([]),
    )

    with pytest.raises(ValueError, match="premises"):
        strategy.forward(question="only question")


def test_inherited_temperature_passes_empty_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When temperatures are unset (None), per-call config is empty so dspy
    inherits the LM's --model-kwargs profile verbatim."""
    strategy = _make_strategy(monkeypatch)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([])
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=_FakeAggregate(),
        interpreter=_FakeInterpreter([_VALID_OUTPUT]),
    )

    strategy.forward(**_INPUTS)

    assert generate.captured_configs == [{}]


def test_explicit_temperature_passes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a temperature is set, it is forced into the per-call config."""
    monkeypatch.setenv("VIREX_BENCH_POT_MAX_ITERS", "3")
    monkeypatch.setenv("VIREX_BENCH_POT_FALLBACK_ON_ERROR", "true")
    monkeypatch.setenv("VIREX_BENCH_POT_GENERATE_TEMPERATURE", "0.7")
    monkeypatch.setenv("VIREX_BENCH_POT_REGENERATE_TEMPERATURE", "0.3")
    strategy = PoTZ3Strategy(_TestSignature)
    generate = _FakeGenerate([_VALID_CODE])
    regenerate = _FakeRegenerate([])
    _wire(
        strategy,
        generate=generate,
        regenerate=regenerate,
        aggregate=_FakeAggregate(),
        interpreter=_FakeInterpreter([_VALID_OUTPUT]),
    )

    strategy.forward(**_INPUTS)

    assert generate.captured_configs == [{"temperature": 0.7}]


def test_report_config_surfaces_resolved_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """report_config includes all POTConfig knobs for the run report."""
    monkeypatch.setenv("VIREX_BENCH_POT_MAX_ITERS", "5")
    monkeypatch.setenv("VIREX_BENCH_POT_FALLBACK_ON_ERROR", "false")
    monkeypatch.delenv("VIREX_BENCH_POT_GENERATE_TEMPERATURE", raising=False)
    monkeypatch.delenv("VIREX_BENCH_POT_REGENERATE_TEMPERATURE", raising=False)
    strategy = PoTZ3Strategy(_TestSignature)

    report = strategy.report_config
    assert report["max_iters"] == 5
    assert report["fallback_on_error"] is False
    assert report["execution_timeout"] == 15.0
    assert report["generate_config"] == {}
    assert report["regenerate_config"] == {}


def test_reasoning_attached_to_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """prediction['reasoning'] is the solver_result string (program + output)."""
    strategy = _make_strategy(monkeypatch)
    aggregate = _FakeAggregate()
    _wire(
        strategy,
        generate=_FakeGenerate([_VALID_CODE]),
        regenerate=_FakeRegenerate([]),
        aggregate=aggregate,
        interpreter=_FakeInterpreter([_VALID_OUTPUT]),
    )

    prediction = strategy.forward(**_INPUTS)

    assert prediction["reasoning"] == aggregate.last_solver_result
