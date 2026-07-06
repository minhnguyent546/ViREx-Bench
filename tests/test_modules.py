"""Tests for ``ThinkingCaptureLM`` and ``DualTask2ChainOfThought``.

The capture LM wraps a base LM and records the last response / native reasoning
emitted across its outputs. The dual-stream CoT module wires that capture into a
``ChainOfThought`` so prompted reasoning and native thinking stay distinct. These
tests drive both with stub base LMs returning canned output shapes (dict with
``text``/``content``/``reasoning_content``, plain strings) -- no real LM calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dspy

from virex_bench.strategies.modules import DualTask2ChainOfThought, ThinkingCaptureLM


class _StubLM:
    """Base LM stub returning canned output items from ``__call__`` / ``acall``."""

    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = list(outputs)
        self.model = "stub-model"
        self.temperature: float | None = None

    def __call__(
        self,
        prompt: str | None = None,
        messages: Sequence[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        return self._outputs

    async def acall(
        self,
        prompt: str | None = None,
        messages: Sequence[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        return self._outputs


class _TestSignature(dspy.Signature):
    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _capture(outputs: list[Any]) -> ThinkingCaptureLM:
    return ThinkingCaptureLM(_StubLM(outputs))  # pyright: ignore[reportArgumentType]


def test_thinking_capture_records_text_and_reasoning_content() -> None:
    messages = [{"role": "user", "content": "hi"}]
    capture = _capture([{"text": "resp", "reasoning_content": "thoughts"}])
    capture(messages=messages)
    assert capture.last_response == "resp"
    assert capture.last_thinking == "thoughts"
    assert capture.last_messages == messages


def test_thinking_capture_records_content_when_no_text_field() -> None:
    capture = _capture([{"content": "alt"}])
    capture(messages=[{"role": "user", "content": "x"}])
    assert capture.last_response == "alt"
    assert capture.last_thinking == ""


def test_thinking_capture_records_plain_string_output() -> None:
    capture = _capture(["raw string"])
    capture(messages=[{"role": "user", "content": "x"}])
    assert capture.last_response == "raw string"
    assert capture.last_thinking == ""


def test_thinking_capture_last_response_is_final_text_without_reasoning() -> None:
    capture = _capture([{"text": "first"}, {"text": "second"}])
    capture(messages=[{"role": "user", "content": "x"}])
    assert capture.last_response == "second"


def test_thinking_capture_getattr_delegates_to_base() -> None:
    capture = _capture([])
    assert capture.model == "stub-model"


def test_thinking_capture_setattr_delegates_non_special_to_base() -> None:
    base = _StubLM([])
    capture = ThinkingCaptureLM(base)  # pyright: ignore[reportArgumentType]
    capture.temperature = 0.7
    # The attribute is set on the wrapped base LM, not the capture wrapper.
    assert base.temperature == 0.7


async def test_thinking_capture_acall_records_async_output() -> None:
    capture = _capture([{"text": "async-resp", "reasoning_content": "async-think"}])
    await capture.acall(messages=[{"role": "user", "content": "x"}])
    assert capture.last_response == "async-resp"
    assert capture.last_thinking == "async-think"


class _CaptureCallingPredict:
    """Fake predict that invokes the passed ``lm`` so the capture wrapper records it."""

    lm = None

    def __init__(self, base_outputs: list[Any]) -> None:
        self._base_outputs = base_outputs

    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        lm = kwargs.get("lm")
        if lm is not None:
            lm(messages=[{"role": "user", "content": "prompt"}])
        return dspy.Prediction(answer="A", reasoning="prompted cot")


def test_dual_cot_forward_attaches_native_thinking_and_messages() -> None:
    base = _StubLM([{"text": "resp", "reasoning_content": "native thoughts"}])
    module = DualTask2ChainOfThought(_TestSignature)
    module.predict = _CaptureCallingPredict(base._outputs)  # type: ignore[assignment]
    result = module.forward(lm=base, premises="p", question="q")

    assert result["answer"] == "A"
    assert result["thinking_content"] == "native thoughts"
    assert result["_lm_response"] == "resp"
    assert result["_lm_messages"] == [{"role": "user", "content": "prompt"}]


def test_dual_cot_forward_without_lm_skips_capture() -> None:
    class _SimplePredict:
        lm = None

        def __call__(self, **_kwargs: Any) -> dspy.Prediction:
            return dspy.Prediction(answer="A", reasoning="r")

    module = DualTask2ChainOfThought(_TestSignature)
    module.predict = _SimplePredict()  # type: ignore[assignment]
    with dspy.context(lm=None):
        result = module.forward(premises="p", question="q")

    assert result["answer"] == "A"
    assert "thinking_content" not in result


class _AsyncCaptureCallingPredict:
    lm = None

    def __init__(self, base_outputs: list[Any]) -> None:
        self._base_outputs = base_outputs

    async def acall(self, **kwargs: Any) -> dspy.Prediction:
        lm = kwargs.get("lm")
        if lm is not None:
            await lm.acall(messages=[{"role": "user", "content": "prompt"}])
        return dspy.Prediction(answer="A", reasoning="r")


async def test_dual_cot_aforward_attaches_native_thinking() -> None:
    base = _StubLM([{"text": "resp", "reasoning_content": "native thoughts"}])
    module = DualTask2ChainOfThought(_TestSignature)
    module.predict = _AsyncCaptureCallingPredict(base._outputs)  # type: ignore[assignment]
    result = await module.aforward(lm=base, premises="p", question="q")

    assert result["answer"] == "A"
    assert result["thinking_content"] == "native thoughts"
