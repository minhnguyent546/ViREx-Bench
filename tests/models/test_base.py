"""Tests for ``BaseLM`` retry layer: transient errors are retried, permanent ones are not."""

from __future__ import annotations

from typing import Any

import dspy
import litellm.exceptions
import pytest

from virex_bench.models.base import BaseLM


def _make_lm() -> BaseLM:
    return BaseLM(model="openai/test-model", api_key="k")


def _fast_retry_env(monkeypatch: pytest.MonkeyPatch, *, max_retries: int = 3) -> None:
    monkeypatch.setenv("VIREX_BENCH_LM_MAX_RETRIES", str(max_retries))
    monkeypatch.setenv("VIREX_BENCH_LM_RETRY_MIN_WAIT", "0")
    monkeypatch.setenv("VIREX_BENCH_LM_RETRY_MAX_WAIT", "0")
    monkeypatch.setenv("VIREX_BENCH_LM_RETRY_JITTER", "0")


def _fake_forward_factory(fail_times: int, exc: BaseException, return_value: str):
    state = {"calls": 0}

    def fake_forward(self: Any, prompt: Any = None, messages: Any = None, **kwargs: Any) -> str:
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc
        return return_value

    return fake_forward, state


def _fake_aforward_factory(fail_times: int, exc: BaseException, return_value: str):
    state = {"calls": 0}

    async def fake_aforward(
        self: Any, prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> str:
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc
        return return_value

    return fake_aforward, state


def test_forward_retries_transient_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _fast_retry_env(monkeypatch, max_retries=3)
    fake_forward, state = _fake_forward_factory(
        2, litellm.exceptions.Timeout("t", "test-model", "openai"), "ok"
    )
    monkeypatch.setattr(dspy.LM, "forward", fake_forward)

    result = _make_lm().forward(messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert state["calls"] == 3


def test_forward_reraises_permanent_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fast_retry_env(monkeypatch, max_retries=3)
    fake_forward, state = _fake_forward_factory(
        10, litellm.exceptions.AuthenticationError("bad key", "openai", "test-model"), "ok"
    )
    monkeypatch.setattr(dspy.LM, "forward", fake_forward)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        _make_lm().forward(messages=[{"role": "user", "content": "hi"}])

    # Not in the retryable set -> a single attempt, no retries.
    assert state["calls"] == 1


async def test_aforward_retries_transient_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fast_retry_env(monkeypatch, max_retries=3)
    fake_aforward, state = _fake_aforward_factory(
        2, litellm.exceptions.Timeout("t", "test-model", "openai"), "ok"
    )
    monkeypatch.setattr(dspy.LM, "aforward", fake_aforward)

    result = await _make_lm().aforward(messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert state["calls"] == 3


async def test_aforward_reraises_permanent_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fast_retry_env(monkeypatch, max_retries=3)
    fake_aforward, state = _fake_aforward_factory(
        10, litellm.exceptions.AuthenticationError("bad key", "openai", "test-model"), "ok"
    )
    monkeypatch.setattr(dspy.LM, "aforward", fake_aforward)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        await _make_lm().aforward(messages=[{"role": "user", "content": "hi"}])

    assert state["calls"] == 1
