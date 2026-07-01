from typing import Any

import pytest

from virex_bench.evaluation import judge as judge_module


def test_build_judge_lm_uses_generic_judge_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _FakeBaseLM:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.delenv("VIREX_BENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("VIREX_BENCH_JUDGE_API_KEY", "test-judge-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(judge_module, "BaseLM", _FakeBaseLM)

    judge_module.build_judge_lm()

    assert captured_kwargs["model"] == "deepseek/deepseek-v4-flash"
    assert captured_kwargs["api_key"] == "test-judge-key"


def test_build_judge_lm_requires_generic_judge_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIREX_BENCH_JUDGE_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("VIREX_BENCH_JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "legacy-key")

    with pytest.raises(RuntimeError, match="VIREX_BENCH_JUDGE_API_KEY"):
        judge_module.build_judge_lm()
