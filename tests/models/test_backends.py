"""Tests for ``load_backend``: model-string assembly and env-var fallback."""

from __future__ import annotations

import pytest

from virex_bench.models.backends import load_backend


def test_load_backend_assembles_provider_prefixed_model_string() -> None:
    lm = load_backend("openai", "Qwen/Qwen3", api_base="http://x", api_key="k")
    assert lm.model == "openai/Qwen/Qwen3"
    assert lm.kwargs["api_base"] == "http://x"
    assert lm.kwargs["api_key"] == "k"
    assert lm.cache is False


def test_load_backend_falls_back_to_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://from-env")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    lm = load_backend("openai", "m")
    assert lm.kwargs["api_base"] == "http://from-env"
    assert lm.kwargs["api_key"] == "env-key"


def test_load_backend_forwards_cache_flag() -> None:
    lm = load_backend("openai", "m", api_base="http://x", api_key="k", cache=True)
    assert lm.cache is True
