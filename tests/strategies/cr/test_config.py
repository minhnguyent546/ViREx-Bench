"""Tests for env-var resolution in ``_build_cr_config``.

Mirrors ``tests/strategies/tot/test_config.py``: clears the CR env vars,
then asserts defaults and overrides resolve into ``CRStrategy.config``.
"""

import dspy
import pytest

from virex_bench.strategies.cr.common import CRConfig
from virex_bench.strategies.cr.strategy import CRStrategy
from virex_bench.strategies.registry import get_strategy

_CR_ENVS = [
    "VIREX_BENCH_CR_TARGET_PROPOSITIONS",
    "VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS",
    "VIREX_BENCH_CR_VERIFIER_MODE",
    "VIREX_BENCH_CR_PROPOSE_TEMPERATURE",
    "VIREX_BENCH_CR_VERIFY_TEMPERATURE",
    "VIREX_BENCH_CR_N_PROPOSE_SAMPLES",
    "VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD",
]


class _TestSignature(dspy.Signature):
    """Minimal signature for strategy construction in tests."""

    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _clear_cr_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CR_ENVS:
        monkeypatch.delenv(name, raising=False)


def _get_cr_config(monkeypatch: pytest.MonkeyPatch) -> CRConfig:
    """Build a CRStrategy from the current env. Callers are responsible for
    clearing/setting CR env vars beforehand (mirrors the ToT test pattern)."""
    strategy = get_strategy("cr", _TestSignature)
    assert isinstance(strategy, CRStrategy)
    return strategy.config


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    config = _get_cr_config(monkeypatch)
    assert config.target_propositions == 7
    assert config.max_failed_attempts == 6
    assert config.verifier_mode == "multi"
    assert config.n_propose_samples == 3
    # Propose temperature defaults to None -> empty config (inherit LM profile).
    assert config.propose_config == {}
    # Verify temperature defaults to None -> empty config (inherit LM profile).
    assert config.verify_config == {}
    assert config.dedupe_similarity_threshold == 0.9


def test_target_propositions_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_TARGET_PROPOSITIONS", "5")
    config = _get_cr_config(monkeypatch)
    assert config.target_propositions == 5


def test_max_failed_attempts_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS", "8")
    config = _get_cr_config(monkeypatch)
    assert config.max_failed_attempts == 8


def test_verifier_mode_single(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_VERIFIER_MODE", "single")
    config = _get_cr_config(monkeypatch)
    assert config.verifier_mode == "single"


def test_verifier_mode_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_VERIFIER_MODE", "MULTI")
    config = _get_cr_config(monkeypatch)
    assert config.verifier_mode == "multi"


def test_verifier_mode_invalid_choice_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_VERIFIER_MODE", "tri")
    with pytest.raises(ValueError, match="Invalid value"):
        get_strategy("cr", _TestSignature)


def test_temperatures_and_dedupe_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_PROPOSE_TEMPERATURE", "0.4")
    monkeypatch.setenv("VIREX_BENCH_CR_VERIFY_TEMPERATURE", "0.2")
    monkeypatch.setenv("VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD", "0.8")
    config = _get_cr_config(monkeypatch)
    # Temperature env vars resolve into per-call override dicts (empty = inherit).
    assert config.propose_config == {"temperature": 0.4}
    assert config.verify_config == {"temperature": 0.2}
    assert config.dedupe_similarity_threshold == 0.8


def test_n_propose_samples_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_N_PROPOSE_SAMPLES", "6")
    config = _get_cr_config(monkeypatch)
    assert config.n_propose_samples == 6


def test_n_propose_samples_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_N_PROPOSE_SAMPLES", "0")
    with pytest.raises(ValueError, match="n_propose_samples"):
        get_strategy("cr", _TestSignature)


def test_temperature_out_of_bounds_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A temperature outside the 0-2 band fails fast at strategy construction."""
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_PROPOSE_TEMPERATURE", "2.5")
    with pytest.raises(ValueError, match="propose_config"):
        get_strategy("cr", _TestSignature)


def test_report_config_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cr_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_CR_TARGET_PROPOSITIONS", "3")
    monkeypatch.setenv("VIREX_BENCH_CR_VERIFIER_MODE", "single")
    strategy = get_strategy("cr", _TestSignature)
    assert isinstance(strategy, CRStrategy)
    report = strategy.report_config
    assert report["target_propositions"] == 3
    assert report["verifier_mode"] == "single"


def test_strategy_name_and_no_variant_axis() -> None:
    """CR exposes no variant axis; bare ``cr`` is the only name."""
    strategy = get_strategy("cr", _TestSignature)
    assert isinstance(strategy, CRStrategy)
    assert strategy.name == "cr"
    assert strategy.accepts_variant is False
