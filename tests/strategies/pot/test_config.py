"""Tests for env-var resolution in ``build_pot_config``."""

import pytest

from virex_bench.strategies.pot.common import POTConfig, build_pot_config

_POT_ENVS = [
    "VIREX_BENCH_POT_MAX_ITERS",
    "VIREX_BENCH_POT_EXECUTION_TIMEOUT",
    "VIREX_BENCH_POT_GENERATE_TEMPERATURE",
    "VIREX_BENCH_POT_REGENERATE_TEMPERATURE",
    "VIREX_BENCH_POT_FALLBACK_ON_ERROR",
]


def _clear_pot_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _POT_ENVS:
        monkeypatch.delenv(name, raising=False)


def _get_pot_config(monkeypatch: pytest.MonkeyPatch) -> POTConfig:
    """Build a POTConfig from the current env. Callers are responsible for
    clearing/setting POT env vars beforehand."""
    return build_pot_config()


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    config = _get_pot_config(monkeypatch)
    assert config.max_iters == 3
    assert config.execution_timeout == 15.0
    assert config.fallback_on_error is True
    # Temperatures default to None -> empty config (inherit LM profile).
    assert config.generate_config == {}
    assert config.regenerate_config == {}


def test_max_iters_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_MAX_ITERS", "5")
    config = _get_pot_config(monkeypatch)
    assert config.max_iters == 5


def test_execution_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_EXECUTION_TIMEOUT", "120.0")
    config = _get_pot_config(monkeypatch)
    assert config.execution_timeout == 120.0


def test_temperatures_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_GENERATE_TEMPERATURE", "0.8")
    monkeypatch.setenv("VIREX_BENCH_POT_REGENERATE_TEMPERATURE", "0.3")
    config = _get_pot_config(monkeypatch)
    assert config.generate_config == {"temperature": 0.8}
    assert config.regenerate_config == {"temperature": 0.3}


def test_fallback_on_error_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_FALLBACK_ON_ERROR", "0")
    config = _get_pot_config(monkeypatch)
    assert config.fallback_on_error is False


def test_fallback_on_error_true(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_FALLBACK_ON_ERROR", "true")
    config = _get_pot_config(monkeypatch)
    assert config.fallback_on_error is True


def test_fallback_on_error_unparseable_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparseable value falls back to the conservative False (get_bool only
    treats 1/true/yes/on as truthy)."""
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_FALLBACK_ON_ERROR", "maybe")
    config = _get_pot_config(monkeypatch)
    assert config.fallback_on_error is False


def test_max_iters_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_MAX_ITERS", "0")
    with pytest.raises(ValueError, match="max_iters"):
        _get_pot_config(monkeypatch)


def test_execution_timeout_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_EXECUTION_TIMEOUT", "0")
    with pytest.raises(ValueError, match="execution_timeout"):
        _get_pot_config(monkeypatch)


def test_generate_temperature_out_of_bounds_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_GENERATE_TEMPERATURE", "2.5")
    with pytest.raises(ValueError, match="generate_config"):
        _get_pot_config(monkeypatch)


def test_regenerate_temperature_out_of_bounds_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pot_envs(monkeypatch)
    monkeypatch.setenv("VIREX_BENCH_POT_REGENERATE_TEMPERATURE", "-0.1")
    with pytest.raises(ValueError, match="regenerate_config"):
        _get_pot_config(monkeypatch)
