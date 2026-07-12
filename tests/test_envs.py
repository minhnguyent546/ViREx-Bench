"""Tests for the env var access layer (``virex_bench.envs``).

Covers the validation/convert helpers, the lazy module ``__getattr__`` dispatch,
``__dir__``, and ``is_set`` — the paths that read environment state at call time
rather than import time.
"""

import pytest

from virex_bench import envs


def test_env_with_choices_returns_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIREX_BENCH_TEST_VAR", raising=False)
    accessor = envs.env_with_choices("VIREX_BENCH_TEST_VAR", "fallback", ["A", "B"])
    assert accessor() == "fallback"


def test_env_with_choices_case_sensitive_match_returns_original_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIREX_BENCH_TEST_VAR", "A")
    accessor = envs.env_with_choices(
        "VIREX_BENCH_TEST_VAR", "fallback", ["A", "B"], case_sensitive=True
    )
    assert accessor() == "A"


def test_env_with_choices_case_sensitive_rejects_wrong_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Case-sensitive matching means a lowercase value is not a valid choice.
    monkeypatch.setenv("VIREX_BENCH_TEST_VAR", "a")
    accessor = envs.env_with_choices(
        "VIREX_BENCH_TEST_VAR", "fallback", ["A", "B"], case_sensitive=True
    )
    with pytest.raises(ValueError, match="Invalid value 'a'"):
        accessor()


def test_env_with_choices_case_insensitive_match_still_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIREX_BENCH_TEST_VAR", "a")
    accessor = envs.env_with_choices(
        "VIREX_BENCH_TEST_VAR", "fallback", ["A", "B"], case_sensitive=False
    )
    # The match is case-insensitive but the original (un-normalized) value wins.
    assert accessor() == "a"


def test_env_with_choices_invalid_value_raises_with_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIREX_BENCH_TEST_VAR", "Z")
    accessor = envs.env_with_choices(
        "VIREX_BENCH_TEST_VAR", "fallback", ["A", "B"], case_sensitive=False
    )
    with pytest.raises(ValueError, match="Valid options: \\['A', 'B'\\]"):
        accessor()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("0", False),
        ("1", True),
        ("5", True),
    ],
)
def test_maybe_convert_bool(raw: str | None, expected: bool | None) -> None:
    assert envs.maybe_convert_bool(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("TRUE", True),
        (" yes ", True),
        ("on", True),
        ("0", False),
        ("no", False),
        ("anything-else", False),
    ],
)
def test_get_bool_parses_truthy_values(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("VIREX_BENCH_TEST_BOOL", raw)
    assert envs.get_bool("VIREX_BENCH_TEST_BOOL", "0") is expected


def test_get_bool_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIREX_BENCH_TEST_BOOL", raising=False)
    # Default "1" → truthy.
    assert envs.get_bool("VIREX_BENCH_TEST_BOOL", "1") is True


def test_maybe_convert_int_and_float_treat_empty_as_none() -> None:
    assert envs.maybe_convert_int(None) is None
    assert envs.maybe_convert_int("") is None
    assert envs.maybe_convert_int("7") == 7
    assert envs.maybe_convert_float(None) is None
    assert envs.maybe_convert_float("") is None
    assert envs.maybe_convert_float("1.5") == 1.5


def test_module_getattr_resolves_registered_var_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIREX_BENCH_OUTPUT_DIR", "/tmp/virex-out")
    # VIREX_BENCH_OUTPUT_DIR is in ``environment_variables`` — resolved via __getattr__.
    assert envs.VIREX_BENCH_OUTPUT_DIR == "/tmp/virex-out"


def test_module_getattr_unknown_name_raises_attributeerror() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'NOT_A_REAL_VAR'"):
        _ = envs.NOT_A_REAL_VAR  # type: ignore[attr-defined]


def test_module_dir_lists_registered_variable_names() -> None:
    names = dir(envs)
    assert "VIREX_BENCH_LOG_LEVEL" in names
    assert "VIREX_BENCH_OUTPUT_DIR" in names


def test_is_set_reflects_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIREX_BENCH_OUTPUT_DIR", "x")
    assert envs.is_set("VIREX_BENCH_OUTPUT_DIR") is True
    monkeypatch.delenv("VIREX_BENCH_OUTPUT_DIR", raising=False)
    assert envs.is_set("VIREX_BENCH_OUTPUT_DIR") is False
