"""Tests for shared PoT-Z3 helpers in ``common.py``.

Covers the pure-Python helpers (``parse_code``, ``strip_ansi``,
``format_solver_result``) and the :class:`POTConfig` bounds validation.
Config env-var resolution is tested separately in ``test_config.py``.
"""

import pytest

from virex_bench.strategies.pot.common import (
    POTConfig,
    format_solver_result,
    parse_code,
    strip_ansi,
)


def test_strip_ansi_removes_color_codes() -> None:
    assert strip_ansi("\x1b[31mred text\x1b[0m") == "red text"


def test_strip_ansi_removes_style_codes() -> None:
    assert strip_ansi("\x1b[1;32mbold green\x1b[0m") == "bold green"


def test_strip_ansi_passthrough_plain_text() -> None:
    assert strip_ansi("no codes here") == "no codes here"


def test_strip_ansi_empty_string() -> None:
    assert strip_ansi("") == ""


def test_parse_code_raw_code_no_fences() -> None:
    code = 'x = 1\nprint(x)\nresult = {"answer": "Yes"}\nprint(result)'
    parsed, error = parse_code(code)
    assert error is None
    assert parsed == code


def test_parse_code_strips_python_fences() -> None:
    raw = "```python\nx = 1\nprint(x)\n```"
    parsed, error = parse_code(raw)
    assert error is None
    assert parsed == "x = 1\nprint(x)"


def test_parse_code_strips_bare_fences() -> None:
    raw = "```\nx = 1\nprint(x)\n```"
    parsed, error = parse_code(raw)
    assert error is None
    assert parsed == "x = 1\nprint(x)"


def test_parse_code_strips_partial_opening_fence() -> None:
    """Fallback: LLM emitted opening ```python but no closing ```."""
    raw = "```python\nx = 1\nprint(x)"
    parsed, error = parse_code(raw)
    assert error is None
    assert parsed == "x = 1\nprint(x)"


def test_parse_code_strips_trailing_fence_only() -> None:
    raw = "x = 1\nprint(x)\n```"
    parsed, error = parse_code(raw)
    assert error is None
    assert parsed == "x = 1\nprint(x)"


def test_parse_code_drops_commentary_after_separator() -> None:
    raw = "```python\nx = 1\nprint(x)\n```\n---\nThis code does x."
    parsed, error = parse_code(raw)
    assert error is None
    assert parsed == "x = 1\nprint(x)"


def test_parse_code_rejects_empty() -> None:
    parsed, error = parse_code("")
    assert parsed is None
    assert error is not None
    assert "Empty" in error


def test_parse_code_rejects_whitespace_only() -> None:
    parsed, error = parse_code("   \n  \n  ")
    assert parsed is None
    assert error is not None


def test_parse_code_rejects_garbled_single_line_multi_equals() -> None:
    """A single line with multiple '=' is almost certainly a garbled extract."""
    parsed, error = parse_code("answer = result = something")
    assert parsed is None
    assert error is not None
    assert "format" in error.lower()


def test_parse_code_appends_print_for_bare_assignment() -> None:
    """If the last line is a bare `result = {...}` without print, append print(result)."""
    raw = 'x = 1\nresult = {"answer": "Yes"}'
    parsed, error = parse_code(raw)
    assert error is None
    assert parsed is not None
    assert parsed.endswith("print(result)")


def test_parse_code_does_not_append_print_when_already_present() -> None:
    raw = 'x = 1\nresult = {"answer": "Yes"}\nprint(result)'
    parsed, error = parse_code(raw)
    assert error is None
    assert parsed is not None
    # Should not double-append.
    assert parsed.count("print(result)") == 1


def test_parse_code_does_not_append_for_single_line() -> None:
    """Single-line programs (len(lines) == 1) never get the append treatment."""
    raw = 'result = {"answer": "Yes"}'
    parsed, error = parse_code(raw)
    # Single line with one '=' is fine (not garbled), but no print appended.
    assert error is None
    assert parsed == 'result = {"answer": "Yes"}'


def test_format_solver_result_success() -> None:
    rendered = format_solver_result("x = 1", '{"answer": "Yes"}', None)
    assert "Z3 Program:" in rendered
    assert "x = 1" in rendered
    assert "Program Output:" in rendered
    assert '{"answer": "Yes"}' in rendered
    assert "failed" not in rendered


def test_format_solver_result_failure() -> None:
    rendered = format_solver_result("x = 1", None, "NameError: x")
    assert "Z3 Program (failed):" in rendered
    assert "x = 1" in rendered
    assert "Error:" in rendered
    assert "NameError: x" in rendered
    assert "Reason over the premises directly." in rendered


def test_format_solver_result_success_output_empty_string() -> None:
    """An empty-string output (program ran but printed nothing) is still success."""
    rendered = format_solver_result("x = 1", "", None)
    assert "Program Output:" in rendered
    assert "failed" not in rendered


def _valid_config_kwargs() -> dict[str, object]:
    return {
        "max_iters": 3,
        "execution_timeout": 45.0,
        "generate_config": {},
        "regenerate_config": {},
        "fallback_on_error": True,
    }


def test_potconfig_accepts_valid_values() -> None:
    config = POTConfig(**_valid_config_kwargs())
    assert config.max_iters == 3
    assert config.execution_timeout == 45.0
    assert config.generate_config == {}
    assert config.regenerate_config == {}
    assert config.fallback_on_error is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_iters", 0),
        ("max_iters", -1),
        ("execution_timeout", 0.0),
        ("execution_timeout", -1.0),
    ],
)
def test_potconfig_rejects_out_of_bounds(field: str, value: object) -> None:
    kwargs = _valid_config_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        POTConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "temperature"),
    [
        ("generate_config", -0.1),
        ("generate_config", 2.1),
        ("regenerate_config", -0.1),
        ("regenerate_config", 2.1),
    ],
)
def test_potconfig_rejects_temperature_out_of_band(field: str, temperature: float) -> None:
    """A ``temperature`` override outside [0, 2] fails fast."""
    kwargs = _valid_config_kwargs()
    kwargs[field] = {"temperature": temperature}
    with pytest.raises(ValueError, match=rf"{field}\['temperature'\]"):
        POTConfig(**kwargs)  # type: ignore[arg-type]


def test_potconfig_accepts_explicit_override_configs() -> None:
    kwargs = _valid_config_kwargs()
    kwargs["generate_config"] = {"temperature": 0.8}
    kwargs["regenerate_config"] = {"temperature": 0.2}
    config = POTConfig(**kwargs)  # type: ignore[arg-type]
    assert config.generate_config == {"temperature": 0.8}
    assert config.regenerate_config == {"temperature": 0.2}


def test_potconfig_accepts_non_temperature_overrides_unchecked() -> None:
    """Only the well-known ``temperature`` key is bounds-checked; other sampling
    overrides (top_p, top_k, ...) pass through untouched."""
    kwargs = _valid_config_kwargs()
    kwargs["generate_config"] = {"top_p": 0.9, "top_k": 20}
    config = POTConfig(**kwargs)  # type: ignore[arg-type]
    assert config.generate_config == {"top_p": 0.9, "top_k": 20}
