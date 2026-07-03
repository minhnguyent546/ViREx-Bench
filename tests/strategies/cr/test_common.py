"""Tests for shared CR helpers in ``common.py``."""

import pytest

from virex_bench.strategies.cr.common import (
    CRConfig,
    is_empty_or_none_proposition,
    parse_bool,
    render_verdict_buckets,
)

# --- parse_bool --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [True, "True", "TRUE", "true", "yes", "Yes", "1", "t", "y"],
)
def test_parse_bool_truthy(raw: object) -> None:
    assert parse_bool(raw) is True


@pytest.mark.parametrize(
    "raw",
    [False, "False", "FALSE", "false", "no", "No", "0", "f", "n", ""],
)
def test_parse_bool_falsy(raw: object) -> None:
    assert parse_bool(raw) is False


def test_parse_bool_unparseable_defaults_false() -> None:
    """An unparseable verifier verdict defaults to the conservative False."""
    assert parse_bool("maybe") is False
    assert parse_bool(object()) is False


def test_parse_bool_labelled_output() -> None:
    """Peels off a leading label like 'is_entailed: true'."""
    assert parse_bool("is_entailed: true") is True
    assert parse_bool("Verdict: false") is False


# --- is_empty_or_none_proposition -------------------------------------------


def test_is_empty_handles_none_and_whitespace() -> None:
    assert is_empty_or_none_proposition(None) is True
    assert is_empty_or_none_proposition("") is True
    assert is_empty_or_none_proposition("   ") is True


def test_is_empty_catches_vietnamese_sentinels() -> None:
    assert is_empty_or_none_proposition("Không có mệnh đề mới.") is True
    assert is_empty_or_none_proposition("không thể suy luận thêm") is True


def test_is_empty_catches_english_sentinels() -> None:
    assert is_empty_or_none_proposition("No new proposition.") is True
    assert is_empty_or_none_proposition("cannot derive anything") is True


def test_is_empty_keeps_substantive_proposition() -> None:
    assert is_empty_or_none_proposition("Theo tiền đề 3, An là sinh viên.") is False


# --- render_verdict_buckets --------------------------------------------------


def test_render_all_empty_returns_empty_string() -> None:
    assert render_verdict_buckets([], [], []) == ""


def test_render_entailed_only() -> None:
    rendered = render_verdict_buckets(["An là sinh viên.", "An học tại Hà Nội."], [], [])
    assert rendered == (
        "[Mệnh đề được xác nhận 1] An là sinh viên.\n[Mệnh đề được xác nhận 2] An học tại Hà Nội."
    )


def test_render_all_three_buckets() -> None:
    rendered = render_verdict_buckets(
        ["An là sinh viên."],
        ["Bình không phải sinh viên."],
        ["An có thể là giáo viên."],
    )
    assert "[Mệnh đề được xác nhận 1] An là sinh viên." in rendered
    assert "[Mệnh đề bị bác bỏ 1] Bình không phải sinh viên." in rendered
    assert "[Mệnh đề không xác định 1] An có thể là giáo viên." in rendered
    # Empty buckets are omitted, non-empty sections separated by blank line.
    assert rendered.count("\n\n") == 2


# --- CRConfig validation -----------------------------------------------------


def _valid_config_kwargs() -> dict[str, object]:
    return {
        "target_propositions": 7,
        "max_failed_attempts": 6,
        "verifier_mode": "multi",
        # Empty dict = inherit the LM's --model-kwargs profile (the default).
        "propose_config": {},
        "verify_config": {},
        "dedupe_similarity_threshold": 0.9,
    }


def test_crconfig_accepts_valid_values() -> None:
    config = CRConfig(**_valid_config_kwargs())
    assert config.target_propositions == 7
    assert config.verifier_mode == "multi"
    assert config.propose_config == {}
    assert config.verify_config == {}


def test_crconfig_accepts_explicit_override_configs() -> None:
    """A user may opt into role-specific sampling via env vars, which lands here
    as a populated per-call override dict."""
    kwargs = _valid_config_kwargs()
    kwargs["propose_config"] = {"temperature": 0.8}
    kwargs["verify_config"] = {"temperature": 0.2}
    config = CRConfig(**kwargs)
    assert config.propose_config == {"temperature": 0.8}
    assert config.verify_config == {"temperature": 0.2}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_propositions", 0),
        ("max_failed_attempts", 0),
        ("dedupe_similarity_threshold", -0.1),
        ("dedupe_similarity_threshold", 1.1),
    ],
)
def test_crconfig_rejects_out_of_bounds(field: str, value: object) -> None:
    kwargs = _valid_config_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        CRConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "temperature"),
    [
        ("propose_config", -0.1),
        ("propose_config", 2.1),
        ("verify_config", -0.1),
        ("verify_config", 2.1),
    ],
)
def test_crconfig_rejects_temperature_out_of_band(field: str, temperature: float) -> None:
    """A ``temperature`` override outside [0, 2] fails fast."""
    kwargs = _valid_config_kwargs()
    kwargs[field] = {"temperature": temperature}
    with pytest.raises(ValueError, match=rf"{field}\['temperature'\]"):
        CRConfig(**kwargs)  # type: ignore[arg-type]


def test_crconfig_accepts_non_temperature_overrides_unchecked() -> None:
    """Only the well-known ``temperature`` key is bounds-checked; other sampling
    overrides (top_p, top_k, ...) pass through untouched."""
    kwargs = _valid_config_kwargs()
    kwargs["propose_config"] = {"top_p": 0.9, "top_k": 20}  # no temperature key
    config = CRConfig(**kwargs)
    assert config.propose_config == {"top_p": 0.9, "top_k": 20}


def test_crconfig_rejects_unknown_verifier_mode() -> None:
    kwargs = _valid_config_kwargs()
    kwargs["verifier_mode"] = "tri"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="verifier_mode"):
        CRConfig(**kwargs)  # type: ignore[arg-type]
