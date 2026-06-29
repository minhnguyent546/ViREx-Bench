"""Tests for shared ToT helpers in ``common.py``."""

import pytest

from virex_bench.strategies.tot.common import dedupe_thoughts


def test_dedupe_exact_duplicates() -> None:
    """Exact duplicates (after case normalization) are dropped."""
    thoughts = ["same thought", "Same Thought", "SAME THOUGHT", "unique"]
    result = dedupe_thoughts(thoughts)
    assert result == ["same thought", "unique"]


def test_dedupe_near_duplicates_above_threshold() -> None:
    """Near-paraphrases with high Jaccard overlap are dropped."""
    thoughts = [
        "According to premise one all cats are mammals that need food and water",
        "According to premise one all cats are mammals that need food and shelter",
        # Unrelated — low word overlap.
        "The weather forecast predicts heavy rain tomorrow afternoon",
    ]
    result = dedupe_thoughts(thoughts, similarity_threshold=0.7)
    assert len(result) == 2
    assert result[0] == "According to premise one all cats are mammals that need food and water"
    assert result[1] == "The weather forecast predicts heavy rain tomorrow afternoon"


def test_dedupe_keeps_dissimilar_thoughts() -> None:
    """Thoughts with low Jaccard overlap are all kept."""
    thoughts = [
        "All cats are animals",
        "Some dogs are pets",
        "The sky is blue today",
    ]
    result = dedupe_thoughts(thoughts)
    assert len(result) == 3


def test_dedupe_respects_custom_threshold() -> None:
    """A higher threshold keeps near-duplicates that a lower one would drop."""
    # Jaccard = 6/8 = 0.75 — deduped at 0.7, kept at 0.8.
    thoughts = [
        "The cat sat on the mat near the door",
        "The cat sat on the mat by the door",
    ]
    assert len(dedupe_thoughts(thoughts, similarity_threshold=0.7)) == 1
    assert len(dedupe_thoughts(thoughts, similarity_threshold=0.8)) == 2


def test_dedupe_default_threshold_is_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env-backed default keeps moderately similar reasoning branches."""
    monkeypatch.delenv("VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD", raising=False)
    thoughts = [
        "The cat sat on the mat near the door",
        "The cat sat on the mat by the door",
    ]
    assert dedupe_thoughts(thoughts) == thoughts


def test_dedupe_uses_env_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD", "0.7")
    thoughts = [
        "The cat sat on the mat near the door",
        "The cat sat on the mat by the door",
    ]
    assert dedupe_thoughts(thoughts) == ["The cat sat on the mat near the door"]


def test_dedupe_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        dedupe_thoughts(["a"], similarity_threshold=1.1)


def test_dedupe_drops_empty_strings() -> None:
    """Empty and whitespace-only strings are always dropped."""
    thoughts = ["", "   ", "real thought"]
    result = dedupe_thoughts(thoughts)
    assert result == ["real thought"]
