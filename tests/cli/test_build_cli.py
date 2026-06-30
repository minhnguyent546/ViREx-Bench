"""Tests for CLI strategy variant handling."""

import argparse
from types import SimpleNamespace
from typing import Any

import dspy
import pytest

from virex_bench.cli import build_cli


class _TestSignature(dspy.Signature):
    """Minimal signature used to verify task lookup."""

    premises: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def test_version_flag_prints_version_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_cli.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"virex-bench {build_cli.__version__}"


def test_run_parser_accepts_tot_dfs_strategy() -> None:
    parser = build_cli.build_parser()
    args = parser.parse_args(["run", "--model", "test-model", "--strategy", "tot-dfs"])
    assert args.strategy == "tot-dfs"


def test_run_resolves_base_strategy_for_task_signature(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    signature_lookups: list[str] = []
    rationale_lookups: list[str] = []
    strategy_calls: list[tuple[str, type[dspy.Signature], object | None]] = []

    class _FakeTask:
        name = "fake-task"

        def get_signature(self, strategy_name: str) -> type[dspy.Signature]:
            signature_lookups.append(strategy_name)
            return _TestSignature

        def get_rationale_field(self, strategy_name: str) -> object | None:
            rationale_lookups.append(strategy_name)
            return None

    def fake_get_strategy(
        name: str,
        signature: type[dspy.Signature],
        rationale_field: object | None = None,
    ) -> object:
        strategy_calls.append((name, signature, rationale_field))
        return SimpleNamespace(name=name, report_config={})

    def fake_evaluate(*args: Any, **kwargs: Any) -> object:
        return SimpleNamespace(
            task="fake-task",
            model="test-model",
            backend="openai",
            strategy="tot-dfs",
            decoding="single-pass",
            metric="accuracy",
            score=1.0,
            num_examples=1,
        )

    monkeypatch.setattr(build_cli, "set_level", lambda _level: None)
    monkeypatch.setattr(build_cli, "get_task", lambda _name: _FakeTask())
    monkeypatch.setattr(build_cli, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(build_cli, "get_strategy", fake_get_strategy)
    monkeypatch.setattr(build_cli, "get_decoding", lambda *args, **kwargs: object())
    monkeypatch.setattr(build_cli, "evaluate", fake_evaluate)
    monkeypatch.setattr(build_cli, "save_report", lambda *_args, **_kwargs: "")

    args = argparse.Namespace(
        log_level="INFO",
        task="fake-task",
        model="test-model",
        backend="openai",
        api_base=None,
        api_key=None,
        model_kwargs={},
        strategy="tot-dfs",
        decoding="single-pass",
        decoding_num_samples=None,
        num_threads=1,
        max_examples=None,
        output_dir="results",
    )

    build_cli._run(args)  # pyright: ignore[reportPrivateUsage]
    capsys.readouterr()

    assert signature_lookups == ["tot"]
    assert rationale_lookups == ["tot"]
    assert strategy_calls == [("tot-dfs", _TestSignature, None)]
