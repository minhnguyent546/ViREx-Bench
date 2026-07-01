"""Tests for CLI strategy variant handling."""

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from virex_bench.cli import build_cli


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


def test_run_delegates_strategy_construction_to_task(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    strategy_calls: list[str] = []

    class _FakeTask:
        name = "fake-task"

        def get_strategy(self, strategy_name: str) -> object:
            strategy_calls.append(strategy_name)
            return SimpleNamespace(name=strategy_name, report_config={})

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

    assert strategy_calls == ["tot-dfs"]
