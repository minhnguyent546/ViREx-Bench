"""Tests for CLI strategy variant handling."""

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from virex_bench.cli import build_cli


def test_version_flag_prints_version_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(build_cli, "__git_revision__", "abc1234")
    parser = build_cli.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"virex-bench: v{build_cli.__version__}\ngit revision: abc1234"


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

    def fake_set_level(_level: str) -> None:
        return None

    def fake_get_task(_name: str) -> _FakeTask:
        return _FakeTask()

    def fake_get_model(*args: Any, **kwargs: Any) -> object:
        return object()

    def fake_get_decoding(*args: Any, **kwargs: Any) -> object:
        return object()

    def fake_save_report(*args: Any, **kwargs: Any) -> str:
        return ""

    monkeypatch.setattr(build_cli, "set_level", fake_set_level)
    monkeypatch.setattr(build_cli, "get_task", fake_get_task)
    monkeypatch.setattr(build_cli, "get_model", fake_get_model)
    monkeypatch.setattr(build_cli, "get_decoding", fake_get_decoding)
    monkeypatch.setattr(build_cli, "evaluate", fake_evaluate)
    monkeypatch.setattr(build_cli, "save_report", fake_save_report)

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
        self_certainty_borda_power=None,
        num_threads=1,
        max_examples=None,
        output_dir="results",
    )

    build_cli._run(args)  # pyright: ignore[reportPrivateUsage]
    capsys.readouterr()

    assert strategy_calls == ["tot-dfs"]


def test_run_forwards_decoding_kwargs_and_prints_partial_eval_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decoding_calls: list[dict[str, object]] = []

    class _FakeTask:
        name = "fake-task"

        def get_strategy(self, strategy_name: str) -> object:
            return SimpleNamespace(name=strategy_name, report_config={})

    def fake_evaluate(*args: Any, **kwargs: Any) -> object:
        # num_evaluated_examples (2) differs from num_examples (5) to exercise the
        # partial-eval branch of the run summary.
        return SimpleNamespace(
            task="fake-task",
            model="test-model",
            strategy="direct",
            decoding="single-pass",
            metric="accuracy",
            score=0.5,
            num_examples=5,
            num_evaluated_examples=2,
        )

    def fake_get_decoding(_name: str, _strategy: object, **kwargs: object) -> object:
        decoding_calls.append(kwargs)
        return object()

    def fake_set_level(_level: str) -> None:
        return None

    def fake_get_task(_name: str) -> _FakeTask:
        return _FakeTask()

    def fake_get_model(*args: Any, **kwargs: Any) -> object:
        return object()

    def fake_save_report(*args: Any, **kwargs: Any) -> str:
        return ""

    monkeypatch.setattr(build_cli, "set_level", fake_set_level)
    monkeypatch.setattr(build_cli, "get_task", fake_get_task)
    monkeypatch.setattr(build_cli, "get_model", fake_get_model)
    monkeypatch.setattr(build_cli, "get_decoding", fake_get_decoding)
    monkeypatch.setattr(build_cli, "evaluate", fake_evaluate)
    monkeypatch.setattr(build_cli, "save_report", fake_save_report)

    args = argparse.Namespace(
        log_level="INFO",
        task="fake-task",
        model="test-model",
        backend="openai",
        api_base=None,
        api_key=None,
        model_kwargs={},
        strategy="direct",
        decoding="single-pass",
        decoding_num_samples=3,
        self_certainty_borda_power=0.5,
        num_threads=1,
        max_examples=None,
        output_dir="results",
    )

    build_cli._run(args)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out

    # Both decoding kwargs are forwarded to get_decoding.
    assert decoding_calls == [{"num_samples": 3, "borda_power": 0.5}]
    # Partial-eval branch prints "evaluated/total examples".
    assert "2/5 examples" in out


def test_run_parser_rejects_non_positive_max_examples() -> None:
    parser = build_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--model", "m", "--max-examples", "0"])


def test_run_parser_rejects_non_integer_max_examples() -> None:
    parser = build_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--model", "m", "--max-examples", "two"])


def test_run_parser_accepts_positive_int_max_examples() -> None:
    parser = build_cli.build_parser()
    args = parser.parse_args(["run", "--model", "m", "--max-examples", "3"])
    assert args.max_examples == 3


def test_run_parser_rejects_invalid_json_model_kwargs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--model", "m", "--model-kwargs", "not-json"])
    err = capsys.readouterr().err
    assert "must be valid JSON" in err


def test_run_parser_rejects_non_object_model_kwargs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--model", "m", "--model-kwargs", "[1, 2, 3]"])
    err = capsys.readouterr().err
    assert "must be a JSON object" in err


def test_run_parser_accepts_json_object_model_kwargs() -> None:
    parser = build_cli.build_parser()
    args = parser.parse_args(["run", "--model", "m", "--model-kwargs", '{"temperature": 0.5}'])
    assert args.model_kwargs == {"temperature": 0.5}


def test_tasks_handler_prints_registered_tasks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_cli._tasks(argparse.Namespace(list=True))  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert "vietnamese-logical-reasoning" in out
    assert "Available tasks" in out
    assert "NAME" in out and "DESCRIPTION" in out


def test_tasks_handler_no_list_flag_prints_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_cli._tasks(argparse.Namespace(list=False))  # pyright: ignore[reportPrivateUsage]
    assert capsys.readouterr().out == ""


def test_strategies_handler_prints_registered_strategies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_cli._strategies(  # pyright: ignore[reportPrivateUsage]
        argparse.Namespace(list=True)
    )
    out = capsys.readouterr().out
    assert "cot" in out.split()
    assert "direct" in out.split()
    assert "Available strategies" in out
    assert "VARIANTS" in out
    # ToT variants are grouped into a single row under the base name.
    assert "tot" in out.split()
    assert "beam" in out and "dfs" in out and "mcts" in out


def test_decodings_handler_prints_registered_decodings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_cli._decodings(  # pyright: ignore[reportPrivateUsage]
        argparse.Namespace(list=True)
    )
    out = capsys.readouterr().out
    assert "single-pass" in out.split()
    assert "self-consistency" in out.split()
    assert "Available decoding strategies" in out


def test_main_dispatches_tasks_list(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["virex-bench", "tasks", "--list"])
    build_cli.main()
    out = capsys.readouterr().out
    assert "vietnamese-logical-reasoning" in out
