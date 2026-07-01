"""Tests for `virex_bench.logger`: compact location format and query-context tagging."""

from __future__ import annotations

import contextvars
import io
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from virex_bench import logger as virex_logger


def _build_record(
    msg: str = "hello",
    *,
    name: str = "virex_bench.evaluation.evaluate",
    pathname: str = "/pkg/evaluate.py",
    lineno: int = 79,
    func: str = "_process_example",
    level: int = logging.INFO,
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=pathname,
        lineno=lineno,
        msg=msg,
        args=(),
        exc_info=None,
        func=func,
    )


@pytest.fixture
def capture_logger() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    """A logger that formats through `_RecordEnricher` + `_LOG_FORMAT` into a buffer.

    Isolated from the package's global stderr handler (own handler, propagate off) so
    tests can assert on exact rendered lines.
    """
    log = logging.getLogger("virex_bench.tests.test_logger")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(virex_logger._RecordEnricher())  # pyright: ignore[reportPrivateUsage]
    handler.setFormatter(
        logging.Formatter(
            virex_logger._LOG_FORMAT,  # pyright: ignore[reportPrivateUsage]
            datefmt=virex_logger._DATE_FORMAT,  # pyright: ignore[reportPrivateUsage]
        )
    )
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    yield log, stream
    log.removeHandler(handler)


def test_init_logger_returns_child_logger_and_configures_root() -> None:
    log = virex_logger.init_logger("virex_bench.something")
    assert isinstance(log, logging.Logger)
    assert log.name == "virex_bench.something"
    # First use attaches the package handler to the "virex_bench" root.
    assert logging.getLogger("virex_bench").handlers


def test_record_enricher_sets_compact_location_and_empty_query_id() -> None:
    enricher = virex_logger._RecordEnricher()  # pyright: ignore[reportPrivateUsage]
    record = _build_record()

    assert enricher.filter(record) is True

    assert getattr(record, "location", None) == "evaluate._process_example:79"
    assert getattr(record, "query_id", None) == ""


def test_record_enricher_reflects_active_query_context() -> None:
    enricher = virex_logger._RecordEnricher()  # pyright: ignore[reportPrivateUsage]
    record = _build_record()

    with virex_logger.log_query_context("q-1"):
        enricher.filter(record)

    assert getattr(record, "query_id", None) == "[q-1] "


def test_format_renders_compact_location_not_full_logger_name() -> None:
    fmt = logging.Formatter(
        virex_logger._LOG_FORMAT,  # pyright: ignore[reportPrivateUsage]
        datefmt=virex_logger._DATE_FORMAT,  # pyright: ignore[reportPrivateUsage]
    )
    enricher = virex_logger._RecordEnricher()  # pyright: ignore[reportPrivateUsage]
    record = _build_record(msg="hi")
    enricher.filter(record)

    output = fmt.format(record)

    assert "evaluate._process_example:79" in output
    # The full dotted logger name must NOT appear (it's replaced by the compact form).
    assert "virex_bench.evaluation.evaluate" not in output


def test_format_uses_compact_location_in_real_log_lines(
    capture_logger: tuple[logging.Logger, io.StringIO],
) -> None:
    log, stream = capture_logger
    log.info("check location")
    line = stream.getvalue().strip()

    # `module.funcName:lineno`, where module is the call-site file stem ("test_logger").
    assert "test_logger.test_format_uses_compact_location_in_real_log_lines:" in line
    assert "virex_bench.tests.test_logger" not in line


def test_log_query_context_tags_only_inside_block(
    capture_logger: tuple[logging.Logger, io.StringIO],
) -> None:
    log, stream = capture_logger
    log.info("before")
    with virex_logger.log_query_context("q-001"):
        log.info("inside the block")
        log.warning("also inside")
    log.info("after")

    lines = stream.getvalue().splitlines()
    assert "[q-001]" not in lines[0]
    assert "[q-001] inside the block" in lines[1]
    assert "[q-001] also inside" in lines[2]
    assert "[q-001]" not in lines[3]


def test_log_query_context_nested_restores_outer_tag(
    capture_logger: tuple[logging.Logger, io.StringIO],
) -> None:
    log, stream = capture_logger
    with virex_logger.log_query_context("outer"):
        log.info("outer line")
        with virex_logger.log_query_context("inner"):
            log.info("inner line")
        log.info("back to outer")
    log.info("no context")

    lines = stream.getvalue().splitlines()
    assert "[outer] outer line" in lines[0]
    assert "[inner] inner line" in lines[1]
    assert "[outer] back to outer" in lines[2]
    assert "[outer]" not in lines[3]
    assert "[inner]" not in lines[3]


def test_log_query_context_clears_tag_even_when_block_raises() -> None:
    enricher = virex_logger._RecordEnricher()  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(RuntimeError, match="boom"):
        with virex_logger.log_query_context("q-x"):
            raise RuntimeError("boom")

    # The contextvar must be reset on exit, so a subsequent record is untagged.
    record = _build_record()
    enricher.filter(record)
    assert getattr(record, "query_id", None) == ""


def test_query_context_propagates_through_copied_context_into_thread() -> None:
    enricher = virex_logger._RecordEnricher()  # pyright: ignore[reportPrivateUsage]
    captured: list[str | None] = []

    def worker() -> None:
        record = _build_record(msg="from worker")
        enricher.filter(record)
        captured.append(getattr(record, "query_id", None))

    with virex_logger.log_query_context("q-thread"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(contextvars.copy_context().run, worker)
            future.result()

    assert captured == ["[q-thread] "]


def test_query_context_does_not_leak_into_plain_threadpool_worker() -> None:
    """Documents the gotcha that motivated the `copy_context().run(...)` wrapping in
    `decoding/self_consistency.py`: a bare `ThreadPoolExecutor.submit` does NOT inherit
    contextvars, so the worker sees no tag.
    """
    enricher = virex_logger._RecordEnricher()  # pyright: ignore[reportPrivateUsage]
    captured: list[str | None] = []

    def worker() -> None:
        record = _build_record(msg="from worker")
        enricher.filter(record)
        captured.append(getattr(record, "query_id", None))

    with virex_logger.log_query_context("q-thread"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker)
            future.result()

    assert captured == [""]


def test_colored_formatter_wraps_location_in_ansi_without_mutating_record() -> None:
    formatter = virex_logger.ColoredFormatter(
        virex_logger._LOG_FORMAT,  # pyright: ignore[reportPrivateUsage]
        datefmt=virex_logger._DATE_FORMAT,  # pyright: ignore[reportPrivateUsage]
    )
    enricher = virex_logger._RecordEnricher()  # pyright: ignore[reportPrivateUsage]
    record = _build_record(level=logging.WARNING)
    enricher.filter(record)

    output = formatter.format(record)

    reset = virex_logger.ColoredFormatter._RESET  # pyright: ignore[reportPrivateUsage]
    assert reset in output
    assert "evaluate._process_example:79" in output
    # The record itself must be left untouched (coloring is applied to a copy).
    assert getattr(record, "location", None) == "evaluate._process_example:79"
    assert record.levelname == "WARNING"
