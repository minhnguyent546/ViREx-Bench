import contextvars
import logging
import sys
from contextlib import contextmanager

from virex_bench import envs

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(location)s | %(query_id)s%(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LEVELNAME_WIDTH = 8
_configured = False

# Active query id for `log_query_context`; prefixes each log line with ``[<id>]``.
# A contextvar so the tag is isolated per task/worker.
_query_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "virex_bench_query_id", default=None
)


@contextmanager
def log_query_context(query_id: str):
    """Prefix all log messages emitted within this block with ``[<query_id>]``.

    Safe for nesting and async/thread use via contextvars. To carry the tag into a
    `ThreadPoolExecutor` worker, run the callable in a copied context (see
    `decoding/self_consistency.py`).
    """
    token = _query_id_var.set(query_id)
    try:
        yield
    finally:
        _query_id_var.reset(token)


class _RecordEnricher(logging.Filter):
    """Stamp each record with ``location`` (``parent.module::func:lineno``) and ``query_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        module_path = ".".join(record.name.split(".")[-2:])
        record.location = f"{module_path}::{record.funcName}:{record.lineno}"
        query_id = _query_id_var.get()
        record.query_id = f"[{query_id}] " if query_id is not None else ""
        return True


class ColoredFormatter(logging.Formatter):
    """Formatter that wraps the level name in ANSI colors and dims metadata.

    The level name is padded to a fixed width *before* the color codes are added so
    that the invisible escape sequences do not throw off column alignment.
    """

    _LEVEL_COLORS = {
        "DEBUG": "\033[37m",  # white
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    _GREY = "\033[90m"
    _RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str | None = None) -> None:
        # The base format pads the level name itself; this formatter pads + colors it
        # manually, so strip the width spec to avoid double-padding.
        colored_fmt = fmt.replace("%(levelname)-8s", "%(levelname)s")
        super().__init__(colored_fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        original_asctime = getattr(record, "asctime", None)
        # `location` is injected by `_RecordEnricher`.
        original_location = getattr(record, "location", "")

        level_color = self._LEVEL_COLORS.get(record.levelname, "")
        padded_levelname = f"{record.levelname:<{_LEVELNAME_WIDTH}}"
        record.levelname = f"{level_color}{padded_levelname}{self._RESET}"
        record.location = f"{self._GREY}[{original_location}]{self._RESET}"
        try:
            message = super().format(record)
        finally:
            record.levelname = original_levelname
            record.location = original_location
            if original_asctime is not None:
                record.asctime = original_asctime
        return message

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return f"{self._GREY}{super().formatTime(record, datefmt)}{self._RESET}"


def _use_color(stream: object) -> bool:
    if envs.NO_COLOR or envs.VIREX_BENCH_LOG_COLOR == "0":
        return False
    if envs.VIREX_BENCH_LOG_COLOR == "1":
        return True
    isatty = getattr(stream, "isatty", None)
    return isatty is not None and isatty()


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    stream = sys.stderr
    handler = logging.StreamHandler(stream)
    handler.addFilter(_RecordEnricher())
    if _use_color(stream):
        handler.setFormatter(ColoredFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger = logging.getLogger("virex_bench")
    root_logger.addHandler(handler)
    root_logger.setLevel(envs.VIREX_BENCH_LOG_LEVEL)
    root_logger.propagate = False
    _configured = True


def init_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring the package handler on first use.

    Usage:
        from virex_bench.logger import init_logger
        logger = init_logger(__name__)
    """
    _configure_root()
    return logging.getLogger(name)


def set_level(level: int | str) -> None:
    _configure_root()
    logging.getLogger("virex_bench").setLevel(level)
