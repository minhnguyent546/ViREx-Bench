import logging

from virex_bench import envs

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
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
