"""
utils/logger.py
───────────────
Configures structlog for structured, levelled logging across all Jarvis
modules.  Call `get_logger(__name__)` in any module to get a bound logger
that automatically includes the module name and timestamp in every line.

Logs are written both to stdout (for live monitoring) and to the log file
specified in settings.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from config.settings import settings


def _configure_stdlib_logging() -> None:
    """
    Set up the standard-library logging so that structlog can pipe through it.
    We use RotatingFileHandler to cap disk usage and StreamHandler for the
    terminal.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level))

    # Ensure the log directory exists
    log_path: Path = settings.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── File handler — rotates at 10 MB, keeps 5 backups ────
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # ── Stream handler — pretty output to stdout ─────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def configure_logging() -> None:
    """
    Full logging bootstrap.  Call once at application startup (e.g. from
    main.py or orchestrator.py's __main__ block).
    """
    _configure_stdlib_logging()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),        # Human-friendly in dev
            # Swap ConsoleRenderer → JSONRenderer in production:
            # structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger bound to *name*.

    Usage:
        log = get_logger(__name__)
        log.info("Agent started", agent="orchestrator")
    """
    return structlog.get_logger(name)
