"""Logging configuration with dual output: rich stdout + rotating file."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_level: str, log_file: Path) -> None:
    """Configure the root logger with rich stdout and rotating file handlers.

    Args:
        log_level: Logging level string (e.g. "INFO", "DEBUG").
        log_file: Path to the log file. Parent directories are created if needed.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicate output
    root.handlers.clear()

    # Rich stdout handler
    stdout_handler = RichHandler(
        level=level,
        rich_tracebacks=True,
        show_path=False,
    )
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stdout_handler)

    # Rotating file handler (10 MB, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)
