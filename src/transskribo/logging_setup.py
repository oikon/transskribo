"""Logging configuration with dual output: rich stdout + rotating file."""

from __future__ import annotations

import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

# Third-party loggers that add their own handlers or are excessively noisy.
_THIRD_PARTY_LOGGERS = (
    "whisperx",
    "pyannote",
    "speechbrain",
    "lightning",
    "lightning_fabric",
    "torch",
    "torchaudio",
)


def setup_logging(log_level: str, log_file: Path) -> None:
    """Configure the root logger with rich stdout and rotating file handlers.

    Args:
        log_level: Logging level string (e.g. "INFO", "DEBUG").
        log_file: Path to the log file. Parent directories are created if needed.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Route Python warnings through the logging system so they respect log level.
    logging.captureWarnings(True)
    if level > logging.WARNING:
        warnings.filterwarnings("ignore")

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicate output
    root.handlers.clear()

    # Force third-party loggers to respect our level and use our handlers only.
    for name in _THIRD_PARTY_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(level)
        lib_logger.handlers.clear()

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
