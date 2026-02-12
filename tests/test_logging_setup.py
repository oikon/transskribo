"""Tests for logging_setup module."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from transskribo.logging_setup import setup_logging


def test_setup_logging_attaches_two_handlers(tmp_path: Path) -> None:
    """Root logger gets a RichHandler and a RotatingFileHandler."""
    log_file = tmp_path / "logs" / "test.log"
    setup_logging("INFO", log_file)

    root = logging.getLogger()
    assert len(root.handlers) == 2

    handler_types = {type(h) for h in root.handlers}
    assert RichHandler in handler_types
    assert RotatingFileHandler in handler_types

    # Cleanup
    root.handlers.clear()


def test_log_file_created(tmp_path: Path) -> None:
    """Log file and its parent directory are created."""
    log_file = tmp_path / "subdir" / "nested" / "test.log"
    setup_logging("DEBUG", log_file)

    assert log_file.parent.is_dir()

    # Write a message so the file gets created
    logger = logging.getLogger("test_file_created")
    logger.info("test message")

    assert log_file.exists()

    # Cleanup
    logging.getLogger().handlers.clear()


def test_messages_appear_in_log_file(tmp_path: Path) -> None:
    """Messages written via logger appear in the log file."""
    log_file = tmp_path / "test.log"
    setup_logging("DEBUG", log_file)

    logger = logging.getLogger("test_messages")
    logger.info("hello from test")
    logger.warning("warning from test")

    # Flush handlers
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_file.read_text()
    assert "hello from test" in content
    assert "warning from test" in content

    # Cleanup
    logging.getLogger().handlers.clear()


def test_log_level_is_respected(tmp_path: Path) -> None:
    """Messages below the configured level are not written to the file."""
    log_file = tmp_path / "test.log"
    setup_logging("WARNING", log_file)

    logger = logging.getLogger("test_level")
    logger.debug("debug msg")
    logger.info("info msg")
    logger.warning("warning msg")

    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_file.read_text()
    assert "debug msg" not in content
    assert "info msg" not in content
    assert "warning msg" in content

    # Cleanup
    logging.getLogger().handlers.clear()


def test_setup_logging_clears_existing_handlers(tmp_path: Path) -> None:
    """Calling setup_logging twice doesn't duplicate handlers."""
    log_file = tmp_path / "test.log"
    setup_logging("INFO", log_file)
    setup_logging("INFO", log_file)

    root = logging.getLogger()
    assert len(root.handlers) == 2

    # Cleanup
    root.handlers.clear()


def test_root_logger_level_is_set(tmp_path: Path) -> None:
    """Root logger level matches the requested level."""
    log_file = tmp_path / "test.log"
    setup_logging("DEBUG", log_file)

    root = logging.getLogger()
    assert root.level == logging.DEBUG

    # Cleanup
    root.handlers.clear()
