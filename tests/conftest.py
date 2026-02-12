"""Shared test fixtures for Transskribo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomli_w


@pytest.fixture
def tmp_input_dir(tmp_path: Path) -> Path:
    """Create a temporary input directory."""
    d = tmp_path / "input"
    d.mkdir()
    return d


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def sample_config_dict(tmp_input_dir: Path, tmp_output_dir: Path) -> dict[str, Any]:
    """Return a minimal valid config dict."""
    return {
        "input_dir": str(tmp_input_dir),
        "output_dir": str(tmp_output_dir),
        "hf_token": "hf_test_token_123",
    }


@pytest.fixture
def sample_config_file(
    tmp_path: Path, sample_config_dict: dict[str, Any]
) -> Path:
    """Write a sample config TOML file and return its path."""
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(tomli_w.dumps(sample_config_dict).encode())
    return config_path
