"""Tests for the index CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tomli_w
from typer.testing import CliRunner

from transskribo.cli import app

runner = CliRunner()


def _make_config(tmp_path: Path, output_dir: Path) -> Path:
    """Create a minimal config.toml pointing at output_dir."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    config_data: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "hf_token": "hf_test",
    }
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(tomli_w.dumps(config_data).encode())
    return config_path


def _make_config_with_index_section(
    tmp_path: Path, output_dir: Path, index_output_path: str
) -> Path:
    """Create a config.toml with [index] section."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    config_data: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "hf_token": "hf_test",
        "index": {"output_path": index_output_path},
    }
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(tomli_w.dumps(config_data).encode())
    return config_path


def _write_enriched_json(path: Path, source_file: str = "lecture.mp3") -> None:
    """Write a minimal enriched transskribo result JSON."""
    doc = {
        "segments": [{"text": "Hello", "speaker": "SPEAKER_00"}],
        "metadata": {"source_file": source_file},
        "title": "Title",
        "keywords": ["kw"],
        "summary": "Summary.",
        "concepts": {"TestConcept": "A test concept."},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


class TestIndexCommand:
    """Tests for the index CLI command."""

    def test_generates_index_at_default_path(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(output_dir / "result.json")
        config_path = _make_config(tmp_path, output_dir)

        result = runner.invoke(app, ["index", "--config", str(config_path)])
        assert result.exit_code == 0

        index_path = output_dir / "concept-index.md"
        assert index_path.exists()
        content = index_path.read_text(encoding="utf-8")
        assert "TestConcept" in content

    def test_output_flag_overrides_default(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(output_dir / "result.json")
        config_path = _make_config(tmp_path, output_dir)
        custom_path = tmp_path / "custom" / "index.md"

        result = runner.invoke(
            app, ["index", "--config", str(config_path), "--output", str(custom_path)]
        )
        assert result.exit_code == 0
        assert custom_path.exists()
        # Default path should NOT exist
        assert not (output_dir / "concept-index.md").exists()

    def test_config_output_path_used_when_set(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(output_dir / "result.json")
        config_output = str(tmp_path / "from-config.md")
        config_path = _make_config_with_index_section(
            tmp_path, output_dir, config_output
        )

        result = runner.invoke(app, ["index", "--config", str(config_path)])
        assert result.exit_code == 0
        assert Path(config_output).exists()

    def test_cli_output_overrides_config_output(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(output_dir / "result.json")
        config_output = str(tmp_path / "from-config.md")
        config_path = _make_config_with_index_section(
            tmp_path, output_dir, config_output
        )
        cli_output = tmp_path / "from-cli.md"

        result = runner.invoke(
            app, ["index", "--config", str(config_path), "--output", str(cli_output)]
        )
        assert result.exit_code == 0
        assert cli_output.exists()
        assert not Path(config_output).exists()

    def test_empty_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _make_config(tmp_path, output_dir)

        result = runner.invoke(app, ["index", "--config", str(config_path)])
        assert result.exit_code == 0
        # No index file should be generated
        assert not (output_dir / "concept-index.md").exists()

    def test_no_enriched_files(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Write a non-enriched result
        doc = {
            "segments": [{"text": "Hello"}],
            "metadata": {"source_file": "file.mp3"},
        }
        (output_dir / "result.json").write_text(json.dumps(doc), encoding="utf-8")
        config_path = _make_config(tmp_path, output_dir)

        result = runner.invoke(app, ["index", "--config", str(config_path)])
        assert result.exit_code == 0
        assert not (output_dir / "concept-index.md").exists()

    def test_overwrites_existing_index(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(output_dir / "result.json")
        config_path = _make_config(tmp_path, output_dir)

        index_path = output_dir / "concept-index.md"
        index_path.write_text("old content", encoding="utf-8")

        result = runner.invoke(app, ["index", "--config", str(config_path)])
        assert result.exit_code == 0
        content = index_path.read_text(encoding="utf-8")
        assert "TestConcept" in content
        assert "old content" not in content
