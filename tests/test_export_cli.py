"""Tests for the export CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import tomli_w
from typer.testing import CliRunner

from transskribo.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(
    tmp_path: Path,
    output_dir: Path,
    export_section: dict[str, Any] | None = None,
) -> Path:
    """Write a config TOML with export section and return its path."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    config_data: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "hf_token": "hf_test_token_123",
    }
    if export_section is not None:
        config_data["export"] = export_section
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(tomli_w.dumps(config_data).encode())
    return config_path


def _create_result_json(path: Path, enriched: bool = False) -> dict[str, Any]:
    """Create a transskribo result JSON and return the document."""
    doc: dict[str, Any] = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Hello world.", "speaker": "SPEAKER_00"},
        ],
        "words": [],
        "metadata": {
            "source_file": "/input/test.mp3",
            "file_hash": "abc123",
            "duration_secs": 5.0,
        },
    }
    if enriched:
        doc["title"] = "Test Title"
        doc["keywords"] = ["test"]
        doc["summary"] = "Test summary."
        doc["concepts"] = {"test": "A concept"}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


# ---------------------------------------------------------------------------
# No format flag
# ---------------------------------------------------------------------------

class TestExportNoFormat:
    def test_error_when_no_format_flag(self, tmp_path: Path) -> None:
        """Export should fail when no format flag is provided."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        result = runner.invoke(app, ["export", "--config", str(config_path)])
        assert result.exit_code != 0
        assert "At least one format flag" in result.output


# ---------------------------------------------------------------------------
# Batch mode tests
# ---------------------------------------------------------------------------

class TestExportBatchMode:
    @patch("transskribo.docx_writer.generate_docx")
    def test_batch_exports_enriched_jsons(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should discover and export all enriched result JSONs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        _create_result_json(output_dir / "lec1.json", enriched=True)
        _create_result_json(output_dir / "lec2.json", enriched=True)

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 2

    @patch("transskribo.docx_writer.generate_docx")
    def test_batch_skips_non_enriched(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should skip non-enriched files with warning."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        _create_result_json(output_dir / "enriched.json", enriched=True)
        _create_result_json(output_dir / "not_enriched.json", enriched=False)

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 1

    @patch("transskribo.docx_writer.generate_docx")
    def test_batch_skips_already_exported(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should skip files with existing .docx unless --force."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        _create_result_json(output_dir / "result.json", enriched=True)
        # Pre-create the .docx so it looks already exported
        (output_dir / "result.docx").touch()

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx"
        ])
        assert result.exit_code == 0
        mock_docx.assert_not_called()

    @patch("transskribo.docx_writer.generate_docx")
    def test_batch_force_regenerates_existing(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--force should regenerate already-exported files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        _create_result_json(output_dir / "result.json", enriched=True)
        (output_dir / "result.docx").touch()

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx", "--force"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 1

    @patch("transskribo.docx_writer.generate_docx")
    def test_batch_skips_non_transskribo_json(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should ignore non-transskribo JSON files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        # Non-transskribo JSON
        (output_dir / "other.json").write_text(
            json.dumps({"key": "value"}), encoding="utf-8"
        )
        # Valid enriched result JSON
        _create_result_json(output_dir / "result.json", enriched=True)

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 1

    @patch("transskribo.docx_writer.generate_docx")
    def test_batch_skips_transskribo_dir(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should skip files in .transskribo/ directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        transskribo_dir = output_dir / ".transskribo"
        transskribo_dir.mkdir()
        (transskribo_dir / "registry.json").write_text(
            json.dumps({"segments": [], "metadata": {},
                         "title": "x", "keywords": [], "summary": "x", "concepts": {}}),
            encoding="utf-8",
        )

        _create_result_json(output_dir / "result.json", enriched=True)

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 1

    @patch("transskribo.docx_writer.generate_docx", side_effect=RuntimeError("Template error"))
    def test_batch_continues_on_per_file_error(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Per-file export errors should be logged and batch should continue."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        _create_result_json(output_dir / "file1.json", enriched=True)
        _create_result_json(output_dir / "file2.json", enriched=True)

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 2

    @patch("transskribo.docx_writer.generate_docx")
    def test_batch_docx_path_alongside_json(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--docx should generate .docx alongside .json."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        _create_result_json(output_dir / "test.json", enriched=True)

        result = runner.invoke(app, [
            "export", "--config", str(config_path), "--docx"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 1
        docx_path_arg = mock_docx.call_args[0][0]
        assert str(docx_path_arg).endswith(".docx")
        assert str(docx_path_arg) == str(output_dir / "test.docx")


# ---------------------------------------------------------------------------
# Single-file mode tests
# ---------------------------------------------------------------------------

class TestExportSingleFile:
    @patch("transskribo.docx_writer.generate_docx")
    def test_single_file_exports(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file should export a single enriched result JSON."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        json_path = output_dir / "single.json"
        _create_result_json(json_path, enriched=True)

        result = runner.invoke(app, [
            "export", "--config", str(config_path),
            "--file", str(json_path), "--docx"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 1

    @patch("transskribo.docx_writer.generate_docx")
    def test_single_file_skips_non_enriched(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file should skip non-enriched file with warning."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        json_path = output_dir / "plain.json"
        _create_result_json(json_path, enriched=False)

        result = runner.invoke(app, [
            "export", "--config", str(config_path),
            "--file", str(json_path), "--docx"
        ])
        assert result.exit_code == 0
        mock_docx.assert_not_called()

    @patch("transskribo.docx_writer.generate_docx")
    def test_single_file_force_regenerates(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file --force should regenerate existing exports."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        json_path = output_dir / "result.json"
        _create_result_json(json_path, enriched=True)
        (output_dir / "result.docx").touch()

        result = runner.invoke(app, [
            "export", "--config", str(config_path),
            "--file", str(json_path), "--docx", "--force"
        ])
        assert result.exit_code == 0
        assert mock_docx.call_count == 1

    @patch("transskribo.docx_writer.generate_docx")
    def test_single_file_skips_already_exported(
        self,
        mock_docx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file should skip if .docx already exists without --force."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, output_dir)

        json_path = output_dir / "result.json"
        _create_result_json(json_path, enriched=True)
        (output_dir / "result.docx").touch()

        result = runner.invoke(app, [
            "export", "--config", str(config_path),
            "--file", str(json_path), "--docx"
        ])
        assert result.exit_code == 0
        mock_docx.assert_not_called()


# ---------------------------------------------------------------------------
# Config and edge cases
# ---------------------------------------------------------------------------

class TestExportConfig:
    def test_missing_config_file(self, tmp_path: Path) -> None:
        """Export should fail with missing config file."""
        result = runner.invoke(app, [
            "export", "--config", str(tmp_path / "nope.toml"), "--docx"
        ])
        assert result.exit_code != 0
        assert "Config file not found" in result.output
