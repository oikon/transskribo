"""Tests for the enrich CLI command."""

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

def _write_config_with_enrich(
    tmp_path: Path, output_dir: Path, enrich_section: dict[str, Any] | None = None
) -> Path:
    """Write a config TOML with enrich section and return its path."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    config_data: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "hf_token": "hf_test_token_123",
    }
    if enrich_section is not None:
        config_data["enrich"] = enrich_section
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
# Batch mode tests
# ---------------------------------------------------------------------------

class TestEnrichBatchMode:
    @patch("transskribo.enricher.enrich_document")
    def test_batch_enriches_result_jsons(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should discover and enrich all result JSONs in output_dir."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        # Create result files
        _create_result_json(output_dir / "lectures" / "lec1.json")
        _create_result_json(output_dir / "lectures" / "lec2.json")

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, ["enrich", "--config", str(config_path)])
        assert result.exit_code == 0
        assert mock_enrich.call_count == 2

    @patch("transskribo.enricher.enrich_document")
    def test_batch_skips_already_enriched(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should skip already-enriched files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        _create_result_json(output_dir / "enriched.json", enriched=True)
        _create_result_json(output_dir / "not_enriched.json", enriched=False)

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, ["enrich", "--config", str(config_path)])
        assert result.exit_code == 0
        # Only the not-enriched file should be processed
        assert mock_enrich.call_count == 1

    @patch("transskribo.enricher.enrich_document")
    def test_batch_force_re_enriches(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--force should re-enrich already-enriched files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        _create_result_json(output_dir / "enriched.json", enriched=True)

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Re-Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, [
            "enrich", "--config", str(config_path), "--force"
        ])
        assert result.exit_code == 0
        assert mock_enrich.call_count == 1

    @patch("transskribo.enricher.enrich_document")
    def test_batch_skips_non_transskribo_json(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should ignore non-transskribo JSON files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        # Create a non-transskribo JSON (no segments/metadata)
        other_json = output_dir / "other.json"
        other_json.write_text(json.dumps({"key": "value"}), encoding="utf-8")

        # Create a valid result JSON
        _create_result_json(output_dir / "result.json")

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, ["enrich", "--config", str(config_path)])
        assert result.exit_code == 0
        assert mock_enrich.call_count == 1

    @patch("transskribo.enricher.enrich_document")
    def test_batch_skips_transskribo_dir(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch mode should skip files in .transskribo/ directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        # Create a JSON in .transskribo/ (registry.json)
        transskribo_dir = output_dir / ".transskribo"
        transskribo_dir.mkdir()
        (transskribo_dir / "registry.json").write_text(
            json.dumps({"segments": [], "metadata": {}}), encoding="utf-8"
        )

        # Create a valid result JSON outside .transskribo/
        _create_result_json(output_dir / "result.json")

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, ["enrich", "--config", str(config_path)])
        assert result.exit_code == 0
        assert mock_enrich.call_count == 1

    @patch("transskribo.enricher.enrich_document", side_effect=RuntimeError("LLM error"))
    def test_batch_continues_on_per_file_error(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Per-file LLM errors should be logged and batch should continue."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        _create_result_json(output_dir / "file1.json")
        _create_result_json(output_dir / "file2.json")

        result = runner.invoke(app, ["enrich", "--config", str(config_path)])
        assert result.exit_code == 0
        # Both files should be attempted
        assert mock_enrich.call_count == 2

    @patch("transskribo.enricher.enrich_document")
    def test_batch_does_not_generate_docx(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Enrich command should NOT generate .docx files (moved to export)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        _create_result_json(output_dir / "test.json")

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, ["enrich", "--config", str(config_path)])
        assert result.exit_code == 0
        # No .docx file should be created
        docx_files = list(output_dir.rglob("*.docx"))
        assert len(docx_files) == 0


# ---------------------------------------------------------------------------
# Single-file mode tests
# ---------------------------------------------------------------------------

class TestEnrichSingleFile:
    @patch("transskribo.enricher.enrich_document")
    def test_single_file_enriches(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file should enrich a single result JSON."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        json_path = output_dir / "single.json"
        _create_result_json(json_path)

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, [
            "enrich", "--config", str(config_path), "--file", str(json_path)
        ])
        assert result.exit_code == 0
        assert mock_enrich.call_count == 1

    @patch("transskribo.enricher.enrich_document")
    def test_single_file_skips_already_enriched(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file should skip already-enriched file without --force."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        json_path = output_dir / "enriched.json"
        _create_result_json(json_path, enriched=True)

        result = runner.invoke(app, [
            "enrich", "--config", str(config_path), "--file", str(json_path)
        ])
        assert result.exit_code == 0
        mock_enrich.assert_not_called()

    @patch("transskribo.enricher.enrich_document")
    def test_single_file_force_re_enriches(
        self,
        mock_enrich: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file --force should re-enrich already-enriched file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config_with_enrich(tmp_path, output_dir)

        json_path = output_dir / "enriched.json"
        _create_result_json(json_path, enriched=True)

        def enrich_side_effect(doc: dict[str, Any], cfg: Any) -> dict[str, Any]:
            doc["title"] = "Re-Enriched"
            doc["keywords"] = ["test"]
            doc["summary"] = "Summary"
            doc["concepts"] = {"c": "d"}
            return doc

        mock_enrich.side_effect = enrich_side_effect

        result = runner.invoke(app, [
            "enrich", "--config", str(config_path),
            "--file", str(json_path), "--force"
        ])
        assert result.exit_code == 0
        assert mock_enrich.call_count == 1


# ---------------------------------------------------------------------------
# Config and edge cases
# ---------------------------------------------------------------------------

class TestEnrichConfig:
    def test_missing_config_file(self, tmp_path: Path) -> None:
        """Enrich should fail with missing config file."""
        result = runner.invoke(app, [
            "enrich", "--config", str(tmp_path / "nope.toml")
        ])
        assert result.exit_code != 0
        assert "Config file not found" in result.output
