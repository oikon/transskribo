"""Tests for the pipeline CLI command."""

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
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Write a full config TOML and return its path."""
    config_data: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "hf_token": "hf_test_token_123",
        "enrich": {
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "sk-test",
            "llm_model": "gpt-4o-mini",
        },
        "export": {
            "template_path": "templates/basic.docx",
            "transcritor": "Test",
        },
    }
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
# Pipeline command tests
# ---------------------------------------------------------------------------

class TestPipelineCommand:
    @patch("transskribo.cli._export_batch")
    @patch("transskribo.cli._enrich_batch")
    @patch("transskribo.cli._run_pipeline")
    @patch("transskribo.cli.check_ffprobe_available")
    def test_pipeline_calls_all_three_stages(
        self,
        mock_ffprobe: MagicMock,
        mock_run: MagicMock,
        mock_enrich: MagicMock,
        mock_export: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline should call run, enrich, and export in order."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        mock_enrich.return_value = {"enriched": 0, "skipped": 0, "failed": 0}
        mock_export.return_value = {"exported": 0, "skipped": 0, "failed": 0}

        result = runner.invoke(app, ["pipeline", "--config", str(config_path)])
        assert result.exit_code == 0

        mock_run.assert_called_once()
        mock_enrich.assert_called_once()
        mock_export.assert_called_once()

    @patch("transskribo.cli._export_batch")
    @patch("transskribo.cli._enrich_batch")
    @patch("transskribo.cli._run_pipeline")
    @patch("transskribo.cli.check_ffprobe_available")
    def test_enrich_runs_even_if_run_had_failures(
        self,
        mock_ffprobe: MagicMock,
        mock_run: MagicMock,
        mock_enrich: MagicMock,
        mock_export: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Stage 2 (enrich) should run even if stage 1 (run) had partial failures."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        # Simulate run completing with an exception raised and caught internally
        mock_run.return_value = None
        mock_enrich.return_value = {"enriched": 2, "skipped": 0, "failed": 1}
        mock_export.return_value = {"exported": 2, "skipped": 0, "failed": 0}

        result = runner.invoke(app, ["pipeline", "--config", str(config_path)])
        assert result.exit_code == 0

        mock_enrich.assert_called_once()
        mock_export.assert_called_once()

    @patch("transskribo.cli._export_batch")
    @patch("transskribo.cli._enrich_batch")
    @patch("transskribo.cli._run_pipeline")
    @patch("transskribo.cli.check_ffprobe_available")
    def test_export_runs_even_if_enrich_had_failures(
        self,
        mock_ffprobe: MagicMock,
        mock_run: MagicMock,
        mock_enrich: MagicMock,
        mock_export: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Stage 3 (export) should run even if stage 2 (enrich) had partial failures."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        mock_enrich.return_value = {"enriched": 1, "skipped": 0, "failed": 2}
        mock_export.return_value = {"exported": 1, "skipped": 0, "failed": 0}

        result = runner.invoke(app, ["pipeline", "--config", str(config_path)])
        assert result.exit_code == 0

        mock_export.assert_called_once()

    @patch("transskribo.cli._export_batch")
    @patch("transskribo.cli._enrich_batch")
    @patch("transskribo.cli._run_pipeline")
    @patch("transskribo.cli.setup_logging")
    @patch("transskribo.cli.check_ffprobe_available")
    def test_pipeline_logs_final_combined_summary(
        self,
        mock_ffprobe: MagicMock,
        mock_logging: MagicMock,
        mock_run: MagicMock,
        mock_enrich: MagicMock,
        mock_export: MagicMock,
        tmp_path: Path,
        caplog: Any,
    ) -> None:
        """Pipeline should log a final combined summary."""
        import logging

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        mock_enrich.return_value = {"enriched": 3, "skipped": 1, "failed": 0}
        mock_export.return_value = {"exported": 2, "skipped": 1, "failed": 0}

        with caplog.at_level(logging.INFO, logger="transskribo.cli"):
            result = runner.invoke(app, ["pipeline", "--config", str(config_path)])

        assert result.exit_code == 0
        log_text = caplog.text
        assert "Pipeline Complete" in log_text
        assert "3 enriched" in log_text
        assert "2 exported" in log_text

    @patch("transskribo.cli._export_batch")
    @patch("transskribo.cli._enrich_batch")
    @patch("transskribo.cli._run_pipeline")
    @patch("transskribo.cli.setup_logging")
    @patch("transskribo.cli.check_ffprobe_available")
    def test_pipeline_logs_per_stage_summaries(
        self,
        mock_ffprobe: MagicMock,
        mock_logging: MagicMock,
        mock_run: MagicMock,
        mock_enrich: MagicMock,
        mock_export: MagicMock,
        tmp_path: Path,
        caplog: Any,
    ) -> None:
        """Pipeline should log per-stage headers."""
        import logging

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        mock_enrich.return_value = {"enriched": 0, "skipped": 0, "failed": 0}
        mock_export.return_value = {"exported": 0, "skipped": 0, "failed": 0}

        with caplog.at_level(logging.INFO, logger="transskribo.cli"):
            result = runner.invoke(app, ["pipeline", "--config", str(config_path)])

        assert result.exit_code == 0
        log_text = caplog.text
        assert "Stage 1/3: Transcription" in log_text
        assert "Stage 2/3: Enrichment" in log_text
        assert "Stage 3/3: Export" in log_text

    def test_pipeline_missing_config(self, tmp_path: Path) -> None:
        """Pipeline should fail with missing config file."""
        result = runner.invoke(app, [
            "pipeline", "--config", str(tmp_path / "nope.toml")
        ])
        assert result.exit_code != 0
        assert "Config file not found" in result.output

    @patch("transskribo.cli._export_batch")
    @patch("transskribo.cli._enrich_batch")
    @patch("transskribo.cli._run_pipeline")
    @patch("transskribo.cli.check_ffprobe_available")
    def test_pipeline_export_uses_all_formats(
        self,
        mock_ffprobe: MagicMock,
        mock_run: MagicMock,
        mock_enrich: MagicMock,
        mock_export: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline should call export with do_docx=True (all formats)."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        mock_enrich.return_value = {"enriched": 0, "skipped": 0, "failed": 0}
        mock_export.return_value = {"exported": 0, "skipped": 0, "failed": 0}

        result = runner.invoke(app, ["pipeline", "--config", str(config_path)])
        assert result.exit_code == 0

        # Verify export was called with do_docx=True (4th positional arg)
        export_call_args = mock_export.call_args
        assert export_call_args[0][2] is False  # force=False
        assert export_call_args[0][3] is True   # do_docx=True
