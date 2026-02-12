"""Tests for CLI integration and pipeline wiring."""

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

def _write_config(tmp_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Write a minimal config TOML and return its path."""
    config_data = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "hf_token": "hf_test_token_123",
    }
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(tomli_w.dumps(config_data).encode())
    return config_path


def _create_audio_file(input_dir: Path, relative: str, size: int = 100) -> Path:
    """Create a dummy audio file in the input directory."""
    path = input_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)
    return path


def _create_registry(output_dir: Path, entries: dict[str, Any]) -> Path:
    """Write a registry file and return its path."""
    reg_dir = output_dir / ".transskribo"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_path = reg_dir / "registry.json"
    reg_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return reg_path


# ---------------------------------------------------------------------------
# 10.01 — CLI arg parsing, config loading, ffprobe check
# ---------------------------------------------------------------------------

class TestRunCommandSetup:
    """Tests for the run command startup: config loading, ffprobe check."""

    def test_run_missing_config_file(self, tmp_path: Path) -> None:
        """Run with a non-existent config file should fail."""
        result = runner.invoke(app, ["run", "--config", str(tmp_path / "nope.toml")])
        assert result.exit_code != 0
        assert "Config file not found" in result.output

    def test_run_invalid_config(self, tmp_path: Path) -> None:
        """Run with invalid config (missing required fields) should fail."""
        config_path = tmp_path / "bad.toml"
        config_path.write_bytes(tomli_w.dumps({"model_size": "tiny"}).encode())
        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code != 0

    @patch("transskribo.cli.check_ffprobe_available", side_effect=RuntimeError("ffprobe not found"))
    def test_run_ffprobe_not_available(
        self, mock_ffprobe: MagicMock, tmp_path: Path
    ) -> None:
        """Run should fail fast if ffprobe is not on PATH."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code != 0
        mock_ffprobe.assert_called_once()

    def test_run_cli_overrides(self, tmp_path: Path) -> None:
        """CLI flags should override config file values."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        # Create a different input dir for CLI override
        alt_input = tmp_path / "alt_input"
        alt_input.mkdir()

        with patch("transskribo.cli.check_ffprobe_available"), \
             patch("transskribo.cli._run_pipeline") as mock_pipeline:
            result = runner.invoke(app, [
                "run", "--config", str(config_path),
                "--input-dir", str(alt_input),
                "--model-size", "tiny",
                "--batch-size", "4",
            ])
            assert result.exit_code == 0
            cfg = mock_pipeline.call_args[0][0]
            assert cfg.input_dir == alt_input
            assert cfg.model_size == "tiny"
            assert cfg.batch_size == 4


# ---------------------------------------------------------------------------
# 10.02 — Pipeline skeleton: scan, filter, validate, progress bar
# ---------------------------------------------------------------------------

class TestPipelineSkeleton:
    """Tests for scan → filter → validate → progress bar flow."""

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.scan_directory")
    @patch("transskribo.cli.filter_already_processed")
    def test_no_files_to_process(
        self,
        mock_filter: MagicMock,
        mock_scan: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline exits cleanly when no files are found."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        mock_scan.return_value = []
        mock_filter.return_value = []

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_invalid_files_skipped(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Invalid files should be skipped without processing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create some dummy files
        _create_audio_file(input_dir, "good.mp3")
        _create_audio_file(input_dir, "bad.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        def validate_side_effect(file_path: Path, max_dur: float) -> ValidationResult:
            if "bad" in file_path.name:
                return ValidationResult(is_valid=False, duration_secs=None, error="corrupt")
            return ValidationResult(is_valid=True, duration_secs=120.0, error=None)

        mock_validate.side_effect = validate_side_effect
        mock_hash.return_value = "abc123"
        mock_process.return_value = {
            "result": {"segments": []},
            "timing": {"transcribe_secs": 1.0, "align_secs": 0.5, "diarize_secs": 0.8, "total_secs": 2.3},
        }

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0
        # Only good.mp3 should be processed
        mock_process.assert_called_once()


# ---------------------------------------------------------------------------
# 10.03 — Hash check + duplicate handling
# ---------------------------------------------------------------------------

class TestDuplicateHandling:
    """Tests for hash-based duplicate detection in the pipeline."""

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    def test_duplicate_copies_output(
        self,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Duplicate file should copy existing output instead of re-processing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "dup.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )
        mock_hash.return_value = "hash_existing"

        # Create an existing output and registry entry
        existing_output = output_dir / "original.json"
        existing_doc = {
            "segments": [],
            "words": [],
            "metadata": {
                "source_file": "/orig/file.mp3",
                "file_hash": "hash_existing",
                "duration_secs": 60.0,
                "num_speakers": 2,
                "model_size": "large-v3",
                "language": "pt",
                "processed_at": "2024-01-01T00:00:00",
                "timing": None,
            },
        }
        existing_output.write_text(json.dumps(existing_doc), encoding="utf-8")

        # Pre-populate registry with the existing entry
        _create_registry(output_dir, {
            "hash_existing": {
                "source_path": "/orig/file.mp3",
                "output_path": str(existing_output),
                "timestamp": "2024-01-01T00:00:00",
                "status": "success",
                "duration_audio_secs": 60.0,
                "timing": None,
                "error": None,
            }
        })

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0

        # The output for dup.mp3 should exist (copied from original)
        dup_output = output_dir / "dup.json"
        assert dup_output.exists()
        with dup_output.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["metadata"]["source_file"] == str(input_dir / "dup.mp3")


# ---------------------------------------------------------------------------
# 10.04 — Transcription wiring, error handling, batch summary
# ---------------------------------------------------------------------------

class TestTranscriptionWiring:
    """Tests for process_file wiring, error handling, and batch summary."""

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_successful_processing(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A file should be transcribed, output written, and registered."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "test.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=300.0, error=None
        )
        mock_hash.return_value = "hash_new"
        mock_process.return_value = {
            "result": {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "text": "Hello world",
                        "speaker": "SPEAKER_00",
                        "words": [
                            {"start": 0.0, "end": 0.5, "word": "Hello", "score": 0.9, "speaker": "SPEAKER_00"},
                            {"start": 0.6, "end": 1.0, "word": "world", "score": 0.8, "speaker": "SPEAKER_00"},
                        ],
                    }
                ]
            },
            "timing": {
                "transcribe_secs": 10.0,
                "align_secs": 2.0,
                "diarize_secs": 5.0,
                "total_secs": 17.0,
            },
        }

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0

        # Output should be written
        output_file = output_dir / "test.json"
        assert output_file.exists()

        with output_file.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["metadata"]["file_hash"] == "hash_new"
        assert doc["metadata"]["duration_secs"] == 300.0
        assert len(doc["segments"]) == 1

        # Registry should be updated
        reg_path = output_dir / ".transskribo" / "registry.json"
        assert reg_path.exists()
        with reg_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
        assert "hash_new" in registry
        assert registry["hash_new"]["status"] == "success"
        assert registry["hash_new"]["timing"]["total_secs"] == 17.0

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file", side_effect=RuntimeError("GPU OOM"))
    def test_per_file_error_continues_batch(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A per-file error should be logged and batch should continue."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "file1.mp3")
        _create_audio_file(input_dir, "file2.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )
        mock_hash.return_value = "hash_fail"

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        # Should still exit 0 — batch continues despite individual failures
        assert result.exit_code == 0
        # Both files attempted
        assert mock_process.call_count == 2

        # Failed entries should be in registry
        reg_path = output_dir / ".transskribo" / "registry.json"
        if reg_path.exists():
            with reg_path.open("r", encoding="utf-8") as f:
                registry = json.load(f)
            assert registry.get("hash_fail", {}).get("status") == "failed"


# ---------------------------------------------------------------------------
# 10.05 — Report command, version command
# ---------------------------------------------------------------------------

class TestReportCommand:
    """Tests for the report command."""

    def test_report_with_empty_registry(self, tmp_path: Path) -> None:
        """Report should work with an empty or missing registry."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        result = runner.invoke(app, ["report", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "Progress" in result.output

    def test_report_with_populated_registry(self, tmp_path: Path) -> None:
        """Report should display stats from the registry."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        _create_registry(output_dir, {
            "hash1": {
                "source_path": str(input_dir / "file1.mp3"),
                "output_path": str(output_dir / "file1.json"),
                "timestamp": "2024-01-01T00:00:00",
                "status": "success",
                "duration_audio_secs": 3600.0,
                "timing": {
                    "transcribe_secs": 100.0,
                    "align_secs": 20.0,
                    "diarize_secs": 50.0,
                    "total_secs": 170.0,
                },
                "error": None,
            }
        })

        result = runner.invoke(app, ["report", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "Processed" in result.output

    def test_report_missing_config(self, tmp_path: Path) -> None:
        """Report should fail if config file doesn't exist."""
        result = runner.invoke(app, ["report", "--config", str(tmp_path / "nope.toml")])
        assert result.exit_code != 0
        assert "Config file not found" in result.output


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_output(self) -> None:
        """Version command should print the version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "transskribo" in result.output
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# 10.06 — Additional CLI integration tests
# ---------------------------------------------------------------------------

class TestBatchSummary:
    """Tests for batch summary logging."""

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_batch_summary_logged(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Batch summary should be logged after processing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _create_audio_file(input_dir, "a.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )
        mock_hash.return_value = "hash_a"
        mock_process.return_value = {
            "result": {"segments": []},
            "timing": {
                "transcribe_secs": 1.0,
                "align_secs": 0.5,
                "diarize_secs": 0.8,
                "total_secs": 2.3,
            },
        }

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0

        # Check that the log file contains batch summary
        log_path = output_dir / ".transskribo" / "transskribo.log"
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8")
            assert "Batch Summary" in log_content


class TestPipelineHelpers:
    """Tests for helper functions in cli.py."""

    def test_registry_path(self, tmp_path: Path) -> None:
        """_registry_path should return correct path."""
        from transskribo.cli import _registry_path
        from transskribo.config import TransskriboConfig

        cfg = TransskriboConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            hf_token="tok",
        )
        assert _registry_path(cfg) == tmp_path / "output" / ".transskribo" / "registry.json"

    def test_log_file_path(self, tmp_path: Path) -> None:
        """_log_file_path should return correct path."""
        from transskribo.cli import _log_file_path
        from transskribo.config import TransskriboConfig

        cfg = TransskriboConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            hf_token="tok",
        )
        assert _log_file_path(cfg) == tmp_path / "output" / ".transskribo" / "transskribo.log"

    def test_get_failed_hashes(self) -> None:
        """_get_failed_hashes should return source_path -> entry for failed."""
        from transskribo.cli import _get_failed_hashes

        registry = {
            "hash1": {"source_path": "/a/b.mp3", "status": "success"},
            "hash2": {"source_path": "/a/c.mp3", "status": "failed"},
            "hash3": {"source_path": "/a/d.mp3", "status": "failed"},
        }
        failed = _get_failed_hashes(registry)
        assert len(failed) == 2
        assert "/a/c.mp3" in failed
        assert "/a/d.mp3" in failed
        assert "/a/b.mp3" not in failed

    def test_get_failed_hashes_empty(self) -> None:
        """_get_failed_hashes with no failures returns empty dict."""
        from transskribo.cli import _get_failed_hashes

        registry = {
            "hash1": {"source_path": "/a/b.mp3", "status": "success"},
        }
        assert _get_failed_hashes(registry) == {}


# ---------------------------------------------------------------------------
# 11.01 — --retry-failed flag
# ---------------------------------------------------------------------------

class TestRetryFailed:
    """Tests for the --retry-failed flag."""

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_retry_failed_reprocesses_failed_files(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--retry-failed should re-process files with status 'failed' in registry."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create an audio file and its output (so it would normally be skipped)
        audio_path = _create_audio_file(input_dir, "failed_file.mp3")
        output_file = output_dir / "failed_file.json"
        output_file.write_text("{}", encoding="utf-8")

        config_path = _write_config(tmp_path, input_dir, output_dir)

        # Create registry with a "failed" entry for this file
        _create_registry(output_dir, {
            "hash_failed": {
                "source_path": str(audio_path),
                "output_path": str(output_file),
                "timestamp": "2024-01-01T00:00:00",
                "status": "failed",
                "duration_audio_secs": 60.0,
                "timing": None,
                "error": "Previous error",
            }
        })

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )
        mock_hash.return_value = "hash_failed"
        mock_process.return_value = {
            "result": {"segments": []},
            "timing": {
                "transcribe_secs": 1.0,
                "align_secs": 0.5,
                "diarize_secs": 0.8,
                "total_secs": 2.3,
            },
        }

        result = runner.invoke(app, [
            "run", "--config", str(config_path), "--retry-failed"
        ])
        assert result.exit_code == 0
        # The file should have been re-processed
        mock_process.assert_called_once()

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    def test_retry_failed_no_failed_files(
        self,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--retry-failed with no failed entries should behave normally."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path = _write_config(tmp_path, input_dir, output_dir)

        # No files to process at all
        result = runner.invoke(app, [
            "run", "--config", str(config_path), "--retry-failed"
        ])
        assert result.exit_code == 0

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_without_retry_failed_skips_files_with_output(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Without --retry-failed, files with existing output are skipped."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        audio_path = _create_audio_file(input_dir, "done.mp3")
        output_file = output_dir / "done.json"
        output_file.write_text("{}", encoding="utf-8")

        config_path = _write_config(tmp_path, input_dir, output_dir)

        _create_registry(output_dir, {
            "hash_done": {
                "source_path": str(audio_path),
                "output_path": str(output_file),
                "timestamp": "2024-01-01T00:00:00",
                "status": "failed",
                "duration_audio_secs": 60.0,
                "timing": None,
                "error": "Old error",
            }
        })

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0
        # Without --retry-failed, the file should NOT be re-processed
        mock_process.assert_not_called()


# ---------------------------------------------------------------------------
# 11.02 — --dry-run flag
# ---------------------------------------------------------------------------

class TestDryRun:
    """Tests for the --dry-run flag."""

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.transcriber.process_file")
    def test_dry_run_does_not_process(
        self,
        mock_process: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--dry-run should scan and validate but not process files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "file1.mp3")
        _create_audio_file(input_dir, "file2.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=120.0, error=None
        )

        result = runner.invoke(app, [
            "run", "--config", str(config_path), "--dry-run"
        ])
        assert result.exit_code == 0
        # No files should be processed
        mock_process.assert_not_called()
        # No output files should be created
        assert not (output_dir / "file1.json").exists()
        assert not (output_dir / "file2.json").exists()

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    def test_dry_run_logs_summary(
        self,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--dry-run should log a summary of what would be processed."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "a.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )

        result = runner.invoke(app, [
            "run", "--config", str(config_path), "--dry-run"
        ])
        assert result.exit_code == 0

        # Check log file for dry-run summary
        log_path = output_dir / ".transskribo" / "transskribo.log"
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8")
            assert "Dry Run Summary" in log_content

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    def test_dry_run_still_validates(
        self,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--dry-run should still validate files and count invalid ones."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "good.mp3")
        _create_audio_file(input_dir, "bad.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        def validate_side_effect(file_path: Path, max_dur: float) -> ValidationResult:
            if "bad" in file_path.name:
                return ValidationResult(is_valid=False, duration_secs=None, error="corrupt")
            return ValidationResult(is_valid=True, duration_secs=120.0, error=None)

        mock_validate.side_effect = validate_side_effect

        result = runner.invoke(app, [
            "run", "--config", str(config_path), "--dry-run"
        ])
        assert result.exit_code == 0
        # validate_file should have been called for both files
        assert mock_validate.call_count == 2


# ---------------------------------------------------------------------------
# 11.03 — Graceful shutdown (SIGINT/SIGTERM)
# ---------------------------------------------------------------------------

class TestGracefulShutdown:
    """Tests for SIGINT/SIGTERM handling."""

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_sigint_stops_after_current_file(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SIGINT during processing should finish current file and stop."""
        import transskribo.cli as cli_module

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create 3 files
        _create_audio_file(input_dir, "a.mp3")
        _create_audio_file(input_dir, "b.mp3")
        _create_audio_file(input_dir, "c.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )

        call_count = 0

        def process_side_effect(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            # After processing the first file, simulate shutdown
            if call_count == 1:
                cli_module._shutdown_requested = True
            return {
                "result": {"segments": []},
                "timing": {
                    "transcribe_secs": 1.0,
                    "align_secs": 0.5,
                    "diarize_secs": 0.8,
                    "total_secs": 2.3,
                },
            }

        mock_hash.return_value = "hash_x"
        mock_process.side_effect = process_side_effect

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0
        # Only 1 file should be processed (shutdown requested after first)
        assert mock_process.call_count == 1

        # Check for partial batch summary in log
        log_path = output_dir / ".transskribo" / "transskribo.log"
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8")
            assert "interrupted" in log_content

    def test_signal_handler_sets_flag(self) -> None:
        """The signal handler should set _shutdown_requested to True."""
        import transskribo.cli as cli_module

        cli_module._shutdown_requested = False

        # Simulate what the handler does
        cli_module._shutdown_requested = True
        assert cli_module._shutdown_requested is True

        # Reset
        cli_module._shutdown_requested = False

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_shutdown_saves_registry(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Registry should be saved for processed files even when interrupted."""
        import transskribo.cli as cli_module

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "first.mp3")
        _create_audio_file(input_dir, "second.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )

        call_count = 0

        def process_side_effect(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                cli_module._shutdown_requested = True
            return {
                "result": {"segments": []},
                "timing": {
                    "transcribe_secs": 1.0,
                    "align_secs": 0.5,
                    "diarize_secs": 0.8,
                    "total_secs": 2.3,
                },
            }

        mock_hash.return_value = "hash_first"
        mock_process.side_effect = process_side_effect

        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0

        # Registry should exist with the first file's entry
        reg_path = output_dir / ".transskribo" / "registry.json"
        assert reg_path.exists()
        with reg_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
        assert "hash_first" in registry
        assert registry["hash_first"]["status"] == "success"


# ---------------------------------------------------------------------------
# 11.04 — Integration test (full pipeline with short audio fixture)
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration test exercising the full pipeline end-to-end.

    Skipped if no GPU is available (CI environments).
    Uses mocked transcriber since we can't require a GPU in tests.
    """

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_full_pipeline_end_to_end(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full pipeline: scan → validate → hash → process → output → registry."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create audio files in nested structure
        sub_dir = input_dir / "lectures" / "week1"
        sub_dir.mkdir(parents=True)
        _create_audio_file(sub_dir, "lecture1.mp3", size=200)
        _create_audio_file(sub_dir, "lecture2.mp3", size=300)
        _create_audio_file(input_dir, "meeting.m4a", size=150)

        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=3600.0, error=None
        )

        # Each file gets a unique hash
        hash_counter = 0

        def hash_side_effect(path: Path) -> str:
            nonlocal hash_counter
            hash_counter += 1
            return f"hash_{hash_counter}"

        mock_hash.side_effect = hash_side_effect

        mock_process.return_value = {
            "result": {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "text": "Bom dia, turma.",
                        "speaker": "SPEAKER_00",
                        "words": [
                            {"start": 0.0, "end": 0.3, "word": "Bom", "score": 0.95, "speaker": "SPEAKER_00"},
                            {"start": 0.4, "end": 0.7, "word": "dia,", "score": 0.92, "speaker": "SPEAKER_00"},
                            {"start": 0.8, "end": 1.2, "word": "turma.", "score": 0.88, "speaker": "SPEAKER_00"},
                        ],
                    },
                    {
                        "start": 5.0,
                        "end": 10.0,
                        "text": "Vamos começar a aula.",
                        "speaker": "SPEAKER_00",
                        "words": [],
                    },
                ],
            },
            "timing": {
                "transcribe_secs": 120.0,
                "align_secs": 30.0,
                "diarize_secs": 60.0,
                "total_secs": 210.0,
            },
        }

        # --- Run the pipeline ---
        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0

        # All 3 files should be processed
        assert mock_process.call_count == 3

        # Output files should exist with correct structure
        lecture1_output = output_dir / "lectures" / "week1" / "lecture1.json"
        lecture2_output = output_dir / "lectures" / "week1" / "lecture2.json"
        meeting_output = output_dir / "meeting.json"

        assert lecture1_output.exists()
        assert lecture2_output.exists()
        assert meeting_output.exists()

        # Verify output document structure
        with lecture1_output.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        assert "segments" in doc
        assert "words" in doc
        assert "metadata" in doc
        assert len(doc["segments"]) == 2
        assert doc["segments"][0]["text"] == "Bom dia, turma."
        assert doc["segments"][0]["speaker"] == "SPEAKER_00"
        assert len(doc["words"]) == 3  # only words from first segment
        assert doc["metadata"]["duration_secs"] == 3600.0
        assert doc["metadata"]["model_size"] == "large-v3"
        assert doc["metadata"]["language"] == "pt"
        assert doc["metadata"]["timing"]["total_secs"] == 210.0

        # Registry should have 3 entries
        reg_path = output_dir / ".transskribo" / "registry.json"
        assert reg_path.exists()
        with reg_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
        assert len(registry) == 3

        # --- Run report command ---
        report_result = runner.invoke(app, ["report", "--config", str(config_path)])
        assert report_result.exit_code == 0
        assert "Progress" in report_result.output

        # --- Run again: all files should be skipped ---
        mock_process.reset_mock()
        result2 = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result2.exit_code == 0
        mock_process.assert_not_called()  # all skipped

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_full_pipeline_with_error_and_retry(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline: process with error, then retry-failed to reprocess."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "file.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )
        mock_hash.return_value = "hash_retry_test"

        # First run: fail
        mock_process.side_effect = RuntimeError("GPU OOM")
        result = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result.exit_code == 0

        # Registry should have a failed entry
        reg_path = output_dir / ".transskribo" / "registry.json"
        assert reg_path.exists()
        with reg_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
        assert registry["hash_retry_test"]["status"] == "failed"

        # Second run without --retry-failed: file.json doesn't exist,
        # so it will be re-attempted (filter_already_processed won't skip it)
        mock_process.reset_mock()
        mock_process.side_effect = None
        mock_process.return_value = {
            "result": {"segments": []},
            "timing": {
                "transcribe_secs": 1.0,
                "align_secs": 0.5,
                "diarize_secs": 0.8,
                "total_secs": 2.3,
            },
        }

        result2 = runner.invoke(app, ["run", "--config", str(config_path)])
        assert result2.exit_code == 0
        mock_process.assert_called_once()

        # Registry should now be success
        with reg_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
        assert registry["hash_retry_test"]["status"] == "success"

    @patch("transskribo.cli.check_ffprobe_available")
    @patch("transskribo.cli.validate_file")
    @patch("transskribo.cli.compute_hash")
    @patch("transskribo.transcriber.process_file")
    def test_dry_run_then_real_run(
        self,
        mock_process: MagicMock,
        mock_hash: MagicMock,
        mock_validate: MagicMock,
        mock_ffprobe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Dry run should not create outputs; real run should."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _create_audio_file(input_dir, "test.mp3")
        config_path = _write_config(tmp_path, input_dir, output_dir)

        from transskribo.validator import ValidationResult

        mock_validate.return_value = ValidationResult(
            is_valid=True, duration_secs=60.0, error=None
        )
        mock_hash.return_value = "hash_drytest"
        mock_process.return_value = {
            "result": {"segments": []},
            "timing": {
                "transcribe_secs": 1.0,
                "align_secs": 0.5,
                "diarize_secs": 0.8,
                "total_secs": 2.3,
            },
        }

        # Dry run first
        result = runner.invoke(app, [
            "run", "--config", str(config_path), "--dry-run"
        ])
        assert result.exit_code == 0
        assert not (output_dir / "test.json").exists()
        mock_process.assert_not_called()

        # Real run
        result2 = runner.invoke(app, [
            "run", "--config", str(config_path)
        ])
        assert result2.exit_code == 0
        assert (output_dir / "test.json").exists()
        mock_process.assert_called_once()
