"""Tests for reporter statistics and formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from transskribo.reporter import (
    compute_statistics,
    compute_timing_statistics,
    format_report,
    per_directory_breakdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_UNSET: Any = object()


def _make_entry(
    source: str = "/input/file.mp3",
    output: str = "/output/file.json",
    status: str = "success",
    duration: float | None = 120.0,
    timing: dict[str, float] | None | Any = _UNSET,
    error: str | None = None,
) -> dict[str, Any]:
    """Helper to build a registry entry dict."""
    if timing is _UNSET:
        if status == "success":
            timing = {
                "transcribe_secs": 10.0,
                "align_secs": 2.0,
                "diarize_secs": 5.0,
                "total_secs": 20.0,
            }
        else:
            timing = None
    entry: dict[str, Any] = {
        "source_path": source,
        "output_path": output,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "status": status,
        "duration_audio_secs": duration,
        "timing": timing,
        "error": error,
    }
    return entry


@pytest.fixture
def sample_registry() -> dict[str, Any]:
    """Registry with 3 successful and 1 failed entry."""
    return {
        "hash1": _make_entry(
            source="/input/lectures/lec1.mp3",
            output="/output/lectures/lec1.json",
            duration=3600.0,
            timing={"transcribe_secs": 60.0, "align_secs": 10.0, "diarize_secs": 30.0, "total_secs": 110.0},
        ),
        "hash2": _make_entry(
            source="/input/lectures/lec2.mp3",
            output="/output/lectures/lec2.json",
            duration=1800.0,
            timing={"transcribe_secs": 30.0, "align_secs": 5.0, "diarize_secs": 15.0, "total_secs": 55.0},
        ),
        "hash3": _make_entry(
            source="/input/meetings/meet1.mp3",
            output="/output/meetings/meet1.json",
            duration=900.0,
            timing={"transcribe_secs": 15.0, "align_secs": 3.0, "diarize_secs": 8.0, "total_secs": 28.0},
        ),
        "hash4": _make_entry(
            source="/input/meetings/meet2.mp3",
            output="/output/meetings/meet2.json",
            status="failed",
            duration=None,
            timing=None,
            error="OOM error",
        ),
    }


@pytest.fixture
def empty_registry() -> dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# compute_statistics tests
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    def test_empty_registry(self, empty_registry: dict[str, Any]) -> None:
        stats = compute_statistics(empty_registry)
        assert stats["total_files"] == 0
        assert stats["processed"] == 0
        assert stats["failed"] == 0
        assert stats["remaining"] == 0
        assert stats["total_audio_duration_processed"] == 0.0

    def test_counts_processed_and_failed(self, sample_registry: dict[str, Any]) -> None:
        stats = compute_statistics(sample_registry)
        assert stats["processed"] == 3
        assert stats["failed"] == 1

    def test_total_audio_duration(self, sample_registry: dict[str, Any]) -> None:
        stats = compute_statistics(sample_registry)
        assert stats["total_audio_duration_processed"] == 3600.0 + 1800.0 + 900.0

    def test_with_input_dir(self, tmp_path: Path) -> None:
        """When input_dir is given, count total files by scanning."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create some audio files
        (input_dir / "a.mp3").write_bytes(b"fake")
        (input_dir / "b.wav").write_bytes(b"fake")
        (input_dir / "c.txt").write_bytes(b"ignore")

        registry: dict[str, Any] = {
            "h1": _make_entry(
                source=str(input_dir / "a.mp3"),
                output=str(output_dir / "a.json"),
            ),
        }

        stats = compute_statistics(registry, input_dir=input_dir, output_dir=output_dir)
        assert stats["total_files"] == 2  # a.mp3 and b.wav
        assert stats["processed"] == 1
        assert stats["remaining"] == 1  # 2 total - 1 processed - 0 failed

    def test_remaining_calculation(self) -> None:
        """Remaining = total - processed - failed."""
        registry = {
            "h1": _make_entry(status="success"),
            "h2": _make_entry(status="failed", duration=None, timing=None),
        }
        # Without input_dir, total_files is 0 so remaining is 0
        stats = compute_statistics(registry)
        assert stats["remaining"] == 0

    def test_only_failed(self) -> None:
        registry = {
            "h1": _make_entry(status="failed", duration=None, timing=None, error="err"),
        }
        stats = compute_statistics(registry)
        assert stats["processed"] == 0
        assert stats["failed"] == 1
        assert stats["total_audio_duration_processed"] == 0.0


# ---------------------------------------------------------------------------
# compute_timing_statistics tests
# ---------------------------------------------------------------------------


class TestComputeTimingStatistics:
    def test_empty_registry(self, empty_registry: dict[str, Any]) -> None:
        ts = compute_timing_statistics(empty_registry)
        assert ts["stages"] == {}
        assert ts["avg_total_secs"] == 0.0
        assert ts["speed_ratio"] == 0.0

    def test_single_entry(self) -> None:
        registry = {
            "h1": _make_entry(
                duration=600.0,
                timing={"transcribe_secs": 10.0, "align_secs": 2.0, "diarize_secs": 5.0, "total_secs": 20.0},
            ),
        }
        ts = compute_timing_statistics(registry)
        assert ts["stages"]["transcribe_secs"]["avg"] == 10.0
        assert ts["stages"]["align_secs"]["avg"] == 2.0
        assert ts["stages"]["diarize_secs"]["avg"] == 5.0
        assert ts["avg_total_secs"] == 20.0
        assert ts["speed_ratio"] == 600.0 / 20.0

    def test_multiple_entries_averages(self, sample_registry: dict[str, Any]) -> None:
        ts = compute_timing_statistics(sample_registry)
        # Only 3 success entries have timing
        stages = ts["stages"]

        # transcribe: 60, 30, 15 -> avg=35, min=15, max=60
        assert stages["transcribe_secs"]["avg"] == pytest.approx(35.0)
        assert stages["transcribe_secs"]["min"] == 15.0
        assert stages["transcribe_secs"]["max"] == 60.0

        # align: 10, 5, 3 -> avg=6.0, min=3, max=10
        assert stages["align_secs"]["avg"] == pytest.approx(6.0)
        assert stages["align_secs"]["min"] == 3.0
        assert stages["align_secs"]["max"] == 10.0

        # total: 110, 55, 28 -> avg=64.33
        assert ts["avg_total_secs"] == pytest.approx(64.333, rel=0.01)

    def test_skips_failed_entries(self) -> None:
        registry = {
            "h1": _make_entry(status="success", duration=100.0),
            "h2": _make_entry(status="failed", duration=None, timing=None),
        }
        ts = compute_timing_statistics(registry)
        assert ts["total_times_count"] == 1

    def test_speed_ratio(self) -> None:
        """Speed ratio = total audio duration / total processing time."""
        registry = {
            "h1": _make_entry(duration=600.0, timing={"transcribe_secs": 5.0, "align_secs": 1.0, "diarize_secs": 2.0, "total_secs": 10.0}),
            "h2": _make_entry(duration=300.0, timing={"transcribe_secs": 3.0, "align_secs": 1.0, "diarize_secs": 1.0, "total_secs": 5.0}),
        }
        ts = compute_timing_statistics(registry)
        # (600 + 300) / (10 + 5) = 900 / 15 = 60.0
        assert ts["speed_ratio"] == pytest.approx(60.0)

    def test_no_timing_data(self) -> None:
        """Entries without timing are ignored."""
        registry = {
            "h1": _make_entry(duration=100.0, timing=None),
        }
        ts = compute_timing_statistics(registry)
        assert ts["stages"] == {}
        assert ts["avg_total_secs"] == 0.0

    def test_no_duration_for_speed_ratio(self) -> None:
        """If no audio durations, speed ratio is 0."""
        registry = {
            "h1": _make_entry(
                duration=None,
                timing={"transcribe_secs": 5.0, "align_secs": 1.0, "diarize_secs": 2.0, "total_secs": 10.0},
            ),
        }
        ts = compute_timing_statistics(registry)
        assert ts["speed_ratio"] == 0.0


# ---------------------------------------------------------------------------
# per_directory_breakdown tests
# ---------------------------------------------------------------------------


class TestPerDirectoryBreakdown:
    def test_empty_registry(self, empty_registry: dict[str, Any]) -> None:
        bd = per_directory_breakdown(empty_registry)
        assert bd == {}

    def test_groups_by_top_level_dir(self, sample_registry: dict[str, Any]) -> None:
        bd = per_directory_breakdown(sample_registry, input_dir=Path("/input"))
        assert "lectures" in bd
        assert "meetings" in bd
        assert bd["lectures"]["processed"] == 2
        assert bd["meetings"]["processed"] == 1
        assert bd["meetings"]["failed"] == 1

    def test_audio_duration_by_dir(self, sample_registry: dict[str, Any]) -> None:
        bd = per_directory_breakdown(sample_registry, input_dir=Path("/input"))
        assert bd["lectures"]["total_audio_secs"] == 3600.0 + 1800.0
        assert bd["meetings"]["total_audio_secs"] == 900.0

    def test_avg_processing_time(self, sample_registry: dict[str, Any]) -> None:
        bd = per_directory_breakdown(sample_registry, input_dir=Path("/input"))
        # lectures: (110 + 55) / 2 = 82.5
        assert bd["lectures"]["avg_processing_secs"] == pytest.approx(82.5)
        # meetings: only 28 from success (failed has no timing)
        assert bd["meetings"]["avg_processing_secs"] == pytest.approx(28.0)

    def test_root_level_files(self) -> None:
        """Files directly in input_dir go to '.' directory."""
        registry = {
            "h1": _make_entry(source="/input/file.mp3", duration=60.0),
        }
        bd = per_directory_breakdown(registry, input_dir=Path("/input"))
        assert "." in bd
        assert bd["."]["processed"] == 1

    def test_without_input_dir(self) -> None:
        """Without input_dir, uses parent directory name."""
        registry = {
            "h1": _make_entry(source="/some/path/lectures/file.mp3"),
        }
        bd = per_directory_breakdown(registry)
        assert "lectures" in bd

    def test_sorted_output(self) -> None:
        registry = {
            "h1": _make_entry(source="/input/zzz/a.mp3"),
            "h2": _make_entry(source="/input/aaa/b.mp3"),
        }
        bd = per_directory_breakdown(registry, input_dir=Path("/input"))
        keys = list(bd.keys())
        assert keys == ["aaa", "zzz"]


# ---------------------------------------------------------------------------
# format_report tests
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_produces_string_output(self) -> None:
        report = format_report(
            stats={"total_files": 10, "processed": 5, "failed": 1, "remaining": 4, "total_audio_duration_processed": 3600.0},
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_progress_data(self) -> None:
        report = format_report(
            stats={"total_files": 100, "processed": 50, "failed": 2, "remaining": 48, "total_audio_duration_processed": 7200.0},
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        assert "100" in report
        assert "50" in report
        assert "48" in report

    def test_contains_timing_data(self) -> None:
        report = format_report(
            stats={"total_files": 10, "processed": 5, "failed": 0, "remaining": 5, "total_audio_duration_processed": 0.0},
            timing_stats={
                "stages": {
                    "transcribe_secs": {"avg": 30.0, "min": 10.0, "max": 50.0},
                    "align_secs": {"avg": 5.0, "min": 2.0, "max": 8.0},
                    "diarize_secs": {"avg": 15.0, "min": 5.0, "max": 25.0},
                },
                "avg_total_secs": 50.0,
                "speed_ratio": 12.0,
            },
            breakdown={},
        )
        assert "Transcribe" in report
        assert "Align" in report
        assert "Diarize" in report

    def test_contains_breakdown_data(self) -> None:
        report = format_report(
            stats={"total_files": 10, "processed": 5, "failed": 0, "remaining": 5, "total_audio_duration_processed": 0.0},
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={
                "lectures": {"processed": 3, "failed": 0, "total_audio_secs": 1800.0, "avg_processing_secs": 30.0},
                "meetings": {"processed": 2, "failed": 0, "total_audio_secs": 900.0, "avg_processing_secs": 20.0},
            },
        )
        assert "lectures" in report
        assert "meetings" in report

    def test_empty_stats(self) -> None:
        report = format_report(
            stats={"total_files": 0, "processed": 0, "failed": 0, "remaining": 0, "total_audio_duration_processed": 0.0},
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        assert "Progress" in report

    def test_eta_shown_when_applicable(self) -> None:
        report = format_report(
            stats={"total_files": 100, "processed": 50, "failed": 0, "remaining": 50, "total_audio_duration_processed": 0.0},
            timing_stats={"stages": {}, "avg_total_secs": 60.0, "speed_ratio": 0.0},
            breakdown={},
        )
        # 50 remaining * 60s avg = 3000s = 50m 0s
        assert "50m" in report

    def test_duration_formatting(self) -> None:
        report = format_report(
            stats={"total_files": 1, "processed": 1, "failed": 0, "remaining": 0, "total_audio_duration_processed": 7200.0},
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        # 7200s = 2h 0m
        assert "2h" in report

    def test_enrichment_progress_shown(self) -> None:
        report = format_report(
            stats={
                "total_files": 10, "processed": 5, "failed": 0, "remaining": 5,
                "total_audio_duration_processed": 0.0,
                "enriched": 3, "not_enriched": 2,
            },
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        assert "Enriched" in report
        assert "3 / 5" in report

    def test_enrichment_not_shown_when_zero(self) -> None:
        report = format_report(
            stats={
                "total_files": 10, "processed": 5, "failed": 0, "remaining": 5,
                "total_audio_duration_processed": 0.0,
                "enriched": 0, "not_enriched": 0,
            },
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        # "Enriched" should not appear in the row if there are no enrichable files
        # The word "Enriched" as a row label won't appear
        assert "0 / 0" not in report

    def test_export_docx_progress_shown(self) -> None:
        report = format_report(
            stats={
                "total_files": 10, "processed": 5, "failed": 0, "remaining": 5,
                "total_audio_duration_processed": 0.0,
                "enriched": 4, "not_enriched": 1,
                "exported_docx": 3,
            },
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        assert "Exported (docx)" in report
        assert "3 / 4" in report

    def test_transcribed_progress_shown(self) -> None:
        report = format_report(
            stats={
                "total_files": 10, "processed": 7, "failed": 0, "remaining": 3,
                "total_audio_duration_processed": 0.0,
                "enriched": 0, "not_enriched": 0,
            },
            timing_stats={"stages": {}, "avg_total_secs": 0.0, "speed_ratio": 0.0},
            breakdown={},
        )
        assert "Transcribed" in report
        assert "7 / 10" in report


# ---------------------------------------------------------------------------
# Enrichment stats tests
# ---------------------------------------------------------------------------


class TestEnrichmentStatistics:
    def test_counts_enriched_and_not_enriched(self, tmp_path: Path) -> None:
        """compute_statistics should count enriched vs not-enriched result JSONs."""
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create enriched result JSON
        enriched = {
            "segments": [{"text": "hello", "speaker": "S1"}],
            "metadata": {"source_file": "a.mp3"},
            "title": "Test",
            "keywords": ["k"],
            "summary": "s",
            "concepts": {"c": "d"},
        }
        (output_dir / "enriched.json").write_text(json.dumps(enriched), encoding="utf-8")

        # Create not-enriched result JSON
        not_enriched = {
            "segments": [{"text": "hello", "speaker": "S1"}],
            "metadata": {"source_file": "b.mp3"},
        }
        (output_dir / "not_enriched.json").write_text(json.dumps(not_enriched), encoding="utf-8")

        # Create non-transskribo JSON (should be ignored)
        other = {"key": "value"}
        (output_dir / "other.json").write_text(json.dumps(other), encoding="utf-8")

        # Create audio files in input for total count
        (input_dir / "a.mp3").write_bytes(b"fake")
        (input_dir / "b.mp3").write_bytes(b"fake")

        stats = compute_statistics({}, input_dir=input_dir, output_dir=output_dir)
        assert stats["enriched"] == 1
        assert stats["not_enriched"] == 1

    def test_enrichment_counts_skip_transskribo_dir(self, tmp_path: Path) -> None:
        """Files in .transskribo/ should be skipped for enrichment counting."""
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Put a JSON in .transskribo/ — it should be ignored
        ts_dir = output_dir / ".transskribo"
        ts_dir.mkdir()
        registry_like = {"segments": [], "metadata": {}}
        (ts_dir / "registry.json").write_text(json.dumps(registry_like), encoding="utf-8")

        stats = compute_statistics({}, output_dir=output_dir)
        assert stats["enriched"] == 0
        assert stats["not_enriched"] == 0

    def test_enrichment_counts_without_output_dir(self) -> None:
        """Without output_dir, enrichment counts should be 0."""
        stats = compute_statistics({})
        assert stats["enriched"] == 0
        assert stats["not_enriched"] == 0

    def test_mixed_enrichment_states(self, tmp_path: Path) -> None:
        """Test with multiple files in various enrichment states."""
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        base_doc = {"segments": [{"text": "t"}], "metadata": {"src": "x"}}

        # 3 enriched
        for i in range(3):
            doc = {
                **base_doc,
                "title": f"Title {i}",
                "keywords": [f"k{i}"],
                "summary": f"s{i}",
                "concepts": {f"c{i}": f"d{i}"},
            }
            (output_dir / f"enriched_{i}.json").write_text(json.dumps(doc), encoding="utf-8")

        # 2 not enriched
        for i in range(2):
            (output_dir / f"plain_{i}.json").write_text(json.dumps(base_doc), encoding="utf-8")

        stats = compute_statistics({}, output_dir=output_dir)
        assert stats["enriched"] == 3
        assert stats["not_enriched"] == 2


class TestExportStatistics:
    def test_counts_exported_docx(self, tmp_path: Path) -> None:
        """compute_statistics should count .docx files alongside enriched JSONs."""
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        enriched_doc = {
            "segments": [{"text": "hello", "speaker": "S1"}],
            "metadata": {"source_file": "a.mp3"},
            "title": "Test",
            "keywords": ["k"],
            "summary": "s",
            "concepts": {"c": "d"},
        }

        # Enriched with docx export
        (output_dir / "exported.json").write_text(json.dumps(enriched_doc), encoding="utf-8")
        (output_dir / "exported.docx").touch()

        # Enriched without docx export
        (output_dir / "not_exported.json").write_text(json.dumps(enriched_doc), encoding="utf-8")

        stats = compute_statistics({}, output_dir=output_dir)
        assert stats["enriched"] == 2
        assert stats["exported_docx"] == 1

    def test_non_enriched_not_counted_for_export(self, tmp_path: Path) -> None:
        """Non-enriched files should not count as exported even if .docx exists."""
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        not_enriched = {
            "segments": [{"text": "hello"}],
            "metadata": {"source_file": "a.mp3"},
        }
        (output_dir / "plain.json").write_text(json.dumps(not_enriched), encoding="utf-8")
        # Even if a stray .docx exists, it shouldn't count
        (output_dir / "plain.docx").touch()

        stats = compute_statistics({}, output_dir=output_dir)
        assert stats["exported_docx"] == 0

    def test_export_stats_without_output_dir(self) -> None:
        """Without output_dir, export counts should be 0."""
        stats = compute_statistics({})
        assert stats["exported_docx"] == 0

    def test_export_stats_skip_transskribo_dir(self, tmp_path: Path) -> None:
        """Files in .transskribo/ should be skipped for export counting."""
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        ts_dir = output_dir / ".transskribo"
        ts_dir.mkdir()
        enriched_doc = {
            "segments": [], "metadata": {},
            "title": "t", "keywords": [], "summary": "s", "concepts": {},
        }
        (ts_dir / "data.json").write_text(json.dumps(enriched_doc), encoding="utf-8")
        (ts_dir / "data.docx").touch()

        stats = compute_statistics({}, output_dir=output_dir)
        assert stats["exported_docx"] == 0
