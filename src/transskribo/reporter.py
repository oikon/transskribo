"""Summary reports and independent statistics."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from transskribo.scanner import scan_directory

ENRICHMENT_KEYS = ("title", "keywords", "summary", "concepts")


def compute_statistics(
    registry: dict[str, Any],
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Calculate overall processing statistics from the registry.

    Args:
        registry: The hash registry (hash -> entry dict).
        input_dir: If provided, scan to count total files in input.
        output_dir: Required with input_dir to build output paths for scanning.

    Returns:
        Dict with keys: total_files, processed, failed, skipped, duplicates,
        remaining, total_audio_duration_processed, total_audio_duration_discovered.
    """
    processed = 0
    failed = 0
    duplicates = 0
    total_audio_processed = 0.0

    # Track unique source paths to count duplicates
    seen_sources: set[str] = set()

    for entry in registry.values():
        status = entry.get("status", "")
        source = entry.get("source_path", "")
        duration = entry.get("duration_audio_secs")

        if status == "success":
            processed += 1
            if duration is not None:
                total_audio_processed += duration
        elif status == "failed":
            failed += 1

        if source:
            seen_sources.add(source)

    # Scan input directory for total count if provided
    total_files = 0
    total_audio_discovered = 0.0
    if input_dir is not None and output_dir is not None:
        all_files = scan_directory(input_dir, output_dir)
        total_files = len(all_files)

    remaining = max(0, total_files - processed - failed) if total_files > 0 else 0

    # Count enrichment and export status by scanning result JSONs in output_dir
    enriched = 0
    not_enriched = 0
    exported_docx = 0
    if output_dir is not None and output_dir.exists():
        enriched, not_enriched, exported_docx = _count_enrichment_and_exports(output_dir)

    return {
        "total_files": total_files,
        "processed": processed,
        "failed": failed,
        "skipped": 0,
        "duplicates": duplicates,
        "remaining": remaining,
        "total_audio_duration_processed": total_audio_processed,
        "total_audio_duration_discovered": total_audio_discovered,
        "enriched": enriched,
        "not_enriched": not_enriched,
        "exported_docx": exported_docx,
    }


def _count_enrichment_and_exports(output_dir: Path) -> tuple[int, int, int]:
    """Count enriched vs not-enriched result JSONs and exported artifacts in output_dir."""
    enriched = 0
    not_enriched = 0
    exported_docx = 0
    transskribo_dir = output_dir / ".transskribo"

    for json_path in output_dir.rglob("*.json"):
        # Skip files inside .transskribo/
        try:
            json_path.relative_to(transskribo_dir)
            continue
        except ValueError:
            pass

        try:
            with json_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Only count transskribo result files
        if "segments" not in doc or "metadata" not in doc:
            continue

        if all(key in doc for key in ENRICHMENT_KEYS):
            enriched += 1
            # Check if corresponding .docx exists
            docx_path = json_path.with_suffix(".docx")
            if docx_path.exists():
                exported_docx += 1
        else:
            not_enriched += 1

    return enriched, not_enriched, exported_docx


def compute_timing_statistics(registry: dict[str, Any]) -> dict[str, Any]:
    """Calculate timing statistics from per-file timing data in the registry.

    Returns:
        Dict with per-stage avg/min/max, average total, speed ratio, and ETA fields.
        Returns empty sub-dicts if no timing data is available.
    """
    stage_names = ["transcribe_secs", "align_secs", "diarize_secs"]
    stage_values: dict[str, list[float]] = {s: [] for s in stage_names}
    total_times: list[float] = []
    audio_durations: list[float] = []
    processing_times: list[float] = []

    for entry in registry.values():
        if entry.get("status") != "success":
            continue
        timing = entry.get("timing")
        if timing is None:
            continue

        for stage in stage_names:
            val = timing.get(stage)
            if val is not None:
                stage_values[stage].append(val)

        total = timing.get("total_secs")
        if total is not None:
            total_times.append(total)
            processing_times.append(total)

        duration = entry.get("duration_audio_secs")
        if duration is not None:
            audio_durations.append(duration)

    stages: dict[str, dict[str, float]] = {}
    for stage in stage_names:
        values = stage_values[stage]
        if values:
            stages[stage] = {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    avg_total = sum(total_times) / len(total_times) if total_times else 0.0

    # Speed ratio: sum(audio_duration) / sum(processing_time)
    speed_ratio = 0.0
    if processing_times and audio_durations:
        total_audio = sum(audio_durations)
        total_proc = sum(processing_times)
        if total_proc > 0:
            speed_ratio = total_audio / total_proc

    return {
        "stages": stages,
        "avg_total_secs": avg_total,
        "speed_ratio": speed_ratio,
        "total_times_count": len(total_times),
    }


def per_directory_breakdown(
    registry: dict[str, Any],
    input_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Group stats and timing by top-level subdirectory.

    The top-level subdirectory is determined from the source_path relative
    to input_dir. If input_dir is None, uses the first path component
    of source_path.

    Returns:
        Dict mapping directory name to stats dict with processed, failed,
        total_audio_secs, and timing averages.
    """
    dirs: dict[str, dict[str, Any]] = {}

    for entry in registry.values():
        source = entry.get("source_path", "")
        if not source:
            continue

        source_path = Path(source)

        # Determine top-level subdirectory
        if input_dir is not None:
            try:
                relative = source_path.relative_to(input_dir)
                parts = relative.parts
                dir_name = parts[0] if len(parts) > 1 else "."
            except ValueError:
                dir_name = source_path.parent.name or "."
        else:
            dir_name = source_path.parent.name or "."

        if dir_name not in dirs:
            dirs[dir_name] = {
                "processed": 0,
                "failed": 0,
                "total_audio_secs": 0.0,
                "timing_totals": [],
            }

        bucket = dirs[dir_name]
        status = entry.get("status", "")

        if status == "success":
            bucket["processed"] += 1
            duration = entry.get("duration_audio_secs")
            if duration is not None:
                bucket["total_audio_secs"] += duration
        elif status == "failed":
            bucket["failed"] += 1

        timing = entry.get("timing")
        if timing is not None and "total_secs" in timing:
            bucket["timing_totals"].append(timing["total_secs"])

    # Compute averages
    result: dict[str, dict[str, Any]] = {}
    for dir_name, bucket in sorted(dirs.items()):
        totals = bucket.pop("timing_totals")
        bucket["avg_processing_secs"] = sum(totals) / len(totals) if totals else 0.0
        result[dir_name] = bucket

    return result


def format_report(
    stats: dict[str, Any],
    timing_stats: dict[str, Any],
    breakdown: dict[str, dict[str, Any]],
) -> str:
    """Render a formatted report using rich tables.

    Args:
        stats: Output from compute_statistics().
        timing_stats: Output from compute_timing_statistics().
        breakdown: Output from per_directory_breakdown().

    Returns:
        Formatted string suitable for printing.
    """
    console = Console(file=StringIO(), force_terminal=False, width=100)

    # --- Progress section ---
    progress_table = Table(title="Progress", show_header=True, header_style="bold")
    progress_table.add_column("Metric", style="cyan")
    progress_table.add_column("Value", justify="right")

    progress_table.add_row("Total files", str(stats.get("total_files", 0)))
    progress_table.add_row("Processed", str(stats.get("processed", 0)))
    progress_table.add_row("Failed", str(stats.get("failed", 0)))
    progress_table.add_row("Remaining", str(stats.get("remaining", 0)))

    total_audio = stats.get("total_audio_duration_processed", 0.0)
    progress_table.add_row("Audio processed", _format_duration(total_audio))

    # Pipeline stage progress
    transcribed = stats.get("processed", 0)
    total = stats.get("total_files", 0)
    enriched = stats.get("enriched", 0)
    not_enriched = stats.get("not_enriched", 0)
    enrich_total = enriched + not_enriched
    exported_docx = stats.get("exported_docx", 0)

    if total > 0:
        progress_table.add_row("Transcribed", f"{transcribed} / {total}")
    if enrich_total > 0:
        progress_table.add_row("Enriched", f"{enriched} / {enrich_total}")
    if enriched > 0:
        progress_table.add_row("Exported (docx)", f"{exported_docx} / {enriched}")

    # ETA
    avg_total = timing_stats.get("avg_total_secs", 0.0)
    remaining_count = stats.get("remaining", 0)
    if avg_total > 0 and remaining_count > 0:
        eta_secs = avg_total * remaining_count
        progress_table.add_row("Estimated remaining time", _format_duration(eta_secs))

    console.print(progress_table)

    # --- Timing section ---
    stages = timing_stats.get("stages", {})
    if stages:
        timing_table = Table(title="Timing (per file)", show_header=True, header_style="bold")
        timing_table.add_column("Stage", style="cyan")
        timing_table.add_column("Avg", justify="right")
        timing_table.add_column("Min", justify="right")
        timing_table.add_column("Max", justify="right")

        stage_labels = {
            "transcribe_secs": "Transcribe",
            "align_secs": "Align",
            "diarize_secs": "Diarize",
        }

        for stage_key, label in stage_labels.items():
            if stage_key in stages:
                s = stages[stage_key]
                timing_table.add_row(
                    label,
                    f"{s['avg']:.1f}s",
                    f"{s['min']:.1f}s",
                    f"{s['max']:.1f}s",
                )

        avg_total = timing_stats.get("avg_total_secs", 0.0)
        if avg_total > 0:
            timing_table.add_row("Total", f"{avg_total:.1f}s", "", "")

        speed = timing_stats.get("speed_ratio", 0.0)
        if speed > 0:
            timing_table.add_row("Speed ratio", f"{speed:.1f}x", "", "")

        console.print(timing_table)

    # --- Per-directory section ---
    if breakdown:
        dir_table = Table(title="Per-Directory Breakdown", show_header=True, header_style="bold")
        dir_table.add_column("Directory", style="cyan")
        dir_table.add_column("Processed", justify="right")
        dir_table.add_column("Failed", justify="right")
        dir_table.add_column("Audio", justify="right")
        dir_table.add_column("Avg Time", justify="right")

        for dir_name, dir_stats in breakdown.items():
            dir_table.add_row(
                dir_name,
                str(dir_stats.get("processed", 0)),
                str(dir_stats.get("failed", 0)),
                _format_duration(dir_stats.get("total_audio_secs", 0.0)),
                f"{dir_stats.get('avg_processing_secs', 0.0):.1f}s",
            )

        console.print(dir_table)

    output = console.file
    assert isinstance(output, StringIO)
    return output.getvalue()


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
