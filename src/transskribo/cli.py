"""CLI entry point for Transskribo."""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Optional

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from transskribo import __version__
from transskribo.config import TransskriboConfig, load_config, load_enrich_config, load_export_config, merge_config
from transskribo.hasher import compute_hash, load_registry, lookup_hash, register_hash, save_registry
from transskribo.logging_setup import setup_logging
from transskribo.output import build_output_document, copy_duplicate_output, write_output
from transskribo.reporter import (
    compute_statistics,
    compute_timing_statistics,
    format_report,
    per_directory_breakdown,
)
from transskribo.scanner import AudioFile, filter_already_processed, scan_directory
from transskribo.validator import check_ffprobe_available, validate_file

logger = logging.getLogger(__name__)

# Flag for graceful shutdown — set by signal handler
_shutdown_requested = False

app = typer.Typer(
    name="transskribo",
    help="Batch audio transcription and speaker diarization using WhisperX.",
    invoke_without_command=True,
    no_args_is_help=True,
)


def _build_config(
    config_path: Path,
    input_dir: Optional[str],
    output_dir: Optional[str],
    model_size: Optional[str],
    batch_size: Optional[int],
) -> TransskriboConfig:
    """Load TOML config and merge with CLI overrides."""
    file_config = load_config(config_path)
    cli_overrides: dict[str, Any] = {}
    if input_dir is not None:
        cli_overrides["input_dir"] = input_dir
    if output_dir is not None:
        cli_overrides["output_dir"] = output_dir
    if model_size is not None:
        cli_overrides["model_size"] = model_size
    if batch_size is not None:
        cli_overrides["batch_size"] = batch_size
    return merge_config(file_config, cli_overrides)


def _registry_path(cfg: TransskriboConfig) -> Path:
    """Return the path to the hash registry file."""
    return cfg.output_dir / ".transskribo" / "registry.json"


def _log_file_path(cfg: TransskriboConfig) -> Path:
    """Return the path to the log file."""
    return cfg.output_dir / ".transskribo" / "transskribo.log"


def _resolve_config_path(config: Optional[str]) -> Path:
    """Resolve the config file path, falling back to ./config.toml."""
    if config is not None:
        return Path(config)
    default = Path("config.toml")
    if default.exists():
        return default
    typer.echo(
        "Error: No --config given and no config.toml found in current directory",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", help="Path to TOML config file (default: ./config.toml)"),
    input_dir: Optional[str] = typer.Option(None, "--input-dir", help="Override input directory"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Override output directory"),
    model_size: Optional[str] = typer.Option(None, "--model-size", help="Override model size"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Re-process files that previously failed"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan and validate without processing"),
    max_files: int = typer.Option(0, "--max-files", help="Stop after N successful transcriptions (0 = no limit)"),
    max_processing_minutes: float = typer.Option(0, "--max-processing-minutes", help="Stop after M minutes of processing time (0 = no limit)"),
) -> None:
    """Run batch transcription processing."""
    config_path = _resolve_config_path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    try:
        cfg = _build_config(config_path, input_dir, output_dir, model_size, batch_size)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    setup_logging(cfg.log_level, _log_file_path(cfg))

    # Fail fast if ffprobe is not installed
    try:
        check_ffprobe_available()
    except RuntimeError as e:
        logger.error("%s", e)
        raise typer.Exit(code=1)

    logger.info("Transskribo v%s — starting batch processing", __version__)
    logger.info("Input: %s", cfg.input_dir)
    logger.info("Output: %s", cfg.output_dir)
    logger.info("Model: %s (compute=%s, device=%s)", cfg.model_size, cfg.compute_type, cfg.device)
    if retry_failed:
        logger.info("Retry-failed mode: will re-process previously failed files")
    if dry_run:
        logger.info("Dry-run mode: will scan and validate only, no processing")
    if max_files > 0:
        logger.info("Max files limit: %d", max_files)
    if max_processing_minutes > 0:
        logger.info("Max processing minutes limit: %.1f", max_processing_minutes)

    # --- Pipeline ---
    _run_pipeline(
        cfg,
        retry_failed=retry_failed,
        dry_run=dry_run,
        max_files=max_files,
        max_processing_minutes=max_processing_minutes,
    )


def _run_pipeline(
    cfg: TransskriboConfig,
    *,
    retry_failed: bool = False,
    dry_run: bool = False,
    max_files: int = 0,
    max_processing_minutes: float = 0,
) -> None:
    """Execute the full processing pipeline."""
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False

    # Install signal handlers for graceful shutdown
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_shutdown(signum: int, frame: FrameType | None) -> None:
        global _shutdown_requested  # noqa: PLW0603
        if _shutdown_requested:
            # Second signal — force exit immediately
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise KeyboardInterrupt
        _shutdown_requested = True
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — will finish current file and exit (press again to force quit)", sig_name)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        _run_pipeline_inner(
            cfg,
            retry_failed=retry_failed,
            dry_run=dry_run,
            max_files=max_files,
            max_processing_minutes=max_processing_minutes,
        )
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)


def _get_failed_hashes(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a mapping of source_path -> entry for failed entries."""
    failed: dict[str, dict[str, Any]] = {}
    for file_hash, entry in registry.items():
        if entry.get("status") == "failed":
            source = entry.get("source_path", "")
            if source:
                failed[source] = entry
    return failed


def _run_pipeline_inner(
    cfg: TransskriboConfig,
    *,
    retry_failed: bool = False,
    dry_run: bool = False,
    max_files: int = 0,
    max_processing_minutes: float = 0,
) -> None:
    """Inner pipeline logic, separated for signal handler cleanup."""
    batch_start = time.monotonic()

    # Load registry early (needed for retry-failed)
    reg_path = _registry_path(cfg)
    registry = load_registry(reg_path)

    # Scan and filter
    all_files = scan_directory(cfg.input_dir, cfg.output_dir)
    pending_files = filter_already_processed(all_files)
    skipped_existing = len(all_files) - len(pending_files)

    # If retry-failed, re-include files that have status "failed" in registry
    if retry_failed:
        failed_sources = _get_failed_hashes(registry)
        if failed_sources:
            # Find files that were filtered out (already have output) but are failed
            already_done = [f for f in all_files if f not in pending_files]
            retry_files: list[AudioFile] = []
            for f in already_done:
                if str(f.path) in failed_sources:
                    retry_files.append(f)
            # Also check pending files against failed sources (their output
            # might not exist, but they might still be in registry as failed)
            if retry_files:
                logger.info(
                    "Retrying %d previously failed files", len(retry_files)
                )
                pending_files = retry_files + pending_files
                skipped_existing -= len(retry_files)

    logger.info(
        "Found %d files total, %d already processed, %d to process",
        len(all_files),
        skipped_existing,
        len(pending_files),
    )

    if not pending_files:
        logger.info("No files to process. Exiting.")
        return

    # Validate files
    valid_files: list[tuple[AudioFile, float | None]] = []
    invalid_count = 0

    for audio_file in pending_files:
        vr = validate_file(audio_file.path, cfg.max_duration_hours)
        if vr.is_valid:
            valid_files.append((audio_file, vr.duration_secs))
        else:
            invalid_count += 1
            logger.warning(
                "Skipping invalid file %s: %s",
                audio_file.relative_path,
                vr.error,
            )

    if not valid_files:
        logger.info("No valid files to process after validation.")
        return

    logger.info(
        "%d files passed validation, %d rejected",
        len(valid_files),
        invalid_count,
    )

    # Dry-run: report what would be processed and exit
    if dry_run:
        logger.info("--- Dry Run Summary ---")
        logger.info("Would process %d files:", len(valid_files))
        for audio_file, duration_secs in valid_files:
            dur_str = f" ({duration_secs:.1f}s)" if duration_secs else ""
            logger.info("  %s%s", audio_file.relative_path, dur_str)
        logger.info("Skipped (already done): %d", skipped_existing)
        logger.info("Invalid (rejected): %d", invalid_count)
        return

    # Process files with progress bar
    processed_count = 0
    failed_count = 0
    duplicate_count = 0
    stop_reason: str | None = None
    loop_start = time.monotonic()
    max_processing_secs = max_processing_minutes * 60 if max_processing_minutes > 0 else 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[current_file]}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Processing",
            total=len(valid_files),
            current_file="",
        )

        for audio_file, duration_secs in valid_files:
            # Check for graceful shutdown before starting next file
            if _shutdown_requested:
                stop_reason = "Stopped: interrupted by signal"
                logger.info("Shutdown requested — stopping before %s", audio_file.relative_path)
                break

            # Check max-files limit
            if max_files > 0 and processed_count >= max_files:
                stop_reason = f"Stopped: max-files limit ({max_files}) reached"
                logger.info("%s", stop_reason)
                break

            # Check max-processing-minutes limit
            if max_processing_secs > 0:
                elapsed = time.monotonic() - loop_start
                if elapsed >= max_processing_secs:
                    stop_reason = f"Stopped: max-processing-minutes limit ({max_processing_minutes}) exceeded"
                    logger.info("%s", stop_reason)
                    break

            progress.update(task, current_file=str(audio_file.relative_path))

            try:
                result = _process_single_file(
                    audio_file, duration_secs, cfg, registry, reg_path
                )
                if result == "processed":
                    processed_count += 1
                elif result == "duplicate":
                    duplicate_count += 1
            except Exception:
                failed_count += 1
                logger.exception(
                    "Error processing %s", audio_file.relative_path
                )
                # Register failure
                try:
                    file_hash = compute_hash(audio_file.path)
                    now = datetime.now(timezone.utc).isoformat()
                    register_hash(
                        registry,
                        file_hash,
                        source_path=str(audio_file.path),
                        output_path=str(audio_file.output_path),
                        timestamp=now,
                        status="failed",
                        duration_audio_secs=duration_secs,
                        error="Processing error (see log)",
                    )
                    save_registry(registry, reg_path)
                except Exception:
                    logger.exception("Failed to register error for %s", audio_file.relative_path)

            progress.advance(task)

    batch_secs = time.monotonic() - batch_start

    # Batch summary
    summary_label = "Partial Batch Summary (interrupted)" if stop_reason else "Batch Summary"
    logger.info("--- %s ---", summary_label)
    logger.info("Processed: %d", processed_count)
    logger.info("Failed: %d", failed_count)
    logger.info("Skipped (already done): %d", skipped_existing)
    logger.info("Invalid (rejected): %d", invalid_count)
    logger.info("Duplicates: %d", duplicate_count)
    logger.info("Total time: %.1fs", batch_secs)
    if stop_reason:
        logger.info("%s", stop_reason)


def _process_single_file(
    audio_file: AudioFile,
    duration_secs: float | None,
    cfg: TransskriboConfig,
    registry: dict[str, Any],
    registry_path: Path,
) -> str:
    """Process a single file: hash check, transcribe or copy duplicate.

    Returns "processed" or "duplicate".
    """
    # Compute hash
    file_hash = compute_hash(audio_file.path)

    # Check for duplicate
    existing = lookup_hash(registry, file_hash)
    if existing is not None:
        existing_output = Path(existing["output_path"])
        if existing_output.exists():
            logger.info(
                "Duplicate detected for %s (matches %s), copying output",
                audio_file.relative_path,
                existing["source_path"],
            )
            copy_duplicate_output(
                existing_output,
                audio_file.output_path,
                new_source_file=str(audio_file.path),
            )
            now = datetime.now(timezone.utc).isoformat()
            register_hash(
                registry,
                file_hash,
                source_path=str(audio_file.path),
                output_path=str(audio_file.output_path),
                timestamp=now,
                status="success",
                duration_audio_secs=duration_secs,
            )
            save_registry(registry, registry_path)
            return "duplicate"

    # Transcribe
    logger.info("Processing %s", audio_file.relative_path)

    # Import transcriber only when needed (it imports torch/whisperx)
    from transskribo.transcriber import process_file

    result_data = process_file(audio_file.path, cfg)
    transcription_result = result_data["result"]
    timing = result_data["timing"]

    # Build and write output
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "source_file": str(audio_file.path),
        "file_hash": file_hash,
        "duration_secs": duration_secs,
        "model_size": cfg.model_size,
        "language": cfg.language,
        "processed_at": now,
        "timing": timing,
    }
    document = build_output_document(transcription_result, metadata)
    write_output(document, audio_file.output_path)

    # Register in hash registry
    register_hash(
        registry,
        file_hash,
        source_path=str(audio_file.path),
        output_path=str(audio_file.output_path),
        timestamp=now,
        status="success",
        duration_audio_secs=duration_secs,
        timing=timing,
    )
    save_registry(registry, registry_path)

    logger.info(
        "Completed %s (%.1fs)",
        audio_file.relative_path,
        timing["total_secs"],
    )
    return "processed"


@app.command()
def enrich(
    config: Optional[str] = typer.Option(None, "--config", help="Path to TOML config file (default: ./config.toml)"),
    file: Optional[str] = typer.Option(None, "--file", help="Path to a single result.json to enrich"),
    force: bool = typer.Option(False, "--force", help="Re-enrich already-enriched files"),
) -> None:
    """Enrich transcription results with LLM-extracted metadata."""
    config_path = _resolve_config_path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    file_config = load_config(config_path)
    enrich_cfg = load_enrich_config(file_config, {})

    output_dir = Path(file_config.get("output_dir", "output"))

    # Import enricher lazily to keep CLI startup fast
    from transskribo.enricher import enrich_document, is_enriched

    if file is not None:
        # Single-file mode
        _enrich_single_file(Path(file), enrich_cfg, force, enrich_document, is_enriched)
    else:
        # Batch mode
        _enrich_batch(output_dir, enrich_cfg, force, enrich_document, is_enriched)


def _is_transskribo_result(document: dict[str, Any]) -> bool:
    """Check if a JSON document is a transskribo result (has segments and metadata)."""
    return "segments" in document and "metadata" in document


def _enrich_single_file(
    file_path: Path,
    enrich_cfg: Any,
    force: bool,
    enrich_document_fn: Any,
    is_enriched_fn: Any,
) -> None:
    """Enrich a single result JSON file."""
    import json

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return

    with file_path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    if not force and is_enriched_fn(document):
        logger.info("Already enriched, skipping: %s", file_path)
        return

    try:
        document = enrich_document_fn(document, enrich_cfg)
        _write_json_atomic(document, file_path)
        logger.info("Enriched: %s", file_path)
    except Exception:
        logger.exception("Error enriching %s", file_path)


def _enrich_batch(
    output_dir: Path,
    enrich_cfg: Any,
    force: bool,
    enrich_document_fn: Any,
    is_enriched_fn: Any,
) -> None:
    """Enrich all result JSON files in the output directory."""
    import json

    if not output_dir.exists():
        logger.error("Output directory not found: %s", output_dir)
        return

    # Find all .json files recursively, skip .transskribo/ directory
    json_files: list[Path] = []
    for json_path in sorted(output_dir.rglob("*.json")):
        # Skip files inside the .transskribo directory
        try:
            json_path.relative_to(output_dir / ".transskribo")
            continue
        except ValueError:
            pass
        json_files.append(json_path)

    enriched_count = 0
    skipped_count = 0
    failed_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[current_file]}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Enriching",
            total=len(json_files),
            current_file="",
        )

        for json_path in json_files:
            progress.update(task, current_file=json_path.name)

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    document = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Cannot read %s, skipping", json_path)
                failed_count += 1
                progress.advance(task)
                continue

            # Only process transskribo result files
            if not _is_transskribo_result(document):
                progress.advance(task)
                continue

            if not force and is_enriched_fn(document):
                skipped_count += 1
                progress.advance(task)
                continue

            try:
                document = enrich_document_fn(document, enrich_cfg)
                _write_json_atomic(document, json_path)

                enriched_count += 1
                logger.info("Enriched: %s", json_path)
            except Exception:
                failed_count += 1
                logger.exception("Error enriching %s", json_path)

            progress.advance(task)

    logger.info("--- Enrich Summary ---")
    logger.info("Enriched: %d", enriched_count)
    logger.info("Skipped (already enriched): %d", skipped_count)
    logger.info("Failed: %d", failed_count)


def _write_json_atomic(document: dict[str, Any], output_path: Path) -> None:
    """Write JSON document atomically."""
    from transskribo.output import write_output

    write_output(document, output_path)


@app.command(name="export")
def export_cmd(
    config: Optional[str] = typer.Option(None, "--config", help="Path to TOML config file (default: ./config.toml)"),
    file: Optional[str] = typer.Option(None, "--file", help="Path to a single result.json to export"),
    force: bool = typer.Option(False, "--force", help="Regenerate already-exported files"),
    docx: bool = typer.Option(False, "--docx", help="Generate .docx files"),
) -> None:
    """Generate output artifacts from enriched transcription results."""
    if not docx:
        typer.echo("Error: At least one format flag is required (e.g. --docx)", err=True)
        raise typer.Exit(code=1)

    config_path = _resolve_config_path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    file_config = load_config(config_path)
    export_cfg = load_export_config(file_config, {})

    output_dir = Path(file_config.get("output_dir", "output"))

    # Import lazily to keep CLI startup fast
    from transskribo.docx_writer import generate_docx, remap_speakers
    from transskribo.enricher import group_speaker_turns, is_enriched

    if file is not None:
        _export_single_file(
            Path(file), export_cfg, force, docx,
            is_enriched, group_speaker_turns, remap_speakers, generate_docx,
        )
    else:
        _export_batch(
            output_dir, export_cfg, force, docx,
            is_enriched, group_speaker_turns, remap_speakers, generate_docx,
        )


def _export_single_file(
    file_path: Path,
    export_cfg: Any,
    force: bool,
    do_docx: bool,
    is_enriched_fn: Any,
    group_speaker_turns_fn: Any,
    remap_speakers_fn: Any,
    generate_docx_fn: Any,
) -> None:
    """Export a single result JSON file."""
    import json

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return

    with file_path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    if not is_enriched_fn(document):
        logger.warning("Not enriched, skipping: %s", file_path)
        return

    try:
        if do_docx:
            docx_path = file_path.with_suffix(".docx")
            if docx_path.exists() and not force:
                logger.info("Already exported, skipping: %s", docx_path)
                return
            source_name = Path(
                document.get("metadata", {}).get("source_file", file_path.name)
            ).stem
            concepts = {
                "title": document.get("title", ""),
                "keywords": document.get("keywords", []),
                "summary": document.get("summary", ""),
                "concepts": document.get("concepts", {}),
            }
            turns = group_speaker_turns_fn(document)
            turns = remap_speakers_fn(turns, document)
            generate_docx_fn(docx_path, source_name, concepts, turns, export_cfg)
            logger.info("Exported: %s", docx_path)
    except Exception:
        logger.exception("Error exporting %s", file_path)


def _export_batch(
    output_dir: Path,
    export_cfg: Any,
    force: bool,
    do_docx: bool,
    is_enriched_fn: Any,
    group_speaker_turns_fn: Any,
    remap_speakers_fn: Any,
    generate_docx_fn: Any,
) -> None:
    """Export all enriched result JSON files in the output directory."""
    import json

    if not output_dir.exists():
        logger.error("Output directory not found: %s", output_dir)
        return

    # Find all .json files recursively, skip .transskribo/ directory
    json_files: list[Path] = []
    for json_path in sorted(output_dir.rglob("*.json")):
        try:
            json_path.relative_to(output_dir / ".transskribo")
            continue
        except ValueError:
            pass
        json_files.append(json_path)

    exported_count = 0
    skipped_count = 0
    failed_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[current_file]}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Exporting",
            total=len(json_files),
            current_file="",
        )

        for json_path in json_files:
            progress.update(task, current_file=json_path.name)

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    document = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Cannot read %s, skipping", json_path)
                failed_count += 1
                progress.advance(task)
                continue

            # Only process transskribo result files
            if not _is_transskribo_result(document):
                progress.advance(task)
                continue

            # Skip non-enriched files with warning
            if not is_enriched_fn(document):
                logger.warning("Not enriched, skipping: %s", json_path)
                skipped_count += 1
                progress.advance(task)
                continue

            try:
                file_exported = False
                if do_docx:
                    docx_path = json_path.with_suffix(".docx")
                    if docx_path.exists() and not force:
                        skipped_count += 1
                        progress.advance(task)
                        continue
                    source_name = Path(
                        document.get("metadata", {}).get("source_file", json_path.name)
                    ).stem
                    concepts = {
                        "title": document.get("title", ""),
                        "keywords": document.get("keywords", []),
                        "summary": document.get("summary", ""),
                        "concepts": document.get("concepts", {}),
                    }
                    turns = group_speaker_turns_fn(document)
                    turns = remap_speakers_fn(turns, document)
                    generate_docx_fn(docx_path, source_name, concepts, turns, export_cfg)
                    file_exported = True

                if file_exported:
                    exported_count += 1
                    logger.info("Exported: %s", json_path)
            except Exception:
                failed_count += 1
                logger.exception("Error exporting %s", json_path)

            progress.advance(task)

    logger.info("--- Export Summary ---")
    logger.info("Exported: %d", exported_count)
    logger.info("Skipped: %d", skipped_count)
    logger.info("Failed: %d", failed_count)


@app.command()
def report(
    config: Optional[str] = typer.Option(None, "--config", help="Path to TOML config file (default: ./config.toml)"),
) -> None:
    """Print processing statistics (no GPU needed)."""
    config_path = _resolve_config_path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    try:
        cfg = _build_config(config_path, None, None, None, None)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    reg_path = _registry_path(cfg)
    registry = load_registry(reg_path)

    stats = compute_statistics(registry, cfg.input_dir, cfg.output_dir)
    timing_stats = compute_timing_statistics(registry)
    breakdown = per_directory_breakdown(registry, cfg.input_dir)
    formatted = format_report(stats, timing_stats, breakdown)

    typer.echo(formatted)


@app.command()
def version() -> None:
    """Print version information."""
    typer.echo(f"transskribo {__version__}")


if __name__ == "__main__":
    app()
