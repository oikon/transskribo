"""CLI entry point for Transskribo."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from transskribo import __version__
from transskribo.config import TransskriboConfig, load_config, merge_config
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


@app.command()
def run(
    config: str = typer.Option(..., "--config", help="Path to TOML config file"),
    input_dir: Optional[str] = typer.Option(None, "--input-dir", help="Override input directory"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Override output directory"),
    model_size: Optional[str] = typer.Option(None, "--model-size", help="Override model size"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size"),
) -> None:
    """Run batch transcription processing."""
    config_path = Path(config)
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

    # --- Pipeline ---
    _run_pipeline(cfg)


def _run_pipeline(cfg: TransskriboConfig) -> None:
    """Execute the full processing pipeline."""
    batch_start = time.monotonic()

    # Scan and filter
    all_files = scan_directory(cfg.input_dir, cfg.output_dir)
    pending_files = filter_already_processed(all_files)
    skipped_existing = len(all_files) - len(pending_files)

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

    # Load registry
    reg_path = _registry_path(cfg)
    registry = load_registry(reg_path)

    # Process files with progress bar
    processed_count = 0
    failed_count = 0
    duplicate_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[current_file]}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Processing",
            total=len(valid_files),
            current_file="",
        )

        for audio_file, duration_secs in valid_files:
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
    logger.info("--- Batch Summary ---")
    logger.info("Processed: %d", processed_count)
    logger.info("Failed: %d", failed_count)
    logger.info("Skipped (already done): %d", skipped_existing)
    logger.info("Invalid (rejected): %d", invalid_count)
    logger.info("Duplicates: %d", duplicate_count)
    logger.info("Total time: %.1fs", batch_secs)


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
def report(
    config: str = typer.Option(..., "--config", help="Path to TOML config file"),
) -> None:
    """Print processing statistics (no GPU needed)."""
    config_path = Path(config)
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
