# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-02-13

### Added

- Split enrich into separate `enrich` and `export` commands with independent configs

### Changed

- Improve enrichment system prompt with structured output spec and constraints
- Show file counts (M/N) instead of percentages in progress bars
- Suppress third-party noise (warnings, logs, prints) when log level is ERROR+
- Update README features and improve description clarity

### Fixed

- Crash in `remap_speakers` when speaker is None
- Force quit on second Ctrl+C during processing

## [1.0.0-alpha] - 2026-02-13

### Added

- Core batch transcription pipeline with WhisperX (transcribe, align, diarize)
- CLI with `run`, `report`, `enrich` commands via Typer
- TOML configuration with CLI override merging
- Directory scanning for audio/video files with 14 supported formats
- ffprobe-based file validation (corrupt, zero-length, no audio stream)
- SHA-256 duplicate detection with JSON hash registry
- Per-stage model lifecycle to avoid GPU OOM (load/unload Whisper and pyannote separately)
- Rich progress bars and formatted statistics reporting
- Graceful shutdown with Ctrl+C signal handling
- `--retry-failed` and `--dry-run` CLI options
- `--max-files` and `--max-processing-minutes` batch limit controls
- LLM-based enrichment with structured outputs (title, keywords, summary, concepts)
- DOCX document generation from enrichment results
- File and stdout logging with configurable log levels
- Google Colab setup guide
