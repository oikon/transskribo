# Requirements

Status key: `[ ]` = not started, `[x]` = done

## 1. Project Setup

- [x] 1.01 — Create `pyproject.toml` with project metadata, dependencies (whisperx, typer, rich, tomli-w), dev dependencies (pytest, pytest-cov, ruff, pyright), and `[project.scripts]` entry point `transskribo = "transskribo.cli:app"`. Create `src/transskribo/__init__.py` with `__version__`. Create `src/transskribo/cli.py` with a minimal Typer app (just `app = typer.Typer()` and a placeholder `run` callback so `--help` works). Create `.gitignore` for Python project (venv, __pycache__, .eggs, dist, output/, input/, *.pyc, .env, .build-sessions/)
- [x] 1.02 — Create `config.example.toml` with all configurable options and comments
- [x] 1.03 — Create `tests/conftest.py` with shared fixtures (tmp dirs, sample config)

**Done when:** `uv sync` installs all deps, `uv run transskribo --help` prints usage, `uv run pytest` runs (even if 0 tests)

## 2. Configuration

- [x] 2.01 — Define `TransskriboConfig` dataclass in `config.py` with fields: input_dir, output_dir, hf_token, model_size (default: "large-v3"), language (default: "pt"), compute_type (default: "float16"), batch_size (default: 8), device (default: "cuda"), log_level (default: "INFO"), max_duration_hours (default: 0, meaning no limit)
- [x] 2.02 — Implement `load_config(path: Path) -> dict` that reads a TOML file and returns a dict
- [x] 2.03 — Implement `merge_config(file_config: dict, cli_overrides: dict) -> TransskriboConfig` that applies defaults, then file config, then CLI overrides, and returns a validated dataclass
- [x] 2.04 — Implement config validation: check input_dir exists, output_dir is creatable, hf_token is non-empty (from config or `HF_TOKEN` env var)
- [x] 2.05 — Write tests for config loading, merging, and validation (valid configs, missing fields, invalid paths, env var fallback)

**Done when:** Config loads from TOML, merges with CLI overrides, validates, and all tests pass.

## 3. Logging

- [x] 3.01 — Implement `setup_logging(log_level: str, log_file: Path)` in `logging_setup.py` that configures a root logger with two handlers: rich stdout handler and rotating file handler. Create log file directory if it doesn't exist
- [x] 3.02 — Write tests: verify both handlers are attached, log file is created, messages appear in both sinks

**Done when:** Calling `setup_logging` produces formatted output to stdout and to a log file. Tests pass.

## 4. File Scanner

- [x] 4.01 — Define `AudioFile` dataclass in `scanner.py` (path, relative_path, output_path, size_bytes) and implement `scan_directory(input_dir: Path, output_dir: Path) -> list[AudioFile]` that walks input_dir recursively, filters for supported extensions (case-insensitive): audio (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.opus`, `.wma`, `.aac`) and video (`.mp4`, `.mkv`, `.avi`, `.webm`, `.mov`, `.wmv`), computes mirrored output paths with `.json` extension. Store the supported extensions as a module-level constant `SUPPORTED_EXTENSIONS`
- [x] 4.02 — Implement `filter_already_processed(files: list[AudioFile]) -> list[AudioFile]` that removes files whose output_path already exists on disk
- [x] 4.03 — Write tests: empty dir, nested dirs, mixed audio/video extensions, case sensitivity (.MP3, .Mp4), unsupported extensions ignored, already-processed filtering

**Done when:** Scanner finds all audio/video files, maps them to output paths, filters processed ones. Tests pass.

## 5. File Validation

- [x] 5.01 — Define `ValidationResult` dataclass in `validator.py` (is_valid: bool, duration_secs: float | None, error: str | None) and implement `check_ffprobe_available()` that verifies ffprobe is on PATH, raising a clear error if not
- [x] 5.02 — Implement `validate_file(file_path: Path, max_duration_hours: float) -> ValidationResult` that: rejects zero-length files (size == 0) without calling ffprobe; runs ffprobe as a subprocess to check the file has at least one audio stream and extract duration; rejects files exceeding max_duration_hours (if > 0); returns ValidationResult with duration on success or error reason on failure
- [x] 5.03 — Write tests: zero-length file, valid audio file, video file with audio stream, file with no audio stream, corrupt/unreadable file, file exceeding max duration, max_duration_hours=0 means no limit. Use small fixture files and/or mock subprocess calls for ffprobe

**Done when:** Validator correctly identifies invalid files (zero-length, corrupt, no audio, over limit) and extracts duration from valid files. Tests pass.

## 6. Hashing & Registry

- [x] 6.01 — Implement `compute_hash(file_path: Path) -> str` in `hasher.py` that streams SHA-256, returns hex digest. Define `RegistryEntry` dataclass with fields: source_path, output_path, timestamp, status, duration_audio_secs (float | None), timing ({transcribe_secs, align_secs, diarize_secs, total_secs} | None), error (str | None)
- [x] 6.02 — Implement `load_registry(registry_path: Path) -> dict` and `save_registry(registry: dict, registry_path: Path)` with atomic write (temp file + rename)
- [x] 6.03 — Implement `lookup_hash(registry: dict, file_hash: str) -> dict | None` that returns the existing entry if hash was seen with status "success", and `register_hash(registry: dict, file_hash: str, ...)` that adds/updates an entry with all fields (source_path, output_path, status, duration_audio_secs, timing, error)
- [x] 6.04 — Write tests: hash determinism, registry CRUD, atomic write safety, lookup hit/miss, register with timing data

**Done when:** Files can be hashed, registry persists to disk atomically, lookups work. Tests pass.

## 7. Transcriber (WhisperX Wrapper)

GPU memory is managed by loading/unloading models per stage within each file.
Whisper + wav2vec2 are loaded for transcription/alignment, then freed before
loading pyannote for diarization. This prevents OOM on 8GB VRAM.

- [x] 7.01 — Implement transcription stage in `transcriber.py`: `load_audio(audio_path, config) -> ndarray` to load audio via `whisperx.load_audio()`, `load_whisper_model(config)` to load WhisperX model, `transcribe(model, audio: ndarray, config) -> dict` to run transcription on the loaded audio array, `align(result, audio: ndarray, config) -> dict` for word-level alignment using the same audio array, `unload_whisper_model(model)` to delete model and call `torch.cuda.empty_cache()`
- [x] 7.02 — Implement diarization stage: `load_diarization_pipeline(config)` to load pyannote with HF token, `diarize(pipeline, audio_path, config) -> dict` to run speaker diarization, `assign_speakers(diarization, aligned) -> dict` to merge labels, `unload_diarization_pipeline(pipeline)` to delete and free VRAM
- [x] 7.03 — Implement `process_file(audio_path: Path, config: TransskriboConfig) -> dict` that orchestrates: load_audio → load whisper → transcribe(model, audio) → align(result, audio) → unload whisper → load pyannote → diarize(pipeline, audio_path) → assign speakers → unload pyannote. Audio is loaded once via `load_audio` and the ndarray is passed to both `transcribe` and `align`. Pyannote loads audio internally so it takes `audio_path`. Collect per-stage timing (transcribe_secs, align_secs, diarize_secs, total_secs) using `time.monotonic()` and return alongside result
- [x] 7.04 — Write mock tests for transcription stage: verify model loaded with correct config, transcribe/align called with correct args, model deleted and cache cleared after unload
- [x] 7.05 — Write mock tests for diarization stage and process_file orchestrator: verify pipeline loaded with HF token, diarize/assign called correctly, VRAM freed, full lifecycle order (load→use→unload per stage), timing values collected, error in one stage doesn't skip cleanup

**Done when:** Transcriber processes a file with explicit model load/unload per stage. Mock tests verify the full lifecycle including VRAM cleanup. Tests pass.

## 8. Output Writer

- [x] 8.01 — Implement `build_output_document(result: dict, metadata: dict) -> dict` in `output.py` that structures the final JSON with three top-level keys: `segments` (list of segments with start, end, text, speaker, words), `words` (flat list of all words with start, end, score, speaker), and `metadata` (dict with fields: source_file, file_hash, duration_secs, num_speakers, model_size, language, processed_at, timing)
- [x] 8.02 — Implement `write_output(document: dict, output_path: Path)` that creates parent directories and writes JSON atomically, and `copy_duplicate_output(source_output: Path, target_output: Path)` that copies an existing output for a duplicate, updating metadata to reflect the new source path
- [x] 8.03 — Write tests: document structure validation, directory creation, atomic write, duplicate copy with metadata update

**Done when:** Output JSON is written correctly with all fields, duplicates are handled. Tests pass.

## 9. Reporter

- [x] 9.01 — Implement `compute_statistics(registry: dict, input_dir: Path | None, output_dir: Path | None) -> dict` in `reporter.py` that calculates: total files in input (reuse `scan_directory` from `scanner.py` if input_dir provided), files processed (succeeded), failed, skipped, duplicates, remaining, total audio duration processed, total audio duration discovered
- [x] 9.02 — Implement `compute_timing_statistics(registry: dict) -> dict` that calculates from per-file timing data: average/min/max per stage (transcribe, align, diarize), average total time per file, processing speed ratio (audio duration / processing time), estimated time remaining (avg total x remaining files)
- [x] 9.03 — Implement `per_directory_breakdown(registry: dict, input_dir: Path | None) -> dict` that groups stats and timing by top-level subdirectory
- [x] 9.04 — Implement `format_report(stats: dict, timing_stats: dict, breakdown: dict) -> str` that renders a formatted report using rich tables: progress section, timing section, per-directory section
- [x] 9.05 — Write tests: stats calculation with various registry states, timing stats with/without timing data, breakdown grouping, report formatting

**Done when:** Statistics include full progress and per-stage timing. Report is readable with rich formatting. Works independently from processing. Tests pass.

## 10. CLI Integration

- [x] 10.01 — Implement `run` command in `cli.py`: parse CLI args (--config, --input-dir, --output-dir, --model-size, --batch-size), load config, set up logging. Call `check_ffprobe_available()` at startup — fail fast if missing
- [x] 10.02 — Wire `run` pipeline skeleton: scan input dir, filter already-processed files, validate each remaining file with `validate_file()` (skip invalid with logged reason), store each file's `ValidationResult.duration_secs` for later use in registry. Iterate over valid files with rich progress bar (overall count, current file name, elapsed time)
- [x] 10.03 — Wire hash check + duplicate handling into pipeline loop: compute hash, check registry, copy existing output if duplicate, update registry
- [x] 10.04 — Wire transcription into pipeline loop: call process_file, write output, register hash with timing data and `duration_audio_secs` (from the `ValidationResult.duration_secs` stored in 10.02), handle per-file errors (log and continue, register with status "failed" and error message). Add batch summary log at end of run (processed, failed, skipped, invalid, duplicates, total time)
- [x] 10.05 — Implement `report` command: load config, scan input dir for total file count, load registry, compute stats + timing stats + breakdown, print rich formatted report (no processing, no GPU needed). Implement `version` command: print version and key dependency versions
- [x] 10.06 — Write tests: CLI arg parsing, run with mocked pipeline, report with fixture registry, ffprobe check at startup

**Done when:** `transskribo run --config config.toml` processes a batch. `transskribo report` prints stats. Tests pass.

## 11. End-to-End & Polish

- [x] 11.01 — Add `--retry-failed` flag to `run` command: add the CLI flag, load failed entries from registry, re-include them in the processing loop (skip the "already processed" filter for files whose hash has status "failed")
- [x] 11.02 — Add `--dry-run` flag to `run` command: scan, validate, and report what would be processed, without actually processing
- [x] 11.03 — Handle SIGINT/SIGTERM gracefully: finish current file, write registry, log partial summary, exit
- [x] 11.04 — Write one integration test with a short audio fixture (< 5 seconds) that exercises the full pipeline end-to-end (can be skipped if no GPU in CI)

## 12. Batch Limit Controls

- [ ] 12.01 — Add `--max-files` CLI-only option to `run` command (int, default 0 = no limit). After each successfully transcribed file (result == "processed"), check if `processed_count >= max_files`. If so, exit the processing loop cleanly. Duplicates, skipped, invalid, and failed files do NOT count toward the limit. Log a message when the limit is reached
- [ ] 12.02 — Add `--max-processing-minutes` CLI-only option to `run` command (float, default 0 = no limit). Before starting each file, check if wall-clock time since the processing loop began (after scan + validation) exceeds `max_processing_minutes * 60` seconds. If so, exit the processing loop cleanly. The current file always finishes — the check is before the next file. Log a message when the limit is reached
- [ ] 12.03 — Update batch summary to indicate the stop reason when a limit was hit (e.g., "Stopped: max-files limit (10) reached" or "Stopped: max-processing-minutes limit (120.0) exceeded")
- [ ] 12.04 — Write tests: `--max-files` stops after N successful transcriptions (not counting duplicates/errors), `--max-processing-minutes` stops after elapsed time, both limits log appropriate messages, limits of 0 mean no limit, interaction with `--dry-run` (limits are irrelevant in dry-run mode), interaction with SIGINT (shutdown takes priority)
