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

- [x] 12.01 — Add `--max-files` CLI-only option to `run` command (int, default 0 = no limit). After each successfully transcribed file (result == "processed"), check if `processed_count >= max_files`. If so, exit the processing loop cleanly. Duplicates, skipped, invalid, and failed files do NOT count toward the limit. Log a message when the limit is reached
- [x] 12.02 — Add `--max-processing-minutes` CLI-only option to `run` command (float, default 0 = no limit). Before starting each file, check if wall-clock time since the processing loop began (after scan + validation) exceeds `max_processing_minutes * 60` seconds. If so, exit the processing loop cleanly. The current file always finishes — the check is before the next file. Log a message when the limit is reached
- [x] 12.03 — Update batch summary to indicate the stop reason when a limit was hit (e.g., "Stopped: max-files limit (10) reached" or "Stopped: max-processing-minutes limit (120.0) exceeded")
- [x] 12.04 — Write tests: `--max-files` stops after N successful transcriptions (not counting duplicates/errors), `--max-processing-minutes` stops after elapsed time, both limits log appropriate messages, limits of 0 mean no limit, interaction with `--dry-run` (limits are irrelevant in dry-run mode), interaction with SIGINT (shutdown takes priority)

## 13. Enrichment & Document Generation

- [x] 13.01 — Add `openai` and `docxtpl` to dependencies in `pyproject.toml`. Define `EnrichConfig` frozen dataclass in `config.py` with fields: llm_base_url (default: "https://api.openai.com/v1"), llm_api_key (default: ""), llm_model (default: "gpt-4o-mini"), template_path (default: Path("templates/basic.docx")), transcritor (default: "Jonas Rodrigues (via IA)"). Implement `load_enrich_config(file_config: dict, cli_overrides: dict) -> EnrichConfig` that reads the `[enrich]` section from the TOML dict and merges CLI overrides. When `llm_api_key` is empty, the `openai` SDK automatically reads `OPENAI_API_KEY` from the environment — no custom env var handling needed. Update `config.example.toml` with the new `[enrich]` section and comments. Write tests for EnrichConfig loading and defaults
- [x] 13.02 — Implement `extract_text(document: dict) -> str` in `enricher.py` that concatenates all segment text from a transcription result JSON (from the `"segments"` key, each segment's `"text"` field) into a single plain text string, separated by spaces. Implement `group_speaker_turns(document: dict) -> list[dict]` that iterates through segments and merges consecutive segments from the same speaker into turns. Each turn is a dict with `"speaker"` (str) and `"texts"` (list[str]) keys. A speaker change starts a new turn
- [x] 13.03 — Implement `is_enriched(document: dict) -> bool` in `enricher.py` that returns True if the document contains all four enrichment keys at the top level: `"title"`, `"keywords"`, `"summary"`, `"concepts"`. Returns False if any key is missing
- [x] 13.04 — Implement `call_llm(text: str, config: EnrichConfig) -> dict` in `enricher.py` that creates an OpenAI client with `base_url=config.llm_base_url` and `api_key=config.llm_api_key`, sends a chat completion request to `config.llm_model` with a system prompt instructing the model to extract title, keywords, summary, and concepts from Brazilian Portuguese transcription text, requesting JSON output. Parse the LLM response as JSON and validate it contains the expected keys (title: str, keywords: list[str], summary: str, concepts: dict[str, str]). Raise a clear error if the API call fails or the response doesn't parse correctly
- [x] 13.05 — Implement `enrich_document(document: dict, config: EnrichConfig) -> dict` in `enricher.py` that orchestrates: extract_text → call_llm → merge enrichment data (title, keywords, summary, concepts) into the document at the top level. Returns the updated document dict
- [x] 13.06 — Implement `generate_docx(output_path: Path, source_name: str, concepts: dict, segments: list[dict], config: EnrichConfig) -> None` in `docx_writer.py` that loads the .docx template from `config.template_path` using `DocxTemplate`, builds the context dict with: `arquivo`=source_name, `transcritor`=config.transcritor, `data_transcricao`=today's date formatted as dd/mm/yyyy, `info`=concepts (the dict with title, keywords, summary, concepts keys), `segmentos`=segments (list of speaker turn dicts). Renders the template and saves to `output_path`. Raises a clear error if the template file doesn't exist
- [x] 13.07 — Implement `enrich` command in `cli.py` with options: `--config` (TOML config path, default: ./config.toml), `--file` (path to a single result.json, optional), `--force` (re-enrich already-enriched files, default: False). In batch mode (no --file): read `output_dir` from top-level config, find all `.json` files recursively in output_dir, filter to only transskribo result files (must have `"segments"` and `"metadata"` top-level keys), skip files in `.transskribo/` directory. For each file: read JSON, check `is_enriched` (skip unless --force), call `enrich_document`, write updated JSON atomically. In single-file mode (--file): enrich just that one file. Log progress per file. Handle per-file errors (log and continue in batch mode). Log batch summary at end (enriched count, skipped count, failed count). NOTE: docx generation has been moved to the `export` command (section 14)
- [x] 13.08 — Write tests for `enricher.py`: extract_text with various segment structures (empty, single segment, multiple segments, missing text field), group_speaker_turns merging logic (same speaker consecutive, alternating speakers, single speaker throughout, empty segments), is_enriched with all keys present / partial keys / no keys, call_llm with mocked OpenAI client (successful JSON response, malformed response, API error), enrich_document integration with mocked call_llm
- [x] 13.09 — Write tests for `docx_writer.py`: generate_docx creates a .docx file at the target path, template not found raises error, context variables are passed correctly (mock DocxTemplate)
- [x] 13.10 — Write tests for `enrich` CLI command: batch mode discovers and enriches result JSONs in output_dir, single-file mode with --file enriches one file, --force re-enriches already-enriched files, skip logic for already-enriched files, per-file error handling continues batch, non-transskribo JSONs are ignored, files in .transskribo/ directory are skipped
- [x] 13.11 — Update `compute_statistics()` in `reporter.py` to include enrichment counts: scan result JSON files in output_dir, count how many have all enrichment keys (title, keywords, summary, concepts) vs how many don't. Add `enriched` and `not_enriched` keys to the stats dict. Update `format_report()` to display enrichment progress in the report output (e.g., "Enriched: 42 / 100"). Write tests for enrichment stats with mixed enriched/non-enriched result files

**Done when:** `transskribo enrich --config config.toml` batch-enriches all result JSONs with LLM-extracted metadata. `transskribo enrich --file result.json` enriches a single file. Already-enriched files are skipped unless `--force`. `transskribo report` shows enrichment progress. Tests pass, lint clean, types clean.

## 14. Export Command & Config Refactor

- [x] 14.01 — Split config: Remove `template_path` and `transcritor` from `EnrichConfig`. Define `ExportConfig` frozen dataclass in `config.py` with fields: template_path (default: Path("templates/basic.docx")), transcritor (default: "Jonas Rodrigues (via IA)"). Implement `load_export_config(file_config: dict, cli_overrides: dict) -> ExportConfig` that reads the `[export]` section from the TOML dict and merges CLI overrides. Add `_EXPORT_DEFAULTS` dict. Update `config.example.toml`: move `template_path` and `transcritor` from `[enrich]` to a new `[export]` section. Write tests for ExportConfig loading and defaults
- [x] 14.02 — Update `docx_writer.py`: change `generate_docx` signature to accept `ExportConfig` instead of `EnrichConfig`. Update imports and all callers
- [x] 14.03 — Refactor `enrich` command in `cli.py`: remove all docx generation logic from both `_enrich_single_file` and `_enrich_batch`. The enrich command should only: read JSON → check `is_enriched` → call `enrich_document` → write updated JSON. Remove `generate_docx` import and `group_speaker_turns` usage from enrich. Remove `remap_speakers` import from enrich functions
- [x] 14.04 — Implement `export` command in `cli.py` with options: `--config` (TOML config path, default: ./config.toml), `--file` (path to a single result.json, optional), `--force` (regenerate already-exported files, default: False), `--docx` (generate .docx files, default: False). At least one format flag (`--docx`) is required — exit with error if none provided. In batch mode (no --file): read `output_dir` from top-level config, find all `.json` files recursively in output_dir, filter to transskribo result files (must have `"segments"` and `"metadata"` keys), skip files in `.transskribo/` directory. For each file: read JSON, check `is_enriched` (skip non-enriched files with warning), check if export artifact already exists (skip unless --force), generate requested artifacts. For `--docx`: call `group_speaker_turns` + `remap_speakers` + `generate_docx` (output .docx at same path as .json but with .docx extension). In single-file mode (--file): export just that one file. Log progress per file. Handle per-file errors (log and continue in batch mode). Log batch summary at end (exported count, skipped count, failed count)
- [x] 14.05 — Update `compute_statistics()` in `reporter.py`: add per-format export counts. Scan result JSONs in output_dir and for each enriched file, check if corresponding artifact files exist (`.docx` alongside `.json`). Add `exported_docx` key to stats dict. Update `format_report()` to display pipeline stage progress: `Transcribed: X / total`, `Enriched: X / transcribed`, `Exported (docx): X / enriched`. Write tests for export stats
- [x] 14.06 — Update existing tests: update `test_enrich_cli.py` to remove assertions about .docx generation from enrich command. Update `test_docx_writer.py` to use `ExportConfig` instead of `EnrichConfig`. Update `test_config.py` with ExportConfig tests. Write new `test_export_cli.py` with tests for: batch mode discovers and exports enriched result JSONs, single-file mode with --file, --force regenerates existing exports, skip logic for non-enriched files (with warning), skip logic for already-exported files, per-file error handling continues batch, --docx generates .docx alongside .json, error when no format flag provided, non-transskribo JSONs ignored, files in .transskribo/ directory skipped

**Done when:** `transskribo export --config config.toml --docx` batch-generates .docx files for all enriched result JSONs. `transskribo export --file result.json --docx` exports a single file. Non-enriched files are skipped with warning. Already-exported files are skipped unless `--force`. `transskribo report` shows full pipeline progress (transcribed → enriched → exported per format). The `enrich` command no longer generates .docx files. Tests pass, lint clean, types clean.
