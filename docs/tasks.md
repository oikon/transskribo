# Build Plan

Each session is designed for one headless Claude Code run. Sessions are
ordered by dependency — later sessions build on earlier ones.

---

## Session 1 — Project Skeleton & Configuration

**Features:** 1.01, 1.02, 1.03, 2.01, 2.02, 2.03, 2.04, 2.05

**Goal:** A working Python project that installs, has a CLI entry point
that prints help, and loads/validates configuration from TOML.

**Preconditions:** None

**Verification:**
- `uv sync` succeeds
- `uv run transskribo --help` shows usage
- `uv run pytest` passes all config tests
- `uv run ruff check src/ tests/` clean
- `uv run pyright src/` clean

---

## Session 2 — Logging, File Scanner & Validation

**Features:** 3.01, 3.02, 4.01, 4.02, 4.03, 5.01, 5.02, 5.03

**Goal:** Logging is configured with dual output. Scanner discovers
audio/video files recursively with expanded format support and filters
already-processed ones. Validator uses ffprobe to reject invalid files
(zero-length, corrupt, no audio stream, over max duration) and extract
duration.

**Preconditions:** Session 1 complete

**Verification:**
- Logging writes to both stdout and file
- Scanner finds .mp3, .m4a, .wav, .flac, .mp4, .mkv, etc. in nested dirs
- Scanner skips files with existing output
- Validator rejects zero-length, corrupt, and no-audio files
- Validator extracts duration and enforces max_duration_hours
- All tests pass, lint clean, types clean

---

## Session 3 — Hashing & Registry

**Features:** 6.01, 6.02, 6.03, 6.04

**Goal:** Files can be hashed, registry tracks processing state with
timing data, atomic writes prevent corruption.

**Preconditions:** Session 2 complete

**Verification:**
- Hash is deterministic for same file
- Registry persists across save/load cycles with timing fields
- Atomic write doesn't corrupt on simulated failure
- All tests pass, lint clean, types clean

---

## Session 4 — Transcriber (WhisperX Wrapper)

**Features:** 7.01, 7.02, 7.03, 7.04, 7.05

**Goal:** WhisperX is wrapped with clean interfaces for each stage.
Models are loaded and unloaded per stage to manage 8GB VRAM.
All interaction is isolated in one module. Tests use mocks.

**Preconditions:** Session 3 complete

**Verification:**
- Transcription stage: load, transcribe, align, unload
- Diarization stage: load, diarize, assign, unload
- process_file orchestrates both stages with per-stage timing
- Mock tests verify correct load/unload lifecycle and VRAM cleanup
- All tests pass, lint clean, types clean

---

## Session 5 — Output Writer & Reporter

**Features:** 8.01, 8.02, 8.03, 9.01, 9.02, 9.03, 9.04, 9.05

**Goal:** Output JSON is structured correctly with all fields.
Duplicate outputs are handled. Reporter computes full stats with
per-stage timing from registry. Rich formatted report works standalone.

**Preconditions:** Session 4 complete

**Verification:**
- Output JSON has segments, words, and metadata sections
- Duplicate files produce copied output with updated metadata
- Reporter generates correct statistics including per-stage timing
- Report shows progress (total/processed/remaining) and ETA
- All tests pass, lint clean, types clean

---

## Session 6 — CLI Integration & Pipeline Wiring

**Features:** 10.01, 10.02, 10.03, 10.04, 10.05, 10.06

**Goal:** The full pipeline works end-to-end from the CLI. Validation
runs before processing. Progress bar shows during processing. Hash-based
duplicate detection works in the loop. Report command works independently.
Batch summary printed at end.

**Preconditions:** Session 5 complete

**Verification:**
- `transskribo run --config config.toml` processes files with progress bar
- Invalid files (corrupt, zero-length, no audio) are skipped with logged reason
- ffprobe check at startup fails fast if not installed
- Duplicate files are detected by hash and output is copied
- Per-file errors are logged and batch continues
- Batch summary printed at end with counts and total time
- `transskribo report --config config.toml` prints statistics with timing
- `transskribo version` prints version info
- All tests pass, lint clean, types clean

---

## Session 7 — Polish & Hardening

**Features:** 11.01, 11.02, 11.03, 11.04

**Goal:** Retry-failed and dry-run modes work. Graceful shutdown on
SIGINT. Integration test exercises the full pipeline.

**Preconditions:** Session 6 complete

**Verification:**
- `--retry-failed` re-processes only failed files
- `--dry-run` scans, validates, and reports without processing
- Ctrl+C finishes current file and exits cleanly
- Integration test passes (or is properly skipped without GPU)
- All tests pass, lint clean, types clean
- Final full run: `uv run ruff check src/ tests/ && uv run pyright src/ && uv run pytest`
