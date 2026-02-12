# Transskribo — Project Instructions

## What This Is

A CLI tool that wraps WhisperX to batch-process thousands of audio files
(lectures and group meetings) with transcription and speaker diarization.
All audio is in Brazilian Portuguese.

## Stack

- **Language:** Python 3.12+
- **CLI framework:** Typer
- **Config format:** TOML (using tomllib + tomli-w)
- **Package manager:** uv
- **Progress/display:** rich
- **Logging:** Python stdlib `logging` with file + stdout handlers
- **Testing:** pytest + pytest-cov
- **Linting:** ruff
- **Type checking:** pyright
- **System dependency:** ffmpeg/ffprobe (required for file validation; already a transitive dep of WhisperX)

## Project Layout

```
transskribo/
├── pyproject.toml
├── config.example.toml        # example configuration
├── src/
│   └── transskribo/
│       ├── __init__.py
│       ├── cli.py             # typer app, entry point
│       ├── config.py          # TOML loading, CLI override merging
│       ├── scanner.py         # directory walking, audio/video file discovery
│       ├── validator.py       # ffprobe-based file validation
│       ├── hasher.py          # SHA-256 hashing, hash registry
│       ├── transcriber.py     # WhisperX wrapper (load, transcribe, align, diarize)
│       ├── output.py          # JSON output writing, directory mirroring
│       ├── reporter.py        # summary reports, independent statistics
│       └── logging_setup.py   # logging configuration
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_scanner.py
│   ├── test_validator.py
│   ├── test_hasher.py
│   ├── test_transcriber.py
│   ├── test_output.py
│   └── test_reporter.py
└── docs/
    ├── design.md
    ├── requirements.md
    └── tasks.md
```

## Conventions

- Use `src/` layout with `transskribo` package
- All modules must have type hints
- Functions should be small and testable in isolation
- WhisperX interaction is isolated in `transcriber.py` — no other module imports whisperx
- Config is loaded once and passed as a dataclass, never accessed as a global
- File paths use `pathlib.Path` everywhere, never raw strings
- Hash registry is a JSON file stored at `<output_dir>/.transskribo/registry.json`
- Registry entries must include per-stage timing data (`transcribe_secs`, `align_secs`, `diarize_secs`, `total_secs`) for reporting
- Logs go to both stdout (with rich formatting) and `<output_dir>/.transskribo/transskribo.log`

## Architectural Constraints

- One file processed at a time (sequential) — no concurrency
- GPU memory: RTX 4060 8GB VRAM — use float16 compute, batch_size=8 as default
- Models are loaded and unloaded per stage within each file to avoid OOM:
  1. Load Whisper → transcribe → align → delete model → `torch.cuda.empty_cache()`
  2. Load pyannote → diarize → assign speakers → delete pipeline → `torch.cuda.empty_cache()`
- Never keep Whisper and pyannote loaded simultaneously
- Every file is validated with ffprobe before processing — reject zero-length, corrupt, no-audio-stream, over max duration
- Supported extensions (case-insensitive): `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.opus`, `.wma`, `.aac`, `.mp4`, `.mkv`, `.avi`, `.webm`, `.mov`, `.wmv`
- Audio files can be 1-3 hours (some longer) — WhisperX handles chunking internally
- Duplicate detection is by SHA-256 hash, computed lazily at processing time
- All state is file-based (JSON registry), no database

## Commands

```bash
uv sync                                # install/update all dependencies
uv run transskribo --help              # show CLI usage
uv run transskribo run --config config.toml   # run batch processing
uv run transskribo report --config config.toml # print stats (no GPU needed)
uv run pytest                          # run all tests
uv run pytest tests/test_config.py     # run one test file
uv run pytest -k "test_merge"          # run tests matching a name
uv run ruff check src/ tests/          # lint check
uv run ruff check --fix src/ tests/    # lint autofix
uv run pyright src/                    # type check
```

## Git Workflow

- Work directly on `main` — no feature branches
- One commit per feature or logical group
- Descriptive commit messages (e.g., "Add config loading with TOML merge and validation")
- Never commit `.env`, audio files, model weights, or `output/`/`input/` dirs

## Session Discipline for Headless Execution

To determine which session you're in: read `claude-progress.txt` (at the repo
root) to see what's been done, then read `docs/tasks.md` to find the next
session whose features are not yet completed.

1. **Read `docs/requirements.md`** — find the next `[ ]` item to implement
2. **Read `docs/design.md`** — understand where the feature fits architecturally
3. **Check `docs/tasks.md`** — confirm the feature belongs to the current session
4. **Implement the feature** — write code in the correct module
5. **Write or update tests** — every feature needs a test
6. **Run `uv run pytest`** — all tests must pass before committing
7. **Run `uv run ruff check src/ tests/`** — no lint errors
8. **Run `uv run pyright src/`** — no type errors
9. **Update `docs/requirements.md`** — change `[ ]` to `[x]` for completed items
10. **Update `claude-progress.txt`** — log what was done
11. **Commit with a descriptive message** — one commit per feature or logical group

## Important Notes

- HuggingFace token is provided via config file or `HF_TOKEN` env var
- Never hardcode tokens or paths
- Never add `.env` files, audio files, or model weights to git
- The `output/` and `input/` directories are not part of the repo
- Keep dependencies minimal — only add what's strictly needed
- Never import `whisperx` or `torch` outside of `transcriber.py`
- Never introduce concurrency — processing is strictly sequential
- Never load Whisper and pyannote models simultaneously
