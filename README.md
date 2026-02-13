# Transskribo

CLI tool for batch audio transcription and speaker diarization using [WhisperX](https://github.com/m-bain/whisperX).

## Why Transskribo

WhisperX gives you great transcription and speaker diarization, but it processes one file at a time and leaves you to manage GPU memory, output files, and retries. When you have thousands of lecture recordings and meeting files to transcribe, that manual overhead adds up fast.

Transskribo wraps WhisperX into a batch pipeline. It scans your directory tree, validates files with ffprobe, detects duplicates by hash, and loads and unloads models to stay within VRAM limits. It writes structured JSON mirroring your input layout and tracks progress in a registry so you can stop and resume at any point. Point it at a folder, run one command, and come back to finished transcripts.

## Features

- Recursive scanning and batch processing of audio and video files
- Speaker diarization via pyannote
- Duplicate detection using SHA-256 hashing
- Graceful shutdown (finish current file on Ctrl+C, press again to force quit)
- Retry failed files with `--retry-failed`
- Dry-run mode to preview what would be processed
- Processing limits (`--max-files`, `--max-processing-minutes`)
- LLM enrichment: titles, keywords, summaries, and concepts from transcripts
- DOCX export with customizable templates
- Per-file timing and summary reports
- TOML configuration with CLI overrides

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support
- [ffmpeg](https://ffmpeg.org/) (for file validation)
- [HuggingFace token](https://huggingface.co/settings/tokens) (for pyannote diarization models)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
git clone https://github.com/yourusername/transskribo.git
cd transskribo
uv sync
```

## Configuration

Copy the example config and edit it:

```bash
cp config.example.toml config.toml
```

```toml
input_dir = "/path/to/audio/files"
output_dir = "/path/to/output"
hf_token = "hf_your_token_here"   # or set HF_TOKEN env var
model_size = "large-v3"
language = "pt"
compute_type = "float16"
batch_size = 8
device = "cuda"

[enrich]
llm_base_url = "https://api.openai.com/v1"
llm_api_key = ""        # or set ENRICH_API_KEY env var
llm_model = "gpt-4o-mini"

[export]
template_path = "templates/basic.docx"
transcritor = "Your Name"
```

See `config.example.toml` for all options.

## Usage

```bash
# Run batch processing
uv run transskribo run --config config.toml

# Preview without processing
uv run transskribo run --config config.toml --dry-run

# Re-process previously failed files
uv run transskribo run --config config.toml --retry-failed

# Override settings from CLI
uv run transskribo run --config config.toml --model-size medium --batch-size 4

# Stop after 10 files or 60 minutes
uv run transskribo run --config config.toml --max-files 10
uv run transskribo run --config config.toml --max-processing-minutes 60

# Enrich transcripts with LLM-extracted metadata
uv run transskribo enrich --config config.toml
uv run transskribo enrich --file output/lecture.json
uv run transskribo enrich --config config.toml --force  # re-enrich all

# Export enriched results to DOCX
uv run transskribo export --config config.toml --docx
uv run transskribo export --file output/lecture.json --docx
uv run transskribo export --config config.toml --docx --force  # regenerate all

# Print processing statistics (no GPU needed)
uv run transskribo report --config config.toml

# Show version
uv run transskribo version
```

If a `config.toml` exists in the current directory, `--config` can be omitted.

## Supported Formats

**Audio:** `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.opus`, `.wma`, `.aac`

**Video:** `.mp4`, `.mkv`, `.avi`, `.webm`, `.mov`, `.wmv`

## Output

Transskribo writes JSON files that mirror your input directory structure. A registry at `<output_dir>/.transskribo/registry.json` tracks processed files for duplicate detection and cross-run resume.

## Development

```bash
uv run pytest                          # run tests
uv run ruff check src/ tests/          # lint
uv run pyright src/                    # type check
```

## Google Colab

See [docs/setup-colab.md](docs/setup-colab.md) for instructions on running Transskribo in Google Colab.

## License

MIT
