"""Directory walking and audio/video file discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    # Audio
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aac",
    # Video
    ".mp4", ".mkv", ".avi", ".webm", ".mov", ".wmv",
})


@dataclass(frozen=True)
class AudioFile:
    """A discovered audio/video file with its computed output path."""

    path: Path
    relative_path: Path
    output_path: Path
    size_bytes: int


def scan_directory(input_dir: Path, output_dir: Path) -> list[AudioFile]:
    """Walk input_dir recursively and return supported audio/video files.

    Each file gets a mirrored output_path under output_dir with a .json extension.
    Results are sorted by relative path for deterministic ordering.
    """
    files: list[AudioFile] = []

    for file_path in input_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative = file_path.relative_to(input_dir)
        output_path = output_dir / relative.with_suffix(".json")

        files.append(AudioFile(
            path=file_path,
            relative_path=relative,
            output_path=output_path,
            size_bytes=file_path.stat().st_size,
        ))

    files.sort(key=lambda f: f.relative_path)
    return files


def filter_already_processed(files: list[AudioFile]) -> list[AudioFile]:
    """Remove files whose output_path already exists on disk."""
    return [f for f in files if not f.output_path.exists()]
