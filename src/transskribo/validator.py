"""File validation using ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating an audio/video file."""

    is_valid: bool
    duration_secs: float | None
    error: str | None


def check_ffprobe_available() -> None:
    """Verify that ffprobe is on PATH. Raises RuntimeError if not found."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe not found on PATH. Install ffmpeg to continue."
        )


def validate_file(
    file_path: Path, max_duration_hours: float
) -> ValidationResult:
    """Validate an audio/video file using ffprobe.

    Checks:
    - File is not zero-length.
    - ffprobe can read the file.
    - File has at least one audio stream.
    - Duration does not exceed max_duration_hours (if > 0).

    Returns a ValidationResult with duration on success or error on failure.
    """
    # Reject zero-length files without calling ffprobe
    if file_path.stat().st_size == 0:
        return ValidationResult(
            is_valid=False, duration_secs=None, error="Zero-length file"
        )

    # Run ffprobe to inspect the file
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            is_valid=False, duration_secs=None, error="ffprobe timed out"
        )
    except OSError as e:
        return ValidationResult(
            is_valid=False, duration_secs=None, error=f"ffprobe error: {e}"
        )

    if result.returncode != 0:
        return ValidationResult(
            is_valid=False,
            duration_secs=None,
            error="ffprobe failed — file may be corrupt or unreadable",
        )

    # Parse ffprobe JSON output
    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ValidationResult(
            is_valid=False,
            duration_secs=None,
            error="ffprobe returned invalid JSON",
        )

    # Check for at least one audio stream
    streams = probe_data.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        return ValidationResult(
            is_valid=False,
            duration_secs=None,
            error="No audio stream found",
        )

    # Extract duration (prefer format-level, fall back to first audio stream)
    duration_secs = _extract_duration(probe_data, audio_streams)
    if duration_secs is None:
        return ValidationResult(
            is_valid=False,
            duration_secs=None,
            error="Could not determine duration",
        )

    # Enforce max duration
    if max_duration_hours > 0:
        max_secs = max_duration_hours * 3600
        if duration_secs > max_secs:
            return ValidationResult(
                is_valid=False,
                duration_secs=duration_secs,
                error=(
                    f"Duration {duration_secs:.1f}s exceeds limit "
                    f"{max_duration_hours}h ({max_secs:.0f}s)"
                ),
            )

    return ValidationResult(
        is_valid=True, duration_secs=duration_secs, error=None
    )


def _extract_duration(
    probe_data: dict[str, object],
    audio_streams: list[dict[str, object]],
) -> float | None:
    """Extract duration in seconds from ffprobe data."""
    # Try format-level duration first
    fmt = probe_data.get("format", {})
    if isinstance(fmt, dict):
        dur_str = fmt.get("duration")
        if dur_str is not None:
            try:
                return float(str(dur_str))
            except (ValueError, TypeError):
                pass

    # Fall back to first audio stream duration
    if audio_streams:
        dur_str = audio_streams[0].get("duration")
        if dur_str is not None:
            try:
                return float(str(dur_str))
            except (ValueError, TypeError):
                pass

    return None
