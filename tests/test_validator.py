"""Tests for validator module."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transskribo.validator import (
    ValidationResult,
    check_ffprobe_available,
    validate_file,
)


# --- check_ffprobe_available ---


def test_ffprobe_available_when_present() -> None:
    """No error when ffprobe is on PATH."""
    with patch("transskribo.validator.shutil.which", return_value="/usr/bin/ffprobe"):
        check_ffprobe_available()  # Should not raise


def test_ffprobe_available_when_missing() -> None:
    """RuntimeError when ffprobe is not on PATH."""
    with patch("transskribo.validator.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffprobe not found"):
            check_ffprobe_available()


# --- validate_file: zero-length ---


def test_validate_zero_length_file(tmp_path: Path) -> None:
    """Zero-length file is rejected without calling ffprobe."""
    f = tmp_path / "empty.mp3"
    f.write_bytes(b"")

    with patch("transskribo.validator.subprocess.run") as mock_run:
        result = validate_file(f, max_duration_hours=0)

    mock_run.assert_not_called()
    assert not result.is_valid
    assert result.error == "Zero-length file"
    assert result.duration_secs is None


# --- validate_file: valid audio ---


def _ffprobe_result(
    streams: list[dict[str, object]] | None = None,
    duration: str = "120.5",
    returncode: int = 0,
) -> MagicMock:
    """Build a mock subprocess.CompletedProcess for ffprobe."""
    if streams is None:
        streams = [{"codec_type": "audio", "duration": "120.5"}]
    data = {
        "streams": streams,
        "format": {"duration": duration},
    }
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = json.dumps(data)
    mock.stderr = ""
    return mock


def test_validate_valid_audio_file(tmp_path: Path) -> None:
    """Valid audio file returns is_valid=True with duration."""
    f = tmp_path / "good.mp3"
    f.write_bytes(b"fake audio data")

    with patch(
        "transskribo.validator.subprocess.run",
        return_value=_ffprobe_result(duration="3600.0"),
    ):
        result = validate_file(f, max_duration_hours=0)

    assert result.is_valid
    assert result.duration_secs == 3600.0
    assert result.error is None


def test_validate_video_with_audio_stream(tmp_path: Path) -> None:
    """Video file with audio stream is accepted."""
    f = tmp_path / "lecture.mp4"
    f.write_bytes(b"fake video data")

    streams = [
        {"codec_type": "video", "duration": "600.0"},
        {"codec_type": "audio", "duration": "600.0"},
    ]
    with patch(
        "transskribo.validator.subprocess.run",
        return_value=_ffprobe_result(streams=streams, duration="600.0"),
    ):
        result = validate_file(f, max_duration_hours=0)

    assert result.is_valid
    assert result.duration_secs == 600.0


# --- validate_file: no audio stream ---


def test_validate_no_audio_stream(tmp_path: Path) -> None:
    """File with no audio stream is rejected."""
    f = tmp_path / "silent_video.mp4"
    f.write_bytes(b"fake video")

    streams = [{"codec_type": "video", "duration": "300.0"}]
    with patch(
        "transskribo.validator.subprocess.run",
        return_value=_ffprobe_result(streams=streams, duration="300.0"),
    ):
        result = validate_file(f, max_duration_hours=0)

    assert not result.is_valid
    assert result.error == "No audio stream found"


# --- validate_file: corrupt file ---


def test_validate_corrupt_file(tmp_path: Path) -> None:
    """Corrupt file (ffprobe returns non-zero) is rejected."""
    f = tmp_path / "corrupt.mp3"
    f.write_bytes(b"not real audio")

    with patch(
        "transskribo.validator.subprocess.run",
        return_value=_ffprobe_result(returncode=1),
    ):
        result = validate_file(f, max_duration_hours=0)

    assert not result.is_valid
    assert "corrupt or unreadable" in (result.error or "")


# --- validate_file: max duration ---


def test_validate_exceeds_max_duration(tmp_path: Path) -> None:
    """File exceeding max_duration_hours is rejected."""
    f = tmp_path / "long.mp3"
    f.write_bytes(b"long audio")

    # 2 hours = 7200 seconds
    with patch(
        "transskribo.validator.subprocess.run",
        return_value=_ffprobe_result(duration="7200.0"),
    ):
        result = validate_file(f, max_duration_hours=1.0)

    assert not result.is_valid
    assert result.duration_secs == 7200.0
    assert "exceeds limit" in (result.error or "")


def test_validate_max_duration_zero_means_no_limit(tmp_path: Path) -> None:
    """max_duration_hours=0 means no limit is enforced."""
    f = tmp_path / "very_long.mp3"
    f.write_bytes(b"very long audio")

    # 10 hours
    with patch(
        "transskribo.validator.subprocess.run",
        return_value=_ffprobe_result(duration="36000.0"),
    ):
        result = validate_file(f, max_duration_hours=0)

    assert result.is_valid
    assert result.duration_secs == 36000.0


def test_validate_within_max_duration(tmp_path: Path) -> None:
    """File within max_duration_hours is accepted."""
    f = tmp_path / "ok.mp3"
    f.write_bytes(b"ok audio")

    with patch(
        "transskribo.validator.subprocess.run",
        return_value=_ffprobe_result(duration="1800.0"),
    ):
        result = validate_file(f, max_duration_hours=1.0)

    assert result.is_valid
    assert result.duration_secs == 1800.0


# --- validate_file: ffprobe timeout ---


def test_validate_ffprobe_timeout(tmp_path: Path) -> None:
    """ffprobe timeout returns invalid result."""
    f = tmp_path / "slow.mp3"
    f.write_bytes(b"slow audio")

    with patch(
        "transskribo.validator.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
    ):
        result = validate_file(f, max_duration_hours=0)

    assert not result.is_valid
    assert "timed out" in (result.error or "")


# --- validate_file: ffprobe invalid JSON ---


def test_validate_ffprobe_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON from ffprobe returns invalid result."""
    f = tmp_path / "badjson.mp3"
    f.write_bytes(b"data")

    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "not json at all"
    mock.stderr = ""

    with patch("transskribo.validator.subprocess.run", return_value=mock):
        result = validate_file(f, max_duration_hours=0)

    assert not result.is_valid
    assert "invalid JSON" in (result.error or "")


# --- validate_file: no duration ---


def test_validate_no_duration(tmp_path: Path) -> None:
    """File with audio stream but no duration is rejected."""
    f = tmp_path / "nodur.mp3"
    f.write_bytes(b"data")

    streams = [{"codec_type": "audio"}]
    data = {"streams": streams, "format": {}}
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps(data)
    mock.stderr = ""

    with patch("transskribo.validator.subprocess.run", return_value=mock):
        result = validate_file(f, max_duration_hours=0)

    assert not result.is_valid
    assert "duration" in (result.error or "").lower()


# --- ValidationResult dataclass ---


def test_validation_result_is_frozen() -> None:
    """ValidationResult is immutable."""
    vr = ValidationResult(is_valid=True, duration_secs=1.0, error=None)
    with pytest.raises(AttributeError):
        vr.is_valid = False  # type: ignore[misc]
