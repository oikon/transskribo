"""Tests for scanner module."""

from __future__ import annotations

from pathlib import Path

from transskribo.scanner import (
    SUPPORTED_EXTENSIONS,
    AudioFile,
    filter_already_processed,
    scan_directory,
)


def test_scan_empty_directory(tmp_input_dir: Path, tmp_output_dir: Path) -> None:
    """Empty input dir returns no files."""
    result = scan_directory(tmp_input_dir, tmp_output_dir)
    assert result == []


def test_scan_finds_audio_files(tmp_input_dir: Path, tmp_output_dir: Path) -> None:
    """Scanner finds common audio files."""
    (tmp_input_dir / "lecture.mp3").write_bytes(b"fake mp3")
    (tmp_input_dir / "meeting.wav").write_bytes(b"fake wav")
    (tmp_input_dir / "talk.flac").write_bytes(b"fake flac")

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    names = {f.relative_path.name for f in result}
    assert names == {"lecture.mp3", "meeting.wav", "talk.flac"}


def test_scan_finds_video_files(tmp_input_dir: Path, tmp_output_dir: Path) -> None:
    """Scanner finds video files with audio."""
    (tmp_input_dir / "recording.mp4").write_bytes(b"fake mp4")
    (tmp_input_dir / "class.mkv").write_bytes(b"fake mkv")

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    names = {f.relative_path.name for f in result}
    assert names == {"recording.mp4", "class.mkv"}


def test_scan_nested_directories(tmp_input_dir: Path, tmp_output_dir: Path) -> None:
    """Scanner walks subdirectories recursively."""
    sub1 = tmp_input_dir / "semester1" / "lectures"
    sub1.mkdir(parents=True)
    (sub1 / "lec01.mp3").write_bytes(b"data")

    sub2 = tmp_input_dir / "semester2"
    sub2.mkdir()
    (sub2 / "lec02.m4a").write_bytes(b"data")

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    relative_paths = {str(f.relative_path) for f in result}
    assert "semester1/lectures/lec01.mp3" in relative_paths
    assert "semester2/lec02.m4a" in relative_paths


def test_scan_output_path_mirrors_structure(
    tmp_input_dir: Path, tmp_output_dir: Path
) -> None:
    """Output paths mirror the input directory structure with .json extension."""
    sub = tmp_input_dir / "course"
    sub.mkdir()
    (sub / "lecture.mp3").write_bytes(b"data")

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    assert len(result) == 1
    assert result[0].output_path == tmp_output_dir / "course" / "lecture.json"


def test_scan_case_insensitive_extensions(
    tmp_input_dir: Path, tmp_output_dir: Path
) -> None:
    """Scanner matches extensions case-insensitively."""
    (tmp_input_dir / "upper.MP3").write_bytes(b"data")
    (tmp_input_dir / "mixed.Mp4").write_bytes(b"data")
    (tmp_input_dir / "lower.wav").write_bytes(b"data")

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    assert len(result) == 3


def test_scan_ignores_unsupported_extensions(
    tmp_input_dir: Path, tmp_output_dir: Path
) -> None:
    """Unsupported extensions are not returned."""
    (tmp_input_dir / "readme.txt").write_bytes(b"text")
    (tmp_input_dir / "data.csv").write_bytes(b"csv")
    (tmp_input_dir / "image.jpg").write_bytes(b"jpg")
    (tmp_input_dir / "good.mp3").write_bytes(b"mp3")

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    assert len(result) == 1
    assert result[0].relative_path.name == "good.mp3"


def test_scan_all_supported_extensions(
    tmp_input_dir: Path, tmp_output_dir: Path
) -> None:
    """Every supported extension is recognized."""
    for ext in SUPPORTED_EXTENSIONS:
        (tmp_input_dir / f"file{ext}").write_bytes(b"data")

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    found_exts = {f.path.suffix.lower() for f in result}
    assert found_exts == SUPPORTED_EXTENSIONS


def test_scan_records_size_bytes(
    tmp_input_dir: Path, tmp_output_dir: Path
) -> None:
    """AudioFile records the file size in bytes."""
    content = b"x" * 1234
    (tmp_input_dir / "sized.mp3").write_bytes(content)

    result = scan_directory(tmp_input_dir, tmp_output_dir)
    assert len(result) == 1
    assert result[0].size_bytes == 1234


def test_filter_already_processed_removes_existing(
    tmp_input_dir: Path, tmp_output_dir: Path
) -> None:
    """Files with existing output are filtered out."""
    (tmp_input_dir / "done.mp3").write_bytes(b"data")
    (tmp_input_dir / "pending.wav").write_bytes(b"data")

    # Create existing output for "done.mp3"
    out = tmp_output_dir / "done.json"
    out.write_text("{}")

    files = scan_directory(tmp_input_dir, tmp_output_dir)
    assert len(files) == 2

    remaining = filter_already_processed(files)
    assert len(remaining) == 1
    assert remaining[0].relative_path.name == "pending.wav"


def test_filter_already_processed_keeps_all_when_none_exist(
    tmp_input_dir: Path, tmp_output_dir: Path
) -> None:
    """All files are kept when no output exists."""
    (tmp_input_dir / "a.mp3").write_bytes(b"data")
    (tmp_input_dir / "b.wav").write_bytes(b"data")

    files = scan_directory(tmp_input_dir, tmp_output_dir)
    remaining = filter_already_processed(files)
    assert len(remaining) == 2


def test_filter_already_processed_empty_list() -> None:
    """Empty input returns empty output."""
    assert filter_already_processed([]) == []


def test_audiofile_is_frozen(tmp_input_dir: Path, tmp_output_dir: Path) -> None:
    """AudioFile is immutable (frozen dataclass)."""
    af = AudioFile(
        path=tmp_input_dir / "test.mp3",
        relative_path=Path("test.mp3"),
        output_path=tmp_output_dir / "test.json",
        size_bytes=100,
    )
    import dataclasses

    assert dataclasses.is_dataclass(af)
    # Frozen dataclass raises FrozenInstanceError on assignment
    try:
        af.size_bytes = 999  # type: ignore[misc]
        assert False, "Should have raised"
    except dataclasses.FrozenInstanceError:
        pass
