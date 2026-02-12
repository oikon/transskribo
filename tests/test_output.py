"""Tests for output writing and document building."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from transskribo.output import build_output_document, copy_duplicate_output, write_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def whisperx_result() -> dict[str, Any]:
    """A minimal WhisperX-like result with segments and words."""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Olá mundo",
                "speaker": "SPEAKER_00",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "Olá", "score": 0.95, "speaker": "SPEAKER_00"},
                    {"start": 1.2, "end": 2.5, "word": "mundo", "score": 0.88, "speaker": "SPEAKER_00"},
                ],
            },
            {
                "start": 3.0,
                "end": 5.0,
                "text": "Tudo bem?",
                "speaker": "SPEAKER_01",
                "words": [
                    {"start": 3.0, "end": 3.8, "word": "Tudo", "score": 0.91, "speaker": "SPEAKER_01"},
                    {"start": 4.0, "end": 5.0, "word": "bem?", "score": 0.87, "speaker": "SPEAKER_01"},
                ],
            },
        ],
    }


@pytest.fixture
def metadata() -> dict[str, Any]:
    """Sample metadata dict."""
    return {
        "source_file": "/input/lecture.mp3",
        "file_hash": "abc123def456",
        "duration_secs": 5.0,
        "num_speakers": 2,
        "model_size": "large-v3",
        "language": "pt",
        "processed_at": "2026-01-01T00:00:00+00:00",
        "timing": {
            "transcribe_secs": 1.5,
            "align_secs": 0.3,
            "diarize_secs": 0.8,
            "total_secs": 3.0,
        },
    }


# ---------------------------------------------------------------------------
# build_output_document tests
# ---------------------------------------------------------------------------


class TestBuildOutputDocument:
    def test_top_level_keys(
        self, whisperx_result: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        doc = build_output_document(whisperx_result, metadata)
        assert set(doc.keys()) == {"segments", "words", "metadata"}

    def test_segments_structure(
        self, whisperx_result: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        doc = build_output_document(whisperx_result, metadata)
        assert len(doc["segments"]) == 2
        seg = doc["segments"][0]
        assert seg["start"] == 0.0
        assert seg["end"] == 2.5
        assert seg["text"] == "Olá mundo"
        assert seg["speaker"] == "SPEAKER_00"
        assert len(seg["words"]) == 2

    def test_words_flat_list(
        self, whisperx_result: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        doc = build_output_document(whisperx_result, metadata)
        assert len(doc["words"]) == 4
        assert doc["words"][0]["word"] == "Olá"
        assert doc["words"][2]["word"] == "Tudo"

    def test_word_fields(
        self, whisperx_result: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        doc = build_output_document(whisperx_result, metadata)
        word = doc["words"][0]
        assert set(word.keys()) == {"start", "end", "word", "score", "speaker"}
        assert word["score"] == 0.95
        assert word["speaker"] == "SPEAKER_00"

    def test_metadata_fields(
        self, whisperx_result: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        doc = build_output_document(whisperx_result, metadata)
        meta = doc["metadata"]
        assert meta["source_file"] == "/input/lecture.mp3"
        assert meta["file_hash"] == "abc123def456"
        assert meta["duration_secs"] == 5.0
        assert meta["num_speakers"] == 2
        assert meta["model_size"] == "large-v3"
        assert meta["language"] == "pt"
        assert meta["processed_at"] == "2026-01-01T00:00:00+00:00"
        assert meta["timing"]["total_secs"] == 3.0

    def test_empty_segments(self, metadata: dict[str, Any]) -> None:
        doc = build_output_document({"segments": []}, metadata)
        assert doc["segments"] == []
        assert doc["words"] == []

    def test_missing_segments_key(self, metadata: dict[str, Any]) -> None:
        doc = build_output_document({}, metadata)
        assert doc["segments"] == []
        assert doc["words"] == []

    def test_segment_without_words(self, metadata: dict[str, Any]) -> None:
        result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "SPEAKER_00"},
            ]
        }
        doc = build_output_document(result, metadata)
        assert len(doc["segments"]) == 1
        assert doc["segments"][0]["words"] == []
        assert doc["words"] == []

    def test_num_speakers_from_segments(self) -> None:
        """When num_speakers is not in metadata, count unique speakers from segments."""
        result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "A", "speaker": "SPEAKER_00", "words": []},
                {"start": 1.0, "end": 2.0, "text": "B", "speaker": "SPEAKER_01", "words": []},
                {"start": 2.0, "end": 3.0, "text": "C", "speaker": "SPEAKER_00", "words": []},
            ]
        }
        meta: dict[str, Any] = {"source_file": "test.mp3", "file_hash": "abc"}
        doc = build_output_document(result, meta)
        assert doc["metadata"]["num_speakers"] == 2


# ---------------------------------------------------------------------------
# write_output tests
# ---------------------------------------------------------------------------


class TestWriteOutput:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        output_path = tmp_path / "output" / "sub" / "file.json"
        doc = {"segments": [], "words": [], "metadata": {}}
        write_output(doc, output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded == doc

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        output_path = tmp_path / "deep" / "nested" / "dir" / "file.json"
        write_output({"key": "value"}, output_path)
        assert output_path.parent.is_dir()
        assert output_path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "file.json"
        write_output({"version": 1}, output_path)
        write_output({"version": 2}, output_path)

        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["version"] == 2

    def test_atomic_write_no_partial_on_error(self, tmp_path: Path) -> None:
        output_path = tmp_path / "file.json"
        write_output({"good": True}, output_path)

        # Try writing something that will fail serialization
        class BadObj:
            pass

        with pytest.raises(TypeError):
            write_output({"bad": BadObj()}, output_path)

        # Original file should still be intact
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded == {"good": True}

    def test_unicode_content(self, tmp_path: Path) -> None:
        output_path = tmp_path / "unicode.json"
        doc = {"text": "Olá, como você está? São Paulo — café"}
        write_output(doc, output_path)

        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["text"] == "Olá, como você está? São Paulo — café"


# ---------------------------------------------------------------------------
# copy_duplicate_output tests
# ---------------------------------------------------------------------------


class TestCopyDuplicateOutput:
    def test_copies_with_updated_source(self, tmp_path: Path) -> None:
        source_path = tmp_path / "original.json"
        target_path = tmp_path / "copy.json"
        doc = {
            "segments": [{"text": "hello"}],
            "words": [],
            "metadata": {
                "source_file": "/input/a.mp3",
                "processed_at": "2026-01-01T00:00:00+00:00",
            },
        }
        write_output(doc, source_path)

        copy_duplicate_output(source_path, target_path, "/input/b.mp3")

        loaded = json.loads(target_path.read_text(encoding="utf-8"))
        assert loaded["metadata"]["source_file"] == "/input/b.mp3"
        assert loaded["metadata"]["processed_at"] != "2026-01-01T00:00:00+00:00"
        assert loaded["segments"] == [{"text": "hello"}]

    def test_creates_target_directories(self, tmp_path: Path) -> None:
        source_path = tmp_path / "original.json"
        target_path = tmp_path / "deep" / "nested" / "copy.json"
        doc = {"segments": [], "words": [], "metadata": {"source_file": "a.mp3", "processed_at": "x"}}
        write_output(doc, source_path)

        copy_duplicate_output(source_path, target_path, "b.mp3")
        assert target_path.exists()

    def test_preserves_segments_and_words(self, tmp_path: Path) -> None:
        source_path = tmp_path / "original.json"
        target_path = tmp_path / "copy.json"
        doc = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Olá", "speaker": "S0", "words": []}],
            "words": [{"word": "Olá", "start": 0.0, "end": 1.0}],
            "metadata": {"source_file": "a.mp3", "processed_at": "x"},
        }
        write_output(doc, source_path)

        copy_duplicate_output(source_path, target_path, "b.mp3")

        loaded = json.loads(target_path.read_text(encoding="utf-8"))
        assert loaded["segments"] == doc["segments"]
        assert loaded["words"] == doc["words"]

    def test_no_metadata_key(self, tmp_path: Path) -> None:
        """If the source document has no metadata key, copy still works."""
        source_path = tmp_path / "original.json"
        target_path = tmp_path / "copy.json"
        doc: dict[str, Any] = {"segments": [], "words": []}
        write_output(doc, source_path)

        copy_duplicate_output(source_path, target_path, "b.mp3")

        loaded = json.loads(target_path.read_text(encoding="utf-8"))
        assert loaded["segments"] == []
