"""JSON output writing and directory mirroring."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_output_document(result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Structure the final JSON output with segments, words, and metadata.

    Args:
        result: The transcription result from WhisperX (contains "segments" with
                nested "words" and speaker labels).
        metadata: Dict with fields: source_file, file_hash, duration_secs,
                  num_speakers, model_size, language, processed_at, timing.

    Returns:
        A dict with three top-level keys: segments, words, metadata.
    """
    segments_raw = result.get("segments", [])

    segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []

    for seg in segments_raw:
        seg_words: list[dict[str, Any]] = []
        for w in seg.get("words", []):
            word_entry = {
                "start": w.get("start"),
                "end": w.get("end"),
                "word": w.get("word", ""),
                "score": w.get("score"),
                "speaker": w.get("speaker"),
            }
            seg_words.append(word_entry)
            all_words.append(word_entry)

        segments.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker"),
            "words": seg_words,
        })

    # Count unique speakers
    speakers = {s.get("speaker") for s in segments if s.get("speaker") is not None}
    metadata_out: dict[str, Any] = {
        "source_file": metadata.get("source_file"),
        "file_hash": metadata.get("file_hash"),
        "duration_secs": metadata.get("duration_secs"),
        "num_speakers": metadata.get("num_speakers", len(speakers)),
        "model_size": metadata.get("model_size"),
        "language": metadata.get("language"),
        "processed_at": metadata.get("processed_at"),
        "timing": metadata.get("timing"),
    }

    return {
        "segments": segments,
        "words": all_words,
        "metadata": metadata_out,
    }


def write_output(document: dict[str, Any], output_path: Path) -> None:
    """Create parent directories and write JSON output atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=output_path.parent, suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def copy_duplicate_output(
    source_output: Path,
    target_output: Path,
    new_source_file: str,
) -> None:
    """Copy an existing output for a duplicate, updating metadata.source_file.

    Args:
        source_output: Path to the existing output JSON.
        target_output: Path where the copy should be written.
        new_source_file: The new source file path to set in metadata.
    """
    with source_output.open("r", encoding="utf-8") as f:
        document: dict[str, Any] = json.load(f)

    if "metadata" in document:
        document["metadata"]["source_file"] = new_source_file
        document["metadata"]["processed_at"] = datetime.now(timezone.utc).isoformat()

    target_output.parent.mkdir(parents=True, exist_ok=True)
    write_output(document, target_output)
