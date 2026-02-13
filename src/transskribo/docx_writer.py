"""Document generation from enriched transcription results."""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from docxtpl import DocxTemplate

from transskribo.config import EnrichConfig


def remap_speakers(
    turns: list[dict[str, Any]],
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    """Remap speaker labels to 'Pessoa 01', 'Pessoa 02', etc. by segment count.

    Speakers are ranked by number of original segments (most segments first).
    Ties are broken alphabetically by original label.

    Args:
        turns: Speaker turns from group_speaker_turns (speaker + texts dicts).
        document: The full transcription document (used to count original segments).

    Returns:
        New list of turn dicts with remapped speaker labels.
    """
    segments = document.get("segments", [])
    counts: Counter[str] = Counter(seg.get("speaker") or "UNKNOWN" for seg in segments)

    # Rank: most segments first, alphabetical tiebreak; exclude UNKNOWN/None
    known = {s for s in counts if s is not None and s != "UNKNOWN"}
    ranked = sorted(known, key=lambda s: (-counts[s], s))
    speaker_map: dict[str, str] = {speaker: f"Pessoa {i + 1:02d}" for i, speaker in enumerate(ranked)}
    speaker_map["UNKNOWN"] = "Pessoa ??"

    return [
        {**turn, "speaker": speaker_map.get(turn.get("speaker") or "UNKNOWN", turn.get("speaker") or "UNKNOWN")}
        for turn in turns
    ]


def generate_docx(
    output_path: Path,
    source_name: str,
    concepts: dict[str, Any],
    segments: list[dict[str, Any]],
    config: EnrichConfig,
) -> None:
    """Load a .docx template, fill it with enrichment data, and save.

    Args:
        output_path: Where to save the rendered .docx file.
        source_name: Name of the source audio file.
        concepts: Dict with title, keywords, summary, concepts keys.
        segments: List of speaker turn dicts (speaker + texts).
        config: Enrich configuration with template_path and transcritor.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    template_file = config.template_path
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    doc = DocxTemplate(str(template_file))

    context = {
        "arquivo": source_name,
        "transcritor": config.transcritor,
        "data_transcricao": date.today().strftime("%d/%m/%Y"),
        "info": concepts,
        "segmentos": segments,
    }

    doc.render(context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
