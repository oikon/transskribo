"""Document generation from enriched transcription results."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from docxtpl import DocxTemplate

from transskribo.config import EnrichConfig


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
