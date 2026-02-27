"""Cross-file concept index generation from enriched transcription results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transskribo.enricher import is_enriched

logger = logging.getLogger(__name__)


def collect_concepts(
    output_dir: Path,
) -> dict[str, list[tuple[str, str]]]:
    """Scan output_dir for enriched result JSONs and collect concepts.

    Returns a dict mapping concept name to a sorted list of
    (file_label, explanation) tuples. File label is derived from
    metadata.source_file without extension.
    """
    concepts: dict[str, list[tuple[str, str]]] = {}
    enriched_count = 0

    if not output_dir.exists():
        logger.warning("Output directory not found: %s", output_dir)
        return concepts

    for json_path in sorted(output_dir.rglob("*.json")):
        # Skip files inside the .transskribo directory
        try:
            json_path.relative_to(output_dir / ".transskribo")
            continue
        except ValueError:
            pass

        # Read and validate the JSON file
        try:
            with json_path.open("r", encoding="utf-8") as f:
                document: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Must be a transskribo result file
        if "segments" not in document or "metadata" not in document:
            continue

        # Must be enriched
        if not is_enriched(document):
            continue

        enriched_count += 1

        # Extract file label from metadata.source_file without extension
        source_file = document.get("metadata", {}).get("source_file", "")
        file_label = Path(source_file).stem if source_file else json_path.stem

        # Collect concepts
        file_concepts = document.get("concepts", {})
        if isinstance(file_concepts, dict):
            for concept_name, explanation in file_concepts.items():
                if concept_name not in concepts:
                    concepts[concept_name] = []
                concepts[concept_name].append((file_label, str(explanation)))

    logger.info("Found %d enriched files", enriched_count)

    # Sort bullets within each concept by file label
    for concept_name in concepts:
        concepts[concept_name].sort(key=lambda t: t[0])

    return concepts


def render_index(concepts: dict[str, list[tuple[str, str]]]) -> str:
    """Render the concept dict into a markdown string.

    Concept names sorted A-Z, bullets within each concept sorted
    alphabetically by file label.
    """
    if not concepts:
        return "# Concept Index\n\nNo concepts found.\n"

    lines: list[str] = ["# Concept Index", ""]

    for concept_name in sorted(concepts.keys()):
        lines.append(f"## {concept_name}")
        lines.append("")
        for file_label, explanation in concepts[concept_name]:
            lines.append(f"- **{file_label}**: {explanation}")
        lines.append("")

    return "\n".join(lines)


def generate_index(output_dir: Path, output_path: Path) -> int:
    """Orchestrate concept index generation.

    Collects concepts from enriched files, renders markdown, and writes
    to output_path. Always overwrites existing file.

    Returns the number of unique concepts indexed.
    """
    concepts = collect_concepts(output_dir)

    if not concepts:
        logger.info("No concepts found — index not generated")
        return 0

    markdown = render_index(concepts)

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(markdown, encoding="utf-8")

    concept_count = len(concepts)
    logger.info(
        "Generated concept index: %d concepts, written to %s",
        concept_count,
        output_path,
    )

    return concept_count
