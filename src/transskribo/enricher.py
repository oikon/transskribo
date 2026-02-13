"""LLM-powered concept extraction from transcription results."""

from __future__ import annotations

import json
import logging
from typing import Any

from transskribo.config import EnrichConfig

logger = logging.getLogger(__name__)

ENRICHMENT_KEYS = ("title", "keywords", "summary", "concepts")

_SYSTEM_PROMPT = """Você é um assistente especializado em análise de transcrições em português brasileiro.

Dada uma transcrição de áudio, extraia as seguintes informações em formato JSON:

1. "title": Um título descritivo e conciso para o conteúdo (string)
2. "keywords": Uma lista de 5-10 palavras-chave relevantes (lista de strings)
3. "summary": Um resumo de 2-4 parágrafos do conteúdo (string)
4. "concepts": Um dicionário com os principais conceitos discutidos, onde cada chave é o nome do conceito e o valor é uma breve explicação (dicionário string->string)

Responda APENAS com o JSON válido, sem texto adicional."""


def extract_text(document: dict[str, Any]) -> str:
    """Concatenate all segment text from a transcription result into plain text."""
    segments = document.get("segments", [])
    texts: list[str] = []
    for seg in segments:
        text = seg.get("text", "")
        if text:
            texts.append(text)
    return " ".join(texts)


def group_speaker_turns(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker segments into speaker turns.

    Each turn is a dict with "speaker" (str) and "texts" (list[str]) keys.
    A speaker change starts a new turn.
    """
    segments = document.get("segments", [])
    turns: list[dict[str, Any]] = []
    current_speaker: str | None = None
    current_texts: list[str] = []

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")

        if speaker != current_speaker:
            if current_texts:
                turns.append({
                    "speaker": current_speaker or "UNKNOWN",
                    "texts": current_texts,
                })
            current_speaker = speaker
            current_texts = [text] if text else []
        else:
            if text:
                current_texts.append(text)

    # Don't forget the last turn
    if current_texts:
        turns.append({
            "speaker": current_speaker or "UNKNOWN",
            "texts": current_texts,
        })

    return turns


def is_enriched(document: dict[str, Any]) -> bool:
    """Return True if the document contains all four enrichment keys."""
    return all(key in document for key in ENRICHMENT_KEYS)


def call_llm(text: str, config: EnrichConfig) -> dict[str, Any]:
    """Call an OpenAI-compatible LLM to extract structured metadata.

    Args:
        text: Plain text from the transcription.
        config: Enrich configuration with LLM endpoint details.

    Returns:
        Dict with keys: title (str), keywords (list[str]),
        summary (str), concepts (dict[str, str]).

    Raises:
        ValueError: If the LLM response cannot be parsed or is missing keys.
        RuntimeError: If the API call fails.
    """
    from openai import OpenAI

    try:
        client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key or None,
        )
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}") from e

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM response is not valid JSON: {e}") from e

    # Validate expected keys
    missing = [k for k in ENRICHMENT_KEYS if k not in parsed]
    if missing:
        raise ValueError(f"LLM response missing keys: {', '.join(missing)}")

    return {
        "title": parsed["title"],
        "keywords": parsed["keywords"],
        "summary": parsed["summary"],
        "concepts": parsed["concepts"],
    }


def enrich_document(
    document: dict[str, Any], config: EnrichConfig
) -> dict[str, Any]:
    """Orchestrate enrichment: extract text, call LLM, merge results.

    Returns the updated document dict with enrichment keys added at top level.
    """
    text = extract_text(document)
    enrichment = call_llm(text, config)

    document["title"] = enrichment["title"]
    document["keywords"] = enrichment["keywords"]
    document["summary"] = enrichment["summary"]
    document["concepts"] = enrichment["concepts"]

    return document
