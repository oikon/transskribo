"""LLM-powered concept extraction from transcription results."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from transskribo.config import EnrichConfig

logger = logging.getLogger(__name__)

ENRICHMENT_KEYS = ("title", "keywords", "summary", "concepts")

_SYSTEM_PROMPT = """Você é um assistente especializado em análise de transcrições em português brasileiro.

A partir da transcrição abaixo, extraia informações estruturadas.

Instruções:
- Ignore repetições, hesitações e ruídos típicos de fala.
- Foque nos temas centrais e argumentos principais.
- Não invente informações que não estejam explicitamente ou implicitamente presentes.

Retorne exclusivamente JSON válido com a seguinte estrutura:

{
  "title": string,
  "keywords": string[],           // 5 a 15 palavras-chave concisas
  "summary": string,              // aproximadamente 200 palavras
  "concepts": [
    {
      "name": string,
      "explanation": string       // explicação breve (1–3 frases)
    }
  ]                                // até 15 conceitos
}
"""


class Concept(BaseModel):
    """A single concept extracted from the transcription."""

    name: str
    explanation: str


class EnrichmentResult(BaseModel):
    """Schema for LLM-extracted metadata from transcriptions."""

    title: str
    keywords: list[str]
    summary: str
    concepts: list[Concept]


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

    Uses Structured Outputs (Pydantic schema) for guaranteed schema compliance.

    Args:
        text: Plain text from the transcription.
        config: Enrich configuration with LLM endpoint details.

    Returns:
        Dict with keys: title (str), keywords (list[str]),
        summary (str), concepts (dict[str, str]).

    Raises:
        ValueError: If the LLM response is refused or empty.
        RuntimeError: If the API call fails.
    """
    from openai import LengthFinishReasonError, OpenAI

    try:
        client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key or None,
        )
        completion = client.chat.completions.parse(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format=EnrichmentResult,
        )
    except LengthFinishReasonError as e:
        raise ValueError("LLM output truncated (max tokens reached)") from e
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}") from e

    message = completion.choices[0].message

    if message.refusal:
        raise ValueError(f"LLM refused the request: {message.refusal}")

    if not message.parsed:
        raise ValueError("LLM returned empty response")

    result = message.parsed
    return {
        "title": result.title,
        "keywords": result.keywords,
        "summary": result.summary,
        "concepts": {c.name: c.explanation for c in result.concepts},
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
