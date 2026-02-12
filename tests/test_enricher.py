"""Tests for enricher module: text extraction, speaker turns, LLM call, enrichment."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transskribo.config import EnrichConfig
from transskribo.enricher import (
    call_llm,
    enrich_document,
    extract_text,
    group_speaker_turns,
    is_enriched,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document() -> dict[str, Any]:
    """A typical transcription result JSON."""
    return {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Bom dia, turma.", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "Vamos começar a aula.", "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 15.0, "text": "Tenho uma pergunta.", "speaker": "SPEAKER_01"},
            {"start": 15.0, "end": 20.0, "text": "Pode perguntar.", "speaker": "SPEAKER_00"},
        ],
        "words": [],
        "metadata": {
            "source_file": "/input/aula.mp3",
            "file_hash": "abc123",
            "duration_secs": 20.0,
        },
    }


@pytest.fixture
def enrich_config() -> EnrichConfig:
    return EnrichConfig(
        llm_base_url="https://api.test.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )


# ---------------------------------------------------------------------------
# extract_text tests
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_multiple_segments(self, sample_document: dict[str, Any]) -> None:
        result = extract_text(sample_document)
        assert result == "Bom dia, turma. Vamos começar a aula. Tenho uma pergunta. Pode perguntar."

    def test_empty_segments(self) -> None:
        doc: dict[str, Any] = {"segments": []}
        assert extract_text(doc) == ""

    def test_no_segments_key(self) -> None:
        doc: dict[str, Any] = {}
        assert extract_text(doc) == ""

    def test_single_segment(self) -> None:
        doc: dict[str, Any] = {"segments": [{"text": "Hello world."}]}
        assert extract_text(doc) == "Hello world."

    def test_missing_text_field(self) -> None:
        doc: dict[str, Any] = {"segments": [{"start": 0.0, "end": 1.0}]}
        assert extract_text(doc) == ""

    def test_empty_text_field(self) -> None:
        doc: dict[str, Any] = {"segments": [{"text": ""}, {"text": "Hello"}]}
        assert extract_text(doc) == "Hello"


# ---------------------------------------------------------------------------
# group_speaker_turns tests
# ---------------------------------------------------------------------------

class TestGroupSpeakerTurns:
    def test_alternating_speakers(self, sample_document: dict[str, Any]) -> None:
        turns = group_speaker_turns(sample_document)
        assert len(turns) == 3
        # First turn: SPEAKER_00 with 2 consecutive segments
        assert turns[0]["speaker"] == "SPEAKER_00"
        assert turns[0]["texts"] == ["Bom dia, turma.", "Vamos começar a aula."]
        # Second turn: SPEAKER_01
        assert turns[1]["speaker"] == "SPEAKER_01"
        assert turns[1]["texts"] == ["Tenho uma pergunta."]
        # Third turn: SPEAKER_00 again
        assert turns[2]["speaker"] == "SPEAKER_00"
        assert turns[2]["texts"] == ["Pode perguntar."]

    def test_single_speaker_throughout(self) -> None:
        doc: dict[str, Any] = {
            "segments": [
                {"text": "First.", "speaker": "SPEAKER_00"},
                {"text": "Second.", "speaker": "SPEAKER_00"},
                {"text": "Third.", "speaker": "SPEAKER_00"},
            ]
        }
        turns = group_speaker_turns(doc)
        assert len(turns) == 1
        assert turns[0]["speaker"] == "SPEAKER_00"
        assert turns[0]["texts"] == ["First.", "Second.", "Third."]

    def test_empty_segments(self) -> None:
        doc: dict[str, Any] = {"segments": []}
        assert group_speaker_turns(doc) == []

    def test_no_segments_key(self) -> None:
        doc: dict[str, Any] = {}
        assert group_speaker_turns(doc) == []

    def test_missing_speaker_field(self) -> None:
        doc: dict[str, Any] = {
            "segments": [
                {"text": "Hello."},
                {"text": "World."},
            ]
        }
        turns = group_speaker_turns(doc)
        assert len(turns) == 1
        assert turns[0]["speaker"] == "UNKNOWN"

    def test_empty_text_segments(self) -> None:
        doc: dict[str, Any] = {
            "segments": [
                {"text": "", "speaker": "SPEAKER_00"},
                {"text": "Hello.", "speaker": "SPEAKER_00"},
            ]
        }
        turns = group_speaker_turns(doc)
        assert len(turns) == 1
        assert turns[0]["texts"] == ["Hello."]


# ---------------------------------------------------------------------------
# is_enriched tests
# ---------------------------------------------------------------------------

class TestIsEnriched:
    def test_all_keys_present(self) -> None:
        doc: dict[str, Any] = {
            "title": "Test",
            "keywords": ["a"],
            "summary": "Summary",
            "concepts": {"x": "y"},
            "segments": [],
            "metadata": {},
        }
        assert is_enriched(doc) is True

    def test_partial_keys(self) -> None:
        doc: dict[str, Any] = {
            "title": "Test",
            "keywords": ["a"],
            "segments": [],
            "metadata": {},
        }
        assert is_enriched(doc) is False

    def test_no_enrichment_keys(self) -> None:
        doc: dict[str, Any] = {
            "segments": [],
            "metadata": {},
        }
        assert is_enriched(doc) is False

    def test_empty_doc(self) -> None:
        assert is_enriched({}) is False


# ---------------------------------------------------------------------------
# call_llm tests
# ---------------------------------------------------------------------------

class TestCallLlm:
    def test_successful_response(self, enrich_config: EnrichConfig) -> None:
        llm_response = {
            "title": "Aula de Introdução",
            "keywords": ["educação", "introdução"],
            "summary": "Resumo da aula.",
            "concepts": {"educação": "Processo de aprendizagem"},
        }

        mock_message = MagicMock()
        mock_message.content = json.dumps(llm_response)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            result = call_llm("Bom dia, turma.", enrich_config)

        assert result["title"] == "Aula de Introdução"
        assert result["keywords"] == ["educação", "introdução"]
        assert result["summary"] == "Resumo da aula."
        assert result["concepts"] == {"educação": "Processo de aprendizagem"}

        mock_openai_cls.assert_called_once_with(
            base_url="https://api.test.com/v1", api_key="test-key"
        )

    def test_malformed_json_response(self, enrich_config: EnrichConfig) -> None:
        mock_message = MagicMock()
        mock_message.content = "not valid json"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            with pytest.raises(ValueError, match="not valid JSON"):
                call_llm("text", enrich_config)

    def test_missing_keys_in_response(self, enrich_config: EnrichConfig) -> None:
        # Response has title but missing other keys
        mock_message = MagicMock()
        mock_message.content = json.dumps({"title": "Test"})
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            with pytest.raises(ValueError, match="missing keys"):
                call_llm("text", enrich_config)

    def test_api_error(self, enrich_config: EnrichConfig) -> None:
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            mock_openai_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="LLM API call failed"):
                call_llm("text", enrich_config)

    def test_empty_response(self, enrich_config: EnrichConfig) -> None:
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            with pytest.raises(ValueError, match="empty response"):
                call_llm("text", enrich_config)


# ---------------------------------------------------------------------------
# enrich_document tests
# ---------------------------------------------------------------------------

class TestEnrichDocument:
    def test_enriches_document(
        self, sample_document: dict[str, Any], enrich_config: EnrichConfig
    ) -> None:
        llm_result = {
            "title": "Aula de Teste",
            "keywords": ["teste", "aula"],
            "summary": "Resumo da aula de teste.",
            "concepts": {"teste": "Verificação"},
        }

        with patch("transskribo.enricher.call_llm", return_value=llm_result) as mock_llm:
            result = enrich_document(sample_document, enrich_config)

        assert result["title"] == "Aula de Teste"
        assert result["keywords"] == ["teste", "aula"]
        assert result["summary"] == "Resumo da aula de teste."
        assert result["concepts"] == {"teste": "Verificação"}
        # Original keys should still be present
        assert "segments" in result
        assert "metadata" in result

        # call_llm should have received the concatenated text
        mock_llm.assert_called_once()
        text_arg = mock_llm.call_args[0][0]
        assert "Bom dia, turma." in text_arg
