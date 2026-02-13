"""Tests for docx_writer module: template rendering and document generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transskribo.config import EnrichConfig
from transskribo.docx_writer import generate_docx, remap_speakers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def enrich_config(tmp_path: Path) -> EnrichConfig:
    """EnrichConfig with a template path pointing to a temp file."""
    template_path = tmp_path / "template.docx"
    template_path.touch()
    return EnrichConfig(
        llm_base_url="https://api.test.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        template_path=template_path,
        transcritor="Test Transcriber",
    )


@pytest.fixture
def sample_concepts() -> dict[str, Any]:
    return {
        "title": "Aula de Teste",
        "keywords": ["teste", "aula"],
        "summary": "Resumo da aula.",
        "concepts": {"teste": "Conceito de teste"},
    }


@pytest.fixture
def sample_segments() -> list[dict[str, Any]]:
    return [
        {"speaker": "SPEAKER_00", "texts": ["Bom dia.", "Vamos começar."]},
        {"speaker": "SPEAKER_01", "texts": ["Tenho uma pergunta."]},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateDocx:
    def test_creates_docx_file(
        self,
        tmp_path: Path,
        enrich_config: EnrichConfig,
        sample_concepts: dict[str, Any],
        sample_segments: list[dict[str, Any]],
    ) -> None:
        """generate_docx should create a .docx file at the target path."""
        output_path = tmp_path / "output" / "test.docx"

        with patch("transskribo.docx_writer.DocxTemplate") as mock_tpl_cls:
            mock_doc = MagicMock()
            mock_tpl_cls.return_value = mock_doc

            generate_docx(output_path, "aula.mp3", sample_concepts, sample_segments, enrich_config)

            mock_tpl_cls.assert_called_once_with(str(enrich_config.template_path))
            mock_doc.render.assert_called_once()
            mock_doc.save.assert_called_once_with(str(output_path))

    def test_template_not_found_raises(
        self,
        tmp_path: Path,
        sample_concepts: dict[str, Any],
        sample_segments: list[dict[str, Any]],
    ) -> None:
        """generate_docx should raise FileNotFoundError if template doesn't exist."""
        config = EnrichConfig(
            template_path=tmp_path / "nonexistent.docx",
        )
        output_path = tmp_path / "output.docx"

        with pytest.raises(FileNotFoundError, match="Template file not found"):
            generate_docx(output_path, "test.mp3", sample_concepts, sample_segments, config)

    def test_context_variables_passed(
        self,
        tmp_path: Path,
        enrich_config: EnrichConfig,
        sample_concepts: dict[str, Any],
        sample_segments: list[dict[str, Any]],
    ) -> None:
        """generate_docx should pass correct context to the template."""
        output_path = tmp_path / "test.docx"

        with patch("transskribo.docx_writer.DocxTemplate") as mock_tpl_cls:
            mock_doc = MagicMock()
            mock_tpl_cls.return_value = mock_doc

            generate_docx(output_path, "aula.mp3", sample_concepts, sample_segments, enrich_config)

            context = mock_doc.render.call_args[0][0]
            assert context["arquivo"] == "aula.mp3"
            assert context["transcritor"] == "Test Transcriber"
            assert "data_transcricao" in context
            assert context["info"] == sample_concepts
            assert context["segmentos"] == sample_segments

    def test_creates_parent_directories(
        self,
        tmp_path: Path,
        enrich_config: EnrichConfig,
        sample_concepts: dict[str, Any],
        sample_segments: list[dict[str, Any]],
    ) -> None:
        """generate_docx should create parent directories if they don't exist."""
        output_path = tmp_path / "deep" / "nested" / "dir" / "test.docx"

        with patch("transskribo.docx_writer.DocxTemplate") as mock_tpl_cls:
            mock_doc = MagicMock()
            mock_tpl_cls.return_value = mock_doc

            generate_docx(output_path, "aula.mp3", sample_concepts, sample_segments, enrich_config)

            assert output_path.parent.exists()


class TestRemapSpeakers:
    def test_ranks_by_segment_count(self) -> None:
        """Speaker with most segments becomes 'Pessoa 01'."""
        document = {
            "segments": [
                {"speaker": "SPEAKER_01", "text": "a"},
                {"speaker": "SPEAKER_00", "text": "b"},
                {"speaker": "SPEAKER_00", "text": "c"},
                {"speaker": "SPEAKER_00", "text": "d"},
                {"speaker": "SPEAKER_01", "text": "e"},
            ],
        }
        turns = [
            {"speaker": "SPEAKER_01", "texts": ["a"]},
            {"speaker": "SPEAKER_00", "texts": ["b", "c", "d"]},
            {"speaker": "SPEAKER_01", "texts": ["e"]},
        ]
        result = remap_speakers(turns, document)
        # SPEAKER_00 has 3 segments -> Pessoa 01, SPEAKER_01 has 2 -> Pessoa 02
        assert result[0]["speaker"] == "Pessoa 02"
        assert result[1]["speaker"] == "Pessoa 01"
        assert result[2]["speaker"] == "Pessoa 02"

    def test_alphabetical_tiebreak(self) -> None:
        """Speakers with equal segment counts are ordered alphabetically."""
        document = {
            "segments": [
                {"speaker": "SPEAKER_02", "text": "a"},
                {"speaker": "SPEAKER_00", "text": "b"},
            ],
        }
        turns = [
            {"speaker": "SPEAKER_02", "texts": ["a"]},
            {"speaker": "SPEAKER_00", "texts": ["b"]},
        ]
        result = remap_speakers(turns, document)
        # Both have 1 segment, SPEAKER_00 < SPEAKER_02 alphabetically
        assert result[0]["speaker"] == "Pessoa 02"  # SPEAKER_02
        assert result[1]["speaker"] == "Pessoa 01"  # SPEAKER_00

    def test_single_speaker(self) -> None:
        """Single speaker becomes 'Pessoa 01'."""
        document = {
            "segments": [
                {"speaker": "SPEAKER_00", "text": "a"},
                {"speaker": "SPEAKER_00", "text": "b"},
            ],
        }
        turns = [{"speaker": "SPEAKER_00", "texts": ["a", "b"]}]
        result = remap_speakers(turns, document)
        assert result[0]["speaker"] == "Pessoa 01"

    def test_empty_segments(self) -> None:
        """Empty document returns empty turns."""
        document: dict[str, Any] = {"segments": []}
        turns: list[dict[str, Any]] = []
        result = remap_speakers(turns, document)
        assert result == []

    def test_preserves_texts(self) -> None:
        """Remapping should not alter the texts in turns."""
        document = {
            "segments": [
                {"speaker": "SPEAKER_00", "text": "Hello"},
                {"speaker": "SPEAKER_01", "text": "World"},
            ],
        }
        turns = [
            {"speaker": "SPEAKER_00", "texts": ["Hello"]},
            {"speaker": "SPEAKER_01", "texts": ["World"]},
        ]
        result = remap_speakers(turns, document)
        assert result[0]["texts"] == ["Hello"]
        assert result[1]["texts"] == ["World"]

    def test_three_speakers(self) -> None:
        """Three speakers ranked correctly."""
        document = {
            "segments": [
                {"speaker": "C", "text": "1"},
                {"speaker": "C", "text": "2"},
                {"speaker": "C", "text": "3"},
                {"speaker": "A", "text": "4"},
                {"speaker": "A", "text": "5"},
                {"speaker": "B", "text": "6"},
            ],
        }
        turns = [
            {"speaker": "C", "texts": ["1", "2", "3"]},
            {"speaker": "A", "texts": ["4", "5"]},
            {"speaker": "B", "texts": ["6"]},
        ]
        result = remap_speakers(turns, document)
        # C=3 segs -> Pessoa 01, A=2 -> Pessoa 02, B=1 -> Pessoa 03
        assert result[0]["speaker"] == "Pessoa 01"
        assert result[1]["speaker"] == "Pessoa 02"
        assert result[2]["speaker"] == "Pessoa 03"

    def test_unknown_speaker_becomes_pessoa_question_mark(self) -> None:
        """UNKNOWN speakers become 'Pessoa ??' and don't affect numbering."""
        document = {
            "segments": [
                {"speaker": "SPEAKER_00", "text": "a"},
                {"speaker": "SPEAKER_00", "text": "b"},
                {"text": "c"},  # missing speaker -> UNKNOWN
                {"speaker": "SPEAKER_01", "text": "d"},
            ],
        }
        turns = [
            {"speaker": "SPEAKER_00", "texts": ["a", "b"]},
            {"speaker": "UNKNOWN", "texts": ["c"]},
            {"speaker": "SPEAKER_01", "texts": ["d"]},
        ]
        result = remap_speakers(turns, document)
        # SPEAKER_00=2 segs -> Pessoa 01, SPEAKER_01=1 -> Pessoa 02
        assert result[0]["speaker"] == "Pessoa 01"
        assert result[1]["speaker"] == "Pessoa ??"
        assert result[2]["speaker"] == "Pessoa 02"
