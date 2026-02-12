"""Tests for docx_writer module: template rendering and document generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transskribo.config import EnrichConfig
from transskribo.docx_writer import generate_docx


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
