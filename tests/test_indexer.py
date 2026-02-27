"""Tests for indexer module — concept index generation."""

from __future__ import annotations

import json
from pathlib import Path

from transskribo.indexer import collect_concepts, generate_index, render_index


def _write_enriched_json(
    path: Path,
    source_file: str = "lecture01.mp3",
    concepts: dict[str, str] | None = None,
) -> None:
    """Write a minimal enriched transskribo result JSON."""
    if concepts is None:
        concepts = {"Recursion": "A function that calls itself."}
    doc = {
        "segments": [{"text": "Hello", "speaker": "SPEAKER_00"}],
        "metadata": {"source_file": source_file},
        "title": "Test Title",
        "keywords": ["test"],
        "summary": "A test summary.",
        "concepts": concepts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _write_non_enriched_json(path: Path) -> None:
    """Write a transskribo result without enrichment keys."""
    doc = {
        "segments": [{"text": "Hello"}],
        "metadata": {"source_file": "file.mp3"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _write_non_transskribo_json(path: Path) -> None:
    """Write a JSON file that is not a transskribo result."""
    doc = {"some_key": "some_value"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


class TestCollectConcepts:
    """Tests for collect_concepts()."""

    def test_empty_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = collect_concepts(output_dir)
        assert result == {}

    def test_nonexistent_output_dir(self, tmp_path: Path) -> None:
        result = collect_concepts(tmp_path / "nonexistent")
        assert result == {}

    def test_single_enriched_file(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "lecture01.json",
            source_file="lecture01.mp3",
            concepts={"Recursion": "Calls itself", "OOP": "Object-oriented"},
        )
        result = collect_concepts(output_dir)
        assert "Recursion" in result
        assert "OOP" in result
        assert result["Recursion"] == [("lecture01", "Calls itself")]
        assert result["OOP"] == [("lecture01", "Object-oriented")]

    def test_multiple_files_group_concepts(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "lec01.json",
            source_file="lec01.mp3",
            concepts={"Recursion": "Self-calling function"},
        )
        _write_enriched_json(
            output_dir / "lec02.json",
            source_file="lec02.mp3",
            concepts={"Recursion": "Base case needed", "Sorting": "Arranging elements"},
        )
        result = collect_concepts(output_dir)
        assert len(result["Recursion"]) == 2
        assert ("lec01", "Self-calling function") in result["Recursion"]
        assert ("lec02", "Base case needed") in result["Recursion"]
        assert len(result["Sorting"]) == 1

    def test_file_label_from_metadata_source_file(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "result.json",
            source_file="/data/input/my_lecture.wav",
            concepts={"Topic": "Explanation"},
        )
        result = collect_concepts(output_dir)
        assert result["Topic"] == [("my_lecture", "Explanation")]

    def test_skips_non_enriched_files(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_non_enriched_json(output_dir / "not_enriched.json")
        _write_enriched_json(
            output_dir / "enriched.json",
            concepts={"Topic": "Explanation"},
        )
        result = collect_concepts(output_dir)
        assert len(result) == 1
        assert "Topic" in result

    def test_skips_transskribo_directory(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / ".transskribo" / "registry.json",
            concepts={"Hidden": "Should not appear"},
        )
        _write_enriched_json(
            output_dir / "visible.json",
            concepts={"Visible": "Should appear"},
        )
        result = collect_concepts(output_dir)
        assert "Hidden" not in result
        assert "Visible" in result

    def test_skips_non_transskribo_jsons(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_non_transskribo_json(output_dir / "other.json")
        result = collect_concepts(output_dir)
        assert result == {}

    def test_bullets_sorted_by_file_label(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "z_file.json",
            source_file="z_lecture.mp3",
            concepts={"Topic": "From Z"},
        )
        _write_enriched_json(
            output_dir / "a_file.json",
            source_file="a_lecture.mp3",
            concepts={"Topic": "From A"},
        )
        result = collect_concepts(output_dir)
        labels = [t[0] for t in result["Topic"]]
        assert labels == ["a_lecture", "z_lecture"]

    def test_handles_corrupt_json(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        corrupt = output_dir / "corrupt.json"
        corrupt.write_text("not valid json{{{", encoding="utf-8")
        _write_enriched_json(
            output_dir / "good.json",
            concepts={"Topic": "Good"},
        )
        result = collect_concepts(output_dir)
        assert "Topic" in result


class TestRenderIndex:
    """Tests for render_index()."""

    def test_empty_concepts(self) -> None:
        result = render_index({})
        assert "# Concept Index" in result
        assert "No concepts found" in result

    def test_single_concept_single_file(self) -> None:
        concepts = {"Recursion": [("lecture01", "A function that calls itself.")]}
        result = render_index(concepts)
        assert "# Concept Index" in result
        assert "## Recursion" in result
        assert "- **lecture01**: A function that calls itself." in result

    def test_single_concept_multiple_files(self) -> None:
        concepts = {
            "Recursion": [
                ("lec01", "Self-calling function"),
                ("lec02", "Needs a base case"),
            ]
        }
        result = render_index(concepts)
        assert "- **lec01**: Self-calling function" in result
        assert "- **lec02**: Needs a base case" in result

    def test_concepts_sorted_alphabetically(self) -> None:
        concepts = {
            "Zebra": [("file1", "Last animal")],
            "Apple": [("file1", "A fruit")],
            "Mango": [("file1", "Another fruit")],
        }
        result = render_index(concepts)
        apple_pos = result.index("## Apple")
        mango_pos = result.index("## Mango")
        zebra_pos = result.index("## Zebra")
        assert apple_pos < mango_pos < zebra_pos

    def test_bullets_sorted_by_file_label(self) -> None:
        concepts = {
            "Topic": [
                ("a_file", "From A"),
                ("z_file", "From Z"),
            ]
        }
        result = render_index(concepts)
        a_pos = result.index("**a_file**")
        z_pos = result.index("**z_file**")
        assert a_pos < z_pos


class TestGenerateIndex:
    """Tests for generate_index()."""

    def test_creates_file_at_target_path(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "file.json",
            concepts={"Topic": "Explanation"},
        )
        index_path = tmp_path / "index.md"
        count = generate_index(output_dir, index_path)
        assert count == 1
        assert index_path.exists()
        content = index_path.read_text(encoding="utf-8")
        assert "## Topic" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "file.json",
            concepts={"Topic": "Explanation"},
        )
        index_path = tmp_path / "nested" / "deep" / "index.md"
        count = generate_index(output_dir, index_path)
        assert count == 1
        assert index_path.exists()

    def test_returns_correct_concept_count(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "file.json",
            concepts={"A": "First", "B": "Second", "C": "Third"},
        )
        count = generate_index(output_dir, tmp_path / "index.md")
        assert count == 3

    def test_no_enriched_files_returns_zero(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        index_path = tmp_path / "index.md"
        count = generate_index(output_dir, index_path)
        assert count == 0
        assert not index_path.exists()

    def test_overwrites_existing_index(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        _write_enriched_json(
            output_dir / "file.json",
            concepts={"NewConcept": "New explanation"},
        )
        index_path = tmp_path / "index.md"
        index_path.write_text("old content", encoding="utf-8")
        generate_index(output_dir, index_path)
        content = index_path.read_text(encoding="utf-8")
        assert "NewConcept" in content
        assert "old content" not in content
