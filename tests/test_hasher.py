"""Tests for hasher module — hashing, registry CRUD, and atomic writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from transskribo.hasher import (
    RegistryEntry,
    compute_hash,
    load_registry,
    lookup_hash,
    register_hash,
    save_registry,
)


# -- compute_hash tests --


class TestComputeHash:
    def test_deterministic_same_content(self, tmp_path: Path) -> None:
        """Same file content produces the same hash."""
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio content 12345")
        assert compute_hash(f) == compute_hash(f)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different file content produces different hashes."""
        f1 = tmp_path / "a.mp3"
        f2 = tmp_path / "b.mp3"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_hash(f1) != compute_hash(f2)

    def test_same_content_different_names(self, tmp_path: Path) -> None:
        """Files with identical content but different names produce the same hash."""
        f1 = tmp_path / "copy1.wav"
        f2 = tmp_path / "copy2.wav"
        content = b"identical audio bytes"
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert compute_hash(f1) == compute_hash(f2)

    def test_hash_is_hex_string(self, tmp_path: Path) -> None:
        """Hash is a 64-character hex string (SHA-256)."""
        f = tmp_path / "test.flac"
        f.write_bytes(b"some bytes")
        h = compute_hash(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_file_has_valid_hash(self, tmp_path: Path) -> None:
        """Empty file produces a valid (deterministic) hash."""
        f = tmp_path / "empty.mp3"
        f.write_bytes(b"")
        h = compute_hash(f)
        assert len(h) == 64


# -- RegistryEntry tests --


class TestRegistryEntry:
    def test_frozen(self) -> None:
        entry = RegistryEntry(
            source_path="/a.mp3",
            output_path="/a.json",
            timestamp="2026-01-01T00:00:00",
            status="success",
        )
        with pytest.raises(AttributeError):
            entry.status = "failed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        entry = RegistryEntry(
            source_path="/a.mp3",
            output_path="/a.json",
            timestamp="2026-01-01T00:00:00",
            status="success",
        )
        assert entry.duration_audio_secs is None
        assert entry.timing is None
        assert entry.error is None

    def test_with_timing(self) -> None:
        timing = {
            "transcribe_secs": 10.0,
            "align_secs": 2.0,
            "diarize_secs": 5.0,
            "total_secs": 17.0,
        }
        entry = RegistryEntry(
            source_path="/a.mp3",
            output_path="/a.json",
            timestamp="2026-01-01T00:00:00",
            status="success",
            duration_audio_secs=3600.0,
            timing=timing,
        )
        assert entry.timing == timing
        assert entry.duration_audio_secs == 3600.0


# -- load_registry / save_registry tests --


class TestRegistryPersistence:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Loading a non-existent registry returns an empty dict."""
        reg = load_registry(tmp_path / "nonexistent.json")
        assert reg == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Registry survives a save/load cycle."""
        reg_path = tmp_path / ".transskribo" / "registry.json"
        registry: dict[str, Any] = {
            "abc123": {
                "source_path": "/input/a.mp3",
                "output_path": "/output/a.json",
                "timestamp": "2026-01-01T00:00:00",
                "status": "success",
                "duration_audio_secs": 120.5,
                "timing": {
                    "transcribe_secs": 10.0,
                    "align_secs": 2.0,
                    "diarize_secs": 5.0,
                    "total_secs": 17.0,
                },
                "error": None,
            }
        }
        save_registry(registry, reg_path)
        loaded = load_registry(reg_path)
        assert loaded == registry

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_registry creates intermediate directories."""
        reg_path = tmp_path / "deep" / "nested" / "registry.json"
        save_registry({"k": "v"}, reg_path)
        assert reg_path.exists()

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        """Saving overwrites the previous registry content."""
        reg_path = tmp_path / "registry.json"
        save_registry({"old": "data"}, reg_path)
        save_registry({"new": "data"}, reg_path)
        loaded = load_registry(reg_path)
        assert loaded == {"new": "data"}

    def test_atomic_write_no_partial_on_error(self, tmp_path: Path) -> None:
        """If writing fails, the original file is untouched."""
        reg_path = tmp_path / "registry.json"
        original = {"original": "data"}
        save_registry(original, reg_path)

        # Attempt to save something that will fail during json.dump
        bad_registry: dict[str, Any] = {"key": object()}
        with pytest.raises(TypeError):
            save_registry(bad_registry, reg_path)

        # Original file should be intact
        loaded = load_registry(reg_path)
        assert loaded == original

    def test_atomic_write_cleanup_on_os_error(self, tmp_path: Path) -> None:
        """Temp file is cleaned up if rename fails."""
        reg_path = tmp_path / "registry.json"
        registry = {"test": "data"}

        with patch("transskribo.hasher.Path.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                save_registry(registry, reg_path)

        # No leftover temp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


# -- lookup_hash tests --


class TestLookupHash:
    def test_lookup_hit(self) -> None:
        """Lookup returns entry when hash exists with status 'success'."""
        registry: dict[str, Any] = {
            "abc123": {
                "source_path": "/a.mp3",
                "output_path": "/a.json",
                "timestamp": "2026-01-01T00:00:00",
                "status": "success",
            }
        }
        result = lookup_hash(registry, "abc123")
        assert result is not None
        assert result["source_path"] == "/a.mp3"

    def test_lookup_miss(self) -> None:
        """Lookup returns None when hash is not in registry."""
        registry: dict[str, Any] = {}
        assert lookup_hash(registry, "nonexistent") is None

    def test_lookup_failed_status(self) -> None:
        """Lookup returns None when hash exists but status is 'failed'."""
        registry: dict[str, Any] = {
            "abc123": {
                "source_path": "/a.mp3",
                "output_path": "/a.json",
                "timestamp": "2026-01-01T00:00:00",
                "status": "failed",
                "error": "OOM",
            }
        }
        assert lookup_hash(registry, "abc123") is None


# -- register_hash tests --


class TestRegisterHash:
    def test_register_new_entry(self) -> None:
        """Register adds a new entry to the registry."""
        registry: dict[str, Any] = {}
        register_hash(
            registry,
            "hash1",
            source_path="/input/a.mp3",
            output_path="/output/a.json",
            timestamp="2026-01-01T00:00:00",
            status="success",
            duration_audio_secs=120.0,
            timing={
                "transcribe_secs": 10.0,
                "align_secs": 2.0,
                "diarize_secs": 5.0,
                "total_secs": 17.0,
            },
        )
        assert "hash1" in registry
        entry = registry["hash1"]
        assert entry["source_path"] == "/input/a.mp3"
        assert entry["status"] == "success"
        assert entry["duration_audio_secs"] == 120.0
        assert entry["timing"]["transcribe_secs"] == 10.0

    def test_register_overwrites_existing(self) -> None:
        """Re-registering a hash updates the entry."""
        registry: dict[str, Any] = {
            "hash1": {
                "source_path": "/old.mp3",
                "output_path": "/old.json",
                "timestamp": "2026-01-01T00:00:00",
                "status": "failed",
                "error": "OOM",
            }
        }
        register_hash(
            registry,
            "hash1",
            source_path="/new.mp3",
            output_path="/new.json",
            timestamp="2026-01-02T00:00:00",
            status="success",
            duration_audio_secs=300.0,
        )
        assert registry["hash1"]["status"] == "success"
        assert registry["hash1"]["source_path"] == "/new.mp3"
        assert registry["hash1"]["error"] is None

    def test_register_failed_with_error(self) -> None:
        """Register a failed entry with an error message."""
        registry: dict[str, Any] = {}
        register_hash(
            registry,
            "hash2",
            source_path="/input/b.mp3",
            output_path="/output/b.json",
            timestamp="2026-01-01T00:00:00",
            status="failed",
            error="CUDA out of memory",
        )
        entry = registry["hash2"]
        assert entry["status"] == "failed"
        assert entry["error"] == "CUDA out of memory"
        assert entry["timing"] is None

    def test_register_preserves_other_entries(self) -> None:
        """Registering a new hash doesn't affect existing entries."""
        registry: dict[str, Any] = {
            "existing": {"source_path": "/x.mp3", "status": "success"}
        }
        register_hash(
            registry,
            "new_hash",
            source_path="/y.mp3",
            output_path="/y.json",
            timestamp="2026-01-01T00:00:00",
            status="success",
        )
        assert "existing" in registry
        assert "new_hash" in registry


# -- Full roundtrip test --


class TestFullRoundtrip:
    def test_hash_register_save_load_lookup(self, tmp_path: Path) -> None:
        """End-to-end: hash a file, register it, save, load, lookup."""
        # Create a test file
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake audio data for roundtrip test")

        # Hash it
        file_hash = compute_hash(audio)

        # Register it
        registry: dict[str, Any] = {}
        register_hash(
            registry,
            file_hash,
            source_path=str(audio),
            output_path=str(tmp_path / "test.json"),
            timestamp="2026-01-01T12:00:00",
            status="success",
            duration_audio_secs=600.0,
            timing={
                "transcribe_secs": 30.0,
                "align_secs": 5.0,
                "diarize_secs": 10.0,
                "total_secs": 45.0,
            },
        )

        # Save to disk
        reg_path = tmp_path / ".transskribo" / "registry.json"
        save_registry(registry, reg_path)

        # Load from disk
        loaded = load_registry(reg_path)

        # Lookup should find the entry
        result = lookup_hash(loaded, file_hash)
        assert result is not None
        assert result["status"] == "success"
        assert result["duration_audio_secs"] == 600.0
        assert result["timing"]["total_secs"] == 45.0

        # Different hash should miss
        assert lookup_hash(loaded, "nonexistent_hash") is None
