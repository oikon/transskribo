"""SHA-256 hashing and hash registry for duplicate detection."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegistryEntry:
    """A single entry in the hash registry."""

    source_path: str
    output_path: str
    timestamp: str
    status: str
    duration_audio_secs: float | None = None
    timing: dict[str, float] | None = None
    error: str | None = None


def compute_hash(file_path: Path) -> str:
    """Stream SHA-256 hash of a file and return the hex digest."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_registry(registry_path: Path) -> dict[str, Any]:
    """Load the hash registry from disk. Returns empty dict if not found."""
    if not registry_path.exists():
        return {}
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict[str, Any], registry_path: Path) -> None:
    """Save the registry to disk atomically (write temp file, then rename)."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=registry_path.parent, suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        tmp_path.replace(registry_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def lookup_hash(
    registry: dict[str, Any], file_hash: str
) -> dict[str, Any] | None:
    """Return the registry entry if hash was seen with status 'success'."""
    entry = registry.get(file_hash)
    if entry is not None and entry.get("status") == "success":
        return entry
    return None


def register_hash(
    registry: dict[str, Any],
    file_hash: str,
    *,
    source_path: str,
    output_path: str,
    timestamp: str,
    status: str,
    duration_audio_secs: float | None = None,
    timing: dict[str, float] | None = None,
    error: str | None = None,
) -> None:
    """Add or update a registry entry for the given hash."""
    entry = RegistryEntry(
        source_path=source_path,
        output_path=output_path,
        timestamp=timestamp,
        status=status,
        duration_audio_secs=duration_audio_secs,
        timing=timing,
        error=error,
    )
    registry[file_hash] = asdict(entry)
