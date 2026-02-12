"""Configuration loading, merging, and validation."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TransskriboConfig:
    """Immutable configuration for a Transskribo run."""

    input_dir: Path
    output_dir: Path
    hf_token: str
    model_size: str = "large-v3"
    language: str = "pt"
    compute_type: str = "float16"
    batch_size: int = 8
    device: str = "cuda"
    log_level: str = "INFO"
    max_duration_hours: float = 0


@dataclass(frozen=True)
class EnrichConfig:
    """Immutable configuration for the enrich command."""

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    template_path: Path = Path("templates/basic.docx")
    transcritor: str = "Jonas Rodrigues (via IA)"


_DEFAULTS: dict[str, Any] = {
    "model_size": "large-v3",
    "language": "pt",
    "compute_type": "float16",
    "batch_size": 8,
    "device": "cuda",
    "log_level": "INFO",
    "max_duration_hours": 0,
}

_ENRICH_DEFAULTS: dict[str, Any] = {
    "llm_base_url": "https://api.openai.com/v1",
    "llm_api_key": "",
    "llm_model": "gpt-4o-mini",
    "template_path": "templates/basic.docx",
    "transcritor": "Jonas Rodrigues (via IA)",
}


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML config file and return a dict."""
    with path.open("rb") as f:
        return tomllib.load(f)


def merge_config(
    file_config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> TransskriboConfig:
    """Merge defaults, file config, and CLI overrides into a validated config.

    Priority: defaults < file config < CLI overrides.
    The HF token falls back to the HF_TOKEN environment variable.
    """
    merged: dict[str, Any] = {**_DEFAULTS}
    merged.update({k: v for k, v in file_config.items() if v is not None})
    merged.update({k: v for k, v in cli_overrides.items() if v is not None})

    # Resolve HF token from env if not in config
    if not merged.get("hf_token"):
        env_token = os.environ.get("HF_TOKEN", "")
        if env_token:
            merged["hf_token"] = env_token

    # Convert string paths to Path objects
    if "input_dir" in merged:
        merged["input_dir"] = Path(merged["input_dir"])
    if "output_dir" in merged:
        merged["output_dir"] = Path(merged["output_dir"])

    return _validate(merged)


def _validate(merged: dict[str, Any]) -> TransskriboConfig:
    """Validate the merged config and return a TransskriboConfig."""
    errors: list[str] = []

    # Required fields
    if "input_dir" not in merged:
        errors.append("input_dir is required")
    if "output_dir" not in merged:
        errors.append("output_dir is required")
    if not merged.get("hf_token"):
        errors.append(
            "hf_token is required (set in config file or HF_TOKEN env var)"
        )

    if errors:
        raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))

    input_dir = merged["input_dir"]
    if isinstance(input_dir, Path) and not input_dir.is_dir():
        raise ValueError(f"input_dir does not exist: {input_dir}")

    output_dir = merged["output_dir"]
    if isinstance(output_dir, Path):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create output_dir: {output_dir}") from e

    return TransskriboConfig(
        input_dir=merged["input_dir"],
        output_dir=merged["output_dir"],
        hf_token=merged["hf_token"],
        model_size=merged.get("model_size", _DEFAULTS["model_size"]),
        language=merged.get("language", _DEFAULTS["language"]),
        compute_type=merged.get("compute_type", _DEFAULTS["compute_type"]),
        batch_size=int(merged.get("batch_size", _DEFAULTS["batch_size"])),
        device=merged.get("device", _DEFAULTS["device"]),
        log_level=merged.get("log_level", _DEFAULTS["log_level"]),
        max_duration_hours=float(
            merged.get("max_duration_hours", _DEFAULTS["max_duration_hours"])
        ),
    )


def load_enrich_config(
    file_config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> EnrichConfig:
    """Load enrich config from the [enrich] section of a TOML dict.

    Priority: defaults < file config [enrich] section < CLI overrides.
    The API key falls back to the ENRICH_API_KEY environment variable.
    """
    enrich_section = file_config.get("enrich", {})
    if not isinstance(enrich_section, dict):
        enrich_section = {}

    merged: dict[str, Any] = {**_ENRICH_DEFAULTS}
    merged.update({k: v for k, v in enrich_section.items() if v is not None})
    merged.update({k: v for k, v in cli_overrides.items() if v is not None})

    # Resolve API key from env if not in config
    if not merged.get("llm_api_key"):
        env_key = os.environ.get("ENRICH_API_KEY", "")
        if env_key:
            merged["llm_api_key"] = env_key

    # Convert template_path to Path
    if "template_path" in merged:
        merged["template_path"] = Path(merged["template_path"])

    return EnrichConfig(
        llm_base_url=merged.get("llm_base_url", _ENRICH_DEFAULTS["llm_base_url"]),
        llm_api_key=merged.get("llm_api_key", _ENRICH_DEFAULTS["llm_api_key"]),
        llm_model=merged.get("llm_model", _ENRICH_DEFAULTS["llm_model"]),
        template_path=merged.get("template_path", Path(_ENRICH_DEFAULTS["template_path"])),
        transcritor=merged.get("transcritor", _ENRICH_DEFAULTS["transcritor"]),
    )
