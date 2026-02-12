"""Tests for configuration loading, merging, and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomli_w

from transskribo.config import (
    EnrichConfig,
    TransskriboConfig,
    load_config,
    load_enrich_config,
    merge_config,
)


class TestLoadConfig:
    """Tests for load_config()."""

    def test_loads_valid_toml(self, sample_config_file: Path) -> None:
        result = load_config(sample_config_file)
        assert isinstance(result, dict)
        assert "hf_token" in result

    def test_loads_all_fields(self, tmp_path: Path) -> None:
        data = {
            "input_dir": "/some/path",
            "output_dir": "/other/path",
            "hf_token": "hf_abc",
            "model_size": "small",
            "batch_size": 4,
        }
        path = tmp_path / "config.toml"
        path.write_bytes(tomli_w.dumps(data).encode())
        result = load_config(path)
        assert result["model_size"] == "small"
        assert result["batch_size"] == 4

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.toml")


class TestMergeConfig:
    """Tests for merge_config()."""

    def test_defaults_applied(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        config = merge_config(sample_config_dict, {})
        assert config.model_size == "large-v3"
        assert config.language == "pt"
        assert config.compute_type == "float16"
        assert config.batch_size == 8
        assert config.device == "cuda"
        assert config.log_level == "INFO"
        assert config.max_duration_hours == 0

    def test_file_config_overrides_defaults(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        sample_config_dict["model_size"] = "small"
        sample_config_dict["batch_size"] = 4
        config = merge_config(sample_config_dict, {})
        assert config.model_size == "small"
        assert config.batch_size == 4

    def test_cli_overrides_file_config(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        sample_config_dict["model_size"] = "small"
        config = merge_config(
            sample_config_dict, {"model_size": "medium"}
        )
        assert config.model_size == "medium"

    def test_cli_none_values_ignored(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        sample_config_dict["model_size"] = "small"
        config = merge_config(
            sample_config_dict, {"model_size": None}
        )
        assert config.model_size == "small"

    def test_returns_frozen_dataclass(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        config = merge_config(sample_config_dict, {})
        assert isinstance(config, TransskriboConfig)
        with pytest.raises(AttributeError):
            config.model_size = "small"  # type: ignore[misc]

    def test_paths_are_path_objects(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        config = merge_config(sample_config_dict, {})
        assert isinstance(config.input_dir, Path)
        assert isinstance(config.output_dir, Path)

    def test_hf_token_from_env(
        self,
        tmp_input_dir: Path,
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        file_config: dict[str, Any] = {
            "input_dir": str(tmp_input_dir),
            "output_dir": str(tmp_output_dir),
        }
        config = merge_config(file_config, {})
        assert config.hf_token == "hf_from_env"

    def test_file_token_overrides_env(
        self,
        sample_config_dict: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        config = merge_config(sample_config_dict, {})
        assert config.hf_token == "hf_test_token_123"

    def test_cli_token_overrides_all(
        self,
        sample_config_dict: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        config = merge_config(
            sample_config_dict, {"hf_token": "hf_from_cli"}
        )
        assert config.hf_token == "hf_from_cli"


class TestConfigValidation:
    """Tests for config validation."""

    def test_missing_input_dir_raises(
        self, tmp_output_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="input_dir is required"):
            merge_config(
                {"output_dir": str(tmp_output_dir), "hf_token": "hf_x"},
                {},
            )

    def test_missing_output_dir_raises(
        self, tmp_input_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="output_dir is required"):
            merge_config(
                {"input_dir": str(tmp_input_dir), "hf_token": "hf_x"},
                {},
            )

    def test_missing_hf_token_raises(
        self,
        tmp_input_dir: Path,
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(ValueError, match="hf_token is required"):
            merge_config(
                {
                    "input_dir": str(tmp_input_dir),
                    "output_dir": str(tmp_output_dir),
                },
                {},
            )

    def test_nonexistent_input_dir_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="input_dir does not exist"):
            merge_config(
                {
                    "input_dir": str(tmp_path / "does_not_exist"),
                    "output_dir": str(tmp_path / "output"),
                    "hf_token": "hf_x",
                },
                {},
            )

    def test_output_dir_created_if_missing(
        self, tmp_input_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "new_output" / "nested"
        config = merge_config(
            {
                "input_dir": str(tmp_input_dir),
                "output_dir": str(out),
                "hf_token": "hf_x",
            },
            {},
        )
        assert out.is_dir()
        assert config.output_dir == out

    def test_valid_config_succeeds(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        config = merge_config(sample_config_dict, {})
        assert config.hf_token == "hf_test_token_123"
        assert config.model_size == "large-v3"

    def test_batch_size_converted_to_int(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        sample_config_dict["batch_size"] = "16"
        config = merge_config(sample_config_dict, {})
        assert config.batch_size == 16
        assert isinstance(config.batch_size, int)

    def test_max_duration_hours_converted_to_float(
        self, sample_config_dict: dict[str, Any]
    ) -> None:
        sample_config_dict["max_duration_hours"] = 3
        config = merge_config(sample_config_dict, {})
        assert config.max_duration_hours == 3.0
        assert isinstance(config.max_duration_hours, float)


class TestEnrichConfig:
    """Tests for EnrichConfig loading and defaults."""

    def test_defaults(self) -> None:
        config = load_enrich_config({}, {})
        assert config.llm_base_url == "https://api.openai.com/v1"
        assert config.llm_api_key == ""
        assert config.llm_model == "gpt-4o-mini"
        assert config.template_path == Path("templates/basic.docx")
        assert config.transcritor == "Jonas Rodrigues (via IA)"

    def test_loads_from_enrich_section(self) -> None:
        file_config: dict[str, Any] = {
            "enrich": {
                "llm_base_url": "http://localhost:11434/v1",
                "llm_api_key": "key-from-file",
                "llm_model": "llama3",
                "template_path": "custom/template.docx",
                "transcritor": "Custom Name",
            }
        }
        config = load_enrich_config(file_config, {})
        assert config.llm_base_url == "http://localhost:11434/v1"
        assert config.llm_api_key == "key-from-file"
        assert config.llm_model == "llama3"
        assert config.template_path == Path("custom/template.docx")
        assert config.transcritor == "Custom Name"

    def test_cli_overrides_file_config(self) -> None:
        file_config: dict[str, Any] = {
            "enrich": {"llm_model": "gpt-4"}
        }
        cli_overrides = {"llm_model": "gpt-3.5-turbo"}
        config = load_enrich_config(file_config, cli_overrides)
        assert config.llm_model == "gpt-3.5-turbo"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENRICH_API_KEY", "key-from-env")
        config = load_enrich_config({}, {})
        assert config.llm_api_key == "key-from-env"

    def test_file_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENRICH_API_KEY", "key-from-env")
        file_config: dict[str, Any] = {
            "enrich": {"llm_api_key": "key-from-file"}
        }
        config = load_enrich_config(file_config, {})
        assert config.llm_api_key == "key-from-file"

    def test_cli_key_overrides_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENRICH_API_KEY", "key-from-env")
        file_config: dict[str, Any] = {
            "enrich": {"llm_api_key": "key-from-file"}
        }
        config = load_enrich_config(file_config, {"llm_api_key": "key-from-cli"})
        assert config.llm_api_key == "key-from-cli"

    def test_template_path_is_path_object(self) -> None:
        config = load_enrich_config({}, {})
        assert isinstance(config.template_path, Path)

    def test_frozen_dataclass(self) -> None:
        config = load_enrich_config({}, {})
        assert isinstance(config, EnrichConfig)
        with pytest.raises(AttributeError):
            config.llm_model = "different"  # type: ignore[misc]

    def test_no_enrich_section_uses_defaults(self) -> None:
        file_config: dict[str, Any] = {"input_dir": "/some/path"}
        config = load_enrich_config(file_config, {})
        assert config.llm_model == "gpt-4o-mini"

    def test_non_dict_enrich_section_uses_defaults(self) -> None:
        file_config: dict[str, Any] = {"enrich": "invalid"}
        config = load_enrich_config(file_config, {})
        assert config.llm_model == "gpt-4o-mini"
