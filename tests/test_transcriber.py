"""Tests for transcriber module — all WhisperX/torch calls are mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transskribo.config import TransskriboConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> TransskriboConfig:
    """Minimal config for transcriber tests."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return TransskriboConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        hf_token="hf_test_token",
        model_size="large-v3",
        language="pt",
        compute_type="float16",
        batch_size=8,
        device="cuda",
    )


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    """Create a dummy audio file path."""
    p = tmp_path / "test.mp3"
    p.write_bytes(b"fake audio data")
    return p


# ---------------------------------------------------------------------------
# 7.04 — Transcription stage tests
# ---------------------------------------------------------------------------


class TestLoadAudio:
    @patch("transskribo.transcriber.whisperx")
    def test_calls_whisperx_load_audio(
        self, mock_wx: MagicMock, audio_path: Path, config: TransskriboConfig
    ) -> None:
        import numpy as np

        mock_wx.load_audio.return_value = np.zeros(16000)
        from transskribo.transcriber import load_audio

        result = load_audio(audio_path, config)
        mock_wx.load_audio.assert_called_once_with(str(audio_path))
        assert result is not None


class TestLoadWhisperModel:
    @patch("transskribo.transcriber.whisperx")
    def test_loads_model_with_correct_config(
        self, mock_wx: MagicMock, config: TransskriboConfig
    ) -> None:
        mock_model = MagicMock()
        mock_wx.load_model.return_value = mock_model
        from transskribo.transcriber import load_whisper_model

        result = load_whisper_model(config)
        mock_wx.load_model.assert_called_once_with(
            "large-v3",
            "cuda",
            compute_type="float16",
            language="pt",
        )
        assert result is mock_model


class TestTranscribe:
    @patch("transskribo.transcriber.whisperx")
    def test_transcribe_calls_model_with_correct_args(
        self, mock_wx: MagicMock, config: TransskriboConfig
    ) -> None:
        import numpy as np

        mock_model = MagicMock()
        audio = np.zeros(16000)
        expected_result = {"segments": [{"text": "hello"}], "language": "pt"}
        mock_model.transcribe.return_value = expected_result

        from transskribo.transcriber import transcribe

        result = transcribe(mock_model, audio, config)
        mock_model.transcribe.assert_called_once_with(
            audio, batch_size=8, language="pt"
        )
        assert result == expected_result


class TestAlign:
    @patch("transskribo.transcriber.torch")
    @patch("transskribo.transcriber.whisperx")
    def test_align_loads_align_model_and_runs(
        self, mock_wx: MagicMock, mock_torch: MagicMock, config: TransskriboConfig
    ) -> None:
        import numpy as np

        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_wx.load_align_model.return_value = (mock_align_model, mock_metadata)
        aligned_result = {"segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}
        mock_wx.align.return_value = aligned_result

        audio = np.zeros(16000)
        input_result: dict[str, Any] = {"segments": [{"text": "hello"}]}

        from transskribo.transcriber import align

        result = align(input_result, audio, config)

        mock_wx.load_align_model.assert_called_once_with(
            language_code="pt", device="cuda"
        )
        mock_wx.align.assert_called_once_with(
            input_result["segments"],
            mock_align_model,
            mock_metadata,
            audio,
            "cuda",
            return_char_alignments=False,
        )
        assert result == aligned_result

    @patch("transskribo.transcriber.torch")
    @patch("transskribo.transcriber.whisperx")
    def test_align_frees_alignment_model(
        self, mock_wx: MagicMock, mock_torch: MagicMock, config: TransskriboConfig
    ) -> None:
        import numpy as np

        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = {"segments": []}

        from transskribo.transcriber import align

        align({"segments": []}, np.zeros(16000), config)
        mock_torch.cuda.empty_cache.assert_called_once()


class TestUnloadWhisperModel:
    @patch("transskribo.transcriber.torch")
    def test_clears_cuda_cache(self, mock_torch: MagicMock) -> None:
        from transskribo.transcriber import unload_whisper_model

        mock_model = MagicMock()
        unload_whisper_model(mock_model)
        mock_torch.cuda.empty_cache.assert_called_once()


# ---------------------------------------------------------------------------
# 7.05 — Diarization stage tests
# ---------------------------------------------------------------------------


class TestLoadDiarizationPipeline:
    @patch("transskribo.transcriber.DiarizationPipeline")
    def test_loads_pipeline_with_hf_token(
        self, mock_dp_cls: MagicMock, config: TransskriboConfig
    ) -> None:
        mock_pipeline = MagicMock()
        mock_dp_cls.return_value = mock_pipeline

        from transskribo.transcriber import load_diarization_pipeline

        result = load_diarization_pipeline(config)
        mock_dp_cls.assert_called_once_with(
            use_auth_token="hf_test_token", device="cuda"
        )
        assert result is mock_pipeline


class TestDiarize:
    def test_diarize_calls_pipeline_with_audio_path(
        self, audio_path: Path, config: TransskriboConfig
    ) -> None:
        mock_pipeline = MagicMock()
        mock_diarize_result = MagicMock()
        mock_pipeline.return_value = mock_diarize_result

        from transskribo.transcriber import diarize

        result = diarize(mock_pipeline, audio_path, config)
        mock_pipeline.assert_called_once_with(str(audio_path))
        assert result is mock_diarize_result


class TestAssignSpeakers:
    @patch("transskribo.transcriber.whisperx")
    def test_assign_speakers_merges_labels(self, mock_wx: MagicMock) -> None:
        mock_diarization = MagicMock()
        aligned: dict[str, Any] = {"segments": [{"text": "hello"}]}
        expected = {"segments": [{"text": "hello", "speaker": "SPEAKER_00"}]}
        mock_wx.assign_word_speakers.return_value = expected

        from transskribo.transcriber import assign_speakers

        result = assign_speakers(mock_diarization, aligned)
        mock_wx.assign_word_speakers.assert_called_once_with(
            mock_diarization, aligned
        )
        assert result == expected


class TestUnloadDiarizationPipeline:
    @patch("transskribo.transcriber.torch")
    def test_clears_cuda_cache(self, mock_torch: MagicMock) -> None:
        from transskribo.transcriber import unload_diarization_pipeline

        mock_pipeline = MagicMock()
        unload_diarization_pipeline(mock_pipeline)
        mock_torch.cuda.empty_cache.assert_called_once()


# ---------------------------------------------------------------------------
# 7.05 — process_file orchestrator tests
# ---------------------------------------------------------------------------


class TestProcessFile:
    @patch("transskribo.transcriber.DiarizationPipeline")
    @patch("transskribo.transcriber.torch")
    @patch("transskribo.transcriber.whisperx")
    def test_full_lifecycle_order(
        self,
        mock_wx: MagicMock,
        mock_torch: MagicMock,
        mock_dp_cls: MagicMock,
        audio_path: Path,
        config: TransskriboConfig,
    ) -> None:
        """Verify the complete load->use->unload lifecycle for both stages."""
        import numpy as np

        # Setup mocks
        mock_audio = np.zeros(16000)
        mock_wx.load_audio.return_value = mock_audio

        mock_model = MagicMock()
        mock_wx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "segments": [{"text": "oi"}],
            "language": "pt",
        }

        mock_align_model = MagicMock()
        mock_align_metadata = MagicMock()
        mock_wx.load_align_model.return_value = (
            mock_align_model,
            mock_align_metadata,
        )
        aligned_result = {"segments": [{"text": "oi", "start": 0.0, "end": 0.5}]}
        mock_wx.align.return_value = aligned_result

        mock_pipeline = MagicMock()
        mock_dp_cls.return_value = mock_pipeline
        mock_diarize_segments = MagicMock()
        mock_pipeline.return_value = mock_diarize_segments

        final_result = {
            "segments": [{"text": "oi", "speaker": "SPEAKER_00"}]
        }
        mock_wx.assign_word_speakers.return_value = final_result

        from transskribo.transcriber import process_file

        output = process_file(audio_path, config)

        # Verify audio loaded first
        mock_wx.load_audio.assert_called_once_with(str(audio_path))

        # Verify stage 1: load whisper -> transcribe -> align -> unload
        mock_wx.load_model.assert_called_once_with(
            "large-v3", "cuda", compute_type="float16", language="pt"
        )
        mock_model.transcribe.assert_called_once()
        mock_wx.load_align_model.assert_called_once()
        mock_wx.align.assert_called_once()

        # Verify stage 2: load diarization -> diarize -> assign -> unload
        mock_dp_cls.assert_called_once_with(
            use_auth_token="hf_test_token", device="cuda"
        )
        mock_pipeline.assert_called_once_with(str(audio_path))
        mock_wx.assign_word_speakers.assert_called_once_with(
            mock_diarize_segments, aligned_result
        )

        # Verify VRAM was freed: empty_cache called for align model (1),
        # whisper unload helper (2), process_file stage 1 finally (3),
        # diarization unload helper (4), process_file stage 2 finally (5)
        assert mock_torch.cuda.empty_cache.call_count == 5

        # Verify output structure
        assert output["result"] == final_result
        assert "timing" in output

    @patch("transskribo.transcriber.DiarizationPipeline")
    @patch("transskribo.transcriber.torch")
    @patch("transskribo.transcriber.whisperx")
    def test_timing_values_collected(
        self,
        mock_wx: MagicMock,
        mock_torch: MagicMock,
        mock_dp_cls: MagicMock,
        audio_path: Path,
        config: TransskriboConfig,
    ) -> None:
        """Verify timing dict has all expected keys with positive values."""
        import numpy as np

        mock_wx.load_audio.return_value = np.zeros(16000)
        mock_model = MagicMock()
        mock_wx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"segments": [], "language": "pt"}
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = {"segments": []}
        mock_pipeline = MagicMock()
        mock_dp_cls.return_value = mock_pipeline
        mock_pipeline.return_value = MagicMock()
        mock_wx.assign_word_speakers.return_value = {"segments": []}

        from transskribo.transcriber import process_file

        output = process_file(audio_path, config)
        timing = output["timing"]

        assert "transcribe_secs" in timing
        assert "align_secs" in timing
        assert "diarize_secs" in timing
        assert "total_secs" in timing
        assert all(isinstance(v, float) for v in timing.values())
        assert all(v >= 0 for v in timing.values())
        assert timing["total_secs"] >= (
            timing["transcribe_secs"]
            + timing["align_secs"]
            + timing["diarize_secs"]
        )

    @patch("transskribo.transcriber.DiarizationPipeline")
    @patch("transskribo.transcriber.torch")
    @patch("transskribo.transcriber.whisperx")
    def test_whisper_cleanup_on_transcribe_error(
        self,
        mock_wx: MagicMock,
        mock_torch: MagicMock,
        mock_dp_cls: MagicMock,
        audio_path: Path,
        config: TransskriboConfig,
    ) -> None:
        """If transcribe fails, Whisper model is still unloaded and VRAM freed."""
        import numpy as np

        mock_wx.load_audio.return_value = np.zeros(16000)
        mock_model = MagicMock()
        mock_wx.load_model.return_value = mock_model
        mock_model.transcribe.side_effect = RuntimeError("OOM")

        from transskribo.transcriber import process_file

        with pytest.raises(RuntimeError, match="OOM"):
            process_file(audio_path, config)

        # Whisper model should still be cleaned up via finally block
        mock_torch.cuda.empty_cache.assert_called()

    @patch("transskribo.transcriber.DiarizationPipeline")
    @patch("transskribo.transcriber.torch")
    @patch("transskribo.transcriber.whisperx")
    def test_diarization_cleanup_on_error(
        self,
        mock_wx: MagicMock,
        mock_torch: MagicMock,
        mock_dp_cls: MagicMock,
        audio_path: Path,
        config: TransskriboConfig,
    ) -> None:
        """If diarization fails, pipeline is still unloaded and VRAM freed."""
        import numpy as np

        mock_wx.load_audio.return_value = np.zeros(16000)
        mock_model = MagicMock()
        mock_wx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"segments": [], "language": "pt"}
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = {"segments": []}

        mock_pipeline = MagicMock()
        mock_dp_cls.return_value = mock_pipeline
        mock_pipeline.side_effect = RuntimeError("Diarization failed")

        from transskribo.transcriber import process_file

        with pytest.raises(RuntimeError, match="Diarization failed"):
            process_file(audio_path, config)

        # Both whisper unload and diarization unload should have called empty_cache
        # (align cleanup + whisper unload helper + stage 1 finally +
        # diarization unload helper + stage 2 finally = 5 calls)
        assert mock_torch.cuda.empty_cache.call_count == 5

    @patch("transskribo.transcriber.DiarizationPipeline")
    @patch("transskribo.transcriber.torch")
    @patch("transskribo.transcriber.whisperx")
    def test_models_never_loaded_simultaneously(
        self,
        mock_wx: MagicMock,
        mock_torch: MagicMock,
        mock_dp_cls: MagicMock,
        audio_path: Path,
        config: TransskriboConfig,
    ) -> None:
        """Verify Whisper is unloaded before pyannote is loaded."""
        import numpy as np

        call_order: list[str] = []

        mock_wx.load_audio.return_value = np.zeros(16000)

        mock_model = MagicMock()
        mock_wx.load_model.side_effect = lambda *a, **kw: (
            call_order.append("load_whisper"),
            mock_model,
        )[-1]
        mock_model.transcribe.return_value = {"segments": [], "language": "pt"}
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = {"segments": []}

        def track_empty_cache() -> None:
            call_order.append("empty_cache")
            return None

        mock_torch.cuda.empty_cache = track_empty_cache

        mock_pipeline = MagicMock()

        def track_load_diarization(*a: Any, **kw: Any) -> MagicMock:
            call_order.append("load_diarization")
            return mock_pipeline

        mock_dp_cls.side_effect = track_load_diarization
        mock_pipeline.return_value = MagicMock()
        mock_wx.assign_word_speakers.return_value = {"segments": []}

        from transskribo.transcriber import process_file

        process_file(audio_path, config)

        # Verify whisper is loaded, then cache cleared (unload), then diarization loaded
        whisper_idx = call_order.index("load_whisper")
        empty_cache_indices = [
            i for i, c in enumerate(call_order) if c == "empty_cache"
        ]
        diarization_idx = call_order.index("load_diarization")

        # At least one empty_cache must occur between whisper load and diarization load
        assert any(
            whisper_idx < idx < diarization_idx for idx in empty_cache_indices
        ), f"Expected empty_cache between whisper and diarization. Order: {call_order}"
