"""WhisperX wrapper — all whisperx and torch interaction is isolated here."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from transskribo.config import TransskriboConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio(audio_path: Path, config: TransskriboConfig) -> Any:
    """Load audio file into a numpy ndarray via whisperx."""
    logger.info("Loading audio: %s", audio_path)
    audio: Any = whisperx.load_audio(str(audio_path))
    return audio


# ---------------------------------------------------------------------------
# Transcription stage
# ---------------------------------------------------------------------------


def load_whisper_model(config: TransskriboConfig) -> Any:
    """Load the WhisperX model for transcription."""
    logger.info(
        "Loading Whisper model: %s (compute=%s, device=%s)",
        config.model_size,
        config.compute_type,
        config.device,
    )
    model: Any = whisperx.load_model(
        config.model_size,
        config.device,
        compute_type=config.compute_type,
        language=config.language,
    )
    return model


def transcribe(model: Any, audio: Any, config: TransskriboConfig) -> dict[str, Any]:
    """Run transcription on loaded audio ndarray."""
    logger.info("Transcribing audio (batch_size=%d)", config.batch_size)
    result: dict[str, Any] = model.transcribe(
        audio, batch_size=config.batch_size, language=config.language
    )
    return result


def align(result: dict[str, Any], audio: Any, config: TransskriboConfig) -> dict[str, Any]:
    """Run word-level forced alignment on the transcription result."""
    logger.info("Aligning transcription")
    model_a, metadata = whisperx.load_align_model(
        language_code=config.language, device=config.device
    )
    aligned: dict[str, Any] = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        config.device,
        return_char_alignments=False,
    )
    del model_a
    torch.cuda.empty_cache()
    return aligned


def unload_whisper_model(model: Any) -> None:
    """Delete the Whisper model and free VRAM."""
    logger.info("Unloading Whisper model")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Diarization stage
# ---------------------------------------------------------------------------


def load_diarization_pipeline(config: TransskriboConfig) -> Any:
    """Load the pyannote diarization pipeline with HF token."""
    logger.info("Loading diarization pipeline")
    pipeline: Any = DiarizationPipeline(
        use_auth_token=config.hf_token, device=config.device
    )
    return pipeline


def diarize(pipeline: Any, audio_path: Path, config: TransskriboConfig) -> Any:
    """Run speaker diarization. Pyannote loads audio internally."""
    logger.info("Running diarization: %s", audio_path)
    diarize_segments: Any = pipeline(str(audio_path))
    return diarize_segments


def assign_speakers(
    diarization: Any, aligned: dict[str, Any]
) -> dict[str, Any]:
    """Merge speaker labels into aligned transcription segments."""
    logger.info("Assigning speakers to segments")
    result: dict[str, Any] = whisperx.assign_word_speakers(diarization, aligned)
    return result


def unload_diarization_pipeline(pipeline: Any) -> None:
    """Delete the diarization pipeline and free VRAM."""
    logger.info("Unloading diarization pipeline")
    del pipeline
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def process_file(
    audio_path: Path, config: TransskriboConfig
) -> dict[str, Any]:
    """Orchestrate the full transcription + diarization pipeline for one file.

    Returns a dict with keys:
      - "result": the final transcription result with speaker labels
      - "timing": dict with transcribe_secs, align_secs, diarize_secs, total_secs
    """
    total_start = time.monotonic()

    # Load audio once (used by transcribe + align)
    audio = load_audio(audio_path, config)

    # --- Stage 1: Transcription + alignment ---
    model = load_whisper_model(config)
    try:
        t0 = time.monotonic()
        raw_result = transcribe(model, audio, config)
        transcribe_secs = time.monotonic() - t0

        t0 = time.monotonic()
        aligned_result = align(raw_result, audio, config)
        align_secs = time.monotonic() - t0
    finally:
        unload_whisper_model(model)

    # --- Stage 2: Diarization + speaker assignment ---
    pipeline = load_diarization_pipeline(config)
    try:
        t0 = time.monotonic()
        diarize_result = diarize(pipeline, audio_path, config)
        diarize_secs = time.monotonic() - t0
    finally:
        unload_diarization_pipeline(pipeline)

    final_result = assign_speakers(diarize_result, aligned_result)

    total_secs = time.monotonic() - total_start

    timing = {
        "transcribe_secs": transcribe_secs,
        "align_secs": align_secs,
        "diarize_secs": diarize_secs,
        "total_secs": total_secs,
    }

    logger.info(
        "Processed %s — transcribe=%.1fs align=%.1fs diarize=%.1fs total=%.1fs",
        audio_path.name,
        transcribe_secs,
        align_secs,
        diarize_secs,
        total_secs,
    )

    return {"result": final_result, "timing": timing}
