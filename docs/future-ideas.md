# Future Development Ideas

## Near-term

### Subtitle Export (SRT/VTT)

Add an `export` command to convert JSON transcription output to subtitle formats (SRT, VTT, ASS) using the word-level timestamps already stored in the output.

```bash
transskribo export result.json --format srt --output lecture.srt
```

**Value**: Direct integration with video editing tools and YouTube uploads.

### Diarization Speaker Limits

Add `min_speakers` / `max_speakers` config options to improve pyannote accuracy per content type.

```toml
[diarization]
min_speakers = 2    # Meetings usually have 2+ speakers
max_speakers = 5    # Lectures rarely exceed 5
```

**Value**: Reduce false speaker splits in single-speaker lectures, improve accuracy in panel discussions.

### Confidence-Based Quality Metrics

Analyze the word-level confidence scores already stored in the JSON output:

- Average confidence score per file
- Low-confidence segment highlighting in docx output
- Flag files with <70% avg confidence for manual review

**Value**: Identify audio quality issues and validate transcription reliability.

## Medium-term

### Per-Directory Enrichment Templates

Allow different LLM extraction prompts per directory or content type:

```toml
[enrich.templates]
lectures = "Extract educational concepts from this lecture transcript"
meetings = "Extract action items and decisions from this meeting"
interviews = "Extract key quotes and themes from this interview"
```

**Value**: One tool handles diverse transcription use cases with tailored extraction.

### Web UI Dashboard

FastAPI + htmx dashboard (no frontend build step) showing:

- Real-time progress (current file, queue depth, ETA)
- Per-directory stats with drill-down
- Failed files list with retry button
- Enrichment queue with LLM usage stats
- Audio player + transcript viewer for results

**Value**: Non-technical users can monitor and control long-running batches.

### Multi-Language Auto-Detection

Use WhisperX's language detection instead of hardcoding `"pt"`:

- Detect language on first 30 seconds of audio
- Add `auto_detect_language` config option
- Store detected language in output metadata

**Value**: Process multilingual lecture series without manual sorting.

## Long-term

### Parallel Processing (Multi-GPU)

Implement a worker pool with GPU affinity:

- Queue files by estimated processing time (prioritize short files)
- Process N files in parallel on multi-GPU setups
- Respect per-worker VRAM limits

**Value**: 4x throughput on a 4-GPU machine.

### Cloud Storage Integration

Support S3/GCS/Azure via `fsspec` filesystem abstraction:

```toml
input_dir = "s3://my-bucket/lectures"
output_dir = "s3://my-bucket/transcripts"
```

**Value**: Process cloud-native data lakes without local disk staging.

### Incremental Diarization for Long Files

Chunk files over 2 hours into overlapping windows for diarization:

- Diarize each chunk independently
- Merge speaker labels across chunks with overlap-based speaker matching
- Avoids OOM on 4+ hour recordings

**Value**: Handle very long recordings without memory issues.

### Smart Priority Queue

Implement priority scoring for file processing order:

- Prioritize small files (quick wins)
- Deprioritize chronic failures
- Support config-based directory weights

**Value**: Optimize for "most files finished fastest" instead of alphabetical order.
