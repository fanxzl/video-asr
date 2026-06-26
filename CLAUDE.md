# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

video-asr — Local video-to-text (ASR) using faster-whisper large-v3 on NVIDIA GPU. Outputs plain text, SRT, or JSON. All processing stays local, data never uploaded.

**One-liner install:**
```powershell
iex (irm https://raw.githubusercontent.com/fanxzl/video-asr/main/scripts/install.ps1)
```

## CLI Entry Point

The package exposes a `transcribe` CLI command via pyproject.toml `[project.scripts]`:

```bash
transcribe "video.mp4"
# or: python -m video_asr "video.mp4"
```

## Key Commands

```bash
# Install in dev mode (from project root)
pip install -e .

# Lint
ruff check src/

# Format
ruff format src/

# Test
pytest

# Transcribe a video
transcribe "video.mp4" --stdout

# GPU model recommendation
transcribe --list-models
transcribe --recommend
```

## Architecture

```
src/video_asr/cli.py  — single-file core (~700 lines)
```

### Pipeline (process_single)

```
video file → ffmpeg extract audio (16kHz mono WAV) → pick backend
  → faster-whisper / openai-whisper / WhisperX
  → segments list → format (txt/srt/json) → write to file with auto-numbering
```

### Key Functions

| Function | Role |
|----------|------|
| `detect_gpu_details()` | Query CUDA: name, VRAM, SM capability |
| `build_recommendations()` | Rank model+precision combos by VRAM fit |
| `transcribe_faster()` | Primary: CTranslate2 backend (fastest) |
| `transcribe_openai()` | Fallback: PyTorch backend |
| `transcribe_whisperx()` | Speaker diarization path |
| `process_single()` | Main pipeline: audio → transcribe → output |
| `segments_to_text/srt/json()` | Output formatters |

### Backend Selection

`--backend auto` (default) tries faster-whisper first, falls back to openai-whisper. `--diarize` forces WhisperX path. `--model` and `--compute-type` control model size and precision.

### Config Persistence

Saved defaults stored at `%APPDATA%/video-asr/config.json`. Applied via `parser.set_defaults(**load_config())` — CLI flags override saved config. Fields: model, compute_type, beam_size, backend, output, output_dir.

### Auto-Numbering

Output files never overwrite. Files matching `stem (N).ext` pattern are scanned, next number assigned.

## Critical Gotchas

1. **Language lock**: Whisper auto-detect locks to first detected language. Chinese video with English segments → English is dropped. Workaround: always use `--language en` for mixed-language content (large-v3 handles Chinese well in English mode).

2. **Model cache**: First run downloads ~3 GB from HuggingFace. Offline after that. Trigger pre-download with `transcribe --list-models`.

3. **VRAM limits**: large-v3 int8 ≈ 1.7 GB, float16 ≈ 3.1 GB, float32 ≈ 6.1 GB. Run `transcribe --list-models` to see what fits the user's GPU.

4. **GPU compatibility**: CTranslate2 (faster-whisper) requires sm_70+. RTX 5060 (sm_120) works with CTranslate2 4.8+. Fallback: `--backend whisper`.

5. **WhisperX diarization**: Requires `pip install video-asr[all]` and a HuggingFace token (HF_TOKEN) with access to pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1.

## Project History

- v1.0.0 — Initial release. Windows-focused. Remote install via `iex (irm ...)`.
- Planned: PyPI publish, macOS/Linux support, streaming/real-time mode.
