# video-asr

> **Local video-to-text (ASR).** Offline transcription using faster-whisper large-v3 on your NVIDIA GPU. Data stays local — nothing uploaded.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **Offline** — whisper large-v3 runs entirely on your GPU, zero cloud dependency
- ⚡ **Fast** — RTX 5060 runs ~4-5× real-time with float16 (~3 GB VRAM)
- 🎬 **Formats** — output as plain text, SRT subtitles, or JSON (with timestamps)
- 🗣️ **Speaker diarization** — optional, via WhisperX
- 🪄 **Never overwrites** — each run gets its own numbered file
- 🔌 **Pipe to AI** — `--stdout` for seamless integration with LLMs

## Quick Install

### Option 1: One-click (Windows)

```powershell
# Download the repo, then:
.\scripts\install.ps1
```

This installs ffmpeg, creates a Python venv, downloads the model, and makes `transcribe` globally available.

### Option 2: pip (cross-platform)

```bash
pip install video-asr
# Ensure ffmpeg is on your PATH
```

### Option 3: uv (lightning fast)

```bash
uv tool install video-asr
```

## Usage

```bash
# Transcribe a video (auto-detect language)
transcribe "my video.mp4"

# Force English mode (recommended for mixed Chinese-English)
transcribe "video.mp4" --language en

# Output SRT subtitles
transcribe "video.mp4" --output srt

# Pipe text directly to another tool
transcribe "video.mp4" --stdout | python analyze.py

# High-precision mode
transcribe "video.mp4" --compute-type float32

# Batch all MP4s in a folder
transcribe "folder/*.mp4"
```

### Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--language, -lang` | auto | Force language (`en` for mixed) |
| `--output, -o` | `txt` | Output format: `txt`, `srt`, or `json` |
| `--output-dir` | `./output` | Where to save files |
| `--stdout, -p` | off | Print to stdout instead of file |
| `--model, -m` | `large-v3` | Model size: tiny/base/small/medium/large-v3 |
| `--compute-type` | `float16` | Precision: `int8`, `float16`, or `float32` |
| `--beam-size` | `8` | Beam search width (higher = more accurate) |
| `--backend, -b` | `auto` | Engine: `auto`, `faster`, `whisper`, `whisperx` |
| `--diarize` | off | Enable speaker diarization |

### Output Files

Each run creates a new file — never overwrites:

```
output/
├── my-video.txt          ← first run
├── my-video (1).txt      ← second run (any model/settings)
└── my-video (2).txt      ← third run
```

## Requirements

- **GPU**: NVIDIA GPU with CUDA 12.x (tested on RTX 5060 8 GB)
- **CPU**: Works but very slow — GPU strongly recommended
- **RAM**: 8+ GB system RAM
- **Disk**: ~3 GB for model cache
- **ffmpeg**: installed and on PATH

## Architecture

```
video-asr/
├── src/video_asr/
│   ├── __init__.py         # package metadata
│   ├── __main__.py         # python -m entry point
│   └── cli.py              # CLI + transcription pipeline
├── scripts/
│   └── install.ps1         # Windows one-click installer
├── .claude/skills/
│   └── video-asr/
│       └── SKILL.md        # Claude Code skill auto-discovery
├── pyproject.toml
├── README.md
└── .gitignore
```

## Language Note

Whisper's auto-detect **locks** to one language for the entire audio. If your video switches between Chinese and English, the model will drop the non-detected segments. **Workaround**: use `--language en` (large-v3 handles Chinese well even in English mode).

## License

MIT
