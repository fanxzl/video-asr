---
name: video-asr
description: "Local video ASR — offline transcription of local video files using faster-whisper large-v3. Outputs plain text, SRT subtitles, or JSON. GPU-accelerated, data stays local."
platforms: [windows]
---

# Video ASR — 视频语音转文本

将本地视频的语音转写为文本。使用 faster-whisper large-v3 在本地 NVIDIA GPU 上离线运行。

## 什么时候用

用户有本地视频文件（.mp4 / .mkv / .avi 等），需要提取语音文案给 AI 做分析。**不支持直接处理视频 URL** — 如需 B站视频，先用 bilibili-download 下载再调用此技能。

## 前置条件

- **安装**: 在项目目录运行 `scripts/install.ps1` 一键安装
- **或**: `pip install video-asr` 后确保有 ffmpeg + CUDA

## 基本用法

```powershell
# 直接 CLI 使用
transcribe "视频.mp4"

# 或指定语言
transcribe "视频.mp4" --language en

# 输出字幕
transcribe "视频.mp4" --output srt
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--language` / `-lang` | 强制指定语言。**中英混合推荐 `en`** |
| `--backend` / `-b` | `auto` / `faster` / `whisper` / `whisperx`。默认 auto |
| `--output` / `-o` | `txt` / `srt` / `json`。默认 txt |
| `--output-dir` | 输出目录。默认 `./output` |
| `--stdout` / `-p` | 输出到 stdout |
| `--model` / `-m` | 模型大小: tiny/base/small/medium/large-v3 |
| `--compute-type` | float16(默认) / int8(省显存) / float32(最高精度) |
| `--beam-size` | 解码 beam size (默认 8) |
| `--diarize` | 说话人分离 (需 whisperx + HF_TOKEN) |

## 输出

每次转写都是新文件，永不覆盖：

```
output/
├── 视频.txt          ← 第一次
├── 视频 (1).txt      ← 第二次
└── 视频 (2).txt      ← 第三次
```

## 注意事项

1. **语言检测锁死**: 自动检测到中文后锁死，后面英文会被过滤。中英混合视频务必 `--language en`
2. **模型缓存**: 首次运行下载 ~3GB 到 HuggingFace 缓存，之后离线可用
3. **GPU**: 需要 NVIDIA GPU + CUDA 12.x。无 GPU 时回退 CPU（极慢）
