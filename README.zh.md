# video-asr

<div align="center">

[English](README.md) | [中文](README.zh.md)

</div>

> **本地视频转文本 (ASR)。** 基于 faster-whisper large-v3 在本地 NVIDIA GPU 上离线运行。数据不出本机，不上传任何内容。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 特点

- 🎯 **纯离线** — whisper large-v3 完全在本地 GPU 运行，无需联网
- ⚡ **速度快** — RTX 5060 float16 实测 ~4-5× 实时率（显存 ~3 GB）
- 🎬 **多种输出** — 纯文本、SRT 字幕、JSON（含时间戳和置信度）
- 🗣️ **说话人分离** — 可选，基于 WhisperX
- 🪄 **永不覆盖** — 每次转写自动编号，旧文件保留
- 🔌 **Pipe 给 AI** — `--stdout` 可直接将文本传给 LLM 分析

## 快速安装

### 方式一：一键安装（Windows）

```powershell
# 下载仓库后，在项目目录运行：
.\scripts\install.ps1
```

自动安装 ffmpeg、创建 Python 虚拟环境、下载模型、注册 `transcribe` 命令。

### 方式二：pip（跨平台）

```bash
pip install video-asr
# 确保 ffmpeg 已在 PATH 中
```

### 方式三：uv（极速）

```bash
uv tool install video-asr
```

## 使用方法

```bash
# 转写视频（自动检测语言）
transcribe "我的视频.mp4"

# 指定英文模式（中英混合视频推荐）
transcribe "视频.mp4" --language en

# 输出 SRT 字幕
transcribe "视频.mp4" --output srt

# Pipe 给 AI 工具
transcribe "视频.mp4" --stdout | python analyze.py

# 高精度模式
transcribe "视频.mp4" --compute-type float32

# 批量处理文件夹内所有 MP4
transcribe "文件夹/*.mp4"
```

### 常用选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--language, -lang` | auto | 强制指定语言（中英混合用 `en`） |
| `--output, -o` | `txt` | 输出格式：`txt` / `srt` / `json` |
| `--output-dir` | `./output` | 输出目录 |
| `--stdout, -p` | 关闭 | 输出到 stdout 而非文件 |
| `--model, -m` | `large-v3` | 模型大小：tiny/base/small/medium/large-v3 |
| `--compute-type` | `float16` | 精度：`int8` / `float16` / `float32` |
| `--beam-size` | `8` | 解码 beam size（越大越精细） |
| `--backend, -b` | `auto` | 后端引擎：`auto` / `faster` / `whisper` / `whisperx` |
| `--diarize` | 关闭 | 启用说话人分离 |
| `--recommend` | — | 检测 GPU，显示推荐模型并选择默认配置 |
| `--list-models` | — | 查看本机可用的所有模型组合 |

### 输出文件

每次转写都是新文件，永不覆盖：

```
output/
├── 我的视频.txt          ← 第一次运行
├── 我的视频 (1).txt      ← 第二次（任意模型/参数）
└── 我的视频 (2).txt      ← 第三次
```

### GPU 检测与模型推荐

```bash
# 查看本机推荐配置
transcribe --list-models

# 交互式选择默认配置
transcribe --recommend
```

示例输出：

```
  GPU: NVIDIA GeForce RTX 5060 Laptop GPU  (8.0 GB)
  VRAM: 8150 MB  |  CUDA: 12.8  |  SM: 12.0

  推荐方案（按质量排序，仅显示本机能跑的）:

        Model        Precision      VRAM Speed    Quality
  ─────────────────────────────────────────────────────
   1. large-v3     float16     ~3.0 GB ★★★☆☆    ★★★★★  ← 推荐
   2. large-v3     float32     ~6.0 GB ★★☆☆☆    ★★★★★
   3. medium       float16     ~1.7 GB ★★★★☆    ★★★★☆
   ...

  选择默认配置 (输入 1-15，或按 Enter 跳过): 1
  ✅ 已保存默认配置: large-v3 / float16
```

## 系统要求

- **GPU**: NVIDIA GPU with CUDA 12.x（RTX 5060 8GB 实测通过）
- **CPU**: 可用但极慢 — 强烈建议使用 GPU
- **内存**: 8+ GB 系统内存
- **硬盘**: ~3 GB 用于模型缓存
- **ffmpeg**: 已安装并在 PATH 中

## 项目结构

```
video-asr/
├── src/video_asr/
│   ├── __init__.py         # 包信息
│   ├── __main__.py         # python -m 入口
│   └── cli.py              # CLI + 转写流水线（核心）
├── scripts/
│   └── install.ps1         # Windows 一键安装脚本
├── .claude/skills/
│   └── video-asr/
│       └── SKILL.md        # Claude Code 技能自动发现
├── pyproject.toml
├── README.md
├── README.zh.md
└── .gitignore
```

## 语言说明

Whisper 的自动检测是**锁定机制**——一旦检测到中文，后面的英文会被直接过滤。如果视频中英混杂：

- ✅ **推荐**: 用 `--language en`，large-v3 的英文模式也能高质量处理中文
- ❌ **不推荐**: 不设语言让 Whisper 自动检测（会丢失英文片段）

## 常见问题

### 输出文字不完整 / 缺少英文？

加 `--language en` 强制英文模式。

### 显存不足？

换小模型或低精度：`transcribe video.mp4 --model medium --compute-type int8`

### "no kernel image" 报错？

CTranslate2 不支持当前 GPU。用 `--backend whisper` 切换到 PyTorch 后端。

### 模型下载慢？

只有第一次需要下载 ~3GB。完成后完全离线。也可先运行 `transcribe --list-models` 触发后台下载。

## 许可

MIT
