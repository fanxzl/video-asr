<#
.SYNOPSIS
    video-asr 一键安装脚本
.DESCRIPTION
    自动处理所有前置依赖（ffmpeg、Python 虚拟环境、模型缓存），
    完成后即可用 `transcribe` 命令转写视频。
.EXAMPLE
    .\install.ps1                              # 默认安装
    .\install.ps1 -NoModelCache                # 跳过模型缓存预热
    .\install.ps1 -InstallDir D:\tools\video-asr  # 自定义安装位置
#>

param(
    [string]$InstallDir = "",

    [switch]$NoModelCache = $false,

    [switch]$SkipFfmpeg = $false
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "video-asr 安装中..."

# --------------- 判断脚本所在位置 ---------------
if (-not $InstallDir) {
    $InstallDir = $PSScriptRoot
}
$InstallDir = Resolve-Path $InstallDir -ErrorAction SilentlyContinue 2>$null
if (-not $InstallDir) {
    $InstallDir = $PSScriptRoot
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  video-asr v1.0.0 安装" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  安装目录: $InstallDir`n"

# --------------- 1. 检查 ffmpeg ---------------
if (-not $SkipFfmpeg) {
    Write-Host "[1/5] 检查 ffmpeg..." -ForegroundColor Yellow
    $ffmpegPath = (Get-Command "ffmpeg" -ErrorAction SilentlyContinue).Source
    if (-not $ffmpegPath) {
        Write-Host "  → 未检测到 ffmpeg，尝试通过 winget 安装..." -ForegroundColor Gray
        try {
            winget install ffmpeg --accept-source-agreements --silent 2>&1 | Out-Null
            # 刷新 PATH 以便当前会话识别
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
            $ffmpegPath = (Get-Command "ffmpeg" -ErrorAction SilentlyContinue).Source
            if ($ffmpegPath) {
                Write-Host "  ✅ ffmpeg 已安装: $ffmpegPath" -ForegroundColor Green
            } else {
                Write-Host "  ⚠  winget 安装完成但未刷新 PATH，重新打开终端即可" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  ⚠  winget 安装失败，请手动安装 ffmpeg:" -ForegroundColor Red
            Write-Host "     winget install ffmpeg" -ForegroundColor Gray
            Write-Host "     或下载: https://ffmpeg.org/download.html" -ForegroundColor Gray
        }
    } else {
        Write-Host "  ✅ ffmpeg: $ffmpegPath" -ForegroundColor Green
    }
} else {
    Write-Host "[1/5] 跳过 ffmpeg 检查" -ForegroundColor Gray
}

# --------------- 2. 检测 CUDA ---------------
Write-Host "[2/5] 检测 GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $cudaInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
        Write-Host "  ✅ $cudaInfo" -ForegroundColor Green
        Write-Host "  → 确保已安装 CUDA 12.x + 驱动 550+。推荐: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠  未检测到 NVIDIA GPU (nvidia-smi not found)" -ForegroundColor Yellow
        Write-Host "  → 转写将回退到 CPU（极慢）。建议安装 CUDA: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ⚠  GPU 检测失败: $_" -ForegroundColor Yellow
}

# --------------- 3. 创建虚拟环境 + 安装依赖 ---------------
Write-Host "[3/5] 创建 Python 虚拟环境..." -ForegroundColor Yellow
$venvPath = Join-Path $InstallDir ".venv_asr"

# 检测系统 Python
$pythonCandidates = @(
    (Get-Command "python3" -ErrorAction SilentlyContinue).Source,
    (Get-Command "python" -ErrorAction SilentlyContinue).Source
)
$pythonPath = $null
foreach ($cmd in $pythonCandidates) {
    if ($cmd -and $cmd -notlike "*Microsoft*" -and $cmd -notlike "*WindowsApp*") {
        try {
            $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver -and [version]$ver -ge [version]"3.10") {
                $pythonPath = $cmd
                Write-Host "  → 使用 Python $ver: $cmd" -ForegroundColor Gray
                break
            }
        } catch {}
    }
}

if (-not $pythonPath) {
    Write-Host "  ❌ 未找到 Python 3.10+，请安装: https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "     安装时务必勾选 'Add Python to PATH'" -ForegroundColor Yellow
    exit 1
}

# 创建虚拟环境
if (Test-Path $venvPath) {
    Write-Host "  → 虚拟环境已存在，跳过创建" -ForegroundColor Gray
} else {
    & $pythonPath -m venv $venvPath
    Write-Host "  ✅ 虚拟环境创建完成" -ForegroundColor Green
}

$pip = Join-Path $venvPath "Scripts" "pip"

# 升级 pip
& $pip install --upgrade pip -q 2>&1 | Out-Null

# 安装依赖
Write-Host "  → 安装 Python 依赖（可能需要几分钟）..." -ForegroundColor Gray
if (Test-Path (Join-Path $InstallDir "requirements.txt")) {
    & $pip install -r (Join-Path $InstallDir "requirements.txt") -q 2>&1
} else {
    # 如果有 pyproject.toml 则本地安装包
    if (Test-Path (Join-Path $InstallDir "pyproject.toml")) {
        & $pip install -e (Join-Path $InstallDir ".") -q 2>&1
    } else {
        & $pip install faster-whisper torch torchaudio --index-url https://download.pytorch.org/whl/cu124 -q 2>&1
    }
}
Write-Host "  ✅ Python 依赖安装完成" -ForegroundColor Green

$pythonVenv = Join-Path $venvPath "Scripts" "python.exe"

# 验证 GPU 可访问
Write-Host "  → 验证 GPU 可访问..." -ForegroundColor Gray
try {
    $gpuOk = & $pythonVenv -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($gpuOk -eq "True") {
        $gpuName = & $pythonVenv -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
        Write-Host "  ✅ CUDA 可用 - $gpuName" -ForegroundColor Green
    } else {
        Write-Host "  ⚠  CUDA 不可用，将使用 CPU" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠  GPU 验证失败: $_" -ForegroundColor Yellow
}

# --------------- 4. 创建启动脚本（cmd 入口） ---------------
Write-Host "[4/5] 创建命令入口..." -ForegroundColor Yellow

# 创建一个 bat 包装器，让 transcribe 命令全局可用
$batPath = Join-Path $InstallDir "transcribe.bat"
@"
@echo off
"%~dp0.venv_asr\Scripts\python.exe" "%~dp0src\video_asr\cli.py" %*
"@ | Out-File -FilePath $batPath -Encoding utf8

# 添加到用户 PATH（下次终端生效）
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    $newPath = "$InstallDir;" + $userPath.TrimEnd(";")
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "  ✅ 已添加到用户 PATH（新终端生效）" -ForegroundColor Green
} else {
    Write-Host "  → PATH 已包含安装目录" -ForegroundColor Gray
}

# 当前会话立即可用
$env:Path = "$InstallDir;$env:Path"

Write-Host "  ✅ 可在终端中直接使用: transcribe <视频文件>" -ForegroundColor Green

# --------------- 5. 模型缓存预热 ---------------
if (-not $NoModelCache) {
    Write-Host "[5/5] 预热模型缓存（首次下载 ~3GB）..." -ForegroundColor Yellow
    Write-Host "  → 这只需一次，后续完全离线使用" -ForegroundColor Gray
    Write-Host "  → 按 Ctrl+C 可跳过（安装后用 --stdout 触发）" -ForegroundColor Gray

    # 生成一段静音音频让 Whisper 加载模型
    $silentWav = Join-Path $env:TEMP "_video_asr_warmup.wav"
    try {
        & $pythonVenv -c @"
import wave, os
sr, dur = 16000, 3
with wave.open(r'$silentWav', 'wb') as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes(b'\x00\x00' * sr * dur)
from faster_whisper import WhisperModel
m = WhisperModel('large-v3', device='cuda', compute_type='float16')
seg, _ = m.transcribe(r'$silentWav', beam_size=5)
list(seg)
os.unlink(r'$silentWav')
print('OK')
"@ 2>&1
        Write-Host "  ✅ 模型缓存就绪" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠  模型预热跳过（可后续首次使用时自动下载）" -ForegroundColor Yellow
        if (Test-Path $silentWav) { Remove-Item $silentWav -Force }
    }
}

# --------------- 6. 模型推荐 ---------------
Write-Host "[6/6] 检测 GPU 并推荐模型..." -ForegroundColor Yellow
try {
    # 检测 GPU
    $hasGpu = & $pythonVenv -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($hasGpu -eq "True") {
        & $pythonVenv -m video_asr --list-models 2>&1
        Write-Host ""
        Write-Host "  选择默认配置 (输入数字后 Enter，或直接 Enter 跳过): " -ForegroundColor Cyan -NoNewline
        $choice = Read-Host
        if ($choice -match '^\d+$') {
            # 从 --recommend 的交互模式提取选择
            $recs = & $pythonVenv -m video_asr --list-models 2>&1 | Where-Object { $_ -match '^\s+\d+\.' }
            $idx = [int]$choice - 1
            if ($idx -ge 0 -and $idx -lt $recs.Count) {
                $line = $recs[$idx]
                if ($line -match '^\s+\d+\.\s+(\S+)\s+(\S+)') {
                    $model = $matches[1]
                    $ct = $matches[2]
                    & $pythonVenv -m video_asr --set-default $model $ct 2>&1
                    Write-Host "  ✅ 默认配置: $model / $ct" -ForegroundColor Green
                }
            }
        } else {
            Write-Host "  → 跳过" -ForegroundColor Gray
        }
    } else {
        Write-Host "  → 未检测到 GPU，跳过推荐" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ⚠  推荐步骤跳过: $_" -ForegroundColor Yellow
} else {
    Write-Host "[5/5] 跳过模型缓存预热" -ForegroundColor Gray
}

# --------------- 完成 ---------------
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  🎉 video-asr 安装完成！" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  使用方式："
Write-Host "    转写视频:    transcribe 视频.mp4"
Write-Host "    指定语言:    transcribe 视频.mp4 --language en"
Write-Host "    输出字幕:    transcribe 视频.mp4 --output srt"
Write-Host "    Pipe 给 AI:  transcribe 视频.mp4 --stdout | python -m ai_analyze"
Write-Host "    批量处理:    transcribe 文件夹/*.mp4"
Write-Host "    高精度:      transcribe 视频.mp4 --compute-type float32"
Write-Host ""
Write-Host "  帮助: transcribe --help"
Write-Host "`n"
