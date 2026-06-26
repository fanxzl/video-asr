"""
video-asr CLI — Transcribe video files to text locally using faster-whisper.

Usage:
  transcribe video.mp4                       # auto-detect language, save to ./output
  transcribe video.mp4 --language en         # force English mode
  transcribe video.mp4 --stdout              # pipe text to stdout
  transcribe folder/*.mp4                    # batch process
  python -m video_asr video.mp4              # same via module
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU detection & model recommendation
# ---------------------------------------------------------------------------

# Measured VRAM (total with inference, MB) on RTX 5060 large-v3.
# Other models estimated proportionally.
_MODEL_VRAM_MB = {
    # (model, compute_type) -> (total_vram_mb, quality, speed)
    ("tiny",   "int8"):   (100,  1, 5),
    ("tiny",   "float16"):(200,  1, 5),
    ("tiny",   "float32"):(400,  1, 5),
    ("base",   "int8"):   (180,  2, 5),
    ("base",   "float16"):(350,  2, 5),
    ("base",   "float32"):(700,  2, 5),
    ("small",  "int8"):   (400,  3, 5),
    ("small",  "float16"):(750,  3, 5),
    ("small",  "float32"):(1500, 3, 4),
    ("medium", "int8"):   (900,  4, 4),
    ("medium", "float16"):(1700, 4, 4),
    ("medium", "float32"):(3400, 4, 3),
    ("large-v3", "int8"):   (1700, 4, 4),
    ("large-v3", "float16"):(3100, 5, 3),
    ("large-v3", "float32"):(6100, 5, 2),
}

def detect_gpu_details():
    """Return dict with GPU info or None if CPU-only."""
    info = {
        "name": None,
        "vram_mb": 0,
        "cuda_version": None,
        "sm": None,
        "has_cuda": False,
    }
    try:
        import torch
        if not torch.cuda.is_available():
            return info
        info["has_cuda"] = True
        info["name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["sm"] = f"{cap[0]}.{cap[1]}"
        props = torch.cuda.get_device_properties(0)
        info["vram_mb"] = props.total_memory // (1024 * 1024)
        # CUDA version
        ver = torch.version.cuda
        info["cuda_version"] = ver
    except Exception:
        pass
    return info


def build_recommendations(vram_mb, reserve_mb=512):
    """
    Return list of (model, compute_type, vram_est, quality, speed)
    sorted by quality descending, filtered to fit available VRAM.

    `vram_mb` is total GPU VRAM. We reserve `reserve_mb` for OS/overhead.
    """
    available = vram_mb - reserve_mb
    results = []
    for (model, ct), (vram, quality, speed) in _MODEL_VRAM_MB.items():
        if available >= vram:
            results.append((model, ct, vram, quality, speed))
    # Sort: quality desc, then speed desc within same quality
    results.sort(key=lambda r: (-r[3], -r[4]))
    return results


def format_recommendations(gpu_info, recs):
    """Build a printable recommendation table."""
    lines = []
    sep = "─" * 60
    lines.append(f"  GPU: {gpu_info['name']}  ({gpu_info['vram_mb'] // 1024}.{gpu_info['vram_mb'] % 1024 // 103} GB)")
    lines.append(f"  VRAM: {gpu_info['vram_mb']} MB  |  CUDA: {gpu_info['cuda_version']}  |  SM: {gpu_info['sm']}")
    lines.append("")
    lines.append("  推荐方案（按质量排序，仅显示本机能跑的）:")
    lines.append("")

    header = f"  {'':>3} {'Model':<12} {'Precision':<10} {'VRAM':>8} {'Speed':<8} {'Quality'}"
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    for i, (model, ct, vram, quality, speed) in enumerate(recs, 1):
        best = i == 1  # top recommendation
        tag = "  ← 推荐" if best else ""
        q_stars = "★" * quality + "☆" * (5 - quality)
        s_stars = "★" * speed + "☆" * (5 - speed)
        vram_str = f"~{vram // 1024}.{vram % 1024 // 103} GB" if vram > 1024 else f"~{vram} MB"
        lines.append(f"  {i:>2}. {model:<12} {ct:<10} {vram_str:>8} {s_stars:<8} {q_stars}{tag}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistent config
# ---------------------------------------------------------------------------

def get_config_dir():
    """Return platform-appropriate config directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(base) / "video-asr"


def load_config():
    """Load saved config or return defaults."""
    cfg_path = get_config_dir() / "config.json"
    defaults = {
        "model": "large-v3",
        "compute_type": "float16",
        "beam_size": 8,
        "backend": "auto",
        "output": "txt",
        "output_dir": "./output",
    }
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            defaults.update(data)
        except Exception:
            pass
    return defaults


def save_config(config):
    """Save config to persistent file."""
    cfg_dir = get_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.json"
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def check_gpu():
    """Check GPU availability and return (ok, message)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA is not available — will use CPU (slow)"
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        msg = f"GPU: {name} (sm_{cap[0]}{cap[1]})"
        return True, msg
    except Exception as e:
        return False, f"GPU check failed: {e}"


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path, sr=16000):
    """Extract 16 kHz mono WAV audio from video using ffmpeg."""
    print(f"[audio] Extracting from {video_path}", file=sys.stderr)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ar", str(sr), "-ac", "1", "-f", "wav", "-sample_fmt", "s16",
             wav_path],
            capture_output=True, check=True,
        )
        return wav_path
    except subprocess.CalledProcessError as e:
        os.unlink(wav_path)
        raise RuntimeError(
            f"ffmpeg audio extraction failed: "
            f"{e.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------

def transcribe_faster(audio_path, model_name="large-v3", compute_type="float16",
                      language=None, beam_size=8):
    """Transcribe using faster-whisper (CTranslate2 backend)."""
    from faster_whisper import WhisperModel

    print(f"[faster-whisper] Loading {model_name} ({compute_type})...",
          file=sys.stderr)
    t0 = time.time()
    model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
    print(f"[faster-whisper] Model loaded in {time.time()-t0:.1f}s",
          file=sys.stderr)

    print(f"[faster-whisper] Transcribing...", file=sys.stderr)
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path, beam_size=beam_size, language=language,
    )
    elapsed = time.time() - t0
    dur = info.duration or 0
    ratio = dur / elapsed if elapsed > 0 else 0
    print(f"[faster-whisper] Done: {elapsed:.1f}s ({ratio:.1f}x real-time)",
          file=sys.stderr)
    return list(segments), info


def transcribe_openai(audio_path, model_name="large-v3", language=None):
    """Transcribe using openai-whisper (PyTorch fallback)."""
    import whisper

    print(f"[openai-whisper] Loading {model_name}...", file=sys.stderr)
    t0 = time.time()
    model = whisper.load_model(model_name, device="cuda")
    print(f"[openai-whisper] Model loaded in {time.time()-t0:.1f}s",
          file=sys.stderr)

    print(f"[openai-whisper] Transcribing...", file=sys.stderr)
    t0 = time.time()
    result = model.transcribe(audio_path, beam_size=5, language=language)
    elapsed = time.time() - t0
    dur = 0
    if result.get("segments"):
        dur = result["segments"][-1].get("end", 0)
    ratio = dur / elapsed if elapsed > 0 else 0
    print(f"[openai-whisper] Done: {elapsed:.1f}s ({ratio:.1f}x real-time)",
          file=sys.stderr)
    return result


def transcribe_whisperx(audio_path, model_name="large-v3", compute_type="float16",
                        hf_token=None, batch_size=8):
    """Transcribe with WhisperX (supports speaker diarization)."""
    import whisperx

    print(f"[WhisperX] Loading {model_name}...", file=sys.stderr)
    t0 = time.time()
    model = whisperx.load_model(model_name, "cuda", compute_type=compute_type)
    print(f"[WhisperX] Model loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    print(f"[WhisperX] Transcribing...", file=sys.stderr)
    t0 = time.time()
    result = model.transcribe(audio_path, batch_size=batch_size)
    print(f"[WhisperX] Done in {time.time()-t0:.1f}s", file=sys.stderr)

    print(f"[WhisperX] Aligning timestamps...", file=sys.stderr)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device="cuda",
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio_path, "cuda",
    )

    if hf_token:
        print(f"[WhisperX] Diarizing speakers...", file=sys.stderr)
        diarize = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device="cuda",
        )
        diarize_segments = diarize(audio_path)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    return result


# ---------------------------------------------------------------------------
# Detect best available backend
# ---------------------------------------------------------------------------

def detect_backend():
    """Return the name of the fastest available backend."""
    try:
        from faster_whisper import WhisperModel  # noqa
        return "faster"
    except ImportError:
        pass
    try:
        import whisper  # noqa
        return "whisper"
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def segments_to_text(segments, diarize=False):
    """Format segments as plain text (one line per segment)."""
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "")
        if diarize and speaker:
            lines.append(f"[{speaker}]: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def segments_to_srt(segments):
    """Format segments as SRT subtitle text."""
    def _fmt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(str(i))
        lines.append(f"{_fmt(seg['start'])} --> {_fmt(seg['end'])}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def segments_to_json(segments, info=None):
    """Format segments as JSON string."""
    data = {
        "segments": [
            {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker", None),
                "confidence": seg.get("confidence", None),
            }
            for seg in segments
        ],
    }
    if info:
        data["language"] = (
            info.language if hasattr(info, "language") else None
        )
        data["duration"] = (
            info.duration if hasattr(info, "duration") else None
        )
    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Single file processing
# ---------------------------------------------------------------------------

def process_single(video_path, args):
    """Transcribe one video file."""
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        print(f"[error] File not found: {video_path}", file=sys.stderr)
        return

    # Extract audio
    try:
        wav_path = extract_audio(video_path)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return

    try:
        # Resolve backend
        if args.backend == "auto":
            detected = detect_backend()
            if detected:
                backend = detected
            else:
                print(
                    "[error] No backend available. "
                    "Install faster-whisper or openai-whisper.",
                    file=sys.stderr,
                )
                return
        else:
            backend = args.backend
        print(f"[backend] {backend}", file=sys.stderr)

        if args.diarize:
            # WhisperX path
            compute_type = (
                args.compute_type if backend == "faster" else "float16"
            )
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            if not hf_token:
                print(
                    "[warn] HF_TOKEN not set — skipping diarization",
                    file=sys.stderr,
                )
            result = transcribe_whisperx(
                wav_path, args.model, compute_type=compute_type,
                hf_token=hf_token, batch_size=args.batch_size,
            )
            segments = result.get("segments", [])
            info = None
            diarize = bool(hf_token)

        elif backend == "faster":
            segments, info = transcribe_faster(
                wav_path, args.model, args.compute_type,
                language=args.language, beam_size=args.beam_size,
            )
            segments = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ]
            diarize = False

        elif backend == "whisper":
            result = transcribe_openai(
                wav_path, args.model, language=args.language,
            )
            segments = result.get("segments", [])
            info = None
            diarize = False

        else:
            print(
                "[error] No backend available.",
                file=sys.stderr,
            )
            return

        # ---- Output ----
        fmt = args.output

        if args.stdout:
            if fmt == "srt":
                sys.stdout.write(segments_to_srt(segments))
            elif fmt == "json":
                sys.stdout.write(
                    segments_to_json(
                        segments, info if backend == "faster" else None
                    )
                )
            else:
                sys.stdout.write(segments_to_text(segments, diarize))
        else:
            output_dir = args.output_dir or "."
            stem = Path(video_path).stem
            if fmt == "srt":
                out_path = Path(output_dir) / f"{stem}.srt"
                content = segments_to_srt(segments)
            elif fmt == "json":
                out_path = Path(output_dir) / f"{stem}.json"
                content = segments_to_json(
                    segments, info if backend == "faster" else None
                )
            else:
                out_path = Path(output_dir) / f"{stem}.txt"
                content = segments_to_text(segments, diarize)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Auto-numbering: never overwrite, always create new file
            existing = sorted(
                out_path.parent.glob(f"{out_path.stem}*{out_path.suffix}")
            )
            if existing:
                max_num = 0
                for f in existing:
                    f_stem = f.stem
                    if f_stem == out_path.stem:
                        max_num = max(max_num, 0)
                    else:
                        m = re.search(r"\((\d+)\)$", f_stem)
                        if m:
                            max_num = max(max_num, int(m.group(1)))
                out_path = (
                    out_path.parent
                    / f"{out_path.stem} ({max_num + 1}){out_path.suffix}"
                )

            out_path.write_text(content, encoding="utf-8")
            print(f"[output] {out_path}", file=sys.stderr)

    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="video-asr — Local video speech-to-text via faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "videos", nargs="*",
        help="Video file path(s); supports glob patterns",
    )

    # Backend
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "faster", "whisper", "whisperx"],
        default="auto",
        help="Transcription backend (default: auto → faster-whisper)",
    )

    # Model
    parser.add_argument(
        "--model", "-m", default="large-v3",
        help="Whisper model size: tiny/base/small/medium/large-v3",
    )
    parser.add_argument(
        "--language", "-lang", default=None,
        help="Force language (auto-detect by default). "
             "Use 'en' for mixed Chinese-English videos.",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["int8", "float16", "float32"],
        help="faster-whisper precision: "
             "float16 (best balance), int8 (less VRAM), float32 (highest)",
    )
    parser.add_argument(
        "--beam-size", type=int, default=8,
        help="Beam search width (default: 8; lower = faster, higher = more accurate)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        default="txt",
        choices=["txt", "srt", "json"],
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output/)",
    )
    parser.add_argument(
        "--stdout", "-p",
        action="store_true",
        help="Print to stdout instead of writing files",
    )

    # Diarization
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires whisperx + HF_TOKEN)",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token for pyannote diarization models",
    )

    # Performance
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="WhisperX batch size (default: 8; reduce to 4 for 8 GB VRAM)",
    )

    # Detection & setup
    parser.add_argument(
        "--recommend", action="store_true",
        help="Detect GPU, show ranked model table, and save your choice",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Show compatible models for this GPU and exit",
    )
    parser.add_argument(
        "--set-default", nargs=2, metavar=("MODEL", "COMPUTE_TYPE"),
        help="Save default model+precision and exit "
             "(e.g. --set-default large-v3 float16)",
    )

    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # ---- Detection / setup commands (no video needed) ----

    # --set-default MODEL COMPUTE_TYPE
    if args.set_default:
        model, ct = args.set_default
        if ct not in ("int8", "float16", "float32"):
            print(f"[error] Invalid compute type: {ct} (use int8/float16/float32)",
                  file=sys.stderr)
            sys.exit(1)
        config = load_config()
        config["model"] = model
        config["compute_type"] = ct
        path = save_config(config)
        print(f"[config] Default saved: {model} / {ct}", file=sys.stderr)
        print(f"[config] File: {path}", file=sys.stderr)
        return

    # Detect GPU
    gpu_info = detect_gpu_details()
    if gpu_info["has_cuda"]:
        print(f"[env] GPU: {gpu_info['name']}  ({gpu_info['vram_mb'] // 1024} GB VRAM)",
              file=sys.stderr)
    else:
        print(f"[env] No CUDA GPU detected — will use CPU (very slow)", file=sys.stderr)

    # --list-models
    if args.list_models:
        if not gpu_info["has_cuda"]:
            print("[error] No GPU detected, cannot recommend models",
                  file=sys.stderr)
            sys.exit(1)
        recs = build_recommendations(gpu_info["vram_mb"])
        print(format_recommendations(gpu_info, recs), file=sys.stderr)
        return

    # --recommend (interactive)
    if args.recommend:
        if not gpu_info["has_cuda"]:
            print("[error] No GPU detected", file=sys.stderr)
            sys.exit(1)
        recs = build_recommendations(gpu_info["vram_mb"])
        print(file=sys.stderr)
        print(format_recommendations(gpu_info, recs), file=sys.stderr)

        while True:
            try:
                print("  选择默认配置 (输入 1-{}，或按 Enter 跳过): ".format(len(recs)),
                      end="", flush=True, file=sys.stderr)
                choice = sys.stdin.readline().strip()
                if not choice:
                    print("  → 跳过，未修改默认配置", file=sys.stderr)
                    break
                idx = int(choice) - 1
                if 0 <= idx < len(recs):
                    model, ct, *_ = recs[idx]
                    config = load_config()
                    config["model"] = model
                    config["compute_type"] = ct
                    path = save_config(config)
                    print(f"\n  ✅ 已保存默认配置: {model} / {ct}", file=sys.stderr)
                    print(f"    配置文件: {path}", file=sys.stderr)
                    print(f"    现在可直接: transcribe 视频.mp4", file=sys.stderr)
                    break
                else:
                    print(f"  ⚠ 请输入 1-{len(recs)} 之间的数字", file=sys.stderr)
            except (ValueError, EOFError):
                print("  → 跳过", file=sys.stderr)
                break
        return

    # ---- Normal transcription mode ----
    if not args.videos:
        print("[error] No video files specified. Use transcribe --help for usage.",
              file=sys.stderr)
        sys.exit(1)
    # Apply saved config as defaults (CLI flags override)
    saved = load_config()
    if not hasattr(args, "_config_loaded"):
        for key, val in saved.items():
            if hasattr(args, key):
                current = getattr(args, key)
                # Only apply if user didn't explicitly set it
                if key == "model" and args.model == "large-v3" and val != "large-v3":
                    setattr(args, key, val)
                if key == "compute_type" and args.compute_type == "float16" and val != "float16":
                    setattr(args, key, val)
                if key == "beam_size" and args.beam_size == 8 and val != 8:
                    setattr(args, key, val)
                if key == "output_dir" and args.output_dir == "./output" and val != "./output":
                    setattr(args, key, val)
        args._config_loaded = True

    # GPU check (already done)
    ok, msg = check_gpu()
    if not ok:
        print(f"[warn] {msg}", file=sys.stderr)

    # Expand patterns
    import glob as glob_mod
    video_files = []
    for pattern in args.videos:
        matched = glob_mod.glob(pattern, recursive=False)
        if matched:
            video_files.extend(Path(p) for p in matched)
        else:
            video_files.append(Path(pattern))

    if not video_files:
        print("[error] No video files matched", file=sys.stderr)
        sys.exit(1)

    print(f"[task] {len(video_files)} video(s)", file=sys.stderr)

    for v in video_files:
        print(f"\n{'='*60}", file=sys.stderr)
        process_single(v, args)
        print(f"{'='*60}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
