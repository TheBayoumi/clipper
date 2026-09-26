from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from .models import ClipCandidate, TranscriptSegment
from .tiktok import create_tiktok_ass


class RenderError(RuntimeError):
    """Raised when FFmpeg cannot produce a clip."""


def _srt_timestamp(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt(
    clip: ClipCandidate,
    segments: Sequence[TranscriptSegment],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    lines: list[str] = []
    index = 1
    for segment in segments:
        start = max(segment.start, clip.start)
        end = min(segment.end, clip.end)
        if end <= start:
            continue
        lines.extend(
            [
                str(index),
                f"{_srt_timestamp(start - clip.start)} --> {_srt_timestamp(end - clip.start)}",
                re.sub(r"^>>\s*", "", segment.text.strip()),
                "",
            ]
        )
        index += 1
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _escape_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_ffmpeg_command(
    source_path: str | Path,
    output_path: str | Path,
    clip: ClipCandidate,
    subtitle_path: str | Path,
    *,
    watermark_path: str | Path | None = None,
    width: int = 1080,
    height: int = 1920,
    editorial_layout: str = "default",
    source_fps: str | None = None,
) -> list[str]:
    preset = os.getenv("CLIPPER_RENDER_PRESET", "ultrafast").strip().lower()
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
        raise RenderError("CLIPPER_RENDER_PRESET is not an allowed x264 preset")
    try:
        crf = int(os.getenv("CLIPPER_RENDER_CRF", "20"))
        threads = int(os.getenv("CLIPPER_RENDER_THREADS", "1"))
    except ValueError as exc:
        raise RenderError("render CRF and threads must be integers") from exc
    if not 16 <= crf <= 28 or not 1 <= threads <= 4:
        raise RenderError("render CRF must be 16-28 and threads must be 1-4")
    escaped_subtitles = _escape_filter_path(Path(subtitle_path))
    is_tiktok = Path(subtitle_path).suffix.lower() == ".ass"
    caption_filter = (
        f"ass='{escaped_subtitles}'"
        if is_tiktok
        else (
            f"subtitles='{escaped_subtitles}':"
            "force_style='FontName=DejaVu Sans,FontSize=10,Alignment=2,"
            "MarginV=28,MarginL=24,MarginR=24,Outline=2,Shadow=0,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000'"
        )
    )
    if is_tiktok:
        if width != 1080 or height != 1920:
            raise RenderError("Style B requires an exact 1080x1920 vertical output")
        if watermark_path is not None:
            raise RenderError("Style B forbids adding any logos or watermarks")
    if source_fps:
        try:
            rate = Fraction(source_fps)
            if not 24 <= float(rate) <= 60:
                raise ValueError("unsupported source frame rate")
        except (ValueError, ZeroDivisionError) as exc:
            raise RenderError("invalid original source frame rate") from exc
        fps = source_fps if float(rate) >= 29 else "30"
    else:
        fps = "30"
    blur_width = max(180, width // 3)
    blur_height = max(320, height // 3)
    base_filter = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={blur_width}:{blur_height}:force_original_aspect_ratio=increase,"
        f"crop={blur_width}:{blur_height},gblur=sigma=18,scale={width}:{height}[bg2];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg2];"
        f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,"
        f"{caption_filter},fps={fps}[captioned]"
    )
    if editorial_layout not in {"default", "tjr-trading-logo-safe"}:
        raise RenderError("unknown editorial layout; never silently bypass logo guard")
    if editorial_layout == "tjr-trading-logo-safe":
        if watermark_path is not None or (width, height) != (1080, 1920):
            raise RenderError("TJR logo-safe crop rejects logos or nonvertical outputs")
        # This known TRiches livestream layout has chart above the webcam,
        # with sponsor strips in the lower-right source region. No pixel of
        # that sponsor area enters the output; no logo is added by Clipper.
        base_filter = (
            "[0:v]split=2[chart][webcam];"
            "[chart]crop=1460:600:280:20,scale=1080:445:flags=lanczos[top];"
            "[webcam]crop=565:335:20:710,scale=1080:640:flags=lanczos[face];"
            "color=c=0x10131a:s=1080x1920:r=30[canvas];"
            "[canvas][top]overlay=0:180[layout];"
            "[layout][face]overlay=0:830,"
            f"{caption_filter},fps={fps}[captioned]"
        )
    inputs = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip.start:.3f}",
        "-i",
        str(source_path),
    ]
    if watermark_path is not None:
        inputs.extend(["-i", str(watermark_path)])
        filter_complex = (
            base_filter
            + ";[1:v]scale=180:-1:force_original_aspect_ratio=decrease[wm];"
            + "[captioned][wm]overlay=W-w-48:48:format=auto,format=yuv420p,setsar=1[v]"
        )
    else:
        filter_complex = base_filter + ";[captioned]format=yuv420p,setsar=1[v]"
    return [
        *inputs,
        "-t",
        f"{clip.duration:.3f}",
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-af",
        "loudnorm=I=-14:LRA=11:TP=-1.5",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-threads",
        str(threads),
        "-c:a",
        "aac",
        "-b:a",
        "320k" if is_tiktok else "192k",
        "-pix_fmt",
        "yuv420p",
        *(
            ["-profile:v", "high", "-x264-params", "aq-mode=3:aq-strength=1.05"]
            if is_tiktok
            else []
        ),
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _source_frame_rate(path: Path) -> str:
    """Read the original cadence so Style B does not create fake 60fps."""
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True, timeout=40,
        )
        streams = json.loads(probe.stdout)["streams"]
        source = streams[0]
        rate = str(source.get("avg_frame_rate") or source.get("r_frame_rate") or "")
        if 24 <= float(Fraction(rate)) <= 60:
            return rate
        raise ValueError("unusable native frame rate")
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            ZeroDivisionError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise RenderError("Style B could not verify the native source frame rate") from exc


class FFmpegRenderer:
    def __init__(self) -> None:
        if not shutil.which("ffmpeg"):
            raise RenderError("ffmpeg is not installed or not on PATH")

    def render(
        self,
        source_path: Path,
        output_path: Path,
        clip: ClipCandidate,
        segments: Sequence[TranscriptSegment],
        watermark_path: Path | None = None,
        editorial_layout: str = "default",
        tiktok_hook: str | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_path = output_path.with_suffix(".srt")
        create_srt(clip, segments, subtitle_path)
        burn_in = (
            create_tiktok_ass(
                clip, segments, output_path.with_suffix(".ass"),
                hook_text=tiktok_hook,
            )
            if tiktok_hook is not None
            else subtitle_path
        )
        native_fps = _source_frame_rate(source_path) if tiktok_hook is not None else None
        command = build_ffmpeg_command(
            source_path,
            output_path,
            clip,
            burn_in,
            watermark_path=watermark_path,
            editorial_layout=editorial_layout,
            source_fps=native_fps,
        )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
        except subprocess.CalledProcessError as exc:
            raise RenderError((exc.stderr or exc.stdout or str(exc))[-2000:]) from exc
        except subprocess.TimeoutExpired as exc:
            raise RenderError("ffmpeg render timed out after 900 seconds") from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RenderError(f"ffmpeg did not create a valid output: {output_path}")
        return output_path
