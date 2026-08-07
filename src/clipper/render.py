from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .models import ClipCandidate, TranscriptSegment


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
) -> list[str]:
    escaped_subtitles = _escape_filter_path(Path(subtitle_path))
    base_filter = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=28[bg2];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg2];"
        f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,"
        f"subtitles='{escaped_subtitles}':"
        "force_style='FontName=DejaVu Sans,FontSize=18,Alignment=2,"
        "MarginV=150,Outline=3,Shadow=0,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000',fps=30[captioned]"
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
            + "[captioned][wm]overlay=W-w-48:48:format=auto,format=yuv420p[v]"
        )
    else:
        filter_complex = base_filter + ";[captioned]format=yuv420p[v]"
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
        "medium",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


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
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_path = output_path.with_suffix(".srt")
        create_srt(clip, segments, subtitle_path)
        command = build_ffmpeg_command(
            source_path,
            output_path,
            clip,
            subtitle_path,
            watermark_path=watermark_path,
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
