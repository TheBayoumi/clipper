from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .captions import create_word_reveal_ass
from .models import ClipCandidate, TranscriptSegment
from .tracking import TrackingPlan, track_face_crop, tracking_expressions


class RenderError(RuntimeError):
    """Raised when FFmpeg cannot produce a clip."""


def _escape_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_ffmpeg_command(
    source_path: str | Path,
    output_path: str | Path,
    clip: ClipCandidate,
    subtitle_path: str | Path,
    *,
    watermark_path: str | Path | None = None,
    tracking_plan: TrackingPlan | None = None,
    zoom_factor: float = 1.12,
    width: int = 1080,
    height: int = 1920,
) -> list[str]:
    escaped_subtitles = _escape_filter_path(Path(subtitle_path))
    blur_width = max(180, width // 3)
    blur_height = max(320, height // 3)
    zoom = tracking_plan.zoom_factor if tracking_plan is not None else zoom_factor
    if zoom > 1.0:
        crop_x, crop_y = tracking_expressions(tracking_plan)
        foreground = (
            f"[fg]crop=iw/{zoom:.6f}:ih/{zoom:.6f}:x='{crop_x}':y='{crop_y}',"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease[fg2];"
        )
    else:
        foreground = f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg2];"

    base_filter = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={blur_width}:{blur_height}:force_original_aspect_ratio=increase,"
        f"crop={blur_width}:{blur_height},gblur=sigma=18,scale={width}:{height}[bg2];"
        + foreground
        + f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,subtitles='{escaped_subtitles}',"
        "fps=30[captioned]"
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
        "ultrafast",
        "-crf",
        "20",
        "-threads",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


class FFmpegRenderer:
    def __init__(
        self,
        *,
        face_tracking: bool = True,
        zoom_factor: float = 1.12,
        face_sample_fps: float = 4.0,
    ) -> None:
        if not shutil.which("ffmpeg"):
            raise RenderError("ffmpeg is not installed or not on PATH")
        if not 1.0 <= zoom_factor <= 1.35:
            raise RenderError("zoom_factor must be between 1.0 and 1.35")
        if not 1.0 <= face_sample_fps <= 12.0:
            raise RenderError("face_sample_fps must be between 1 and 12")
        self.face_tracking = face_tracking
        self.zoom_factor = zoom_factor
        self.face_sample_fps = face_sample_fps

    def render(
        self,
        source_path: Path,
        output_path: Path,
        clip: ClipCandidate,
        segments: Sequence[TranscriptSegment],
        watermark_path: Path | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_path = output_path.with_suffix(".ass")
        create_word_reveal_ass(clip, segments, subtitle_path)
        tracking_plan = (
            track_face_crop(
                source_path,
                clip,
                zoom_factor=self.zoom_factor,
                sample_fps=self.face_sample_fps,
            )
            if self.face_tracking
            else TrackingPlan(self.zoom_factor, 0, 0)
        )
        output_path.with_suffix(".tracking.json").write_text(
            json.dumps(tracking_plan.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        command = build_ffmpeg_command(
            source_path,
            output_path,
            clip,
            subtitle_path,
            watermark_path=watermark_path,
            tracking_plan=tracking_plan,
            zoom_factor=self.zoom_factor,
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
