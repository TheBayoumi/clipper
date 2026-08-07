from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .captions import create_word_reveal_ass
from .models import ClipCandidate, TranscriptSegment
from .tracking import (
    DEFAULT_TARGET_ASPECT,
    TrackingPlan,
    plan_speaker_crop,
    portrait_crop_dimensions,
    tracking_expressions,
)


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
    target_aspect = width / height
    zoom = tracking_plan.zoom_factor if tracking_plan is not None else zoom_factor
    crop_x, crop_y = tracking_expressions(tracking_plan)

    if (
        tracking_plan is not None
        and tracking_plan.source_width > 0
        and tracking_plan.source_height > 0
    ):
        crop_width = tracking_plan.crop_width
        crop_height = tracking_plan.crop_height
        if crop_width <= 0 or crop_height <= 0:
            crop_width, crop_height = portrait_crop_dimensions(
                tracking_plan.source_width,
                tracking_plan.source_height,
                target_aspect=target_aspect,
                zoom_factor=zoom,
            )
        crop_filter = f"crop={crop_width}:{crop_height}:x='{crop_x}':y='{crop_y}'"
    else:
        crop_w = f"trunc(min(iw,ih*{target_aspect:.8f})/{zoom:.6f}/2)*2"
        crop_h = f"trunc(min(ih,iw/{target_aspect:.8f})/{zoom:.6f}/2)*2"
        crop_filter = f"crop=w='{crop_w}':h='{crop_h}':x='{crop_x}':y='{crop_y}'"

    base_filter = (
        f"[0:v]{crop_filter},scale={width}:{height}:flags=lanczos,"
        f"subtitles='{escaped_subtitles}',fps=30[captioned]"
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
        speaker_focus: bool = True,
        zoom_factor: float = 1.12,
        speaker_sample_fps: float = 4.0,
        speaker_switch_margin: float = 1.35,
        speaker_transition_seconds: float = 0.22,
    ) -> None:
        if not shutil.which("ffmpeg"):
            raise RenderError("ffmpeg is not installed or not on PATH")
        if not 1.0 <= zoom_factor <= 1.35:
            raise RenderError("zoom_factor must be between 1.0 and 1.35")
        if not 1.0 <= speaker_sample_fps <= 12.0:
            raise RenderError("speaker_sample_fps must be between 1 and 12")
        if not 1.0 <= speaker_switch_margin <= 3.0:
            raise RenderError("speaker_switch_margin must be between 1.0 and 3.0")
        if not 0.0 <= speaker_transition_seconds <= 1.0:
            raise RenderError("speaker_transition_seconds must be between 0.0 and 1.0")
        self.speaker_focus = speaker_focus
        self.zoom_factor = zoom_factor
        self.speaker_sample_fps = speaker_sample_fps
        self.speaker_switch_margin = speaker_switch_margin
        self.speaker_transition_seconds = speaker_transition_seconds

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
            plan_speaker_crop(
                source_path,
                clip,
                list(segments),
                zoom_factor=self.zoom_factor,
                sample_fps=self.speaker_sample_fps,
                switch_margin=self.speaker_switch_margin,
                transition_seconds=self.speaker_transition_seconds,
            )
            if self.speaker_focus
            else TrackingPlan(
                self.zoom_factor,
                0,
                0,
                target_aspect=DEFAULT_TARGET_ASPECT,
                speaker_focus=False,
            )
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
