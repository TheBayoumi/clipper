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
from .source_fidelity import (
    SourceProfile,
    compare_encoded_to_composition,
    crf_attempts,
    probe_source_profile,
)
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


_ATTENTION_WORDS = {
    "damn",
    "wow",
    "wait",
    "look",
    "risk",
    "sell",
    "buy",
    "entry",
    "stop",
    "profit",
    "loss",
    "fomo",
    "scam",
    "bullshit",
}


def _attention_beats(
    clip: ClipCandidate,
    segments: Sequence[TranscriptSegment],
    *,
    limit: int = 4,
) -> tuple[tuple[float, float], ...]:
    """Find sparse edit beats from authentic word timing, never random keywords alone."""
    words = sorted(
        (
            word
            for segment in segments
            if segment.end > clip.start and segment.start < clip.end
            for word in segment.words
            if word.end > clip.start and word.start < clip.end
        ),
        key=lambda word: (word.start, word.end),
    )
    if not words:
        return ()
    beats: list[tuple[float, float]] = []
    for index, word in enumerate(words):
        relative = max(word.start, clip.start) - clip.start
        if relative > clip.duration - 0.55:
            continue
        previous = words[index - 1] if index else None
        gap = word.start - previous.end if previous is not None else 99.0
        sentence_reset = bool(previous is None or previous.text.rstrip().endswith((".", "!", "?")))
        token = re.sub(r"[^a-z']", "", word.text.lower())
        reaction = token in _ATTENTION_WORDS
        # A beat must be justified by a real pause/sentence transition or a
        # compact reaction/action word. Spacing prevents machine-gun zooms.
        if not (gap >= 0.32 or sentence_reset or reaction):
            continue
        start = max(0.0, relative - (0.04 if reaction else 0.0))
        if beats and start - beats[-1][0] < 3.2:
            continue
        end = min(clip.duration, start + (0.68 if reaction else 0.52))
        if end - start < 0.35:
            continue
        beats.append((round(start, 3), round(end, 3)))
        if len(beats) >= limit:
            break
    return tuple(beats)


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
    crf_override: int | None = None,
    source_profile: SourceProfile | None = None,
    attention_beats: Sequence[tuple[float, float]] = (),
) -> list[str]:
    preset = os.getenv("CLIPPER_RENDER_PRESET", "ultrafast").strip().lower()
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
        raise RenderError("CLIPPER_RENDER_PRESET is not an allowed x264 preset")
    try:
        crf = int(os.getenv("CLIPPER_RENDER_CRF", "20"))
        threads = int(os.getenv("CLIPPER_RENDER_THREADS", "1"))
    except ValueError as exc:
        raise RenderError("render CRF and threads must be integers") from exc
    if crf_override is not None:
        crf = crf_override
    if not 12 <= crf <= 28 or not 1 <= threads <= 4:
        raise RenderError("render CRF must be 12-28 and threads must be 1-4")
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
    if is_tiktok and (width != 1080 or height != 1920):
        raise RenderError("Style B requires an exact 1080x1920 vertical output")
    if source_fps:
        try:
            rate = Fraction(source_fps)
            if not 23 <= float(rate) <= 61:
                raise ValueError("unsupported source frame rate")
        except (ValueError, ZeroDivisionError) as exc:
            raise RenderError("invalid original source frame rate") from exc
        fps = source_fps
    else:
        fps = "30"
    blur_width = max(180, width // 3)
    blur_height = max(320, height // 3)
    for start, end in attention_beats:
        if not (0 <= start < end <= clip.duration):
            raise RenderError("attention beat is outside the selected clip")
    effect_enable = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in attention_beats)
    if is_tiktok and attention_beats and editorial_layout == "default":
        base_filter = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={blur_width}:{blur_height}:force_original_aspect_ratio=increase,"
            f"crop={blur_width}:{blur_height},gblur=sigma=18,scale={width}:{height}[bg2];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgscaled];"
            "[fgscaled]split=2[fgbase][fgzoomsrc];"
            "[fgzoomsrc]scale='ceil(iw*1.025/2)*2':'ceil(ih*1.025/2)*2'[fgzoom];"
            "[bg2][fgbase]overlay=(W-w)/2:(H-h)/2[scene];"
            f"[scene][fgzoom]overlay=(W-w)/2:(H-h)/2:enable='{effect_enable}',"
            f"{caption_filter},fps={fps}:start_time=0[captioned]"
        )
    else:
        base_filter = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={blur_width}:{blur_height}:force_original_aspect_ratio=increase,"
            f"crop={blur_width}:{blur_height},gblur=sigma=18,scale={width}:{height}[bg2];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,"
            f"{caption_filter},fps={fps}:start_time=0[captioned]"
        )
    if editorial_layout not in {
        "default",
        "tjr-trading-logo-safe",
        "tjr-memecoin-logo-safe",
    }:
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
            f"color=c=0x10131a:s=1080x1920:r={fps}[canvas];"
            "[canvas][top]overlay=0:180[layout];"
            "[layout][face]overlay=0:830,"
            f"{caption_filter},fps={fps}:start_time=0[captioned]"
        )
    if editorial_layout == "tjr-memecoin-logo-safe":
        if watermark_path is not None or (width, height) != (1080, 1920):
            raise RenderError("TJR memecoin crop rejects logos or nonvertical outputs")
        # Audited 1920x1080 TJRTrades memecoin layout: isolate the central
        # chart and the bottom-right reaction camera. The Fomo header/sidebar,
        # order-entry panel and chart-provider badge remain outside the crop.
        base_filter = (
            "[0:v]split=2[chart][webcam];"
            "[chart]crop=1140:465:340:155,scale=1080:440:flags=lanczos[top];"
            "[webcam]crop=575:325:1345:755,scale=1080:610:flags=lanczos[face];"
            f"color=c=0x10131a:s=1080x1920:r={fps}[canvas];"
            "[canvas][top]overlay=0:310[layout];"
            "[layout][face]overlay=0:850,"
            f"{caption_filter},fps={fps}:start_time=0[captioned]"
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
        *(
            ["-colorspace", source_profile.color_space]
            if source_profile and source_profile.color_space in {"bt709", "smpte170m"}
            else []
        ),
        *(
            ["-color_trc", source_profile.color_transfer]
            if source_profile and source_profile.color_transfer in {"bt709", "smpte170m"}
            else []
        ),
        *(
            ["-color_primaries", source_profile.color_primaries]
            if source_profile and source_profile.color_primaries in {"bt709", "smpte170m"}
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
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=40,
        )
        streams = json.loads(probe.stdout)["streams"]
        source = streams[0]
        rate = str(source.get("avg_frame_rate") or source.get("r_frame_rate") or "")
        if 23 <= float(Fraction(rate)) <= 61:
            return rate
        raise ValueError("unusable native frame rate")
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        ZeroDivisionError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise RenderError("Style B could not verify the native source frame rate") from exc


class FFmpegRenderer:
    def __init__(self) -> None:
        if not shutil.which("ffmpeg"):
            raise RenderError("ffmpeg is not installed or not on PATH")
        self.quality_results: dict[str, dict[str, object]] = {}

    def render(
        self,
        source_path: Path,
        output_path: Path,
        clip: ClipCandidate,
        segments: Sequence[TranscriptSegment],
        watermark_path: Path | None = None,
        editorial_layout: str = "default",
        tiktok_hook: str | None = None,
        source_profile: SourceProfile | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_path = output_path.with_suffix(".srt")
        create_srt(clip, segments, subtitle_path)
        burn_in = (
            create_tiktok_ass(
                clip,
                segments,
                output_path.with_suffix(".ass"),
                hook_text=tiktok_hook,
            )
            if tiktok_hook is not None
            else subtitle_path
        )
        native = (
            (source_profile or probe_source_profile(source_path))
            if tiktok_hook is not None
            else None
        )
        attention_beats = (
            _attention_beats(clip, segments)
            if tiktok_hook is not None and editorial_layout == "default"
            else ()
        )
        split_montage = editorial_layout in {
            "tjr-trading-logo-safe",
            "tjr-memecoin-logo-safe",
        }
        rates = crf_attempts(native) if native else (None,)
        evidence: list[dict[str, float | int]] = []
        stats_path = output_path.with_suffix(".ssim.txt")
        self.quality_results.pop(str(output_path.resolve()), None)
        for crf in rates:
            command = build_ffmpeg_command(
                source_path,
                output_path,
                clip,
                burn_in,
                watermark_path=watermark_path,
                editorial_layout=editorial_layout,
                source_fps=native.fps if native else None,
                crf_override=crf,
                source_profile=native,
                attention_beats=attention_beats,
            )
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
            except subprocess.CalledProcessError as exc:
                raise RenderError((exc.stderr or exc.stdout or str(exc))[-2000:]) from exc
            except subprocess.TimeoutExpired as exc:
                raise RenderError("ffmpeg render timed out after 900 seconds") from exc
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RenderError(f"ffmpeg did not create a valid output: {output_path}")
            if native is None:
                return output_path
            if crf is None:
                raise RenderError("source-matched encoding requires measured attempt")
            try:
                measured, compared = compare_encoded_to_composition(
                    command,
                    source=source_path,
                    output=output_path,
                    clip_start=clip.start,
                    duration=clip.duration,
                    fps=native.fps,
                    stats_path=stats_path,
                )
            except Exception:
                output_path.unlink(missing_ok=True)
                raise
            evidence.append({"crf": crf, "mean_ssim": measured, "frames_compared": compared})
            if measured < 0.99:
                continue
            report: dict[str, object] = {
                "status": "MEASURED_SOURCE_MATCHED_ENCODING",
                "source_profile": native.as_dict(),
                "output_cadence": native.fps,
                "output_format": "1080x1920 H.264 high yuv420p",
                "campaign_watermark_applied": watermark_path is not None,
                "source_to_delivery_mean_ssim": measured,
                "minimum_mean_ssim": 0.99,
                "compared_frames": compared,
                "accepted_crf": crf,
                "attempts": evidence,
                "output_bytes": output_path.stat().st_size,
                "measurement_method": (
                    "frame-level SSIM against decoded original with the identical "
                    "crop, animated ASS overlay, pixel format and native cadence "
                    "before the final H.264 encode"
                ),
                "editorial_visual_approval": False,
                "edit_plan": {
                    "style": (
                        "split_screen_montage"
                        if split_montage
                        else (
                            "semantic_micro_punch"
                            if attention_beats
                            else "caption_led_no_forced_effect"
                        )
                    ),
                    "punch_scale": 1.025 if attention_beats else 1.0,
                    "attention_beats": [
                        {"start": start, "end": end} for start, end in attention_beats
                    ],
                    "random_effects": False,
                    "editorial_layout": editorial_layout,
                },
            }
            report_path = output_path.with_suffix(".quality.json")
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            self.quality_results[str(output_path.resolve())] = report
            return output_path
        output_path.unlink(missing_ok=True)
        raise RenderError(
            "final render did not match the actual source composition at "
            "mean frame SSIM 0.99 or higher: " + str(evidence)
        )
