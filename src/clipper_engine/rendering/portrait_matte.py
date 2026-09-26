"""Clipper-owned portrait announcement matte with source-synchronized approved typography.

The gameplay profile supplies source transition calibration and approved exact copy;
the renderer owns layout, frame-by-frame state, fidelity checks, and delivery.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .. import media_contract as media
from ..montage import MontageRejection, rate
from ..profiles import CampaignProfile
from . import ffv1

RGB = tuple[int, int, int]

PORTRAIT_REQUIRED_CHECKS = frozenset(
    {
        "portrait_canonical_ffv1_nut",
        "portrait_video_frame_count_exact",
        "portrait_encoded_duration",
        "portrait_dimensions",
        "portrait_frame_rate",
        "approved_text_full_duration",
        "toggle_opening_off",
        "toggle_final_on",
        "portrait_source_audio_only",
        "portrait_ssim",
        "portrait_psnr_db",
        "portrait_color_metadata",
    }
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ease(value: float) -> float:
    t = _clamp(value)
    return t * t * (3.0 - 2.0 * t)


def toggle_progress(frame: int, plan: dict[str, Any], profile: CampaignProfile) -> float:
    """Use the SAME certified source states and exact FFmpeg edit windows as the picture."""
    edit = plan["montage"]
    source_event_start = int(plan["evidence"]["toggle_motion_start_frame"])
    source_event_end = int(plan["evidence"]["toggle_motion_end_frame"])
    if not source_event_start < source_event_end:
        raise MontageRejection("toggle_event_invalid", "source visual-event interval is invalid")
    hook_frames = int(edit["hook"]["frames"])
    source_frames = int(edit["full_source_frames"])
    comparison_frames = int(edit["comparison_frames"])
    comparison_start = hook_frames + source_frames
    ending_start = comparison_start + comparison_frames

    if frame < hook_frames:
        source_frame = int(edit["hook"]["start_frame"]) + frame
        return _ease((source_frame - source_event_start) / (source_event_end - source_event_start))
    if frame < comparison_start:
        source_frame = frame - hook_frames
        return _ease((source_frame - source_event_start) / (source_event_end - source_event_start))
    if frame < ending_start:
        local = frame - comparison_start
        if edit["comparison_mode"] == "wipe":
            start = comparison_frames // 6
            duration = 2 * comparison_frames // 3
            return _ease((local - start) / duration)
        if edit["comparison_mode"] == "cuts":
            elapsed = Fraction(local, 1) / rate(profile)
            return float(
                not (elapsed < Fraction("0.45") or Fraction("1.05") <= elapsed <= Fraction("1.35"))
            )
        raise MontageRejection("comparison_mode", "unknown calibrated comparison edit")

    offset = frame - ending_start
    for shot in edit["ending_shots"]:
        frames = int(shot["frames"])
        if offset < frames:
            # Actual A/B cuts are instantaneous: the title must switch ON
            # on the SAME output frame as the AFTER picture, never three later.
            return 1.0 if shot["state"] == "after" else 0.0
        offset -= frames
    raise MontageRejection("text_timeline_overflow", "title frame lies outside legal edit")


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.FreeTypeFont:
    completed = subprocess.run(
        ["fc-match", "-f", "%{file}", "DejaVu Sans:style=Bold"],
        capture_output=True,
        text=True,
        check=False,
    )
    path = Path(completed.stdout.strip())
    if completed.returncode or not path.is_file():
        raise RuntimeError("Clipper portrait renderer requires a readable bold sans font")
    return ImageFont.truetype(str(path), size)


def _portrait_layout(text: str, profile: CampaignProfile) -> list[str]:
    cfg = profile.config["output"]["portrait_matte"]
    lines = cfg["title_layouts"].get(text)
    if not isinstance(lines, list) or len(lines) != 3:
        raise MontageRejection("portrait_copy", "approved copy has no three-line matte layout")
    if any(not isinstance(line, str) for line in lines):
        raise MontageRejection("portrait_copy", "portrait lines must be literal text")
    if sum(line.count("{Toggle}") for line in lines) != 1:
        raise MontageRejection("portrait_copy", "exactly one verified toggle token is required")
    if " ".join(line.replace("{Toggle}", "Toggle") for line in lines) != text:
        raise MontageRejection("portrait_copy", "reflowed copy differs from the approved line")
    return lines


def _frame_image(
    frame: int,
    plan: dict[str, Any],
    profile: CampaignProfile,
    width: int,
    top_height: int,
    font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
    progress: float,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    cfg = profile.config["output"]["portrait_matte"]
    lines = _portrait_layout(plan["montage"]["approved_on_screen_text"], profile)
    scale = width / 1080
    black_raw = ImageColor.getrgb(str(cfg["background_hex"]))
    black: RGB = (black_raw[0], black_raw[1], black_raw[2])
    white_raw = ImageColor.getrgb(str(cfg["text_hex"]))
    white: RGB = (white_raw[0], white_raw[1], white_raw[2])
    gold_raw = ImageColor.getrgb(str(cfg["accent_hex"]))
    gold: RGB = (gold_raw[0], gold_raw[1], gold_raw[2])
    im = Image.new("RGB", (width, top_height), black)
    draw = ImageDraw.Draw(im)
    y_positions = [round(y * scale) for y in (236, 352, 438)]
    word_box = (0, 0, 0, 0)
    for index, line in enumerate(lines):
        line_font = font if index == 0 else small_font
        if index == 2:
            line_font = _font(max(8, round(54 * scale)))
        if "{Toggle}" not in line:
            line_width = draw.textlength(line, font=line_font)
            if line_width > width * 0.90:
                raise MontageRejection("portrait_copy_overflow", "approved line exceeds matte")
            draw.text(
                ((width - line_width) / 2, y_positions[index]), line, font=line_font, fill=white
            )
            continue

        prefix, suffix = line.split("{Toggle}")
        word = "Toggle"
        left_width = draw.textlength(prefix, font=line_font)
        right_width = draw.textlength(suffix, font=line_font)
        word_width = draw.textlength(word, font=line_font)
        gap = max(2, round(19 * scale))
        padding = max(3, round(20 * scale))
        pill_w = math.ceil(word_width + padding * 2)
        pill_h = round(83 * scale) if index == 0 else round(70 * scale)
        full_width = (
            left_width + (gap if prefix else 0) + pill_w + (gap if suffix else 0) + right_width
        )
        if full_width > width * 0.93:
            raise MontageRejection("portrait_copy_overflow", "toggle line exceeds title matte")
        x = (width - full_width) / 2
        y = y_positions[index]
        if prefix:
            draw.text((x, y), prefix, font=line_font, fill=white)
            x += left_width + gap
        px, py = round(x), y
        off = Image.new("RGB", (pill_w, pill_h), black)
        off_draw = ImageDraw.Draw(off)
        off_draw.rounded_rectangle(
            (0, 0, pill_w - 1, pill_h - 1),
            radius=max(3, round(19 * scale)),
            fill=(40, 43, 47),
            outline=(94, 90, 64),
            width=max(1, round(2 * scale)),
        )
        word_x = (pill_w - word_width) / 2
        off_draw.text((word_x, max(0, round(2 * scale))), word, font=line_font, fill=white)
        on = Image.new("RGB", (pill_w, pill_h), black)
        on_draw = ImageDraw.Draw(on)
        on_draw.rounded_rectangle(
            (0, 0, pill_w - 1, pill_h - 1),
            radius=max(3, round(19 * scale)),
            fill=gold,
        )
        on_draw.text((word_x, max(0, round(2 * scale))), word, font=line_font, fill=(24, 23, 20))
        if progress > 0:
            mask = Image.new("L", (pill_w, pill_h), 0)
            ImageDraw.Draw(mask).rectangle(
                (0, 0, min(pill_w, round(progress * pill_w)), pill_h), fill=255
            )
            off.paste(on, (0, 0), mask)
        im.paste(off, (px, py))
        word_box = (px, py, pill_w, pill_h)
        if suffix:
            draw.text((px + pill_w + gap, y), suffix, font=line_font, fill=white)

    bar_width = max(12, round(172 * scale))
    bx = (width - bar_width) // 2
    by = round(548 * scale)
    if by + max(1, round(4 * scale)) >= top_height:
        raise MontageRejection("portrait_matte", "title bar intrudes into source footage")
    draw.rounded_rectangle(
        (bx, by, bx + bar_width, by + max(1, round(4 * scale))),
        radius=1,
        fill=(60, 56, 42),
    )
    if progress:
        draw.rounded_rectangle(
            (bx, by, bx + round(bar_width * progress), by + max(1, round(4 * scale))),
            radius=1,
            fill=gold,
        )
    return im, word_box


def render_title_frames(
    workspace: Path, plan: dict[str, Any], profile: CampaignProfile
) -> dict[str, Any]:
    cfg = profile.config["output"]["portrait_matte"]
    if cfg.get("text_full_duration") is not True:
        raise MontageRejection("portrait_copy", "approved title must remain for every frame")
    width = int(cfg["width"])
    height = int(cfg["height"])
    source_width = int(profile.config["output"]["width"])
    source_height = int(profile.config["output"]["height"])
    source_display_height = round(width * source_height / source_width)
    top_height = (height - source_display_height) // 2
    if width <= 0 or height <= 0 or top_height <= round(568 * width / 1080):
        raise MontageRejection("portrait_geometry", "no legal title-safe top matte")
    font = _font(max(8, round(64 * width / 1080)))
    small_font = _font(max(8, round(59 * width / 1080)))
    frames = int(plan["montage"]["output_frames"])
    folder = workspace / "approved_title_frames"
    folder.mkdir(parents=True, exist_ok=True)
    sample_indices = {0, 3, 9, 15, 73, 79, 193, 203, 223, 243, 253, 264, 274, 281, frames - 1}
    samples: dict[str, float] = {}
    word_box = (0, 0, 0, 0)
    for n in range(frames):
        progress = toggle_progress(n, plan, profile)
        frame_image, word_box = _frame_image(
            n, plan, profile, width, top_height, font, small_font, progress
        )
        frame_image.save(folder / f"{n:04d}.png", compress_level=3)
        if n in sample_indices:
            samples[str(n)] = round(progress, 6)
    if samples.get("0") != 0 or samples.get(str(frames - 1)) != 1:
        raise MontageRejection("toggle_sync", "opening/final toggle state is incorrect")
    return {
        "folder": str(folder),
        "frame_count": frames,
        "text_visible_frames": frames,
        "source_trigger_frames": [
            int(plan["evidence"]["toggle_motion_start_frame"]),
            int(plan["evidence"]["toggle_motion_end_frame"]),
        ],
        "progress_samples": samples,
        "word_box": list(word_box),
        "top_height": top_height,
        "center_height": source_display_height,
        "canvas": [width, height],
        "approved_copy": plan["montage"]["approved_on_screen_text"],
        "compositing": "approved text only in upper portrait matte",
    }


def render_portrait(
    clean_canonical: Path,
    output_dir: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    source_profile: dict[str, Any],
    metric: Callable[[Path, Path, str, str], float],
) -> dict[str, Any]:
    """Generate source-only-audio, exact-frame FFV1 portrait stage and MP4 delivery."""
    title_qa = render_title_frames(workspace, plan, profile)
    width, height = (int(z) for z in title_qa["canvas"])
    top = int(title_qa["top_height"])
    center = int(title_qa["center_height"])
    frames = int(plan["montage"]["output_frames"])
    fps = rate(profile)
    duration = float(Fraction(frames, 1) / fps)
    canonical = workspace / "portrait_canonical.nut"
    file = output_dir / f"{profile.name}_{plan['montage']['comparison_mode']}_portrait.mp4"
    graph = (
        f"[0:v]scale={width}:{center}:flags=lanczos,setsar=1,"
        f"pad={width}:{height}:0:{top}:color=black[base];"
        f"[base][1:v]overlay=0:0:shortest=1:format=auto,"
        "format=yuv420p[outv]"
    )
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-i",
            str(clean_canonical),
            "-framerate",
            f"{fps.numerator}/{fps.denominator}",
            "-i",
            str(workspace / "approved_title_frames" / "%04d.png"),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            graph,
            "-map",
            "[outv]",
            "-map",
            "0:a:0",
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-threads:v",
            "1",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(profile.config["output"]["audio_sample_rate"]),
            "-ac",
            str(profile.config["output"]["audio_channels"]),
            *media.profile_output_args(source_profile),
            *media.color_metadata_tag_args(source_profile),
            "-t",
            f"{duration:.9f}",
            "-f",
            "nut",
            str(canonical),
        ]
    )
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-i",
            str(canonical),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            str(profile.config["output"]["video_codec"]),
            "-preset",
            str(profile.config["output"]["video_preset"]),
            "-crf",
            str(profile.config["output"]["video_crf"]),
            "-threads:v",
            "2",
            "-r",
            f"{fps.numerator}/{fps.denominator}",
            "-c:a",
            str(profile.config["output"]["audio_codec"]),
            "-b:a",
            "320k",
            "-ar",
            str(profile.config["output"]["audio_sample_rate"]),
            "-ac",
            str(profile.config["output"]["audio_channels"]),
            *media.profile_output_args(source_profile),
            *media.color_metadata_tag_args(source_profile),
            "-video_track_timescale",
            str(media.timing_from_profile(source_profile).track_timescale),
            "-movflags",
            "+faststart",
            "-t",
            f"{duration:.9f}",
            str(file),
        ]
    )
    stage = media.video_profile(canonical, count_frames=True)
    actual = media.video_profile(file, count_frames=True)
    audio = media.audio_profile(file)
    probe = json.loads(
        media.run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(file),
            ]
        ).stdout
    )
    actual_duration = float(probe["format"]["duration"])
    expected_colors = media.source_color_metadata(source_profile)
    actual_colors = media.source_color_metadata(actual)
    ssim = metric(canonical, file, "ssim", r"All:([0-9.]+)")
    psnr = metric(canonical, file, "psnr", r"average:([0-9.]+)")
    checks = {
        "portrait_canonical_ffv1_nut": stage["codec_name"] == "ffv1" and ffv1._is_nut(canonical),
        "portrait_video_frame_count_exact": actual["frame_count"] == frames
        and stage["frame_count"] == frames,
        "portrait_encoded_duration": abs(actual_duration - duration) < 0.04
        and 10 <= actual_duration <= 12,
        "portrait_dimensions": (actual["width"], actual["height"]) == (width, height),
        "portrait_frame_rate": actual["avg_frame_rate"] == f"{fps.numerator}/{fps.denominator}",
        "approved_text_full_duration": title_qa["text_visible_frames"] == frames,
        "toggle_opening_off": title_qa["progress_samples"]["0"] == 0,
        "toggle_final_on": title_qa["progress_samples"][str(frames - 1)] == 1,
        "portrait_source_audio_only": audio["sample_rate"]
        == int(profile.config["output"]["audio_sample_rate"])
        and audio["channels"] == int(profile.config["output"]["audio_channels"]),
        "portrait_ssim": ssim >= 0.96,
        "portrait_psnr_db": psnr >= 35.0,
        "portrait_color_metadata": all(
            actual_colors.get(k) == v for k, v in expected_colors.items()
        ),
    }
    if set(checks) != PORTRAIT_REQUIRED_CHECKS or not all(checks.values()):
        raise RuntimeError(f"portrait QA failed: {checks}")
    hasher = hashlib.sha256()
    with file.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            hasher.update(block)
    return {
        "file": str(file.resolve()),
        "sha256": hasher.hexdigest(),
        "qa": {
            "checks": checks,
            "frame_count": frames,
            "encoded_duration": actual_duration,
            "video_profile": actual,
            "audio_profile": audio,
            "ssim": ssim,
            "psnr_db": psnr,
        },
        "title": title_qa,
        "canonical_ffv1_nut": True,
    }
