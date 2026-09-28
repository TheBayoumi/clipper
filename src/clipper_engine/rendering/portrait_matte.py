"""Clipper-owned portrait announcement matte with source-synchronized approved typography.

The gameplay profile supplies source transition calibration and approved exact copy;
the renderer owns layout, frame-by-frame state, fidelity checks, and delivery.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont, ImageStat

from .. import media_contract as media
from ..montage import MontageRejection, rate, source_frame_count, source_windows
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
        "portrait_mattes_match",
        "portrait_target_size",
        "portrait_color_metadata",
    }
)


SNAPBACK_REQUIRED_CHECKS = frozenset(
    {
        "portrait_snapback_schedule",
        "portrait_snapback_verified_stills",
        "portrait_snapback_text_sync",
        "portrait_snapback_encoded_rewind",
    }
)


REBOUND_REQUIRED_CHECKS = frozenset(
    {
        "portrait_rebound_schedule",
        "portrait_rebound_verified_stills",
        "portrait_rebound_text_sync",
        "portrait_rebound_encoded_phases",
        "portrait_rebound_live_detail_visible",
    }
)


CONTINUOUS_REVEAL_REQUIRED_CHECKS = frozenset(
    {
        "portrait_continuous_reveal_schedule",
        "portrait_continuous_reveal_source_sync",
        "portrait_continuous_reveal_legibility",
        "portrait_continuous_reveal_no_ai",
        "portrait_continuous_reveal_effects_declared",
        "portrait_continuous_reveal_backdrop_visible",
        "portrait_continuous_reveal_toggle_visible",
    }
)


HERO_FOCUS_REQUIRED_CHECKS = frozenset(
    {
        "portrait_hero_focus_schedule",
        "portrait_hero_focus_verified_stills",
        "portrait_hero_focus_text_sync",
        "portrait_hero_focus_encoded_phases",
    }
)


SPOTLIGHT_REQUIRED_CHECKS = frozenset(
    {
        "portrait_spotlight_schedule",
        "portrait_spotlight_verified_stills",
        "portrait_spotlight_text_sync",
        "portrait_spotlight_encoded_state_changes",
    }
)


CASCADE_REQUIRED_CHECKS = frozenset(
    {
        "portrait_cascade_three_panels",
        "portrait_cascade_verified_stills",
        "portrait_cascade_text_sync",
        "portrait_cascade_panels_visible",
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
    if edit.get("type") == "continuous_source_reveal":
        if not 0 <= frame < int(edit["output_frames"]):
            raise MontageRejection("source_timeline", "continuous reveal frame is outside output")
        source_frame = int(edit["source_window"]["start_frame"]) + frame
        return _ease((source_frame - source_event_start) / (source_event_end - source_event_start))
    if not source_event_start < source_event_end:
        raise MontageRejection("toggle_event_invalid", "source visual-event interval is invalid")
    hook_frames = int(edit["hook"]["frames"])
    source_frames = source_frame_count(edit)
    comparison_frames = int(edit["comparison_frames"])
    comparison_start = hook_frames + source_frames
    ending_start = comparison_start + comparison_frames

    if frame < hook_frames:
        source_frame = int(edit["hook"]["start_frame"]) + frame
        return _ease((source_frame - source_event_start) / (source_event_end - source_event_start))
    if frame < comparison_start:
        offset = frame - hook_frames
        for window in source_windows(edit):
            length = int(window["frames"])
            if offset < length:
                source_frame = int(window["start_frame"]) + offset
                return _ease(
                    (source_frame - source_event_start) / (source_event_end - source_event_start)
                )
            offset -= length
        raise MontageRejection("source_timeline", "source excerpt overflow")
    if frame < ending_start:
        local = frame - comparison_start
        if edit["comparison_mode"] == "wipe":
            start = comparison_frames // 6
            duration = 2 * comparison_frames // 3
            return _ease((local - start) / duration)
        if edit["comparison_mode"] == "cascade":
            from . import panel_compositor

            return sum(panel_compositor.stage_progress(local, plan, profile)) / 3
        if edit["comparison_mode"] == "spotlight":
            from . import panel_compositor

            return panel_compositor.spotlight_stage(local, plan, profile)[1]
        if edit["comparison_mode"] == "hero_focus":
            from . import panel_compositor

            return panel_compositor.hero_focus_progress(local, plan, profile)
        if edit["comparison_mode"] == "rebound":
            from . import panel_compositor

            return panel_compositor.rebound_progress(local, plan, profile)
        if edit["comparison_mode"] == "snapback":
            from . import panel_compositor

            return panel_compositor.snapback_progress(local, plan, profile)
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
    fonts: list[ImageFont.FreeTypeFont],
    y_positions: list[int],
    pill_heights: list[int],
    bar_y: int,
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
    word_box = (0, 0, 0, 0)
    for index, line in enumerate(lines):
        line_font = fonts[index]
        if "{Toggle}" not in line:
            line_width = draw.textlength(line, font=line_font)
            if line_width > width * 0.90:
                raise MontageRejection("portrait_copy_overflow", "approved line exceeds matte")
            draw.text(
                ((width - line_width) / 2, y_positions[index]),
                line,
                font=line_font,
                fill=white,
            )
            continue
        prefix, suffix = line.split("{Toggle}")
        word = "Toggle"
        left_width = draw.textlength(prefix, font=line_font)
        right_width = draw.textlength(suffix, font=line_font)
        word_width = draw.textlength(word, font=line_font)
        gap = max(2, round(14 * scale))
        padding = max(3, round(16 * scale))
        pill_w = math.ceil(word_width + padding * 2)
        pill_h = int(pill_heights[index])
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
            radius=max(3, round(15 * scale)),
            fill=(40, 43, 47),
            outline=(94, 90, 64),
            width=max(1, round(2 * scale)),
        )
        word_x = (pill_w - word_width) / 2
        off_draw.text((word_x, max(0, round(1 * scale))), word, font=line_font, fill=white)
        on = Image.new("RGB", (pill_w, pill_h), black)
        on_draw = ImageDraw.Draw(on)
        on_draw.rounded_rectangle(
            (0, 0, pill_w - 1, pill_h - 1),
            radius=max(3, round(15 * scale)),
            fill=gold,
        )
        on_draw.text((word_x, max(0, round(1 * scale))), word, font=line_font, fill=(24, 23, 20))
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
    if bar_y + max(1, round(4 * scale)) >= top_height:
        raise MontageRejection("portrait_matte", "title bar intrudes into source footage")
    draw.rounded_rectangle(
        (bx, bar_y, bx + bar_width, bar_y + max(1, round(4 * scale))),
        radius=1,
        fill=(60, 56, 42),
    )
    if progress:
        draw.rounded_rectangle(
            (bx, bar_y, bx + round(bar_width * progress), bar_y + max(1, round(4 * scale))),
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
    frames = int(plan["montage"]["output_frames"])
    continuous = plan["montage"].get("type") == "continuous_source_reveal"
    if continuous:
        from . import kinetic_reframe

        style = kinetic_reframe.config(profile, frames)
        top_height = int(style["title_top_height"])
        source_display_height = int(style["visual_height"])
        fonts = [_font(max(8, round(int(n) * width / 1080))) for n in style["title_font_sizes"]]
        y_positions = [round(int(n) * width / 1080) for n in style["title_y_positions"]]
        pill_heights = [round(int(n) * width / 1080) for n in style["title_pill_heights"]]
        bar_y = round(int(style["title_bar_y"]) * width / 1080)
        if top_height + source_display_height >= height:
            raise MontageRejection("portrait_geometry", "continuous portrait geometry overflows")
    else:
        source_width = int(profile.config["output"]["width"])
        source_height = int(profile.config["output"]["height"])
        source_display_height = round(width * source_height / source_width)
        top_height = (height - source_display_height) // 2
        if width <= 0 or height <= 0 or top_height <= round(568 * width / 1080):
            raise MontageRejection("portrait_geometry", "no legal title-safe top matte")
        fonts = [
            _font(max(8, round(64 * width / 1080))),
            _font(max(8, round(59 * width / 1080))),
            _font(max(8, round(54 * width / 1080))),
        ]
        y_positions = [round(y * width / 1080) for y in (236, 352, 438)]
        pill_heights = [
            round(83 * width / 1080),
            round(70 * width / 1080),
            round(70 * width / 1080),
        ]
        bar_y = round(548 * width / 1080)

    folder = workspace / "approved_title_frames"
    folder.mkdir(parents=True, exist_ok=True)
    sample_indices = {0, 3, 9, 15, 73, 79, 193, 203, 223, 243, 253, 264, 274, 281, frames - 1}
    mode = plan["montage"]["comparison_mode"]
    if mode == "spotlight":
        from . import panel_compositor

        mode_cfg = panel_compositor.spotlight_config(
            profile, int(plan["montage"]["comparison_frames"])
        )
        start = int(plan["montage"]["hook"]["frames"]) + int(
            plan["montage"]["source_window"]["frames"]
        )
        for i in range(3):
            change = (
                start + i * int(mode_cfg["focus_frames"]) + int(mode_cfg["switch_after_frames"])
            )
            sample_indices.update({change - 1, change})
        sample_indices.add(start + 3 * int(mode_cfg["focus_frames"]))
    if mode == "snapback":
        from . import panel_compositor

        mode_cfg = panel_compositor.snapback_config(
            profile, int(plan["montage"]["comparison_frames"])
        )
        start = int(plan["montage"]["hook"]["frames"]) + int(
            plan["montage"]["source_window"]["frames"]
        )
        rewind = start + int(mode_cfg["after_preview_frames"])
        reveal = rewind + int(mode_cfg["before_hold_frames"])
        switched = reveal + int(mode_cfg["transition_frames"]) - 1
        sample_indices.update({start, rewind - 1, rewind, reveal - 1, reveal, switched})
    if mode == "hero_focus":
        from . import panel_compositor

        mode_cfg = panel_compositor.hero_focus_config(
            profile, int(plan["montage"]["comparison_frames"])
        )
        start = int(plan["montage"]["hook"]["frames"]) + int(
            plan["montage"]["source_window"]["frames"]
        )
        original = start + int(mode_cfg["after_preview_frames"])
        split = original + int(mode_cfg["before_hold_frames"])
        group = split + int(mode_cfg["split_frames"])
        for edge in (start, original, split, group):
            sample_indices.update({edge - 1, edge})
    if mode == "rebound":
        from . import panel_compositor

        mode_cfg = panel_compositor.rebound_config(
            profile, int(plan["montage"]["comparison_frames"])
        )
        start = int(plan["montage"]["hook"]["frames"]) + source_frame_count(plan["montage"])
        original = start + int(mode_cfg["before_hold_frames"])
        flash = original + int(mode_cfg["after_flash_frames"])
        group = flash + int(mode_cfg["split_frames"])
        sample_indices.update({0, int(plan["montage"]["hook"]["frames"]), 35, 54, 145})
        for edge in (start, original, flash, group):
            sample_indices.update({edge - 1, edge})
    if continuous:
        from . import kinetic_reframe

        mode_cfg = kinetic_reframe.config(profile, frames)
        sample_indices.update(int(n) for n in mode_cfg["storyboard_frames"])
        local_start = int(plan["evidence"]["toggle_motion_start_frame"]) - int(
            plan["montage"]["source_window"]["start_frame"]
        )
        local_end = int(plan["evidence"]["toggle_motion_end_frame"]) - int(
            plan["montage"]["source_window"]["start_frame"]
        )
        sample_indices.update({local_start, local_end})

    sample_indices = {n for n in sample_indices if 0 <= n < frames}
    samples: dict[str, float] = {}
    word_box = (0, 0, 0, 0)
    for n in range(frames):
        progress = toggle_progress(n, plan, profile)
        frame_image, word_box = _frame_image(
            n,
            plan,
            profile,
            width,
            top_height,
            fonts,
            y_positions,
            pill_heights,
            bar_y,
            progress,
        )
        frame_image.save(folder / f"{n:04d}.png", compress_level=3)
        if n in sample_indices:
            samples[str(n)] = round(progress, 6)
    opening = 1 if mode == "rebound" else 0
    if samples.get("0") != opening or samples.get(str(frames - 1)) != 1:
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
        "compositing": (
            "compact approved text above enlarged continuous source footage"
            if continuous
            else "approved text only in upper portrait matte"
        ),
    }


def _encoded_montage_storyboard(
    file: Path,
    output_dir: Path,
    indices: list[int],
    captions: tuple[str, ...],
    filename: str,
    fps: float,
) -> dict[str, Any]:
    """Qualifying contact sheet from the finished encoded portrait, not a mockup."""
    if indices != sorted(set(indices)) or len(indices) != 6 or len(captions) != 6:
        raise RuntimeError("actual-render storyboard requires six ordered frames")
    video = media.video_profile(file, count_frames=True)
    width, height = int(video["width"]), int(video["height"])
    filter_select = "+".join(f"eq(n\\,{n})" for n in indices)
    data = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(file),
            "-vf",
            f"select={filter_select},format=rgb24",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(len(indices)),
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    bytes_per = width * height * 3
    if len(data) != bytes_per * len(indices):
        raise RuntimeError("encoded portrait storyboard has missing actual frames")
    tile_w = 320
    tile_h = round(tile_w * height / width)
    label_h = 37
    canvas = Image.new("RGB", (3 * tile_w, 2 * (tile_h + label_h)), (15, 17, 21))
    font = _font(17)
    drawer = ImageDraw.Draw(canvas)
    for i, (n, name) in enumerate(zip(indices, captions, strict=True)):
        frame = Image.frombytes("RGB", (width, height), data[i * bytes_per : (i + 1) * bytes_per])
        thumb = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (i % 3) * tile_w
        y = (i // 3) * (tile_h + label_h)
        canvas.paste(thumb, (x, y))
        drawer.text(
            (x + 10, y + tile_h + 5),
            f"{name} - {n / fps:.2f}s",
            font=font,
            fill=(248, 218, 92),
        )
    path = output_dir / filename
    canvas.save(path, quality=88, optimize=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": str(path.resolve()),
        "sha256": digest,
        "frames": indices,
        "source": "actual_encoded_delivery",
    }


def _encoded_spotlight_storyboard(
    file: Path, output_dir: Path, plan: dict[str, Any], panel_qa: dict[str, Any]
) -> dict[str, Any]:
    indices = [
        0,
        *[int(v) for v in panel_qa["switch_frames"]],
        int(panel_qa["group_start_frame"]) + 2,
        int(plan["montage"]["output_frames"]) - 1,
    ]
    return _encoded_montage_storyboard(
        file,
        output_dir,
        indices,
        (
            "REAL SOURCE HOOK",
            "OPERATOR 1",
            "OPERATOR 2",
            "OPERATOR 3",
            "GROUP PAYOFF",
            "FINAL FRAME",
        ),
        "operator_spotlight_storyboard.jpg",
        float(Fraction(str(plan["source"]["fps"]))),
    )


def _encoded_snapback_storyboard(
    file: Path, output_dir: Path, plan: dict[str, Any], panel_qa: dict[str, Any]
) -> dict[str, Any]:
    rewind, reveal, switched = (int(v) for v in panel_qa["switch_frames"])
    return _encoded_montage_storyboard(
        file,
        output_dir,
        [0, rewind - 1, rewind, reveal - 1, switched, int(plan["montage"]["output_frames"]) - 1],
        (
            "SOURCE HOOK",
            "RESULT FIRST",
            "SNAP BACK",
            "ORIGINAL LOOK",
            "TOGGLE REVEAL",
            "FINAL LOOK",
        ),
        "operator_snapback_storyboard.jpg",
        float(Fraction(str(plan["source"]["fps"]))),
    )


def _encoded_hero_focus_storyboard(
    file: Path, output_dir: Path, plan: dict[str, Any], panel_qa: dict[str, Any]
) -> dict[str, Any]:
    _, original, split, group, _ = (int(n) for n in panel_qa["phase_frames"])
    return _encoded_montage_storyboard(
        file,
        output_dir,
        [0, original - 1, original, split, group, int(plan["montage"]["output_frames"]) - 1],
        (
            "VERIFIED SOURCE HOOK",
            "TRANSFORMED TEASE",
            "ORIGINAL LOOK",
            "CENTER A/B",
            "THREE OPERATORS",
            "FINAL PAYOFF",
        ),
        "operator_hero_focus_storyboard.jpg",
        float(Fraction(str(plan["source"]["fps"]))),
    )


def _encoded_rebound_storyboard(
    file: Path, output_dir: Path, plan: dict[str, Any], panel_qa: dict[str, Any]
) -> dict[str, Any]:
    _, after, split, _, _ = (int(n) for n in panel_qa["phase_frames"])
    hook_end = int(plan["montage"]["hook"]["frames"])
    toggle_mid = hook_end + int(plan["montage"]["source_windows"][0]["frames"]) + 4
    return _encoded_montage_storyboard(
        file,
        output_dir,
        [0, hook_end, toggle_mid, after, split, int(plan["montage"]["output_frames"]) - 1],
        (
            "RESULT FIRST",
            "ORIGINAL LOOK",
            "REAL SOURCE TOGGLE",
            "ENLARGED TRANSFORMED LOOK",
            "LARGE BEFORE / AFTER",
            "FINAL THREE-OPERATOR PAYOFF",
        ),
        "operator_rebound_storyboard.jpg",
        float(Fraction(str(plan["source"]["fps"]))),
    )


def _encoded_continuous_reveal_storyboard(
    file: Path, output_dir: Path, plan: dict[str, Any], profile: CampaignProfile
) -> dict[str, Any]:
    from . import kinetic_reframe

    cfg = kinetic_reframe.config(profile, int(plan["montage"]["output_frames"]))
    return _encoded_montage_storyboard(
        file,
        output_dir,
        [int(n) for n in cfg["storyboard_frames"]],
        (
            "ORIGINAL TRIO",
            "GENTLE PUSH",
            "REAL TOGGLE START",
            "REAL TOGGLE COMPLETE",
            "TRANSFORMED INSPECTION",
            "CLEAN TRANSFORMED FINISH",
        ),
        "operator_continuous_reveal_storyboard.jpg",
        float(Fraction(str(plan["source"]["fps"]))),
    )


def render_portrait(
    clean_canonical: Path,
    output_dir: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    source_profile: dict[str, Any],
    metric: Callable[[Path, Path, str, str], float],
    comparison_stills: tuple[Path, Path] | None = None,
    expected_still_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate source-only-audio, exact-frame FFV1 portrait stage and MP4 delivery."""
    title_qa = render_title_frames(workspace, plan, profile)
    mode = str(plan["montage"]["comparison_mode"])
    cascade = mode == "cascade"
    spotlight = mode == "spotlight"
    snapback = mode == "snapback"
    hero_focus = mode == "hero_focus"
    rebound = mode == "rebound"
    continuous_reveal = mode == "continuous_reveal"
    panel_mode = cascade or spotlight or snapback or hero_focus or rebound
    width, height = (int(z) for z in title_qa["canvas"])
    top = int(title_qa["top_height"])
    center = int(title_qa["center_height"])
    frames = int(plan["montage"]["output_frames"])
    fps = rate(profile)
    duration = float(Fraction(frames, 1) / fps)
    canonical = workspace / "portrait_canonical.nut"
    file = output_dir / f"{profile.name}_{plan['montage']['comparison_mode']}_portrait.mp4"
    background = str(profile.config["output"]["portrait_matte"]["background_hex"])
    panel_qa: dict[str, Any] | None = None
    reframe_qa: dict[str, Any] | None = None
    if continuous_reveal:
        from . import kinetic_reframe

        reveal_cfg = kinetic_reframe.config(profile, frames)
        visual_top = int(reveal_cfg["visual_top"])
        visual_height = int(reveal_cfg["visual_height"])
        if visual_height != center:
            raise MontageRejection(
                "continuous_reveal_layout", "title and portrait reframe geometry disagree"
            )
        reframe_qa = kinetic_reframe.render_frames(
            clean_canonical,
            workspace,
            plan,
            profile,
            width,
            visual_height,
        )
        blur_sigma = float(reveal_cfg["backdrop_blur_sigma"])
        backdrop_brightness = float(reveal_cfg["backdrop_brightness"])
        backdrop_saturation = float(reveal_cfg["backdrop_saturation"])
        graph = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},gblur=sigma={blur_sigma}:steps=2,"
            f"eq=brightness={backdrop_brightness}:saturation={backdrop_saturation}[backdrop];"
            f"[backdrop][2:v]overlay=0:{visual_top}:shortest=1:format=auto[with_video];"
            "[with_video][1:v]overlay=0:0:shortest=1:format=auto,"
            "format=yuv420p[outv]"
        )
    elif panel_mode:
        from . import panel_compositor

        if comparison_stills is None or expected_still_hashes is None:
            raise MontageRejection("panel_stills", "verified source comparison stills required")
        full_progress = [toggle_progress(n, plan, profile) for n in range(frames)]
        panel_qa = panel_compositor.render_panels(
            comparison_stills[0],
            comparison_stills[1],
            workspace,
            plan,
            profile,
            width,
            height - top - center,
            full_progress,
            live_source=clean_canonical if rebound else None,
        )
        if panel_qa["source_still_sha256"] != expected_still_hashes:
            raise MontageRejection("panel_stills", "panels do not match certified comparison")
        graph = (
            f"[0:v]scale={width}:{center}:flags=lanczos,setsar=1,"
            f"pad={width}:{height}:0:{top}:color=0x{background.lstrip('#')}[base];"
            "[base][1:v]overlay=0:0:shortest=1:format=auto[with_title];"
            f"[with_title][2:v]overlay=0:{top + center}:shortest=1:format=auto,"
            "format=yuv420p[outv]"
        )
    else:
        graph = (
            f"[0:v]scale={width}:{center}:flags=lanczos,setsar=1,"
            f"pad={width}:{height}:0:{top}:color=0x{background.lstrip('#')}[base];"
            "[base][1:v]overlay=0:0:shortest=1:format=auto,"
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
            *(
                [
                    "-framerate",
                    f"{fps.numerator}/{fps.denominator}",
                    "-i",
                    str(workspace / "panel_frames" / "%04d.png"),
                ]
                if panel_mode
                else [
                    "-framerate",
                    f"{fps.numerator}/{fps.denominator}",
                    "-i",
                    str(workspace / "reframe_frames" / "%04d.png"),
                ]
                if continuous_reveal
                else []
            ),
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
    matte = profile.config["output"]["portrait_matte"]
    size_target_mb = float(matte["target_size_mb"])
    size_tolerance_mb = float(matte["size_tolerance_mb"])
    audio_kbps = int(matte["audio_kbps"])
    if not (size_target_mb > size_tolerance_mb > 0 and audio_kbps > 0):
        raise MontageRejection("invalid_portrait_target", "size target/tolerance/audio invalid")
    # Decimal MB, with a small MP4 muxing allowance. This is real
    # two-pass encoding, not video padding or artificial filler bytes.
    video_kbps = round((size_target_mb * 8000 / duration - audio_kbps) * 0.995)
    if video_kbps < 150:
        raise MontageRejection("invalid_portrait_target", "bit budget insufficient for video")
    passlog = str(workspace / "portrait_twopass")
    common_video = [
        "-map",
        "0:v:0",
        "-c:v",
        str(profile.config["output"]["video_codec"]),
        "-preset",
        str(profile.config["output"]["video_preset"]),
        "-b:v",
        f"{video_kbps}k",
        "-threads:v",
        "2",
        "-r",
        f"{fps.numerator}/{fps.denominator}",
        *media.profile_output_args(source_profile),
    ]
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
            *common_video,
            "-pass",
            "1",
            "-passlogfile",
            passlog,
            "-an",
            "-f",
            "null",
            os.devnull,
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
            *common_video,
            "-pass",
            "2",
            "-passlogfile",
            passlog,
            "-map",
            "0:a:0",
            "-c:a",
            str(profile.config["output"]["audio_codec"]),
            "-b:a",
            f"{audio_kbps}k",
            "-ar",
            str(profile.config["output"]["audio_sample_rate"]),
            "-ac",
            str(profile.config["output"]["audio_channels"]),
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
    filesize_mb = file.stat().st_size / 1_000_000
    # Validate ENCODED top and bottom mattes at opening, comparison and ending.
    sampled_frames = [0, frames // 2, frames - 1]
    selected_frames = "+".join(f"eq(n\\,{n})" for n in sampled_frames)
    still = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(file),
            "-vf",
            f"select={selected_frames},format=rgb24",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(len(sampled_frames)),
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    frame_bytes = width * height * 3
    if len(still.stdout) != frame_bytes * len(sampled_frames):
        raise RuntimeError("decoded portrait sample count differs from qualified delivery")
    sampled_mattes: list[dict[str, Any]] = []
    backdrop_differences: list[float] = []
    matte_rgb_error = 0
    header_rgb_error = 0
    expected_matte = ImageColor.getrgb(background)
    for index, n in enumerate(sampled_frames):
        rgb = Image.frombytes(
            "RGB",
            (width, height),
            still.stdout[index * frame_bytes : (index + 1) * frame_bytes],
        )
        top_rgb = rgb.getpixel((width // 18, max(2, top // 8)))
        bottom_rgb = rgb.getpixel((width // 18, height - max(2, top // 8)))
        if not isinstance(top_rgb, tuple) or not isinstance(bottom_rgb, tuple):
            raise RuntimeError("decoded RGB matte verification failed")
        delta = max(abs(int(a) - int(b)) for a, b in zip(top_rgb, bottom_rgb, strict=True))
        matte_rgb_error = max(matte_rgb_error, delta)
        top_delta = max(
            abs(int(a) - int(b)) for a, b in zip(top_rgb, expected_matte, strict=True)
        )
        header_rgb_error = max(header_rgb_error, top_delta)
        if continuous_reveal:
            visual_bottom = int(reveal_cfg["visual_top"]) + int(reveal_cfg["visual_height"])
            backdrop_region = rgb.crop((0, visual_bottom, width, height))
            backdrop_mean = ImageStat.Stat(backdrop_region).mean
            backdrop_differences.append(
                round(
                    sum(
                        abs(float(value) - float(expected))
                        for value, expected in zip(backdrop_mean, expected_matte, strict=True)
                    )
                    / 3,
                    4,
                )
            )
        sampled_mattes.append(
            {
                "frame": n,
                "top_rgb": list(top_rgb),
                "bottom_rgb": list(bottom_rgb),
                "max_error": delta,
            }
        )
    top_rgb = tuple(sampled_mattes[0]["top_rgb"])
    bottom_rgb = tuple(sampled_mattes[0]["bottom_rgb"])
    expected_colors = media.source_color_metadata(source_profile)
    actual_colors = media.source_color_metadata(actual)
    ssim = metric(canonical, file, "ssim", r"All:([0-9.]+)")
    psnr = metric(canonical, file, "psnr", r"average:([0-9.]+)")
    mode = str(plan["montage"]["comparison_mode"])
    mode_calibration = profile.config["editorial"].get("mode_timing", {}).get(mode, {})
    max_seconds = float(
        mode_calibration.get(
            "maximum_output_seconds", profile.config["editorial"]["maximum_output_seconds"]
        )
    )
    checks = {
        "portrait_canonical_ffv1_nut": stage["codec_name"] == "ffv1" and ffv1._is_nut(canonical),
        "portrait_video_frame_count_exact": actual["frame_count"] == frames
        and stage["frame_count"] == frames,
        "portrait_encoded_duration": abs(actual_duration - duration) < 0.04
        and float(profile.config["editorial"]["minimum_output_seconds"])
        <= actual_duration
        <= max_seconds,
        "portrait_dimensions": (actual["width"], actual["height"]) == (width, height),
        "portrait_frame_rate": actual["avg_frame_rate"] == f"{fps.numerator}/{fps.denominator}",
        "approved_text_full_duration": title_qa["text_visible_frames"] == frames,
        **(
            {"toggle_opening_after": title_qa["progress_samples"]["0"] == 1}
            if rebound
            else {"toggle_opening_off": title_qa["progress_samples"]["0"] == 0}
        ),
        "toggle_final_on": title_qa["progress_samples"][str(frames - 1)] == 1,
        "portrait_source_audio_only": audio["sample_rate"]
        == int(profile.config["output"]["audio_sample_rate"])
        and audio["channels"] == int(profile.config["output"]["audio_channels"]),
        "portrait_ssim": ssim >= 0.96,
        "portrait_psnr_db": psnr >= 35.0,
        "portrait_mattes_match": (
            header_rgb_error <= 2 if continuous_reveal else matte_rgb_error <= 2
        ),
        "portrait_target_size": abs(filesize_mb - size_target_mb) <= size_tolerance_mb,
        "portrait_color_metadata": all(
            actual_colors.get(k) == v for k, v in expected_colors.items()
        ),
    }
    panel_differences: list[float] = []
    if cascade:
        if panel_qa is None:
            raise RuntimeError("missing cascade panel metadata")
        start = int(plan["montage"]["hook"]["frames"]) + int(
            plan["montage"]["source_window"]["frames"]
        )
        after_frame = start + int(plan["montage"]["comparison_frames"]) - 1
        select = f"select=eq(n\\,{start})+eq(n\\,{after_frame}),format=rgb24"
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(file),
                "-vf",
                select,
                "-fps_mode",
                "passthrough",
                "-frames:v",
                "2",
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        frame_bytes = width * height * 3
        if len(decoded) != 2 * frame_bytes:
            raise RuntimeError("encoded cascade panel frames missing")
        before_rgb = Image.frombytes("RGB", (width, height), decoded[:frame_bytes])
        after_rgb = Image.frombytes("RGB", (width, height), decoded[frame_bytes:])
        for x0, y0, x1, y1 in panel_qa["panel_boxes"]:
            box = (
                int(x0) + 2,
                top + center + int(y0) + 2,
                int(x1) - 2,
                top + center + int(y1) - 2,
            )
            difference = ImageChops.difference(before_rgb.crop(box), after_rgb.crop(box))
            panel_differences.append(round(sum(ImageStat.Stat(difference).mean) / 3, 4))

    spotlight_differences: list[float] = []
    if spotlight:
        if panel_qa is None:
            raise RuntimeError("missing spotlight panel metadata")
        switch_frames = [int(v) for v in panel_qa["switch_frames"]]
        samples = sorted({n for switch in switch_frames for n in (switch - 1, switch)})
        selected = "+".join(f"eq(n\\,{n})" for n in samples)
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(file),
                "-vf",
                f"select={selected},format=rgb24",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(samples)),
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        bytes_per_frame = width * height * 3
        if len(raw) != len(samples) * bytes_per_frame:
            raise RuntimeError("encoded spotlight frame-pair proof is incomplete")
        spotlight_decoded = {
            n: Image.frombytes(
                "RGB",
                (width, height),
                raw[i * bytes_per_frame : (i + 1) * bytes_per_frame],
            )
            for i, n in enumerate(samples)
        }
        x0, y0, x1, y1 = (int(v) for v in panel_qa["focus_box"])
        focus_box = (x0 + 3, top + center + y0 + 3, x1 - 3, top + center + y1 - 3)
        for n in switch_frames:
            pixel_change = ImageChops.difference(
                spotlight_decoded[n - 1].crop(focus_box), spotlight_decoded[n].crop(focus_box)
            )
            spotlight_differences.append(round(sum(ImageStat.Stat(pixel_change).mean) / 3, 4))

    snapback_differences: list[float] = []
    if snapback:
        if panel_qa is None:
            raise RuntimeError("missing snapback panel QA")
        rewind, reveal, switched = (int(v) for v in panel_qa["switch_frames"])
        sample_indices = sorted({rewind - 1, rewind, reveal - 1, switched})
        selected = "+".join(f"eq(n\\,{n})" for n in sample_indices)
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(file),
                "-vf",
                f"select={selected},format=rgb24",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(sample_indices)),
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        bytes_per_frame = width * height * 3
        if len(raw) != len(sample_indices) * bytes_per_frame:
            raise RuntimeError("encoded snapback rewind/reveal sample frames are missing")
        sampled_video = {
            n: Image.frombytes(
                "RGB",
                (width, height),
                raw[i * bytes_per_frame : (i + 1) * bytes_per_frame],
            )
            for i, n in enumerate(sample_indices)
        }
        roi = profile.config["editorial"]["visual_state_roi"]
        box = (
            round(float(roi[0]) * width),
            top + round(float(roi[1]) * center),
            round(float(roi[2]) * width),
            top + round(float(roi[3]) * center),
        )
        for left, right in ((rewind - 1, rewind), (reveal - 1, switched)):
            snapback_image_delta = ImageChops.difference(
                sampled_video[left].crop(box), sampled_video[right].crop(box)
            )
            snapback_differences.append(
                round(sum(ImageStat.Stat(snapback_image_delta).mean) / 3, 4)
            )

    if cascade:
        if panel_qa is None or expected_still_hashes is None:
            raise RuntimeError("missing cascade QA")
        checks["portrait_cascade_three_panels"] = (
            panel_qa["panel_count"] == 3 and panel_qa["frame_count"] == frames
        )
        checks["portrait_cascade_verified_stills"] = (
            panel_qa["source_still_sha256"] == expected_still_hashes
            and panel_qa["source_only"] is True
        )
        checks["portrait_cascade_text_sync"] = panel_qa["text_synced"] is True
        checks["portrait_cascade_panels_visible"] = len(panel_differences) == 3 and all(
            delta > 2.5 for delta in panel_differences
        )
    if spotlight:
        if panel_qa is None or expected_still_hashes is None:
            raise RuntimeError("missing spotlight QA")
        from . import panel_compositor

        cfg = panel_compositor.spotlight_config(profile, int(plan["montage"]["comparison_frames"]))
        compare_start = int(plan["montage"]["hook"]["frames"]) + int(
            plan["montage"]["source_window"]["frames"]
        )
        expected_switches = [
            compare_start + i * int(cfg["focus_frames"]) + int(cfg["switch_after_frames"])
            for i in range(3)
        ]
        checks["portrait_spotlight_schedule"] = (
            panel_qa["focus_order"] == cfg["operator_order"]
            and panel_qa["switch_frames"] == expected_switches
            and panel_qa["group_start_frame"] == compare_start + 3 * int(cfg["focus_frames"])
            and panel_qa["frame_count"] == frames
        )
        checks["portrait_spotlight_verified_stills"] = (
            panel_qa["source_still_sha256"] == expected_still_hashes
            and panel_qa["source_only"] is True
        )
        checks["portrait_spotlight_text_sync"] = panel_qa["text_synced"] is True
        checks["portrait_spotlight_encoded_state_changes"] = len(
            spotlight_differences
        ) == 3 and all(change > 2.5 for change in spotlight_differences)
    if snapback:
        if panel_qa is None or expected_still_hashes is None:
            raise RuntimeError("missing snapback source QA")
        from . import panel_compositor

        cfg = panel_compositor.snapback_config(profile, int(plan["montage"]["comparison_frames"]))
        start = int(plan["montage"]["hook"]["frames"]) + int(
            plan["montage"]["source_window"]["frames"]
        )
        rewind = start + int(cfg["after_preview_frames"])
        reveal = rewind + int(cfg["before_hold_frames"])
        switched = reveal + int(cfg["transition_frames"]) - 1
        checks["portrait_snapback_schedule"] = (
            panel_qa["panel_count"] == 2
            and panel_qa["focus_operator_index"] == cfg["focus_operator_index"]
            and panel_qa["focus_roi"]
            == profile.config["output"]["portrait_matte"]["operator_rois"][
                cfg["focus_operator_index"]
            ]
            and panel_qa["frame_count"] == frames
            and panel_qa["comparison_start_frame"] == start
            and panel_qa["switch_frames"] == [rewind, reveal, switched]
            and panel_qa["sampled_states"][str(rewind - 1)] == 1.0
            and panel_qa["sampled_states"][str(rewind)] == 0.0
            and panel_qa["sampled_states"][str(reveal - 1)] == 0.0
            and panel_qa["sampled_states"][str(switched)] == 1.0
        )
        checks["portrait_snapback_verified_stills"] = (
            panel_qa["source_still_sha256"] == expected_still_hashes
            and panel_qa["source_only"] is True
        )
        checks["portrait_snapback_text_sync"] = panel_qa["text_synced"] is True
        checks["portrait_snapback_encoded_rewind"] = len(snapback_differences) == 2 and all(
            change > 2.5 for change in snapback_differences
        )
    hero_focus_differences: list[float] = []
    if hero_focus:
        if panel_qa is None:
            raise RuntimeError("missing hero-focus panel metadata")
        _, original, split, group, _ = (int(n) for n in panel_qa["phase_frames"])
        pairs = ((original - 1, original), (split - 1, split), (group - 1, group))
        sample_indices = sorted({frame for pair in pairs for frame in pair})
        selected = "+".join(f"eq(n\\,{frame})" for frame in sample_indices)
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(file),
                "-vf",
                f"select={selected},format=rgb24",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(sample_indices)),
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        bytes_per_frame = width * height * 3
        if len(raw) != len(sample_indices) * bytes_per_frame:
            raise RuntimeError("encoded hero-focus phase proof is incomplete")
        decoded_hero: dict[int, Image.Image] = {
            n: Image.frombytes(
                "RGB",
                (width, height),
                raw[i * bytes_per_frame : (i + 1) * bytes_per_frame],
            )
            for i, n in enumerate(sample_indices)
        }
        roi = profile.config["editorial"]["visual_state_roi"]
        visual_box = (
            round(float(roi[0]) * width),
            top + round(float(roi[1]) * center),
            round(float(roi[2]) * width),
            top + round(float(roi[3]) * center),
        )
        for left, right in pairs:
            change = ImageChops.difference(
                decoded_hero[left].crop(visual_box), decoded_hero[right].crop(visual_box)
            )
            hero_focus_differences.append(round(sum(ImageStat.Stat(change).mean) / 3, 4))

    if hero_focus:
        if panel_qa is None or expected_still_hashes is None:
            raise RuntimeError("missing hero-focus source QA")
        from . import panel_compositor

        cfg = panel_compositor.hero_focus_config(profile, int(plan["montage"]["comparison_frames"]))
        start = int(plan["montage"]["hook"]["frames"]) + int(
            plan["montage"]["source_window"]["frames"]
        )
        original = start + int(cfg["after_preview_frames"])
        split = original + int(cfg["before_hold_frames"])
        group = split + int(cfg["split_frames"])
        end = group + int(cfg["group_frames"])
        expected = [start, original, split, group, end]
        samples = panel_qa["sampled_states"]
        checks["portrait_hero_focus_schedule"] = (
            panel_qa["mode"] == "hero_focus"
            and panel_qa["panel_count"] == 3
            and panel_qa["frame_count"] == frames
            and panel_qa["focus_operator_index"] == cfg["focus_operator_index"]
            and panel_qa["focus_roi"] == cfg["operator_rois"][cfg["focus_operator_index"]]
            and panel_qa["phase_frames"] == expected
            and samples[str(original - 1)] == {"phase": "after", "after": 1.0}
            and samples[str(original)] == {"phase": "before", "after": 0.0}
            and samples[str(split)] == {"phase": "split", "after": 0.5}
            and samples[str(group)] == {"phase": "group", "after": 1.0}
        )
        checks["portrait_hero_focus_verified_stills"] = (
            panel_qa["source_still_sha256"] == expected_still_hashes
            and panel_qa["source_only"] is True
        )
        checks["portrait_hero_focus_text_sync"] = panel_qa["text_synced"] is True
        checks["portrait_hero_focus_encoded_phases"] = len(hero_focus_differences) == 3 and all(
            change > 2.5 for change in hero_focus_differences
        )

    rebound_differences: list[float] = []
    rebound_live_difference = 0.0
    if rebound:
        if panel_qa is None or expected_still_hashes is None:
            raise RuntimeError("missing rebound source QA")
        from . import panel_compositor

        cfg = panel_compositor.rebound_config(profile, int(plan["montage"]["comparison_frames"]))
        start = int(plan["montage"]["hook"]["frames"]) + source_frame_count(plan["montage"])
        original = start + int(cfg["before_hold_frames"])
        flash = original + int(cfg["after_flash_frames"])
        group = flash + int(cfg["split_frames"])
        end = group + int(cfg["group_frames"])
        expected = [start, original, flash, group, end]
        samples = panel_qa["sampled_states"]
        checks["portrait_rebound_schedule"] = (
            panel_qa["mode"] == "rebound"
            and panel_qa["panel_count"] == 3
            and panel_qa["frame_count"] == frames
            and panel_qa["focus_operator_index"] == cfg["focus_operator_index"]
            and panel_qa["focus_roi"] == cfg["detail_roi"]
            and panel_qa["phase_frames"] == expected
            and panel_qa["live_detail_frames"] == start + frames - end
            and panel_qa["live_detail_source"] == "clean_canonical_ffv1_nut"
            and samples["0"] == {"phase": "live", "after": 1.0}
            and samples[str(int(plan["montage"]["hook"]["frames"]))]
            == {"phase": "live", "after": 0.0}
            and samples[str(start)] == {"phase": "before", "after": 0.0}
            and samples[str(original)] == {"phase": "after", "after": 1.0}
            and samples[str(flash)] == {"phase": "split", "after": 0.5}
            and samples[str(group)] == {"phase": "group", "after": 1.0}
        )
        checks["portrait_rebound_verified_stills"] = (
            panel_qa["source_still_sha256"] == expected_still_hashes
            and panel_qa["source_only"] is True
        )
        checks["portrait_rebound_text_sync"] = panel_qa["text_synced"] is True
        pairs = ((original - 1, original), (flash - 1, flash), (group - 1, group))
        original_start = int(plan["montage"]["hook"]["frames"])
        indices = sorted({0, original_start, *(n for pair in pairs for n in pair)})
        selected = "+".join(f"eq(n\\,{frame})" for frame in indices)
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(file),
                "-vf",
                f"select={selected},format=rgb24",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(indices)),
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        frame_bytes = width * height * 3
        if len(raw) != len(indices) * frame_bytes:
            raise RuntimeError("encoded rebound detail/phase frames missing")
        decoded_rebound: dict[int, Image.Image] = {
            n: Image.frombytes("RGB", (width, height), raw[i * frame_bytes : (i + 1) * frame_bytes])
            for i, n in enumerate(indices)
        }
        roi = profile.config["editorial"]["visual_state_roi"]
        visual_box = (
            round(float(roi[0]) * width),
            top + round(float(roi[1]) * center),
            round(float(roi[2]) * width),
            top + round(float(roi[3]) * center),
        )
        for left, right in pairs:
            change = ImageChops.difference(
                decoded_rebound[left].crop(visual_box), decoded_rebound[right].crop(visual_box)
            )
            rebound_differences.append(round(sum(ImageStat.Stat(change).mean) / 3, 4))
        x0, y0, x1, y1 = (int(v) for v in panel_qa["focus_box"])
        detail_box = (x0 + 6, top + center + y0 + 6, x1 - 6, top + center + y1 - 6)
        live_delta = ImageChops.difference(
            decoded_rebound[0].crop(detail_box), decoded_rebound[original_start].crop(detail_box)
        )
        rebound_live_difference = round(sum(ImageStat.Stat(live_delta).mean) / 3, 4)
        checks["portrait_rebound_encoded_phases"] = len(rebound_differences) == 3 and all(
            v > 2.5 for v in rebound_differences
        )
        checks["portrait_rebound_live_detail_visible"] = (
            x1 - x0 >= width * 0.8
            and y1 - y0 >= 0.7 * (height - top - center)
            and rebound_live_difference > 2.5
        )

    continuous_reveal_differences: list[float] = []
    if continuous_reveal:
        if reframe_qa is None:
            raise RuntimeError("missing continuous-reveal reframe metadata")
        from . import kinetic_reframe

        reveal_cfg = kinetic_reframe.config(profile, frames)
        visual_height = int(reveal_cfg["visual_height"])
        visual_top = int(reveal_cfg["visual_top"])
        checks["portrait_continuous_reveal_schedule"] = (
            reframe_qa["frame_count"] == frames
            and reframe_qa["keyframes"] == reveal_cfg["keyframes"]
            and reframe_qa["storyboard_frames"] == reveal_cfg["storyboard_frames"]
            and reframe_qa["visual_size"] == [width, visual_height]
            and reframe_qa["visual_top"] == visual_top
            and reframe_qa["backdrop_bottom_height"] == int(
                reveal_cfg["backdrop_bottom_height"]
            )
        )
        checks["portrait_continuous_reveal_source_sync"] = (
            reframe_qa["source_only"] is True
            and reframe_qa["source_frame_grid_exact"] is True
            and reframe_qa["input_frame_count"] == frames
        )
        checks["portrait_continuous_reveal_legibility"] = reframe_qa[
            "minimum_effective_source_width_px"
        ] >= int(reveal_cfg["minimum_effective_source_width"]) and visual_height > round(
            width * int(profile.config["output"]["height"]) / int(profile.config["output"]["width"])
        )
        checks["portrait_continuous_reveal_no_ai"] = reframe_qa["ai_enhancement"] is False
        checks["portrait_continuous_reveal_effects_declared"] = reframe_qa["effects"] == {
            "smooth_source_push_in": True,
            "luminance_flash": False,
            "source_backdrop": {
                "source": "same_canonical_frame",
                "blur_sigma": reveal_cfg["backdrop_blur_sigma"],
                "brightness": reveal_cfg["backdrop_brightness"],
                "saturation": reveal_cfg["backdrop_saturation"],
            },
            "unsharp_mask": {
                "radius": reveal_cfg["sharpen_radius"],
                "percent": reveal_cfg["sharpen_percent"],
                "threshold": reveal_cfg["sharpen_threshold"],
            },
        }
        checks["portrait_continuous_reveal_backdrop_visible"] = (
            len(backdrop_differences) == len(sampled_frames)
            and all(value > 4.0 for value in backdrop_differences)
            and reframe_qa["effects"]["source_backdrop"]["source"] == "same_canonical_frame"
        )
        source_start = int(plan["montage"]["source_window"]["start_frame"])
        trigger_start = int(plan["evidence"]["toggle_motion_start_frame"]) - source_start
        trigger_end = int(plan["evidence"]["toggle_motion_end_frame"]) - source_start
        selected = f"eq(n\\,{trigger_start})+eq(n\\,{trigger_end})"
        raw_reveal = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(file),
                "-vf",
                f"select={selected},format=rgb24",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                "2",
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        reveal_frame_bytes = width * height * 3
        if len(raw_reveal) != 2 * reveal_frame_bytes:
            raise RuntimeError("encoded continuous-reveal toggle frames are incomplete")
        before_toggle = Image.frombytes("RGB", (width, height), raw_reveal[:reveal_frame_bytes])
        after_toggle = Image.frombytes("RGB", (width, height), raw_reveal[reveal_frame_bytes:])
        visual_box = (0, visual_top, width, visual_top + visual_height)
        toggle_delta = ImageChops.difference(
            before_toggle.crop(visual_box), after_toggle.crop(visual_box)
        )
        continuous_reveal_differences.append(round(sum(ImageStat.Stat(toggle_delta).mean) / 3, 4))
        checks["portrait_continuous_reveal_toggle_visible"] = (
            len(continuous_reveal_differences) == 1 and continuous_reveal_differences[0] > 2.5
        )

    required = PORTRAIT_REQUIRED_CHECKS | (
        CASCADE_REQUIRED_CHECKS
        if cascade
        else SPOTLIGHT_REQUIRED_CHECKS
        if spotlight
        else SNAPBACK_REQUIRED_CHECKS
        if snapback
        else HERO_FOCUS_REQUIRED_CHECKS
        if hero_focus
        else REBOUND_REQUIRED_CHECKS
        if rebound
        else CONTINUOUS_REVEAL_REQUIRED_CHECKS
        if continuous_reveal
        else frozenset()
    )
    if rebound:
        required = (required - {"toggle_opening_off"}) | {"toggle_opening_after"}
    if set(checks) != required or not all(checks.values()):
        raise RuntimeError(f"portrait QA failed: {checks}")
    storyboard = (
        _encoded_spotlight_storyboard(file, output_dir, plan, panel_qa)
        if spotlight and panel_qa is not None
        else _encoded_snapback_storyboard(file, output_dir, plan, panel_qa)
        if snapback and panel_qa is not None
        else _encoded_hero_focus_storyboard(file, output_dir, plan, panel_qa)
        if hero_focus and panel_qa is not None
        else _encoded_rebound_storyboard(file, output_dir, plan, panel_qa)
        if rebound and panel_qa is not None
        else _encoded_continuous_reveal_storyboard(file, output_dir, plan, profile)
        if continuous_reveal
        else None
    )
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
            "file_size_mb": filesize_mb,
            "target_size_mb": size_target_mb,
            "size_tolerance_mb": size_tolerance_mb,
            "video_bitrate_kbps": video_kbps,
            "matte_top_rgb": list(top_rgb),
            "matte_bottom_rgb": list(bottom_rgb),
            "matte_rgb_max_error": (
                header_rgb_error if continuous_reveal else matte_rgb_error
            ),
            "matte_sampled_frames": sampled_mattes,
            "continuous_reveal_backdrop_differences": backdrop_differences,
            "cascade_panel_pixel_differences": panel_differences,
            "spotlight_pixel_differences": spotlight_differences,
            "snapback_pixel_differences": snapback_differences,
            "hero_focus_pixel_differences": hero_focus_differences,
            "rebound_pixel_differences": rebound_differences,
            "rebound_live_detail_difference": rebound_live_difference,
            "continuous_reveal_pixel_differences": continuous_reveal_differences,
            "video_profile": actual,
            "audio_profile": audio,
            "ssim": ssim,
            "psnr_db": psnr,
        },
        "title": title_qa,
        "cascade": panel_qa if cascade else None,
        "spotlight": panel_qa if spotlight else None,
        "snapback": panel_qa if snapback else None,
        "hero_focus": panel_qa if hero_focus else None,
        "rebound": panel_qa if rebound else None,
        "continuous_reveal": reframe_qa if continuous_reveal else None,
        "storyboard": storyboard,
        "canonical_ffv1_nut": True,
    }
