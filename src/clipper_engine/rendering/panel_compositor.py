"""Reusable Clipper source-grounded panel and staggered visual-state compositor.

Gameplay/campaign profiles calibrate source ROIs and frame offsets. All frame
construction, lossless comparison staging, compositing and QA remain in Clipper.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageOps

from .. import media_contract as media
from ..montage import MontageRejection, rate
from ..profiles import CampaignProfile


def _ease(value: float) -> float:
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


def config(profile: CampaignProfile, comparison_frames: int) -> dict[str, Any]:
    settings: dict[str, Any] = profile.config["output"]["portrait_matte"]["cascade"]
    regions = settings["operator_rois"]
    boundaries = settings["main_band_boundaries"]
    starts = settings["switch_start_frames"]
    fade = int(settings["transition_frames"])
    if (
        len(regions) != 3
        or len(boundaries) != 4
        or len(starts) != 3
        or len(settings["switch_order"]) != 3
        or sorted(settings["switch_order"]) != [0, 1, 2]
        or not (0 == boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3] == 1)
        or not (0 <= starts[0] < starts[1] < starts[2])
        or fade < 1
        or starts[-1] + fade >= comparison_frames
    ):
        raise MontageRejection("cascade_calibration", "invalid cascade stages, bounds or cadence")
    for region in regions:
        if len(region) != 4 or not (
            0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1
        ):
            raise MontageRejection("cascade_roi", "normalized Operator ROI is invalid")
    return settings


def stage_progress(local_frame: int, plan: dict[str, Any], profile: CampaignProfile) -> list[float]:
    """Operator states for one frame within the exact legal comparison window."""
    cfg = config(profile, int(plan["montage"]["comparison_frames"]))
    values = [0.0, 0.0, 0.0]
    for index, start in zip(cfg["switch_order"], cfg["switch_start_frames"], strict=True):
        values[index] = _ease((local_frame - int(start)) / int(cfg["transition_frames"]))
    return values


def states_for_output(
    frame: int, plan: dict[str, Any], profile: CampaignProfile, global_progress: float
) -> list[float]:
    edit = plan["montage"]
    start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
    end = start + int(edit["comparison_frames"])
    if start <= frame < end:
        return stage_progress(frame - start, plan, profile)
    return [global_progress] * 3


def _crop(image: Image.Image, normalized: list[float]) -> Image.Image:
    w, h = image.size
    left, top, right, bottom = normalized
    return image.crop(
        (
            int(left * w),
            int(top * h),
            max(int(left * w) + 1, int(right * w)),
            max(int(top * h) + 1, int(bottom * h)),
        )
    )


def render_comparison(
    before_path: Path,
    after_path: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    source_profile: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Build the canonical full-frame staggered comparison from verified source stills."""
    frames = int(plan["montage"]["comparison_frames"])
    cfg = config(profile, frames)
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    if before.size != after.size or before.size != (
        int(profile.config["output"]["width"]),
        int(profile.config["output"]["height"]),
    ):
        raise MontageRejection("cascade_stills", "verified stills differ from certified geometry")
    folder = workspace / "cascade_comparison_frames"
    folder.mkdir(exist_ok=True)
    w, h = before.size
    x_boundaries = [round(float(v) * w) for v in cfg["main_band_boundaries"]]
    sampled: dict[str, list[float]] = {}
    sample_set = {0, frames - 1}
    for start in cfg["switch_start_frames"]:
        sample_set.update({int(start), int(start) + int(cfg["transition_frames"])})
    for n in range(frames):
        states = stage_progress(n, plan, profile)
        frame = before.copy()
        for index, progress in enumerate(states):
            x0, x1 = x_boundaries[index : index + 2]
            if progress <= 0:
                continue
            frame.paste(
                Image.blend(before.crop((x0, 0, x1, h)), after.crop((x0, 0, x1, h)), progress),
                (x0, 0),
            )
        frame.save(folder / f"{n:04d}.png", compress_level=3)
        if n in sample_set:
            sampled[str(n)] = [round(v, 6) for v in states]
    target = workspace / "comparison.nut"
    fps = rate(profile)
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-framerate",
            f"{fps.numerator}/{fps.denominator}",
            "-i",
            str(folder / "%04d.png"),
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-threads:v",
            "1",
            *media.profile_output_args(source_profile),
            *media.color_metadata_tag_args(source_profile),
            "-f",
            "nut",
            str(target),
        ]
    )
    measured = media.video_profile(target, count_frames=True)
    exact = measured["frame_count"] == frames and measured["codec_name"] == "ffv1"
    geometry = (measured["width"], measured["height"]) == (w, h)
    if not (exact and geometry):
        raise RuntimeError(f"Clipper cascade canonical comparison failed QA: {measured}")
    return target, {
        "before_frame": int(plan["evidence"]["before_frame"]),
        "after_frame": int(plan["evidence"]["after_frame"]),
        "comparison_mode": "cascade",
        "comparison_frames": frames,
        "frame_count_exact": exact,
        "full_frame": geometry,
        "no_black_bar_layout": True,
        "source_only": True,
        "source_still_sha256": {
            "before": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            "after": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        },
        "operator_cascade_states": sampled,
        "verified_source_regions": cfg["operator_rois"],
    }


def render_panels(
    before_path: Path,
    after_path: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    canvas_width: int,
    bottom_height: int,
    source_progress: list[float],
) -> dict[str, Any]:
    """Generate bottom-matte panel sequence, matched to the same output frame grid."""
    edit = plan["montage"]
    frames = int(edit["output_frames"])
    cfg = config(profile, int(edit["comparison_frames"]))
    if len(source_progress) != frames:
        raise MontageRejection("cascade_timeline", "title and panel timelines are unequal")
    width_scale = canvas_width / 1080
    margin = max(2, round(float(cfg["margin"]) * width_scale))
    gap = max(2, round(float(cfg["gap"]) * width_scale))
    panel_y = max(3, round(float(cfg["panel_top"]) * width_scale))
    panel_height = max(10, round(float(cfg["panel_height"]) * width_scale))
    panel_width = (canvas_width - 2 * margin - 2 * gap) // 3
    if panel_width < 15 or panel_y + panel_height >= bottom_height - 4:
        raise MontageRejection("cascade_layout", "panel cards exceed the portrait bottom matte")
    bg_raw = ImageColor.getrgb(profile.config["output"]["portrait_matte"]["background_hex"])
    bg = tuple(bg_raw[:3])
    gold_raw = ImageColor.getrgb(profile.config["output"]["portrait_matte"]["accent_hex"])
    gold = tuple(gold_raw[:3])
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    if before.size != after.size:
        raise MontageRejection("cascade_stills", "source comparison stills have different geometry")
    before_crops = [
        ImageOps.fit(
            _crop(before, roi), (panel_width, panel_height), method=Image.Resampling.LANCZOS
        )
        for roi in cfg["operator_rois"]
    ]
    after_crops = [
        ImageOps.fit(
            _crop(after, roi), (panel_width, panel_height), method=Image.Resampling.LANCZOS
        )
        for roi in cfg["operator_rois"]
    ]
    radius = max(2, round(12 * width_scale))
    mask = Image.new("L", (panel_width, panel_height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, panel_width - 1, panel_height - 1), radius=radius, fill=255
    )
    folder = workspace / "cascade_panels"
    folder.mkdir(exist_ok=True)
    comparison_start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
    sample_frames = {
        0,
        frames - 1,
        comparison_start,
        comparison_start + int(edit["comparison_frames"]) - 1,
    }
    for start in cfg["switch_start_frames"]:
        sample_frames.update(
            {
                comparison_start + int(start),
                comparison_start + int(start) + int(cfg["transition_frames"]),
            }
        )
    sampled: dict[str, list[float]] = {}
    for n in range(frames):
        states = states_for_output(n, plan, profile, source_progress[n])
        image = Image.new("RGB", (canvas_width, bottom_height), bg)
        draw = ImageDraw.Draw(image)
        for index, progress in enumerate(states):
            left = margin + index * (panel_width + gap)
            if progress <= 0:
                card = before_crops[index]
            elif progress >= 1:
                card = after_crops[index]
            else:
                card = Image.blend(before_crops[index], after_crops[index], progress)
            image.paste(card, (left, panel_y), mask)
            # Gold border indicates the Operators already switched;
            # unmodified before cards use subtle neutral borders.
            outline = tuple(
                round(a * (1 - progress) + b * progress)
                for a, b in zip((83, 87, 91), gold, strict=True)
            )
            draw.rounded_rectangle(
                (left, panel_y, left + panel_width - 1, panel_y + panel_height - 1),
                radius=radius,
                outline=outline,
                width=max(1, round(3 * width_scale)),
            )
        image.save(folder / f"{n:04d}.png", compress_level=3)
        if n in sample_frames:
            sampled[str(n)] = [round(v, 6) for v in states]
    start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
    for n in range(int(edit["comparison_frames"])):
        panel_state = stage_progress(n, plan, profile)
        if abs(sum(panel_state) / 3 - source_progress[start + n]) > 1e-8:
            raise MontageRejection("cascade_text_sync", "title fill differs from panel transitions")
    return {
        "folder": str(folder),
        "frame_count": frames,
        "panel_count": 3,
        "switch_frames": [start + int(v) for v in cfg["switch_start_frames"]],
        "transition_frames": int(cfg["transition_frames"]),
        "sampled_states": sampled,
        "source_still_sha256": {
            "before": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            "after": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        },
        "bottom_height": bottom_height,
        "panel_size": [panel_width, panel_height],
        "panel_boxes": [
            [
                margin + index * (panel_width + gap),
                panel_y,
                margin + index * (panel_width + gap) + panel_width,
                panel_y + panel_height,
            ]
            for index in range(3)
        ],
        "source_only": True,
        "text_synced": True,
    }
