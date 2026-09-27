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
    matte = profile.config["output"]["portrait_matte"]
    settings: dict[str, Any] = {
        **matte["cascade"],
        "operator_rois": matte["operator_rois"],
    }
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


def spotlight_config(profile: CampaignProfile, comparison_frames: int) -> dict[str, Any]:
    """Validate the single-focus/three-operator montage against one profile schedule."""
    matte = profile.config["output"]["portrait_matte"]
    settings: dict[str, Any] = {**matte["spotlight"], "operator_rois": matte["operator_rois"]}
    order = settings["operator_order"]
    segment = int(settings["focus_frames"])
    switch = int(settings["switch_after_frames"])
    group = int(settings["group_frames"])
    if (
        len(settings["operator_rois"]) != 3
        or len(order) != 3
        or sorted(order) != [0, 1, 2]
        or segment < 2
        or not 1 <= switch < segment
        or group < 3
        or segment * len(order) + group != comparison_frames
        or min(
            int(settings["focus_card_width"]),
            int(settings["focus_card_height"]),
            int(settings["group_card_height"]),
            int(settings["margin"]),
            int(settings["gap"]),
            int(settings["panel_top"]),
        )
        < 1
    ):
        raise MontageRejection("spotlight_calibration", "invalid focus, group or frame schedule")
    crop_center = settings.get("focus_crop_center")
    if (
        not isinstance(crop_center, list)
        or len(crop_center) != 2
        or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in crop_center)
    ):
        raise MontageRejection(
            "spotlight_calibration", "source-grounded spotlight focus center is invalid"
        )
    for roi in settings["operator_rois"]:
        if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise MontageRejection("spotlight_roi", "invalid normalized source-only crop")
    return settings


def spotlight_stage(
    local_frame: int, plan: dict[str, Any], profile: CampaignProfile
) -> tuple[int | None, float]:
    """Return one focused Operator and its verified visual state, or the group payoff."""
    frames = int(plan["montage"]["comparison_frames"])
    if not 0 <= local_frame < frames:
        raise MontageRejection("spotlight_timeline", "comparison frame outside planned grid")
    cfg = spotlight_config(profile, frames)
    segment = int(cfg["focus_frames"])
    index = local_frame // segment
    if index >= len(cfg["operator_order"]):
        return None, 1.0
    return int(cfg["operator_order"][index]), float(
        local_frame % segment >= int(cfg["switch_after_frames"])
    )


def snapback_config(profile: CampaignProfile, comparison_frames: int) -> dict[str, Any]:
    """Validate a source-still after->before->after comparison on the exact frame grid."""
    cfg: dict[str, Any] = profile.config["output"]["portrait_matte"]["snapback"]
    if set(cfg) != {
        "after_preview_frames",
        "before_hold_frames",
        "transition_frames",
        "margin",
        "gap",
        "panel_top",
        "panel_height",
    }:
        raise MontageRejection("snapback_calibration", "unsupported snapback calibration fields")
    preview = int(cfg["after_preview_frames"])
    before = int(cfg["before_hold_frames"])
    transition = int(cfg["transition_frames"])
    if (
        preview < 3
        or before < 3
        or transition < 2
        or preview + before + transition + 3 > comparison_frames
        or any(int(cfg[key]) <= 0 for key in ("margin", "gap", "panel_top", "panel_height"))
    ):
        raise MontageRejection("snapback_calibration", "preview/rewind/reveal must fit comparison")
    return cfg


def snapback_progress(local_frame: int, plan: dict[str, Any], profile: CampaignProfile) -> float:
    """One shared source-grounded visual state for picture, title and panel highlights."""
    frames = int(plan["montage"]["comparison_frames"])
    if not 0 <= local_frame < frames:
        raise MontageRejection("snapback_timeline", "comparison frame outside calibrated grid")
    cfg = snapback_config(profile, frames)
    preview = int(cfg["after_preview_frames"])
    before = int(cfg["before_hold_frames"])
    transition = int(cfg["transition_frames"])
    if local_frame < preview:
        return 1.0
    if local_frame < preview + before:
        return 0.0
    if local_frame < preview + before + transition:
        return _ease((local_frame - (preview + before)) / (transition - 1))
    return 1.0


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
    mode = str(plan["montage"]["comparison_mode"])
    if mode not in {"cascade", "spotlight", "snapback"}:
        raise MontageRejection("invalid_comparison_mode", mode)
    cfg = (
        config(profile, frames)
        if mode == "cascade"
        else spotlight_config(profile, frames)
        if mode == "spotlight"
        else snapback_config(profile, frames)
    )
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    if before.size != after.size or before.size != (
        int(profile.config["output"]["width"]),
        int(profile.config["output"]["height"]),
    ):
        raise MontageRejection("cascade_stills", "verified stills differ from certified geometry")
    folder = workspace / "panel_comparison_frames"
    folder.mkdir(exist_ok=True)
    w, h = before.size
    sampled: dict[str, Any] = {}
    sample_set = {0, frames - 1}
    if mode == "cascade":
        x_boundaries = [round(float(v) * w) for v in cfg["main_band_boundaries"]]
        for start in cfg["switch_start_frames"]:
            sample_set.update({int(start), int(start) + int(cfg["transition_frames"])})
    elif mode == "spotlight":
        for i in range(len(cfg["operator_order"])):
            focus_start = i * int(cfg["focus_frames"])
            sample_set.update(
                {
                    focus_start,
                    focus_start + int(cfg["switch_after_frames"]) - 1,
                    focus_start + int(cfg["switch_after_frames"]),
                }
            )
        sample_set.add(int(cfg["focus_frames"]) * len(cfg["operator_order"]))
    else:
        preview = int(cfg["after_preview_frames"])
        reveal = preview + int(cfg["before_hold_frames"])
        sample_set.update(
            {preview - 1, preview, reveal - 1, reveal, reveal + int(cfg["transition_frames"]) - 1}
        )
    for n in range(frames):
        if mode == "cascade":
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
        elif mode == "spotlight":
            focus, progress = spotlight_stage(n, plan, profile)
            frame = before if progress <= 0 else after
        else:
            progress = snapback_progress(n, plan, profile)
            frame = (
                before
                if progress == 0
                else after
                if progress == 1
                else Image.blend(before, after, progress)
            )
        frame.save(folder / f"{n:04d}.png", compress_level=3)
        if n in sample_set:
            sampled[str(n)] = (
                [round(v, 6) for v in states]
                if mode == "cascade"
                else {"focus": focus, "after": progress}
                if mode == "spotlight"
                else {"after": progress}
            )
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
        raise RuntimeError(f"Clipper panel comparison failed full-frame QA: {measured}")
    return target, {
        "before_frame": int(plan["evidence"]["before_frame"]),
        "after_frame": int(plan["evidence"]["after_frame"]),
        "comparison_mode": mode,
        "comparison_frames": frames,
        "frame_count_exact": exact,
        "full_frame": geometry,
        "no_black_bar_layout": True,
        "source_only": True,
        "source_still_sha256": {
            "before": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            "after": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        },
        "panel_comparison_states": sampled,
        **({"verified_source_regions": cfg["operator_rois"]} if mode != "snapback" else {}),
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
    if plan["montage"]["comparison_mode"] == "snapback":
        return render_snapback_panels(
            before_path,
            after_path,
            workspace,
            plan,
            profile,
            canvas_width,
            bottom_height,
            source_progress,
        )
    if plan["montage"]["comparison_mode"] == "spotlight":
        return render_spotlight_panels(
            before_path,
            after_path,
            workspace,
            plan,
            profile,
            canvas_width,
            bottom_height,
            source_progress,
        )
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
    folder = workspace / "panel_frames"
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


def render_spotlight_panels(
    before_path: Path,
    after_path: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    canvas_width: int,
    bottom_height: int,
    source_progress: list[float],
) -> dict[str, Any]:
    """Single-Operator close-ups followed by a three-Operator group payoff.

    Both focus crops and the group cards come exclusively from the exact verified
    source stills also used by Clipper's canonical comparison stage.
    """
    edit = plan["montage"]
    frames = int(edit["output_frames"])
    cfg = spotlight_config(profile, int(edit["comparison_frames"]))
    if len(source_progress) != frames:
        raise MontageRejection("spotlight_timeline", "title and spotlight have different grids")
    scale = canvas_width / 1080
    margin = max(2, round(int(cfg["margin"]) * scale))
    gap = max(2, round(int(cfg["gap"]) * scale))
    y = max(3, round(int(cfg["panel_top"]) * scale))
    wide = min(canvas_width - 2 * margin, round(int(cfg["focus_card_width"]) * scale))
    high = round(int(cfg["focus_card_height"]) * scale)
    group_high = round(int(cfg["group_card_height"]) * scale)
    group_wide = (canvas_width - 2 * margin - 2 * gap) // 3
    if (
        wide < 20
        or high < 20
        or group_wide < 14
        or group_high < 20
        or y + max(high, group_high) >= bottom_height - max(4, round(35 * scale))
    ):
        raise MontageRejection("spotlight_layout", "focus or final-group cards exceed bottom matte")
    focus_x = (canvas_width - wide) // 2
    matte = profile.config["output"]["portrait_matte"]
    bg = ImageColor.getrgb(str(matte["background_hex"]))[:3]
    gold = ImageColor.getrgb(str(matte["accent_hex"]))[:3]
    original = Image.open(before_path).convert("RGB")
    alternate = Image.open(after_path).convert("RGB")
    if original.size != alternate.size:
        raise MontageRejection("spotlight_stills", "certified source still geometry differs")
    crops: dict[str, list[Image.Image]] = {}
    for name, source in (("before", original), ("after", alternate)):
        crops[name + "_focus"] = [
            ImageOps.fit(
                _crop(source, roi),
                (wide, high),
                method=Image.Resampling.LANCZOS,
                centering=(
                    float(cfg["focus_crop_center"][0]),
                    float(cfg["focus_crop_center"][1]),
                ),
            )
            for roi in cfg["operator_rois"]
        ]
        crops[name + "_group"] = [
            ImageOps.fit(
                _crop(source, roi),
                (group_wide, group_high),
                method=Image.Resampling.LANCZOS,
            )
            for roi in cfg["operator_rois"]
        ]
    focus_mask = Image.new("L", (wide, high), 0)
    group_mask = Image.new("L", (group_wide, group_high), 0)
    radius = max(2, round(12 * scale))
    ImageDraw.Draw(focus_mask).rounded_rectangle((0, 0, wide - 1, high - 1), radius, fill=255)
    ImageDraw.Draw(group_mask).rounded_rectangle(
        (0, 0, group_wide - 1, group_high - 1), radius, fill=255
    )
    folder = workspace / "panel_frames"
    folder.mkdir(exist_ok=True)
    compare_start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
    compare_end = compare_start + int(edit["comparison_frames"])
    order = [int(v) for v in cfg["operator_order"]]
    switch = int(cfg["switch_after_frames"])
    segment = int(cfg["focus_frames"])
    group_start = compare_start + len(order) * segment
    samples: dict[str, dict[str, Any]] = {}
    sample_indices = {0, frames - 1, compare_start, group_start, compare_end - 1}
    for pos in range(len(order)):
        sample_indices.update(
            {
                compare_start + pos * segment,
                compare_start + pos * segment + switch - 1,
                compare_start + pos * segment + switch,
            }
        )
    for n in range(frames):
        if compare_start <= n < compare_end:
            focus, after_value = spotlight_stage(n - compare_start, plan, profile)
            if after_value != source_progress[n]:
                raise MontageRejection(
                    "spotlight_text_sync", "verified appearance and highlighted title differ"
                )
        elif n < compare_start:
            # During the retained real footage, cycle a single source-only crop
            # without misrepresenting the source frame's measured visual state.
            focus = order[min(len(order) - 1, n * len(order) // max(1, compare_start))]
            after_value = source_progress[n]
        else:
            focus = None
            after_value = source_progress[n]
        image = Image.new("RGB", (canvas_width, bottom_height), bg)
        draw = ImageDraw.Draw(image)
        if focus is not None:
            if after_value <= 0:
                card = crops["before_focus"][focus]
            elif after_value >= 1:
                card = crops["after_focus"][focus]
            else:
                card = Image.blend(
                    crops["before_focus"][focus], crops["after_focus"][focus], after_value
                )
            image.paste(card, (focus_x, y), focus_mask)
            draw.rounded_rectangle(
                (focus_x, y, focus_x + wide - 1, y + high - 1),
                radius=radius,
                outline=tuple(
                    round(a * (1 - after_value) + b * after_value)
                    for a, b in zip((83, 87, 91), gold, strict=True)
                ),
                width=max(1, round(3 * scale)),
            )
            # Graphic-only step markers, no unapproved campaign wording.
            dot_w = max(3, round(36 * scale))
            dot_gap = max(2, round(14 * scale))
            dots_left = (canvas_width - 3 * dot_w - 2 * dot_gap) // 2
            dots_y = y + high + max(5, round(15 * scale))
            for i, op in enumerate(order):
                draw.rounded_rectangle(
                    (
                        dots_left + i * (dot_w + dot_gap),
                        dots_y,
                        dots_left + i * (dot_w + dot_gap) + dot_w,
                        dots_y + max(2, round(4 * scale)),
                    ),
                    radius=max(1, round(2 * scale)),
                    fill=gold if op == focus else (75, 78, 81),
                )
        else:
            for i, op in enumerate(order):
                left = margin + i * (group_wide + gap)
                if after_value <= 0:
                    card = crops["before_group"][op]
                elif after_value >= 1:
                    card = crops["after_group"][op]
                else:
                    card = Image.blend(
                        crops["before_group"][op], crops["after_group"][op], after_value
                    )
                image.paste(card, (left, y), group_mask)
                draw.rounded_rectangle(
                    (left, y, left + group_wide - 1, y + group_high - 1),
                    radius=radius,
                    outline=gold if after_value >= 1 else (83, 87, 91),
                    width=max(1, round(3 * scale)),
                )
        image.save(folder / f"{n:04d}.png", compress_level=3)
        if n in sample_indices:
            samples[str(n)] = {"focus": focus, "after": round(after_value, 6)}
    return {
        "folder": str(folder),
        "frame_count": frames,
        "mode": "spotlight",
        "source_only": True,
        "text_synced": True,
        "focus_order": order,
        "focus_frames": segment,
        "switch_after_frames": switch,
        "group_frames": int(cfg["group_frames"]),
        "switch_frames": [compare_start + i * segment + switch for i in range(len(order))],
        "group_start_frame": group_start,
        "focus_box": [focus_x, y, focus_x + wide, y + high],
        "sampled_states": samples,
        "source_still_sha256": {
            "before": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            "after": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        },
    }


def render_snapback_panels(
    before_path: Path,
    after_path: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    canvas_width: int,
    bottom_height: int,
    source_progress: list[float],
) -> dict[str, Any]:
    """Two persistent certified full-group frames, highlighting the active visual state.

    Uses no unapproved panel labels or third-party material. Source-to-piece
    equality is enforced by the shared Clipper FFV1 source/canonical stages.
    """
    edit = plan["montage"]
    frames = int(edit["output_frames"])
    cfg = snapback_config(profile, int(edit["comparison_frames"]))
    if len(source_progress) != frames:
        raise MontageRejection("snapback_timeline", "portrait text and panel frames diverged")
    scale = canvas_width / 1080
    margin = max(2, round(int(cfg["margin"]) * scale))
    gap = max(2, round(int(cfg["gap"]) * scale))
    y = max(3, round(int(cfg["panel_top"]) * scale))
    height = max(12, round(int(cfg["panel_height"]) * scale))
    width = (canvas_width - 2 * margin - gap) // 2
    if width < 28 or y + height >= bottom_height - 4:
        raise MontageRejection("snapback_layout", "two comparison panels exceed bottom matte")
    matte = profile.config["output"]["portrait_matte"]
    bg = ImageColor.getrgb(str(matte["background_hex"]))[:3]
    gold = ImageColor.getrgb(str(matte["accent_hex"]))[:3]
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    if before.size != after.size:
        raise MontageRejection("snapback_stills", "certified before/after geometries differ")

    # Contain each entire 16:9 team shot, never crop an Operator out of the comparison.
    def card_image(source: Image.Image) -> Image.Image:
        card = Image.new("RGB", (width, height), bg)
        thumb = ImageOps.contain(source, (width - 8, height - 10), method=Image.Resampling.LANCZOS)
        card.paste(thumb, ((width - thumb.width) // 2, (height - thumb.height) // 2))
        return card

    cards = (card_image(before), card_image(after))
    mask = Image.new("L", (width, height), 0)
    radius = max(2, round(12 * scale))
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius, fill=255)
    folder = workspace / "panel_frames"
    folder.mkdir(exist_ok=True)
    compare_start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
    compare_end = compare_start + int(edit["comparison_frames"])
    preview = int(cfg["after_preview_frames"])
    rewind = compare_start + preview
    reveal = rewind + int(cfg["before_hold_frames"])
    switched = reveal + int(cfg["transition_frames"]) - 1
    sample_frames = {
        0,
        compare_start,
        rewind - 1,
        rewind,
        reveal - 1,
        reveal,
        switched,
        compare_end - 1,
        frames - 1,
    }
    samples: dict[str, float] = {}
    boxes = []
    for i in range(2):
        x = margin + i * (width + gap)
        boxes.append([x, y, x + width, y + height])
    for n in range(frames):
        active = source_progress[n]
        if compare_start <= n < compare_end:
            expected = snapback_progress(n - compare_start, plan, profile)
            if active != expected:
                raise MontageRejection("snapback_text_sync", "title fill contradicts shown state")
        image = Image.new("RGB", (canvas_width, bottom_height), bg)
        draw = ImageDraw.Draw(image)
        for i, card in enumerate(cards):
            x0, y0, x1, y1 = boxes[i]
            image.paste(card, (x0, y0), mask)
            intensity = 1.0 - active if i == 0 else active
            outline = tuple(
                round(a * (1 - intensity) + b * intensity)
                for a, b in zip((83, 87, 91), gold, strict=True)
            )
            draw.rounded_rectangle(
                (x0, y0, x1 - 1, y1 - 1),
                radius=radius,
                outline=outline,
                width=max(1, round(4 * scale)),
            )
        image.save(folder / f"{n:04d}.png", compress_level=3)
        if n in sample_frames:
            samples[str(n)] = round(active, 6)
    return {
        "folder": str(folder),
        "mode": "snapback",
        "frame_count": frames,
        "panel_count": 2,
        "panel_boxes": boxes,
        "source_only": True,
        "text_synced": True,
        "source_still_sha256": {
            "before": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            "after": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        },
        "comparison_start_frame": compare_start,
        "switch_frames": [rewind, reveal, switched],
        "sampled_states": samples,
    }
