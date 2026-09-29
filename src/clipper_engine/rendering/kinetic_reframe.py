"""Clipper-owned deterministic source-native portrait reframing.

Profiles calibrate legal crop, title geometry, mild sharpening and optional
source-reveal styling. Clipper owns interpolation, rendering and source-sync QA.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .. import media_contract as media
from ..montage import MontageRejection
from ..profiles import CampaignProfile

SOURCE_REVEAL_MODES = frozenset({"continuous_reveal", "archive_reveal"})


def _ease(value: float) -> float:
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


def _effective_source_width(
    roi: list[float],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> int:
    crop_width = (roi[2] - roi[0]) * source_width
    crop_height = (roi[3] - roi[1]) * source_height
    return round(min(crop_width, crop_height * target_width / target_height))


def config(
    profile: CampaignProfile,
    output_frames: int,
    mode: str = "continuous_reveal",
) -> dict[str, Any]:
    if mode not in SOURCE_REVEAL_MODES:
        raise MontageRejection("source_reveal_mode", f"unsupported source-reveal mode {mode!r}")
    cfg = profile.config["output"]["portrait_matte"].get(mode)
    common = {
        "keyframes",
        "sharpen_radius",
        "sharpen_percent",
        "sharpen_threshold",
        "minimum_effective_source_width",
        "visual_top",
        "visual_height",
        "title_top_height",
        "backdrop_blur_sigma",
        "backdrop_brightness",
        "backdrop_saturation",
        "title_y_positions",
        "title_bar_y",
        "title_font_sizes",
        "title_pill_heights",
        "storyboard_frames",
    }
    archive = {
        "archive_grayscale",
        "archive_contrast",
        "archive_brightness",
        "archive_flicker_strength",
        "transition_direction",
        "transition_feather_px",
    }
    expected = common | archive if mode == "archive_reveal" else common
    code = f"{mode}_calibration"
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise MontageRejection(code, f"unsupported {mode.replace('_', '-')} calibration")

    keyframes = cfg["keyframes"]
    if not isinstance(keyframes, list) or len(keyframes) < 2:
        raise MontageRejection(code, "at least two crop keyframes are required")
    source_width = int(profile.config["output"]["width"])
    source_height = int(profile.config["output"]["height"])
    portrait = profile.config["output"]["portrait_matte"]
    target_width = int(portrait["width"])
    target_height = int(portrait["height"])
    visual_top = cfg["visual_top"]
    visual_height = cfg["visual_height"]
    title_top_height = cfg["title_top_height"]
    minimum_width = cfg["minimum_effective_source_width"]
    if (
        type(visual_top) is not int
        or type(visual_height) is not int
        or visual_height < 1
        or type(title_top_height) is not int
        or title_top_height < 1
        or visual_top < title_top_height
        or visual_top + visual_height >= target_height
        or type(minimum_width) is not int
        or minimum_width < 1
    ):
        raise MontageRejection(code, "portrait geometry or clarity floor is invalid")

    parsed: list[dict[str, Any]] = []
    previous_frame = -1
    previous_roi: list[float] | None = None
    for item in keyframes:
        if not isinstance(item, dict) or set(item) != {"frame", "roi"}:
            raise MontageRejection(code, "keyframes require only frame and roi")
        frame = item["frame"]
        roi = item["roi"]
        if (
            type(frame) is not int
            or frame <= previous_frame
            or not 0 <= frame < output_frames
            or not isinstance(roi, list)
            or len(roi) != 4
            or any(type(v) not in (int, float) for v in roi)
            or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)
        ):
            raise MontageRejection(code, "crop keyframe leaves the legal frame grid")
        normalized = [float(v) for v in roi]
        if previous_roi is not None and not (
            normalized[0] >= previous_roi[0]
            and normalized[1] >= previous_roi[1]
            and normalized[2] <= previous_roi[2]
            and normalized[3] <= previous_roi[3]
            and max(abs(a - b) for a, b in zip(normalized, previous_roi, strict=True)) <= 0.03
        ):
            raise MontageRejection(code, "source reveal permits only a gentle monotonic push-in")
        if (
            _effective_source_width(
                normalized, source_width, source_height, target_width, int(visual_height)
            )
            < minimum_width
        ):
            raise MontageRejection(code, "crop path falls below the configured source-detail floor")
        parsed.append({"frame": frame, "roi": normalized})
        previous_frame = frame
        previous_roi = normalized
    if parsed[0]["frame"] != 0 or parsed[-1]["frame"] != output_frames - 1:
        raise MontageRejection(code, "crop path must anchor first and final frames")

    storyboard = cfg["storyboard_frames"]
    title_y = cfg["title_y_positions"]
    font_sizes = cfg["title_font_sizes"]
    pill_heights = cfg["title_pill_heights"]
    if (
        not isinstance(storyboard, list)
        or len(storyboard) != 6
        or storyboard != sorted(set(storyboard))
        or any(type(n) is not int or not 0 <= n < output_frames for n in storyboard)
        or not isinstance(title_y, list)
        or len(title_y) != 3
        or any(type(n) is not int or n < 0 or n >= title_top_height for n in title_y)
        or not isinstance(font_sizes, list)
        or len(font_sizes) != 3
        or any(type(n) is not int or n < 8 for n in font_sizes)
        or not isinstance(pill_heights, list)
        or len(pill_heights) != 3
        or any(type(n) is not int or n < 8 for n in pill_heights)
        or type(cfg["title_bar_y"]) is not int
        or not 0 <= cfg["title_bar_y"] < title_top_height - 3
        or type(cfg["backdrop_blur_sigma"]) not in (int, float)
        or not 4 <= float(cfg["backdrop_blur_sigma"]) <= 40
        or type(cfg["backdrop_brightness"]) not in (int, float)
        or not -0.5 <= float(cfg["backdrop_brightness"]) <= 0
        or type(cfg["backdrop_saturation"]) not in (int, float)
        or not 0 <= float(cfg["backdrop_saturation"]) <= 1
        or type(cfg["sharpen_radius"]) not in (int, float)
        or not 0 <= float(cfg["sharpen_radius"]) <= 1
        or type(cfg["sharpen_percent"]) is not int
        or not 0 <= cfg["sharpen_percent"] <= 25
        or type(cfg["sharpen_threshold"]) is not int
        or not 0 <= cfg["sharpen_threshold"] <= 8
    ):
        raise MontageRejection(code, "title, review or mild-sharpen calibration is invalid")

    if mode == "archive_reveal" and (
        cfg["archive_grayscale"] is not True
        or type(cfg["archive_contrast"]) not in (int, float)
        or not 1 <= float(cfg["archive_contrast"]) <= 1.4
        or type(cfg["archive_brightness"]) not in (int, float)
        or not 0.8 <= float(cfg["archive_brightness"]) <= 1
        or type(cfg["archive_flicker_strength"]) not in (int, float)
        or not 0 <= float(cfg["archive_flicker_strength"]) <= 0.05
        or cfg["transition_direction"] != "left_to_right"
        or type(cfg["transition_feather_px"]) is not int
        or not 0 <= int(cfg["transition_feather_px"]) <= 240
    ):
        raise MontageRejection(code, "archive-film style calibration is invalid")

    return {
        **cfg,
        "keyframes": parsed,
        "storyboard_frames": list(storyboard),
        "backdrop_bottom_height": target_height - visual_top - visual_height,
    }


def _roi_for_frame(frame: int, keyframes: list[dict[str, Any]]) -> list[float]:
    for left, right in pairwise(keyframes):
        if frame <= int(right["frame"]):
            start = int(left["frame"])
            finish = int(right["frame"])
            progress = _ease((frame - start) / (finish - start))
            return [
                float(a) + (float(b) - float(a)) * progress
                for a, b in zip(left["roi"], right["roi"], strict=True)
            ]
    return list(keyframes[-1]["roi"])


def _crop(image: Image.Image, roi: list[float]) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = roi
    return image.crop(
        (
            round(x0 * width),
            round(y0 * height),
            max(round(x0 * width) + 1, round(x1 * width)),
            max(round(y0 * height) + 1, round(y1 * height)),
        )
    )


def _archive_monochrome(image: Image.Image, frame: int, cfg: dict[str, Any]) -> Image.Image:
    archived = ImageOps.grayscale(image).convert("RGB")
    archived = ImageEnhance.Contrast(archived).enhance(float(cfg["archive_contrast"]))
    flicker = 1 + float(cfg["archive_flicker_strength"]) * math.sin(frame * 1.7)
    archived = ImageEnhance.Brightness(archived).enhance(float(cfg["archive_brightness"]) * flicker)
    return archived


def _wipe_mask(size: tuple[int, int], progress: float, feather_px: int) -> Image.Image:
    width, height = size
    if progress <= 0:
        return Image.new("L", size, 0)
    if progress >= 1:
        return Image.new("L", size, 255)
    edge = progress * width
    half = feather_px / 2
    left = edge - half
    right = edge + half
    row = bytearray(width)
    for x in range(width):
        if x <= left:
            row[x] = 255
        elif x >= right:
            row[x] = 0
        elif feather_px == 0:
            row[x] = 255 if x < edge else 0
        else:
            row[x] = round(255 * (right - x) / feather_px)
    return Image.frombytes("L", (width, 1), bytes(row)).resize(
        (width, height), Image.Resampling.NEAREST
    )


def _styled_frame(
    image: Image.Image,
    frame: int,
    plan: dict[str, Any],
    cfg: dict[str, Any],
    mode: str,
) -> tuple[Image.Image, dict[str, Any]]:
    if mode != "archive_reveal":
        return image, {"phase": "source_color", "color_reveal": 1.0}

    source_start = int(plan["montage"]["source_window"]["start_frame"])
    trigger_start = int(plan["evidence"]["toggle_motion_start_frame"]) - source_start
    trigger_end = int(plan["evidence"]["toggle_motion_end_frame"]) - source_start
    archived = _archive_monochrome(image, frame, cfg)
    if frame < trigger_start:
        return archived, {"phase": "archive_monochrome", "color_reveal": 0.0}
    if frame >= trigger_end:
        return image, {"phase": "source_color", "color_reveal": 1.0}

    progress = _ease((frame - trigger_start) / (trigger_end - trigger_start))
    mask = _wipe_mask(image.size, progress, int(cfg["transition_feather_px"]))
    return Image.composite(image, archived, mask), {
        "phase": "color_wipe",
        "color_reveal": round(progress, 6),
    }


def render_frames(
    clean_canonical: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    canvas_width: int,
    visual_height: int,
) -> dict[str, Any]:
    frames = int(plan["montage"]["output_frames"])
    mode = str(plan["montage"]["comparison_mode"])
    cfg = config(profile, frames, mode)
    if visual_height != int(cfg["visual_height"]):
        raise MontageRejection(
            f"{mode}_layout", "portrait visual height differs from profile calibration"
        )
    video = media.video_profile(clean_canonical, count_frames=True)
    source_width = int(video["width"])
    source_height = int(video["height"])
    if video["frame_count"] != frames or (source_width, source_height) != (
        int(profile.config["output"]["width"]),
        int(profile.config["output"]["height"]),
    ):
        raise MontageRejection(
            f"{mode}_source",
            "portrait reframe input differs from the certified continuous source window",
        )

    frame_bytes = source_width * source_height * 3
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-i",
            str(clean_canonical),
            "-map",
            "0:v:0",
            "-frames:v",
            str(frames),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("source-reveal canonical frame stream is unavailable")

    folder = workspace / "reframe_frames"
    folder.mkdir(exist_ok=True)
    sample_indices = {int(item["frame"]) for item in cfg["keyframes"]} | set(
        int(n) for n in cfg["storyboard_frames"]
    )
    source_start = int(plan["montage"]["source_window"]["start_frame"])
    trigger_start = int(plan["evidence"]["toggle_motion_start_frame"]) - source_start
    trigger_end = int(plan["evidence"]["toggle_motion_end_frame"]) - source_start
    sample_indices.update({trigger_start, (trigger_start + trigger_end) // 2, trigger_end})
    sampled: dict[str, list[float]] = {}
    style_samples: dict[str, dict[str, Any]] = {}
    minimum_effective = source_width
    for frame in range(frames):
        raw = process.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            process.kill()
            raise RuntimeError("source reveal lost a canonical source frame")
        source = Image.frombytes("RGB", (source_width, source_height), raw)
        roi = _roi_for_frame(frame, cfg["keyframes"])
        minimum_effective = min(
            minimum_effective,
            _effective_source_width(roi, source_width, source_height, canvas_width, visual_height),
        )
        reframed = ImageOps.fit(
            _crop(source, roi),
            (canvas_width, visual_height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        if int(cfg["sharpen_percent"]) > 0:
            reframed = reframed.filter(
                ImageFilter.UnsharpMask(
                    radius=float(cfg["sharpen_radius"]),
                    percent=int(cfg["sharpen_percent"]),
                    threshold=int(cfg["sharpen_threshold"]),
                )
            )
        reframed, style_state = _styled_frame(reframed, frame, plan, cfg, mode)
        reframed.save(folder / f"{frame:04d}.png", compress_level=3)
        if frame in sample_indices:
            sampled[str(frame)] = [round(value, 6) for value in roi]
            style_samples[str(frame)] = style_state

    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        message = process.stderr.read().decode(errors="replace")[-500:] if process.stderr else ""
        raise RuntimeError(f"source-reveal canonical decode failed: {message}")
    if process.stderr is not None:
        process.stderr.close()

    digest = hashlib.sha256()
    with clean_canonical.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)

    effects: dict[str, Any] = {
        "smooth_source_push_in": True,
        "luminance_flash": False,
        "source_backdrop": {
            "source": "same_canonical_frame",
            "blur_sigma": cfg["backdrop_blur_sigma"],
            "brightness": cfg["backdrop_brightness"],
            "saturation": cfg["backdrop_saturation"],
        },
        "unsharp_mask": {
            "radius": cfg["sharpen_radius"],
            "percent": cfg["sharpen_percent"],
            "threshold": cfg["sharpen_threshold"],
        },
    }
    if mode == "archive_reveal":
        effects["archive_style"] = {
            "grayscale_before_toggle": cfg["archive_grayscale"],
            "contrast": cfg["archive_contrast"],
            "brightness": cfg["archive_brightness"],
            "flicker_strength": cfg["archive_flicker_strength"],
            "transition_direction": cfg["transition_direction"],
            "transition_feather_px": cfg["transition_feather_px"],
            "trigger_frames": [trigger_start, trigger_end],
        }

    return {
        "mode": mode,
        "folder": str(folder),
        "frame_count": frames,
        "input_frame_count": int(video["frame_count"]),
        "source_frame_grid_exact": int(video["frame_count"]) == frames,
        "source_only": True,
        "ai_enhancement": False,
        "input_canonical_sha256": digest.hexdigest(),
        "visual_size": [canvas_width, visual_height],
        "visual_top": int(cfg["visual_top"]),
        "backdrop_bottom_height": int(cfg["backdrop_bottom_height"]),
        "minimum_effective_source_width_px": minimum_effective,
        "keyframes": cfg["keyframes"],
        "storyboard_frames": cfg["storyboard_frames"],
        "sampled_rois": sampled,
        "style_samples": style_samples,
        "effects": effects,
    }
