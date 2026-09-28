"""Clipper-owned deterministic source-native kinetic portrait reframing.

Profiles provide only crop/effect calibration. This module owns frame-grid
rendering, crop interpolation, mild delivery sharpening, and source-sync QA.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from .. import media_contract as media
from ..montage import MontageRejection
from ..profiles import CampaignProfile


def _ease(value: float) -> float:
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


def config(profile: CampaignProfile, output_frames: int) -> dict[str, Any]:
    cfg = profile.config["output"]["portrait_matte"].get("impact_cut")
    expected = {
        "keyframes",
        "flash_frames",
        "flash_strength",
        "sharpen_radius",
        "sharpen_percent",
        "sharpen_threshold",
        "minimum_source_crop_width",
        "bottom_matte_height",
        "storyboard_frames",
    }
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise MontageRejection("impact_cut_calibration", "unsupported Impact Cut calibration")

    keyframes = cfg["keyframes"]
    if not isinstance(keyframes, list) or len(keyframes) < 2:
        raise MontageRejection("impact_cut_calibration", "at least two crop keyframes required")
    parsed: list[dict[str, Any]] = []
    previous = -1
    source_width = int(profile.config["output"]["width"])
    minimum_width = int(cfg["minimum_source_crop_width"])
    for item in keyframes:
        if not isinstance(item, dict) or set(item) != {"frame", "roi"}:
            raise MontageRejection("impact_cut_calibration", "keyframes require frame and roi")
        frame = item["frame"]
        roi = item["roi"]
        if (
            type(frame) is not int
            or frame <= previous
            or not 0 <= frame < output_frames
            or not isinstance(roi, list)
            or len(roi) != 4
            or any(type(v) not in (int, float) for v in roi)
            or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)
            or round((float(roi[2]) - float(roi[0])) * source_width) < minimum_width
        ):
            raise MontageRejection(
                "impact_cut_calibration", "crop keyframe leaves frame grid or clarity bounds"
            )
        parsed.append({"frame": frame, "roi": [float(v) for v in roi]})
        previous = frame
    if parsed[0]["frame"] != 0 or parsed[-1]["frame"] != output_frames - 1:
        raise MontageRejection(
            "impact_cut_calibration", "crop path must anchor first and final output frames"
        )

    flash_frames = cfg["flash_frames"]
    storyboard = cfg["storyboard_frames"]
    if (
        not isinstance(flash_frames, list)
        or any(type(n) is not int or not 0 <= n < output_frames for n in flash_frames)
        or flash_frames != sorted(set(flash_frames))
        or not isinstance(storyboard, list)
        or len(storyboard) != 6
        or storyboard != sorted(set(storyboard))
        or any(type(n) is not int or not 0 <= n < output_frames for n in storyboard)
        or type(cfg["bottom_matte_height"]) is not int
        or cfg["bottom_matte_height"] < 32
        or type(cfg["minimum_source_crop_width"]) is not int
        or cfg["minimum_source_crop_width"] < 1
        or type(cfg["flash_strength"]) not in (int, float)
        or not 0 <= float(cfg["flash_strength"]) <= 0.2
        or type(cfg["sharpen_radius"]) not in (int, float)
        or not 0 < float(cfg["sharpen_radius"]) <= 2
        or type(cfg["sharpen_percent"]) is not int
        or not 0 <= cfg["sharpen_percent"] <= 100
        or type(cfg["sharpen_threshold"]) is not int
        or not 0 <= cfg["sharpen_threshold"] <= 8
    ):
        raise MontageRejection("impact_cut_calibration", "invalid effect or review calibration")

    return {
        **cfg,
        "keyframes": parsed,
        "flash_frames": list(flash_frames),
        "storyboard_frames": list(storyboard),
    }


def _roi_for_frame(frame: int, keyframes: list[dict[str, Any]]) -> list[float]:
    for left, right in zip(keyframes, keyframes[1:], strict=True):
        if frame <= int(right["frame"]):
            start = int(left["frame"])
            finish = int(right["frame"])
            if finish == start:
                return list(right["roi"])
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


def render_frames(
    clean_canonical: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    canvas_width: int,
    visual_height: int,
) -> dict[str, Any]:
    frames = int(plan["montage"]["output_frames"])
    cfg = config(profile, frames)
    if visual_height <= 0:
        raise MontageRejection("impact_cut_layout", "portrait visual area is empty")

    video = media.video_profile(clean_canonical, count_frames=True)
    source_width = int(video["width"])
    source_height = int(video["height"])
    if video["frame_count"] != frames or (source_width, source_height) != (
        int(profile.config["output"]["width"]),
        int(profile.config["output"]["height"]),
    ):
        raise MontageRejection(
            "impact_cut_source", "kinetic reframe input differs from canonical source grid"
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
        raise RuntimeError("Impact Cut canonical frame stream is unavailable")

    folder = workspace / "kinetic_frames"
    folder.mkdir(exist_ok=True)
    sampled: dict[str, list[float]] = {}
    sample_indices = {int(item["frame"]) for item in cfg["keyframes"]} | set(
        int(n) for n in cfg["flash_frames"]
    )
    minimum_crop = source_width
    white = Image.new("RGB", (canvas_width, visual_height), (255, 255, 255))
    for frame in range(frames):
        raw = process.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            process.kill()
            raise RuntimeError("Impact Cut lost a canonical source frame")
        source = Image.frombytes("RGB", (source_width, source_height), raw)
        roi = _roi_for_frame(frame, cfg["keyframes"])
        minimum_crop = min(minimum_crop, round((roi[2] - roi[0]) * source_width))
        reframed = ImageOps.fit(
            _crop(source, roi),
            (canvas_width, visual_height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        reframed = reframed.filter(
            ImageFilter.UnsharpMask(
                radius=float(cfg["sharpen_radius"]),
                percent=int(cfg["sharpen_percent"]),
                threshold=int(cfg["sharpen_threshold"]),
            )
        )
        if frame in cfg["flash_frames"]:
            reframed = Image.blend(reframed, white, float(cfg["flash_strength"]))
        reframed.save(folder / f"{frame:04d}.png", compress_level=3)
        if frame in sample_indices:
            sampled[str(frame)] = [round(value, 6) for value in roi]

    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        message = process.stderr.read().decode(errors="replace")[-500:] if process.stderr else ""
        raise RuntimeError(f"Impact Cut canonical decode failed: {message}")
    if process.stderr is not None:
        process.stderr.close()

    digest = hashlib.sha256()
    with clean_canonical.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)

    return {
        "mode": "impact_cut",
        "folder": str(folder),
        "frame_count": frames,
        "input_frame_count": int(video["frame_count"]),
        "source_frame_grid_exact": int(video["frame_count"]) == frames,
        "source_only": True,
        "ai_enhancement": False,
        "input_canonical_sha256": digest.hexdigest(),
        "visual_size": [canvas_width, visual_height],
        "bottom_matte_height": int(cfg["bottom_matte_height"]),
        "minimum_source_crop_width_px": minimum_crop,
        "keyframes": cfg["keyframes"],
        "flash_frames": cfg["flash_frames"],
        "storyboard_frames": cfg["storyboard_frames"],
        "sampled_rois": sampled,
        "effects": {
            "kinetic_crop": True,
            "luminance_flash": {
                "frames": cfg["flash_frames"],
                "strength": cfg["flash_strength"],
            },
            "unsharp_mask": {
                "radius": cfg["sharpen_radius"],
                "percent": cfg["sharpen_percent"],
                "threshold": cfg["sharpen_threshold"],
            },
        },
    }
