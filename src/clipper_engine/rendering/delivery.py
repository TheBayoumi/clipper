from __future__ import annotations

from fractions import Fraction
from typing import Any

from . import portrait_layout

_SOURCE_DIMENSIONS = {None, "source", "auto"}


def _dimension(value: Any, source_value: int, *, label: str) -> int:
    if value in _SOURCE_DIMENSIONS:
        return int(source_value)
    resolved = int(value)
    if resolved <= 0:
        raise RuntimeError(f"output.{label} must be positive")
    if resolved % 2:
        raise RuntimeError(f"output.{label} must be even for yuv420p delivery")
    return resolved


def profile_for_output(
    source_profile: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = dict(config.get("output") or {})
    width = _dimension(
        settings.get("width"),
        int(source_profile["width"]),
        label="width",
    )
    height = _dimension(
        settings.get("height"),
        int(source_profile["height"]),
        label="height",
    )
    ratio = Fraction(width, height)

    profile = dict(source_profile)
    profile["width"] = width
    profile["height"] = height
    profile["sample_aspect_ratio"] = "1:1"
    profile["display_aspect_ratio"] = f"{ratio.numerator}:{ratio.denominator}"
    return profile


def composition_required(
    source_profile: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    delivery_profile = profile_for_output(source_profile, config)
    return (
        int(delivery_profile["width"]),
        int(delivery_profile["height"]),
    ) != (
        int(source_profile["width"]),
        int(source_profile["height"]),
    )


def full_frame_filter(
    width: int,
    height: int,
    *,
    input_label: str = "0:v",
    output_label: str = "delivery",
) -> str:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise RuntimeError("delivery geometry must use positive even dimensions")

    blur_width = max(180, (width // 3) // 2 * 2)
    blur_height = max(320, (height // 3) // 2 * 2)
    return (
        f"[{input_label}]split=2[delivery_bg][delivery_fg];"
        f"[delivery_bg]scale={blur_width}:{blur_height}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={blur_width}:{blur_height},gblur=sigma=18,"
        f"scale={width}:{height}:flags=lanczos[delivery_bg2];"
        f"[delivery_fg]scale={width}:{height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos[delivery_fg2];"
        "[delivery_bg2][delivery_fg2]"
        f"overlay=(W-w)/2:(H-h)/2:shortest=1,"
        f"setsar=1[{output_label}]"
    )


def _portrait_blurred_filter(
    *,
    input_label: str,
    output_label: str,
    width: int,
    height: int,
    visual_left: int,
    visual_top: int,
    visual_width: int,
    visual_height: int,
    blur_sigma: float,
) -> str:
    if blur_sigma <= 0:
        raise RuntimeError("portrait_layout.background_blur_sigma must be positive")
    blur_width = max(180, (width // 3) // 2 * 2)
    blur_height = max(320, (height // 3) // 2 * 2)
    return (
        f"[{input_label}]split=2[portrait_bg][portrait_fg];"
        f"[portrait_bg]scale={blur_width}:{blur_height}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={blur_width}:{blur_height},"
        f"gblur=sigma={blur_sigma:g},"
        f"scale={width}:{height}:flags=lanczos[portrait_bg2];"
        f"[portrait_fg]scale={visual_width}:{visual_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos"
        "[portrait_fg2];"
        "[portrait_bg2][portrait_fg2]"
        f"overlay={visual_left}:{visual_top}:shortest=1,"
        f"setsar=1[{output_label}]"
    )


def composition_filter(
    source_profile: dict[str, Any],
    config: dict[str, Any],
    *,
    input_label: str = "0:v",
    output_label: str = "delivery",
) -> tuple[str, dict[str, Any]]:
    profile = profile_for_output(source_profile, config)
    width = int(profile["width"])
    height = int(profile["height"])
    layout = portrait_layout.resolve(config, width, height)
    if layout is None:
        return full_frame_filter(
            width,
            height,
            input_label=input_label,
            output_label=output_label,
        ), {
            "mode": "full_frame_fit_blurred_background",
            "background_mode": "blurred_source",
            "source_foreground_full_frame": True,
            "source_foreground_crop_used": False,
        }

    visual_width = int(layout["visual_width"])
    visual_height = int(layout["visual_height"])
    visual_left = int(layout["visual_left"])
    visual_top = int(layout["visual_top"])
    background_mode = str(layout.get("background_mode") or "solid")

    if background_mode == "blurred_source":
        blur_sigma = float(layout.get("background_blur_sigma", 18.0))
        graph = _portrait_blurred_filter(
            input_label=input_label,
            output_label=output_label,
            width=width,
            height=height,
            visual_left=visual_left,
            visual_top=visual_top,
            visual_width=visual_width,
            visual_height=visual_height,
            blur_sigma=blur_sigma,
        )
        return graph, {
            "mode": "portrait_matte_full_frame",
            "background_mode": background_mode,
            "background_blur_sigma": blur_sigma,
            "source_foreground_full_frame": True,
            "source_foreground_crop_used": False,
            "portrait_layout": layout,
        }

    if background_mode != "solid":
        raise RuntimeError("portrait_layout.background_mode must be 'solid' or 'blurred_source'")

    background = str(layout.get("background_hex") or "#0F1115").lstrip("#")
    graph = (
        f"[{input_label}]scale={visual_width}:{visual_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:{visual_left}:{visual_top}:"
        f"color=0x{background},setsar=1[{output_label}]"
    )
    return graph, {
        "mode": "portrait_matte_full_frame",
        "background_mode": background_mode,
        "background_hex": f"#{background}",
        "source_foreground_full_frame": True,
        "source_foreground_crop_used": False,
        "portrait_layout": layout,
    }
