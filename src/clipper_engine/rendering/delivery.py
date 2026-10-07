from __future__ import annotations

from fractions import Fraction
from typing import Any

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
    width = _dimension(settings.get("width"), int(source_profile["width"]), label="width")
    height = _dimension(settings.get("height"), int(source_profile["height"]), label="height")
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
        f"overlay=(W-w)/2:(H-h)/2:shortest=1,setsar=1[{output_label}]"
    )
