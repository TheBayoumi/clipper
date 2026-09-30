"""Shared Clipper portrait source-band layout contract.

Campaign profiles calibrate one portrait layout. Rendering capabilities consume the
resolved geometry; they do not redefine source placement or title-safe regions.
"""

from __future__ import annotations

from typing import Any

from ..montage import MontageRejection
from ..profiles import CampaignProfile


def resolve(
    profile: CampaignProfile,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    """Resolve and validate the campaign-wide portrait layout for an output canvas."""
    matte = profile.config["output"]["portrait_matte"]
    raw = matte.get("layout")
    if raw is None:
        return None
    expected = {
        "visual_left",
        "visual_top",
        "visual_width",
        "visual_height",
        "title_top_height",
        "title_y_positions",
        "title_bar_y",
        "title_font_sizes",
        "title_pill_heights",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise MontageRejection("portrait_geometry", "unsupported shared portrait calibration")

    calibration_width = int(raw["visual_width"])
    if calibration_width < 1:
        raise MontageRejection("portrait_geometry", "portrait calibration width is invalid")
    scale = width / calibration_width
    values: dict[str, Any] = {
        "visual_left": round(int(raw["visual_left"]) * scale),
        "visual_top": round(int(raw["visual_top"]) * scale),
        "visual_width": round(int(raw["visual_width"]) * scale),
        "visual_height": round(int(raw["visual_height"]) * scale),
        "title_top_height": round(int(raw["title_top_height"]) * scale),
        "title_y_positions": [round(int(n) * scale) for n in raw["title_y_positions"]],
        "title_bar_y": round(int(raw["title_bar_y"]) * scale),
        "title_font_sizes": [max(8, round(int(n) * scale)) for n in raw["title_font_sizes"]],
        "title_pill_heights": [round(int(n) * scale) for n in raw["title_pill_heights"]],
    }

    source_width = int(profile.config["output"]["width"])
    source_height = int(profile.config["output"]["height"])
    expected_source_height = round(width * source_height / source_width)
    expected_bottom = height - values["visual_top"] - values["visual_height"]
    if (
        values["visual_left"] != 0
        or values["visual_width"] != width
        or values["visual_top"] != values["title_top_height"]
        or values["visual_height"] != expected_source_height
        or values["visual_top"] < 1
        or expected_bottom < 1
        or len(values["title_y_positions"]) != 3
        or len(values["title_font_sizes"]) != 3
        or len(values["title_pill_heights"]) != 3
        or values["title_bar_y"] >= values["title_top_height"]
        or any(n < 0 or n >= values["title_top_height"] for n in values["title_y_positions"])
    ):
        raise MontageRejection(
            "portrait_geometry",
            "shared portrait band leaves the legal source-native 9:16 layout",
        )
    return {**values, "bottom_height": expected_bottom}
