"""Fail-closed check for the historical dark, AI-generated subtitle panel.

This is a conservative, layout-specific guard; it does not certify spelling.
Humans still inspect the physical Atomic Habits cover before publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageStat


def region_brightness(
    image: Image.Image, left: float, top: float, right: float, bottom: float
) -> float:
    w, h = image.size
    patch = image.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))
    return sum(ImageStat.Stat(patch).mean) / 3


def contains_old_subtitle_panel(frame: Path) -> bool:
    with Image.open(frame) as img:
        rgb = img.convert("RGB")
    if rgb.width < 700 or rgb.height < 1200:
        raise ValueError("book intro keyframe below expected 9:16 Qwen resolution")
    # The defective source had a black translucent strip over warm timber at
    # approximately 88-91% image height. Test three narrow horizontal bands,
    # rather than the whole scene where the character's dark hoodie is valid.
    for top in (0.885, 0.898, 0.907):
        middle = region_brightness(rgb, 0.39, top, 0.61, top + 0.008)
        left = region_brightness(rgb, 0.03, top, 0.11, top + 0.008)
        right = region_brightness(rgb, 0.88, top, 0.96, top + 0.008)
        if middle < 34 and max(left, right) > 95 and (left + right) / 2 > middle * 2:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, required=True)
    args = parser.parse_args()
    if contains_old_subtitle_panel(args.frame):
        raise SystemExit("::error::Qwen book frame contains baked caption panel; reject before GPU")
    print("BOOK_SOURCE_NO_OLD_BAKED_SUBTITLE_PANEL")
    print("BOOK_TITLE_SPELLING_AND_MOTION_REQUIRE_HUMAN_REVIEW")


if __name__ == "__main__":
    main()
