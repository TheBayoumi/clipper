from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve(
    config: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any] | None:
    raw = dict(config.get("output", {}).get("portrait_layout") or {})
    if raw.get("enabled") is not True:
        return None
    if int(raw.get("width", width)) != width or int(raw.get("height", height)) != height:
        raise RuntimeError("portrait layout geometry differs from configured delivery geometry")

    visual_left = int(raw["visual_left"])
    visual_top = int(raw["visual_top"])
    visual_width = int(raw["visual_width"])
    visual_height = int(raw["visual_height"])
    title_top_height = int(raw["title_top_height"])
    if (
        visual_left < 0
        or visual_top < 0
        or visual_width <= 0
        or visual_height <= 0
        or visual_left + visual_width > width
        or visual_top + visual_height > height
        or title_top_height <= 0
        or title_top_height > visual_top
    ):
        raise RuntimeError("portrait layout leaves the configured delivery canvas")

    positions = dict(raw.get("title_y_positions") or {})
    font_sizes = dict(raw.get("title_font_sizes") or {})
    for count in ("1", "2", "3"):
        ys = [int(item) for item in positions.get(count) or []]
        sizes = [int(item) for item in font_sizes.get(count) or []]
        if len(ys) != int(count) or len(sizes) != int(count):
            raise RuntimeError(f"portrait title calibration for {count} lines is invalid")
        if any(value <= 0 or value >= title_top_height for value in ys):
            raise RuntimeError("portrait title y-position leaves the title-safe region")
        if any(value < 8 for value in sizes):
            raise RuntimeError("portrait title font calibration is too small")

    return {
        **raw,
        "width": width,
        "height": height,
        "visual_left": visual_left,
        "visual_top": visual_top,
        "visual_width": visual_width,
        "visual_height": visual_height,
        "title_top_height": title_top_height,
        "title_y_positions": positions,
        "title_font_sizes": font_sizes,
    }


def _font(font_path: Path, size: int) -> Any:
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise RuntimeError("portrait title layout requires Pillow") from exc
    return ImageFont.truetype(str(font_path), size)


def _partitions(words: list[str], count: int) -> list[list[str]]:
    if count == 1:
        return [[" ".join(words)]]
    if count == 2:
        return [
            [" ".join(words[:index]), " ".join(words[index:])]
            for index in range(1, len(words))
        ]
    if count == 3:
        return [
            [
                " ".join(words[:left]),
                " ".join(words[left:right]),
                " ".join(words[right:]),
            ]
            for left in range(1, len(words) - 1)
            for right in range(left + 1, len(words))
        ]
    return []


def title_lines(text: str, font_path: Path, layout: dict[str, Any]) -> list[dict[str, Any]]:
    words = text.strip().split()
    if not words:
        raise RuntimeError("portrait title is empty")
    safe_width = int(
        int(layout["width"]) * float(layout.get("title_max_width_fraction", 0.88))
    )
    positions = dict(layout["title_y_positions"])
    font_sizes = dict(layout["title_font_sizes"])

    for count in range(1, 4):
        sizes = [int(value) for value in font_sizes[str(count)]]
        candidates: list[tuple[float, list[dict[str, Any]]]] = []
        for partition in _partitions(words, count):
            rows: list[dict[str, Any]] = []
            widths: list[float] = []
            valid = True
            for index, line in enumerate(partition):
                font = _font(font_path, sizes[index])
                width = float(font.getlength(line))
                if width > safe_width:
                    valid = False
                    break
                widths.append(width)
                rows.append(
                    {
                        "text": line,
                        "font_size": sizes[index],
                        "y": int(positions[str(count)][index]),
                        "width": round(width, 3),
                    }
                )
            if valid:
                balance = max(widths) - min(widths) if widths else 0.0
                candidates.append((max(widths, default=0.0) + 0.20 * balance, rows))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    raise RuntimeError("portrait title cannot fit the calibrated three-line safe region")
