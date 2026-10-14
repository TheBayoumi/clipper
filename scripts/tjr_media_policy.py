"""Shared TJR source-media eligibility policy.

This module is deliberately dependency-free so the exact same HD predicate can
run on GitHub and inside the isolated Modal acquisition worker.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MIN_HD_SHORT_SIDE = 720
MIN_HD_LONG_SIDE = 1280


def production_hd_dimensions(width: int, height: int) -> bool:
    """Return True only for a real >=1280x720 or >=720x1280 video frame."""
    if width <= 0 or height <= 0:
        return False
    short_side, long_side = sorted((int(width), int(height)))
    return short_side >= MIN_HD_SHORT_SIDE and long_side >= MIN_HD_LONG_SIDE


def production_hd_format(item: Mapping[str, Any]) -> bool:
    """Apply the same production-HD rule to one yt-dlp format record."""
    if item.get("vcodec") in ("none", None):
        return False
    try:
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
    except (TypeError, ValueError):
        return False
    return production_hd_dimensions(width, height)


def has_production_hd_video_stream(streams: object) -> bool:
    """Apply the shared HD rule to ffprobe stream metadata."""
    if not isinstance(streams, list):
        return False
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        try:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if production_hd_dimensions(width, height):
            return True
    return False
