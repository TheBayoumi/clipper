#!/usr/bin/env python3
"""Cookie-free original YouTube egress probe on the user's existing Modal account.

GitHub Actions controls the task. Modal supplies a different network path,
not a different video source. No original bytes or browser cookies are uploaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

app = modal.App("clipper-tjr-official-youtube-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deno.land/install.sh | sh",
        "ln -sf /root/.deno/bin/deno /usr/local/bin/deno",
    )
    .pip_install("yt-dlp[default]>=2026.7.4,<2027")
)


@app.function(image=image, timeout=180, cpu=1)
def inspect_original_youtube(candidates: list[dict[str, str]]) -> dict[str, Any]:
    """Inspect only feed-listed original video IDs and their exact expected owners."""
    import yt_dlp

    approved = {
        "UCGHBUXjDCeiIXNdKR0HUZnA",
        "UCZen39LQJPx04GjPj7FOMcw",
    }
    attempts: list[dict[str, str]] = []
    for candidate in candidates[:3]:
        video_id = candidate["video_id"]
        channel_id = candidate["channel_id"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        if channel_id not in approved or len(video_id) != 11:
            attempts.append({"url": url, "reason": "SOURCE_NOT_ALLOWLISTED"})
            continue
        try:
            with yt_dlp.YoutubeDL(
                {
                    "quiet": True,
                    "no_warnings": True,
                    "skip_download": True,
                    "noplaylist": True,
                    "socket_timeout": 20,
                    "retries": 1,
                }
            ) as downloader:
                metadata = downloader.extract_info(url, download=False)
            if not isinstance(metadata, dict):
                raise RuntimeError("YouTube returned no usable metadata")
            if metadata.get("id") != video_id or metadata.get("channel_id") != channel_id:
                attempts.append({"url": url, "reason": "VIDEO_OWNER_MISMATCH"})
                continue
            if metadata.get("live_status") == "is_live":
                attempts.append({"url": url, "reason": "ONGOING_LIVESTREAM"})
                continue
            if float(metadata.get("duration") or 0) < 90:
                attempts.append({"url": url, "reason": "SHORT_OR_UNKNOWN_DURATION"})
                continue
            formats = metadata.get("formats") or []
            hd = any(
                isinstance(item, dict)
                and int(item.get("height") or 0) >= 720
                and item.get("vcodec") not in ("none", None)
                and str(item.get("url") or "").startswith("https://")
                for item in formats
            )
            if not hd:
                attempts.append({"url": url, "reason": "NO_DOWNLOADABLE_HD_FORMAT"})
                continue
            return {
                "status": "EXACT_OFFICIAL_YOUTUBE_HD_ACCESSIBLE",
                "source_url": url,
                "source_video_id": video_id,
                "source_channel_id": channel_id,
                "duration": float(metadata.get("duration") or 0),
                "title": str(metadata.get("title") or "")[:160],
                "attempts": attempts,
            }
        except Exception as exc:
            reason = (
                "YOUTUBE_IP_OR_LOGIN_CHALLENGE"
                if "Sign in to confirm" in str(exc)
                else type(exc).__name__
            )
            attempts.append({"url": url, "reason": reason})
    return {"status": "NO_ACCESSIBLE_ORIGINAL_YOUTUBE", "attempts": attempts}


@app.local_entrypoint()
def main() -> None:
    from scripts.tjr_youtube_preview import (
        constrain_official_sources,
        discover_official_uploads,
    )

    root = Path("tjr-modal-probe")
    root.mkdir(parents=True, exist_ok=True)
    output = root / "verified-original-egress.json"
    try:
        candidates, discovery_failures = discover_official_uploads()
        official = constrain_official_sources(candidates, requested_id=None)
        inputs = [
            {"video_id": item.video_id, "channel_id": item.channel_id} for item in official[:3]
        ]
        if not inputs:
            raise RuntimeError("No feed-confirmed campaign YouTube videos")
        result = inspect_original_youtube.remote(inputs)
        result["discovery_failures"] = discovery_failures
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result["status"] != "EXACT_OFFICIAL_YOUTUBE_HD_ACCESSIBLE":
            raise RuntimeError("Modal network also cannot fetch these official original videos")
        print("Verified original YouTube URL:", result["source_url"])
    except Exception:
        if not output.is_file():
            output.write_text(
                json.dumps({"status": "MODAL_PROBE_FAILED_BEFORE_REMOTE_RESULT"}, indent=2) + "\n",
                encoding="utf-8",
            )
        raise
