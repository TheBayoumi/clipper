#!/usr/bin/env python3
"""Try the actual official YouTube watch page in runner Chrome, without account cookies.

This is a supplemental transport diagnostic and original-media attempt. It
never treats successful channel discovery or generated tokens as a video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from scripts.tjr_quality import probe_original
from scripts.tjr_youtube_preview import (
    CHANNELS,
    OfficialVideo,
    constrain_official_sources,
    discover_official_uploads,
)

LOGGER = logging.getLogger("tjr-browser")
PLAYER = re.compile(r"(?:var\s+)?ytInitialPlayerResponse\s*=\s*")
MAX_HTML_BYTES = 9_000_000


def parse_player_response(html: str) -> dict[str, object] | None:
    """Extract original player JSON from Chrome's rendered YouTube watch DOM."""
    for match in PLAYER.finditer(html):
        try:
            payload, _ = json.JSONDecoder().raw_decode(html[match.end() :])
            if isinstance(payload, dict):
                return payload
        except ValueError:
            continue
    return None


def trusted_media_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and (host == "googlevideo.com" or host.endswith(".googlevideo.com"))
        and parsed.path.startswith("/videoplayback")
    )


def select_browser_formats(
    player: dict[str, object],
) -> tuple[str, str] | None:
    """Accept ONLY HTTPS GoogleVideo direct streams from the verified watch page."""
    streams = player.get("streamingData")
    if not isinstance(streams, dict):
        return None
    entries = streams.get("adaptiveFormats")
    if not isinstance(entries, list):
        return None
    formats = [item for item in entries if isinstance(item, dict)]
    video = sorted(
        (
            item
            for item in formats
            if trusted_media_url(item.get("url"))
            and str(item.get("mimeType", "")).startswith("video/")
            and int(item.get("height") or 0) >= 720
        ),
        key=lambda item: (
            item.get("mimeType", "").startswith('video/mp4; codecs="avc1'),
            int(item.get("height") or 0),
            int(item.get("bitrate") or 0),
        ),
        reverse=True,
    )
    audio = sorted(
        (
            item
            for item in formats
            if trusted_media_url(item.get("url"))
            and str(item.get("mimeType", "")).startswith("audio/")
        ),
        key=lambda item: (
            str(item.get("mimeType", "")).startswith("audio/mp4"),
            int(item.get("bitrate") or 0),
        ),
        reverse=True,
    )
    if video and audio:
        return str(video[0]["url"]), str(audio[0]["url"])
    return None


def browser_html(video: OfficialVideo, browser: Path, profile: Path) -> str:
    profile.mkdir(parents=True, exist_ok=True)
    command = [
        str(browser),
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--virtual-time-budget=6500",
        f"--user-data-dir={profile}",
        "--dump-dom",
        video.url,
    ]
    result = subprocess.run(command, capture_output=True, timeout=65, check=False)
    if result.returncode:
        raise RuntimeError(f"Chrome watch page failed with exit {result.returncode}")
    if len(result.stdout) > MAX_HTML_BYTES:
        raise RuntimeError("Chrome returned oversized watch page")
    return result.stdout.decode("utf-8", errors="replace")


def fetch_browser_original(video_url: str, audio_url: str, output: Path) -> str:
    """Fetch only the limited 14-minute excerpt; never print signed media URLs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-headers",
        "Referer: https://www.youtube.com/\r\n",
        "-i",
        video_url,
        "-headers",
        "Referer: https://www.youtube.com/\r\n",
        "-i",
        audio_url,
        "-t",
        "840",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        str(output),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=1100, check=False)
        if result.returncode != 0:
            return f"googlevideo_fetch_failed_exit_{result.returncode}"
        probe_original(output)
        return "CAPTURED_HD_ORIGINAL"
    except subprocess.TimeoutExpired:
        return "googlevideo_download_timeout"
    except (RuntimeError, ValueError):
        return "downloaded_media_not_verified_HD"


def run_probe(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    report = root / "browser-probe.json"
    source_manifest = root / "browser-source.json"
    results: list[dict[str, str]] = []
    browser_name = os.environ.get("YT_DLP_WPC_BROWSER_PATH", "").strip()
    browser = Path(browser_name)
    if not browser.is_file():
        report.write_text(
            json.dumps({"status": "MISSING_CHROME", "attempts": []}, indent=2) + "\n"
        )
        return report
    candidates, failures = discover_official_uploads()
    requested = os.environ.get("TJR_SOURCE_VIDEO_ID", "").strip()
    selected = constrain_official_sources(candidates, requested)[:2]
    for index, video in enumerate(selected):
        record = {
            "source_url": video.url,
            "feed_channel_id": video.channel_id,
            "browser_playability": "UNAVAILABLE",
        }
        try:
            html = browser_html(video, browser, root / f"browser-profile-{index}")
            player = parse_player_response(html)
            if player is None:
                record["browser_playability"] = "NO_PLAYER_JSON"
                results.append(record)
                continue
            playability = player.get("playabilityStatus")
            details = player.get("videoDetails")
            status = (
                str(playability.get("status") or "UNKNOWN")
                if isinstance(playability, dict)
                else "UNKNOWN"
            )
            record["browser_playability"] = status
            if not isinstance(details, dict):
                results.append(record)
                continue
            if (
                details.get("videoId") != video.video_id
                or details.get("channelId") != video.channel_id
                or video.channel_id not in CHANNELS
            ):
                record["browser_playability"] = "REJECTED_SOURCE_OWNER_MISMATCH"
                results.append(record)
                continue
            if status != "OK":
                results.append(record)
                continue
            formats = select_browser_formats(player)
            if formats is None:
                record["browser_playability"] = "PLAYABLE_BUT_NO_DIRECT_HD_FORMATS"
                results.append(record)
                continue
            destination = root / "work" / f"{video.video_id}.mkv"
            transfer_status = fetch_browser_original(*formats, destination)
            record["browser_transfer"] = transfer_status
            results.append(record)
            if transfer_status != "CAPTURED_HD_ORIGINAL":
                destination.unlink(missing_ok=True)
                continue
            with destination.open("rb") as media:
                digest = hashlib.file_digest(media, "sha256").hexdigest()
            source_manifest.write_text(
                json.dumps(
                    {
                        "video_id": video.video_id,
                        "channel_id": video.channel_id,
                        "public_video_url": video.url,
                        "published": video.published,
                        "title": str(details.get("title") or video.title),
                        "duration": int(details.get("lengthSeconds") or 0),
                        "source_path": str(destination.resolve()),
                        "source_sha256": digest,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            break
        except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as exc:
            record["browser_playability"] = type(exc).__name__
            results.append(record)
    state = "CAPTURED_HD_ORIGINAL" if source_manifest.is_file() else "NO_BROWSER_ORIGINAL"
    report.write_text(
        json.dumps(
            {
                "status": state,
                "attempts": results,
                "discovery_failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Chrome original acquisition: %s", state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_probe(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
