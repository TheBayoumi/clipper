#!/usr/bin/env python3
"""Cookie-free original YouTube egress probe on the user's existing Modal account.

GitHub Actions controls the task. Modal supplies a different network path,
not a different video source. No original bytes or browser cookies are uploaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

app = modal.App("clipper-tjr-official-youtube-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip", "ca-certificates", "ffmpeg")
    .run_commands(
        "curl -fsSL https://deno.land/install.sh | sh",
        "ln -sf /root/.deno/bin/deno /usr/local/bin/deno",
    )
    .pip_install("yt-dlp[default]>=2026.7.4,<2027")
)


volume = modal.Volume.from_name("clipper-tjr-source-transport", create_if_missing=True)


@app.function(
    image=image, volumes={"/tjr-media": volume}, timeout=1600, cpu=2, memory=2048
)
def inspect_original_youtube(
    candidates: list[dict[str, str]], run_key: str = ""
) -> dict[str, Any]:
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
            # Formats advertised in metadata are not proof that the signed
            # GoogleVideo endpoint actually serves bytes on this network.
            import tempfile

            with tempfile.TemporaryDirectory(prefix="tjr-modal-youtube-") as scratch:
                with yt_dlp.YoutubeDL(
                    {
                        "quiet": True,
                        "no_warnings": True,
                        "noplaylist": True,
                        "test": True,
                        "format": "bv*[height>=720]/b[height>=720]",
                        "outtmpl": str(Path(scratch) / "original.%(ext)s"),
                        "retries": 1,
                        "socket_timeout": 20,
                    }
                ) as probe:
                    transfer_status = probe.download([url])
                actual_bytes = sum(
                    path.stat().st_size
                    for path in Path(scratch).glob("original.*")
                    if path.is_file()
                )
                if transfer_status != 0 or actual_bytes < 1024:
                    attempts.append({"url": url, "reason": "HD_MEDIA_BYTES_NOT_VERIFIED"})
                    continue
            result: dict[str, Any] = {
                "status": "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED",
                "source_url": url,
                "source_video_id": video_id,
                "source_channel_id": channel_id,
                "duration": float(metadata.get("duration") or 0),
                "title": str(metadata.get("title") or "")[:160],
                "attempts": attempts,
            }
            # Keep playback validation and full download on the SAME egress IP.
            # A second .remote() call may be assigned to a blocked worker.
            if run_key:
                result["staging"] = stage_official_original.local(result, run_key)
            return result
        except Exception as exc:
            reason = (
                "YOUTUBE_IP_OR_LOGIN_CHALLENGE"
                if "Sign in to confirm" in str(exc)
                else type(exc).__name__
            )
            attempts.append({"url": url, "reason": reason})
    return {"status": "NO_ACCESSIBLE_ORIGINAL_YOUTUBE", "attempts": attempts}


@app.function(image=image, volumes={"/tjr-media": volume}, timeout=1600, cpu=2, memory=2048)
def stage_official_original(selected: dict[str, Any], run_key: str) -> dict[str, Any]:
    """Download actual allowlisted original bytes via the verified Modal network."""
    import hashlib
    import re
    import subprocess
    from pathlib import Path

    video_id = str(selected["source_video_id"])
    channel_id = str(selected["source_channel_id"])
    if (
        channel_id not in {"UCGHBUXjDCeiIXNdKR0HUZnA", "UCZen39LQJPx04GjPj7FOMcw"}
        or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id)
        or not re.fullmatch(r"\d{4,20}-\d{1,4}", run_key)
    ):
        raise RuntimeError("staging rejected an unapproved channel, ID or run identifier")
    url = f"https://www.youtube.com/watch?v={video_id}"
    if url != selected.get("source_url"):
        raise RuntimeError("original source URL mismatch")
    folder = Path("/tjr-media") / "runs" / run_key
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--merge-output-format",
        "mp4",
        "--download-sections",
        "*00:00:00-00:14:00",
        "-f",
        "bv*[height>=720][height<=1080]+ba/b[height>=720]/bv*+ba/b",
        "--no-part",
        "-o",
        str(folder / "original.%(ext)s"),
        url,
    ]
    try:
        outcome = subprocess.run(command, capture_output=True, timeout=1330, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("original YouTube excerpt download exceeded the remote limit") from exc
    if outcome.returncode:
        raise RuntimeError(
            f"original YouTube download failed on verified network: exit {outcome.returncode}"
        )
    source_files = [
        item
        for item in folder.glob("original.*")
        if item.is_file() and item.suffix.lower() in {".mp4", ".mkv", ".webm"}
    ]
    if len(source_files) != 1:
        raise RuntimeError("source download did not produce exactly one media file")
    original = source_files[0]
    inspect = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(original),
        ],
        capture_output=True,
        text=True,
        timeout=50,
        check=True,
    )
    streams = json.loads(inspect.stdout)["streams"]
    if not any(
        stream.get("codec_type") == "video" and int(stream.get("height") or 0) >= 720
        for stream in streams
    ):
        raise RuntimeError("staged original has no HD video stream")
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise RuntimeError("staged original is missing its original audio")
    with original.open("rb") as media:
        digest = hashlib.file_digest(media, "sha256").hexdigest()
    volume.commit()
    return {
        "status": "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED",
        "video_id": video_id,
        "channel_id": channel_id,
        "public_video_url": url,
        "source_remote_path": f"runs/{run_key}/{original.name}",
        "source_sha256": digest,
        "size_bytes": original.stat().st_size,
        "title": str(selected.get("title") or ""),
        "duration": float(selected.get("duration") or 0),
    }


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
        stage_media = os.getenv("TJR_MODAL_STAGE_ORIGINAL") == "1"
        run_key = (
            os.environ["GITHUB_RUN_ID"]
            + "-"
            + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
            if stage_media
            else ""
        )
        region_attempts: list[dict[str, str]] = []
        result: dict[str, Any] = {
            "status": "NO_ACCESSIBLE_ORIGINAL_YOUTUBE",
            "attempts": [],
        }
        # Regional egress is tested at most three times; stop immediately on
        # a real HD transfer. Never treat a metadata-only hit as successful.
        for region in (None, "eu-west", "us-west"):
            provider = (
                inspect_original_youtube
                if region is None
                else inspect_original_youtube.with_options(region=region)
            )
            try:
                candidate = provider.remote(inputs, run_key)
                result = candidate
                if candidate.get("status") == "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED" and (
                    not stage_media
                    or candidate.get("staging", {}).get("status")
                    == "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED"
                ):
                    result["selected_modal_region"] = region or "default"
                    break
                region_attempts.append(
                    {"region": region or "default", "status": str(candidate.get("status"))}
                )
            except Exception as exc:
                region_attempts.append(
                    {"region": region or "default", "error": type(exc).__name__}
                )
        result["region_attempts"] = region_attempts
        result["discovery_failures"] = discovery_failures
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result["status"] != "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED":
            raise RuntimeError(
                "No tested Modal region could fetch HD bytes from the official YouTube video"
            )
        print("Verified original YouTube URL:", result["source_url"])
        if stage_media:
            staging = result.get("staging")
            if (
                not isinstance(staging, dict)
                or staging.get("status") != "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED"
            ):
                raise RuntimeError("No same-worker verified HD original was staged")
            (root / "staged-original.json").write_text(
                json.dumps(staging, indent=2) + "\n",
                encoding="utf-8",
            )
            print("Verified original media staged on the same Modal worker")
    except Exception:
        if not output.is_file():
            output.write_text(
                json.dumps({"status": "MODAL_PROBE_FAILED_BEFORE_REMOTE_RESULT"}, indent=2) + "\n",
                encoding="utf-8",
            )
        raise
