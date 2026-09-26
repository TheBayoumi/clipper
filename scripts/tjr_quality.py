#!/usr/bin/env python3
"""Fail-closed technical QA for the Reach TJR weekly preview artifacts.

This verifies media and campaign-brief invariants, not human editorial or
campaign approval. Never treat this report as permission to publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

OFFICIAL_TJR_CHANNEL = "UCGHBUXjDCeiIXNdKR0HUZnA"
OFFICIAL_TJR_CHANNELS = [OFFICIAL_TJR_CHANNEL, "UCZen39LQJPx04GjPj7FOMcw"]
EXPECTED_SIZE = (1080, 1920)
MIN_SECONDS = 20.0
MAX_SECONDS = 42.0


class QualityError(ValueError):
    """A clip or its campaign configuration failed mandatory technical QA."""


def check_campaign_brief(path: Path) -> dict[str, Any]:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise QualityError("campaign brief is not an object")
    channels = data.get("source_channel_ids")
    pinned_channel = (
        isinstance(channels, list)
        and len(channels) == 1
        and channels[0] in OFFICIAL_TJR_CHANNELS
        and bool(data.get("source_media_urls"))
    )
    if channels != OFFICIAL_TJR_CHANNELS and not pinned_channel:
        raise QualityError("source must be restricted to Reach's two listed TJR YouTube channels")
    video_ids = data.get("allowed_video_ids", [])
    if (
        not isinstance(video_ids, list)
        or not video_ids
        or not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_-]{11}", v) for v in video_ids)
        or len(video_ids) != len(set(video_ids))
    ):
        raise QualityError("allowed_video_ids must be unique YouTube video IDs")
    mirrors = data.get("source_media_urls") or {}
    if not isinstance(mirrors, dict):
        raise QualityError("source_media_urls must be a mapping")
    if mirrors:
        if len(video_ids) != 1 or set(mirrors) != set(video_ids):
            raise QualityError("mirrored media must match the pinned TJR video ID")
        for url in mirrors.values():
            parsed = urlparse(url) if isinstance(url, str) else None
            if (
                parsed is None
                or parsed.scheme != "https"
                or parsed.hostname != "drive.google.com"
                or not re.fullmatch(r"/file/d/[A-Za-z0-9_-]+/view/?", parsed.path)
            ):
                raise QualityError("mirror must be a Google Drive file/view HTTPS URL")
    if data.get("rights_confirmed") is not True:
        raise QualityError("campaign clipping permission has not been verified")
    if data.get("watermark_text") or data.get("watermark_url"):
        raise QualityError("TJR campaign prohibits added logos and watermarks")
    if "#TJR" not in data.get("required_hashtags", []):
        raise QualityError("mandatory #TJR hashtag is missing")
    if int(data.get("min_clip_seconds", -1)) != 20:
        raise QualityError("expected 20 second minimum clip length")
    if int(data.get("max_clip_seconds", -1)) != 42:
        raise QualityError("expected 42 second maximum clip length")
    return data


def _resolve_approved_youtube_video(video_id: str) -> str:
    """Check a selected original's ID against the TWO real official channel feeds."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise QualityError("TJR_SOURCE_VIDEO_ID must be an 11-character YouTube video ID")
    from scripts.tjr_youtube_preview import discover_official_uploads

    listings, _failures = discover_official_uploads()
    channels = {item.channel_id for item in listings if item.video_id == video_id}
    if len(channels) != 1 or not channels.issubset(set(OFFICIAL_TJR_CHANNELS)):
        raise QualityError(
            "selected video ID was not confirmed in either current Reach-listed "
            "official YouTube channel feed; do not substitute an unrelated mirror"
        )
    return channels.pop()


def prepare_staged_brief(template: Path, output: Path) -> Path:
    """Prepare a source-ID- and SHA-pinned brief after explicit user verification."""
    if os.getenv("TJR_BUDGET_CONFIRMED") != "true" or os.getenv("TJR_SOURCE_VERIFIED") != "true":
        raise QualityError("confirm live campaign budget and authentic TJR source")
    sha = os.getenv("TJR_SOURCE_MEDIA_SHA256", "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        raise QualityError("TJR_SOURCE_MEDIA_SHA256 must be a 64-character hex digest")
    data = check_campaign_brief(template)
    media_url = os.getenv("TJR_SOURCE_MEDIA_URL", "").strip()
    # Reject malformed mirror URLs before making external source-discovery requests.
    if not re.fullmatch(r"https://drive[.]google[.]com/file/d/[A-Za-z0-9_-]+/view/?(?:[?].*)?", media_url):
        raise QualityError("mirror must be a Google Drive file/view HTTPS URL")
    video_id = os.getenv("TJR_SOURCE_VIDEO_ID", "").strip()
    channel_id = _resolve_approved_youtube_video(video_id)
    data["source_channel_ids"] = [channel_id]
    data["allowed_video_ids"] = [video_id]
    data["source_media_urls"] = {video_id: media_url}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        check_campaign_brief(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def probe_original(path: Path) -> dict[str, int]:
    """Reject source files below 720p before delivering 1080p output."""
    if not path.is_file() or path.stat().st_size == 0:
        raise QualityError("downloaded original TJR source is missing")
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        streams = json.loads(probe.stdout)["streams"]
        video = next(item for item in streams if item.get("codec_type") == "video")
        width, height = int(video["width"]), int(video["height"])
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise QualityError("unable to probe staged original footage") from exc
    except (ValueError, KeyError, TypeError, StopIteration) as exc:
        raise QualityError("invalid original video metadata") from exc
    if min(width, height) < 720 or max(width, height) < 1280:
        raise QualityError("original footage is below 720p HD")
    return {"width": width, "height": height}


def probe_video(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise QualityError(f"clip is empty or missing: {path}")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise QualityError(f"ffprobe failed for {path.name}: {exc}") from exc
    try:
        info = json.loads(result.stdout)
        videos = [s for s in info["streams"] if s["codec_type"] == "video"]
        audios = [s for s in info["streams"] if s["codec_type"] == "audio"]
        if len(videos) != 1 or len(audios) < 1:
            raise QualityError("clip needs exactly one video stream and an audio stream")
        video = videos[0]
        audio = audios[0]
        if (video["width"], video["height"]) != EXPECTED_SIZE:
            raise QualityError("clip must be 1080x1920 true 9:16")
        if video["codec_name"] != "h264" or video.get("pix_fmt") != "yuv420p":
            raise QualityError("clip must be broadly compatible H.264 yuv420p")
        if audio["codec_name"] != "aac":
            raise QualityError("clip audio must use AAC")
        if Fraction(video.get("sample_aspect_ratio", "1:1").replace(":", "/")) != 1:
            raise QualityError("clip must have square pixels")
        fps = Fraction(video.get("avg_frame_rate", "0/1"))
        if not 29 <= float(fps) <= 61:
            raise QualityError("clip frame rate must be between 29 and 61 fps")
        duration = float(info["format"]["duration"])
        if not MIN_SECONDS - 0.5 <= duration <= MAX_SECONDS + 0.5:
            raise QualityError(f"clip duration outside {MIN_SECONDS}-{MAX_SECONDS}s: {duration}")
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        if isinstance(exc, QualityError):
            raise
        raise QualityError(f"invalid ffprobe metadata: {exc}") from exc
    return {
        "file": path.name,
        "width": video["width"],
        "height": video["height"],
        "fps": float(fps),
        "duration_seconds": round(duration, 2),
        "video_codec": video["codec_name"],
        "audio_codec": audio["codec_name"],
        "bytes": path.stat().st_size,
    }


def check_full_decode(path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise QualityError(f"full video/audio decode failed: {path.name}: {exc}") from exc


def validate_artifacts(brief: Path, artifact_root: Path) -> dict[str, Any]:
    config = check_campaign_brief(brief)
    manifests = sorted(artifact_root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise QualityError(f"expected exactly one manifest; got {len(manifests)}")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise QualityError(f"pipeline errors: {manifest['errors']}")
    discovered = manifest.get("discovered_videos", [])
    if not discovered or any(
        video.get("channel_id") not in OFFICIAL_TJR_CHANNELS for video in discovered
    ):
        raise QualityError("all discovered sources must be one of Reach's TJR YouTube channels")
    planned = manifest.get("planned_clips") or []
    rendered = manifest.get("rendered_clips") or []
    if not planned or len(planned) != len(rendered):
        raise QualityError("every planned clip must render and at least one is required")
    run_dir = manifest_path.parent.resolve()
    source_details: dict[str, Any] = {"mode": "public YouTube"}
    mirrors = config.get("source_media_urls") or {}
    if mirrors:
        expected = os.getenv("TJR_SOURCE_MEDIA_SHA256", "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise QualityError("staged source SHA-256 secret is missing")
        video_id = config["allowed_video_ids"][0]
        original = run_dir / "work" / video_id / "source.mp4"
        if not original.is_file():
            raise QualityError("staged original not found in campaign run")
        with original.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise QualityError("staged original hash differs from approved media")
        source_details = {
            "mode": "SHA-256 pinned mirror",
            "sha256": actual,
            "source_video_id": video_id,
            "source_channel_id": config["source_channel_ids"][0],
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
        }
        source_details.update(probe_original(original))
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for item in rendered:
        output = Path(item["output_path"]).resolve()
        if not output.is_relative_to(run_dir / "clips"):
            raise QualityError("manifest clip path escapes the run's clips directory")
        source_id = item.get("video_id", "")
        source_url = item.get("source_url", "")
        if f"watch?v={source_id}" not in source_url:
            raise QualityError("clip has no verifiable YouTube source link")
        identity = (source_id, float(item["start"]), float(item["end"]))
        if identity in seen:
            raise QualityError("duplicate source time range in run")
        seen.add(identity)
        subtitles = output.with_suffix(".srt")
        if not subtitles.is_file() or not subtitles.read_text(encoding="utf-8").strip():
            raise QualityError(f"missing burned-caption SRT: {subtitles.name}")
        inspected = probe_video(output)
        check_full_decode(output)
        inspected.update(
            {
                "source_url": source_url,
                "source_start": float(item["start"]),
                "source_end": float(item["end"]),
            }
        )
        results.append(inspected)
    return {
        "status": "TECHNICAL_QA_PASSED__HUMAN_REVIEW_REQUIRED",
        "campaign": "reach-tjr-weekly",
        "source": source_details,
        "clips": results,
        "manual_checks": [
            "Confirm TJR appears in every video, and clips preserve the original context.",
            "Inspect every frame for pre-existing source logos and forbidden watermarks.",
            "Review subtitle accuracy, framing, hook, audio and financial claims.",
            "Verify source permissions, live campaign budget, #TJR, account branding and audience.",
            "Publish manually; submit the public URL to Whop within 30 minutes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output-brief", type=Path)
    args = parser.parse_args()
    try:
        if args.prepare:
            if args.output_brief is None:
                raise QualityError("--output-brief is required for --prepare")
            prepare_staged_brief(args.brief, args.output_brief)
            print("Verified staged-source brief ready (media URL not logged).")
            return 0
        if args.artifact_root is None:
            raise QualityError("--artifact-root is required for QA")
        report = validate_artifacts(args.brief, args.artifact_root)
    except (QualityError, OSError, json.JSONDecodeError) as exc:
        print(f"TJR QA FAILED: {exc}", file=sys.stderr)
        return 1
    manifests = list(args.artifact_root.glob("*/manifest.json"))
    output = manifests[0].parent / "tjr-qa-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
