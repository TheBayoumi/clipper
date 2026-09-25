#!/usr/bin/env python3
"""Fail-closed technical QA for the Reach TJR weekly preview artifacts.

This verifies media and campaign-brief invariants, not human editorial or
campaign approval. Never treat this report as permission to publish.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

OFFICIAL_TJR_CHANNEL = "UCGHBUXjDCeiIXNdKR0HUZnA"
EXPECTED_SIZE = (1080, 1920)
MIN_SECONDS = 20.0
MAX_SECONDS = 42.0


class QualityError(ValueError):
    """A clip or its campaign configuration failed mandatory technical QA."""


def check_campaign_brief(path: Path) -> dict[str, Any]:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise QualityError("campaign brief is not an object")
    if data.get("source_channel_ids") != [OFFICIAL_TJR_CHANNEL]:
        raise QualityError("source must be restricted to TJR's verified YouTube channel")
    video_ids = data.get("allowed_video_ids", [])
    if (
        not isinstance(video_ids, list)
        or not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_-]{11}", v) for v in video_ids)
        or len(video_ids) != len(set(video_ids))
    ):
        raise QualityError("allowed_video_ids must be unique YouTube video IDs")
    if data.get("source_media_urls"):
        raise QualityError("unverified direct source media is not permitted")
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
    check_campaign_brief(brief)
    manifests = sorted(artifact_root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise QualityError(f"expected exactly one manifest; got {len(manifests)}")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise QualityError(f"pipeline errors: {manifest['errors']}")
    discovered = manifest.get("discovered_videos", [])
    if not discovered or any(
        video.get("channel_id") != OFFICIAL_TJR_CHANNEL for video in discovered
    ):
        raise QualityError("all discovered sources must be TJR's verified channel")
    planned = manifest.get("planned_clips") or []
    rendered = manifest.get("rendered_clips") or []
    if not planned or len(planned) != len(rendered):
        raise QualityError("every planned clip must render and at least one is required")
    run_dir = manifest_path.parent.resolve()
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
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    try:
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
