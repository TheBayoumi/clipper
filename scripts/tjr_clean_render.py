"""Render clean source-only portrait clips without ASR, subtitles, graphics or LLM calls.

Consumes an immutable, hash-bound list of *already selected* source timestamps.
This deliberately does not import the editorial/ASR renderer or touch its caches.
All outputs remain private review drafts; nothing here grants publication rights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_STATUS = "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED"
_POLICY = {
    "asr": False,
    "model_inference": False,
    "text_overlays": False,
    "subtitles": False,
    "captions": False,
    "hooks": False,
    "watermark": False,
    "visual_effects": False,
    "automatic_publication": False,
}


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"CLEAN_INVALID_JSON_OBJECT: {path.name}")
    return obj


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def validate_inputs(
    selection: dict[str, Any], staging: dict[str, Any], source: Path
) -> tuple[list[dict[str, Any]], float]:
    if selection.get("schema") != "clipper-clean-source-ranges-v1":
        raise ValueError("CLEAN_INVALID_SELECTION_SCHEMA")
    if staging.get("status") != _SOURCE_STATUS:
        raise ValueError("CLEAN_UNVERIFIED_ORIGINAL_STATUS")
    video = selection.get("video_id")
    channel = selection.get("channel_id")
    sha = selection.get("source_sha256")
    if (
        not isinstance(video, str)
        or not _VIDEO_ID.fullmatch(video)
        or not isinstance(channel, str)
        or not channel.startswith("UC")
        or not isinstance(sha, str)
        or not _SHA256.fullmatch(sha)
        or any(staging.get(key) != value for key, value in (
            ("video_id", video), ("channel_id", channel), ("source_sha256", sha)
        ))
    ):
        raise ValueError("CLEAN_SOURCE_PROVENANCE_MISMATCH")
    if not source.is_file() or _sha256(source) != sha:
        raise ValueError("CLEAN_ORIGINAL_BYTES_SHA256_MISMATCH")
    if type(selection.get("selection_run_id")) is not int:
        raise ValueError("CLEAN_SELECTION_RUN_ID_MISSING")
    if type(selection.get("verified_original_run_id")) is not int:
        raise ValueError("CLEAN_ORIGINAL_RUN_ID_MISSING")
    rows = selection.get("clips")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
        raise ValueError("CLEAN_INVALID_SELECTION_COUNT")
    source_info = _probe(source)
    duration = float(source_info["format"]["duration"])
    kinds = [stream.get("codec_type") for stream in source_info.get("streams", [])]
    if "video" not in kinds or "audio" not in kinds:
        raise ValueError("CLEAN_SOURCE_REQUIRES_VIDEO_AND_AUDIO")
    seen: set[str] = set()
    safe_ranges: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("CLEAN_INVALID_CLIP_RECORD")
        clip_id = row.get("id")
        start, end = row.get("start_seconds"), row.get("end_seconds")
        if (
            clip_id != f"clean-{index:02d}"
            or clip_id in seen
            or type(start) not in (int, float)
            or type(end) not in (int, float)
            or not 0 <= start < end <= duration + 0.05
            or end - start < 1
        ):
            raise ValueError(f"CLEAN_INVALID_SOURCE_RANGE: #{index}")
        seen.add(clip_id)
        safe_ranges.append({"id": clip_id, "start": float(start), "end": float(end)})
    return safe_ranges, duration


def clean_ffmpeg_command(
    source: Path, destination: Path, *, start: float, end: float
) -> list[str]:
    """Center-crop the native video only. No subtitles, overlays or visual effects."""
    if not 0 <= start < end:
        raise ValueError("CLEAN_INVALID_CUT_TIMES")
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0", "-sn", "-dn",
        "-vf",
        (
            "crop=trunc(min(iw\\,ih*9/16)/2)*2:"
            "trunc(min(ih\\,iw*16/9)/2)*2:(iw-ow)/2:(ih-oh)/2,"
            "scale=1080:1920:flags=lanczos,setsar=1"
        ),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-map_metadata", "-1", "-movflags", "+faststart",
        str(destination),
    ]


def render(selection_path: Path, staging_path: Path, source: Path, output: Path) -> Path:
    selection = _read_json(selection_path)
    staging = _read_json(staging_path)
    clips, source_duration = validate_inputs(selection, staging, source)
    clips_root = output / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    proof: list[dict[str, Any]] = []
    for item in clips:
        destination = clips_root / f"{item['id']}-{selection['video_id']}.mp4"
        subprocess.run(
            clean_ffmpeg_command(
                source, destination, start=item["start"], end=item["end"]
            ),
            check=True,
        )
        info = _probe(destination)
        streams = info.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        observed = float(info["format"]["duration"])
        if (
            video is None
            or audio is None
            or video.get("width") != 1080
            or video.get("height") != 1920
            or video.get("codec_name") != "h264"
            or audio.get("codec_name") != "aac"
            or any(s.get("codec_type") in ("subtitle", "attachment") for s in streams)
            or abs(observed - (item["end"] - item["start"])) > 0.3
            or destination.stat().st_size < 1024
        ):
            raise RuntimeError(f"CLEAN_OUTPUT_MEDIA_VERIFICATION_FAILED: {item['id']}")
        # Full-decode every delivered file rather than trusting the container header.
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(destination),
             "-f", "null", "-"],
            check=True, capture_output=True, text=True,
        )
        record = {
            **item,
            "file": str(destination.relative_to(output)),
            "duration_seconds": observed,
            "width": 1080,
            "height": 1920,
            "video_codec": video["codec_name"],
            "audio_codec": audio["codec_name"],
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "full_decode_passed": True,
        }
        proof.append(record)
        print(
            f"CLEAN_CLIP_RENDERED: {item['id']} "
            f"{item['start']:.2f}-{item['end']:.2f}s "
            f"bytes={record['size_bytes']}",
            flush=True,
        )
    report = {
        "schema": "clipper-clean-render-report-v1",
        "status": "TECHNICAL_QA_PASSED__HUMAN_REVIEW_REQUIRED",
        "source_sha256": selection["source_sha256"],
        "video_id": selection["video_id"],
        "channel_id": selection["channel_id"],
        "selection_run_id": selection["selection_run_id"],
        "verified_original_run_id": selection["verified_original_run_id"],
        "original_duration_seconds": source_duration,
        "verified_clean_clip_count": len(proof),
        "policy": _POLICY,
        "frame_policy": "static_center_crop_without_added_text_or_effects",
        "source_embedded_text": "not_removed",
        "clips": proof,
    }
    path = output / "clean-render-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CLEAN_RENDER_COMPLETE: {len(proof)} MP4s; ASR=0; OVERLAYS=0", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.selection, args.staging, args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
