"""Evidence-gated review of actual TJR production artifacts.

Never substitute a green workflow for validated MP4s or human editorial approval.
This module reads only inert artifact files and produces actionable, durable JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

OFFICIAL_CHANNELS = {
    "UCZen39LQJPx04GjPj7FOMcw",
    "UCGHBUXjDCeiIXNdKR0HUZnA",
}
MIN_SSIM = 0.99
MAX_GENERIC_FRACTION = 0.25


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 8_000_000:
        raise ValueError("missing or oversized evidence file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence must be a JSON object")
    return value


def checked_path(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("missing evidence path")
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError("missing or escaping evidence path")
    return path


def probe_media(path: Path, *, full_decode: bool = False) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=50,
    )
    streams = json.loads(result.stdout)["streams"]
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ValueError("one video and one audio stream are required")
    if full_decode:
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
            timeout=180,
        )
    return {
        "width": video[0]["width"],
        "height": video[0]["height"],
        "fps": float(Fraction(video[0]["avg_frame_rate"])),
        "video_codec": video[0]["codec_name"],
        "audio_codec": audio[0]["codec_name"],
    }


def valid_publication_date(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        return date >= datetime(2026, 9, 1, tzinfo=UTC)
    except ValueError:
        return False


def inspect_artifact(
    folder: Path,
    probe: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "artifact": folder.name,
        "rendered_mp4_count": 0,
        "technically_verified_mp4_count": 0,
        "generic_hooks": 0,
        "minimum_ssim": None,
        "issues": [],
    }
    issues: list[str] = entry["issues"]
    reports = list(folder.rglob("tjr-youtube-qa-report.json"))
    if len(reports) != 1:
        for name in ("verified-original-egress.json", "source-acquisition-errors.json"):
            diagnostics = list(folder.rglob(name))
            if not diagnostics:
                continue
            try:
                data = read_json(diagnostics[0])
                detail = str(data.get("status") or data.get("error") or "")
                issue = (
                    "YOUTUBE_EGRESS_BLOCKED"
                    if "BOT_CHALLENGE" in detail or "403" in detail
                    else "ACQUISITION_OR_RENDER_FAILED"
                )
                issues.append(issue)
            except (ValueError, json.JSONDecodeError):
                issues.append("INVALID_FAILURE_DIAGNOSTIC")
            break
        if not issues:
            issues.append("MISSING_PRODUCTION_REPORT")
        return entry

    base = reports[0].parent
    try:
        report = read_json(reports[0])
        entry["channel_id"] = report.get("source_channel_id")
        if report.get("source_channel_id") not in OFFICIAL_CHANNELS:
            issues.append("UNVERIFIED_CHANNEL")
        if not re.fullmatch(r"[0-9a-f]{64}", str(report.get("source_sha256") or "")):
            issues.append("UNVERIFIED_SOURCE_HASH")
        if not valid_publication_date(report.get("source_published_at")):
            issues.append("SOURCE_DATE_UNVERIFIED")
        coverage = read_json(base / "source-analysis-coverage.json")
        original = float(coverage.get("reported_original_seconds") or 0)
        scanned = float(coverage.get("analyzed_source_seconds") or 0)
        if (
            coverage.get("full_source_analyzed") is not True
            or original < 90
            or scanned + 30 < original
        ):
            issues.append("INCOMPLETE_SOURCE_COVERAGE")
        clips = report.get("clips")
        if not isinstance(clips, list) or not clips:
            issues.append("NO_RENDERED_CLIPS")
            return entry
        seen: set[tuple[str, float, float]] = set()
        min_ssim = 1.0
        for clip in clips:
            if not isinstance(clip, dict):
                issues.append("INVALID_CLIP_REPORT")
                continue
            entry["rendered_mp4_count"] += 1
            if str(clip.get("hook_candidate") or "").startswith("THE MOMENT:"):
                entry["generic_hooks"] += 1
            try:
                mp4 = checked_path(base, clip.get("file"))
                for field in (
                    "srt",
                    "ass_sidecar",
                    "source_matched_quality",
                    "contact_sheet",
                    "preview",
                ):
                    checked_path(base, clip.get(field))
                checked_path(base, str(clip["file"]).removesuffix(".mp4") + ".ssim.txt")
                overlay = clip.get("overlay_acceptance") or {}
                if (
                    overlay.get("style") != "B2"
                    or abs(
                        float(overlay.get("persistent_hook_seconds") or 0)
                        - float(clip["duration_seconds"])
                    )
                    > 0.15
                    or int(overlay.get("spoken_word_highlight_events") or 0) < 1
                ):
                    issues.append("CAPTION_OR_HOOK_TIMING_FAILED")
                quality = read_json(checked_path(base, clip["source_matched_quality"]))
                ssim = float(quality.get("source_to_delivery_mean_ssim") or 0)
                min_ssim = min(min_ssim, ssim)
                if (
                    quality.get("status") != "MEASURED_SOURCE_MATCHED_ENCODING"
                    or ssim < MIN_SSIM
                    or int(quality.get("compared_frames") or 0) < 1
                ):
                    issues.append("SOURCE_FIDELITY_FAILED")
                actual = probe(mp4)
                native_fps = float(Fraction(report["source_profile"]["fps"]))
                if (
                    (int(actual["width"]), int(actual["height"])) != (1080, 1920)
                    or actual["video_codec"] != "h264"
                    or actual["audio_codec"] != "aac"
                    or abs(float(actual["fps"]) - native_fps) > 0.04
                ):
                    issues.append("INVALID_REAL_MEDIA_PROFILE")
                identity = (
                    str(report.get("source_url") or ""),
                    float(clip["source_start_seconds"]),
                    float(clip["source_end_seconds"]),
                )
                if identity in seen or identity[1] >= identity[2]:
                    issues.append("DUPLICATE_OR_INVALID_CLIP_WINDOW")
                seen.add(identity)
                if clip.get("review_required") is not True:
                    issues.append("MISSING_HUMAN_REVIEW_GATE")
                entry["technically_verified_mp4_count"] += 1
            except (
                KeyError,
                ValueError,
                TypeError,
                ZeroDivisionError,
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                issues.append("MISSING_OR_INVALID_CLIP_EVIDENCE")
        entry["minimum_ssim"] = round(min_ssim, 6)
        if entry["generic_hooks"] / len(clips) > MAX_GENERIC_FRACTION:
            issues.append("GENERIC_HOOK_OVERUSE")
    except (ValueError, TypeError, json.JSONDecodeError):
        issues.append("INVALID_SOURCE_OR_COVERAGE_REPORT")
    entry["issues"] = sorted(set(issues))
    return entry


def review_run(
    root: Path,
    *,
    expected_channels: int,
    probe: Callable[[Path], dict[str, Any]],
    head_sha: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    folders = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    channels = [inspect_artifact(folder, probe) for folder in folders]
    issues = {issue for channel in channels for issue in channel["issues"]}
    if len(channels) != expected_channels:
        issues.add("MISSING_CHANNEL_ARTIFACT")
    ids = [channel.get("channel_id") for channel in channels if channel.get("channel_id")]
    if len(ids) != len(set(ids)):
        issues.add("DUPLICATE_CHANNEL_ARTIFACT")
    actions: list[str] = []
    if any("EGRESS" in issue for issue in issues):
        actions.append("REPAIR_YOUTUBE_SOURCE_TRANSPORT")
    if "GENERIC_HOOK_OVERUSE" in issues:
        actions.append("IMPROVE_GROUNDED_CREATIVE_HOOKS")
    if any("SOURCE" in issue for issue in issues):
        actions.append("REPAIR_SOURCE_OR_PROVENANCE_VALIDATION")
    if any("CLIP" in issue or "QA" in issue or "CAPTION" in issue for issue in issues):
        actions.append("REPAIR_RENDER_OR_QA")
    if issues and not actions:
        actions.append("INSPECT_ATTACHED_DIAGNOSTICS")
    if not issues:
        actions.append("REQUEST_HUMAN_VISUAL_AND_EDITORIAL_APPROVAL")
    return {
        "head_sha": head_sha,
        "run_id": run_id,
        "audited_production_run_id": os.getenv("TJR_FEEDBACK_SOURCE_RUN_ID") or run_id,
        "status": ("BLOCKED" if issues else "TECHNICAL_QA_PASSED__HUMAN_REVIEW_REQUIRED"),
        "expected_channels": expected_channels,
        "technically_verified_mp4_count": sum(
            item["technically_verified_mp4_count"] for item in channels
        ),
        "channels": channels,
        "issues": sorted(issues),
        "next_actions": actions,
        "automatic_publication_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-channels", type=int, choices=(1, 2), required=True)
    parser.add_argument("--full-decode", action="store_true")
    args = parser.parse_args()

    def inspect(path: Path) -> dict[str, Any]:
        return probe_media(path, full_decode=args.full_decode)

    result = review_run(
        args.artifact_root,
        expected_channels=args.expected_channels,
        probe=inspect,
        head_sha=os.getenv("GITHUB_SHA", ""),
        run_id=os.getenv("GITHUB_RUN_ID", ""),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "feedback-report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "## TJR real-media evidence gate",
        "Status: **" + result["status"] + "**",
        "Audit commit: " + result["head_sha"] + " · Audit run: " + result["run_id"],
        "Original production run: " + result["audited_production_run_id"],
        "Technically verified MP4s: " + str(result["technically_verified_mp4_count"]),
        "Issues: " + (", ".join(result["issues"]) if result["issues"] else "none"),
        "Next: " + ", ".join(result["next_actions"]),
        "Publication is blocked pending human visual and editorial approval.",
    ]
    (args.output / "feedback-summary.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)
    return 1 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
