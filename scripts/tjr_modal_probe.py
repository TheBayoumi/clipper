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
MAX_MODAL_EGRESS_ATTEMPTS = 3
# Reuse the original Clipper Modal media image/provider that successfully
# acquired YouTube masters in August, not bare yt-dlp in Debian/Deno.
image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.12")
    .entrypoint([])
    .apt_install("ffmpeg", "git", "ca-certificates")
    .uv_pip_install(
        "yt-dlp[default]>=2026.7.4,<2027",
        "bgutil-ytdlp-pot-provider==1.3.1",
    )
    .run_commands(
        "git clone --depth 1 --branch 1.3.1 "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "
        "/root/bgutil-ytdlp-pot-provider",
        "cd /root/bgutil-ytdlp-pot-provider/server && npm ci && npx tsc",
    )
)

PROVIDER_HOME = "/root/bgutil-ytdlp-pot-provider/server"
PROVIDER_ARG = f"youtubepot-bgutilscript:server_home={PROVIDER_HOME}"
# Independent provider-backed and plain-client transports. A bad/expired
# PO token must not suppress a working default client on the same route.
ACQUISITION_STRATEGIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "bgutil_default_mweb",
        (
            "--extractor-args",
            "youtube:player_client=default,mweb",
            "--extractor-args",
            PROVIDER_ARG,
        ),
    ),
    # This is deliberately the second strategy: if the provider-backed mweb
    # attempt hits LOGIN_REQUIRED, test an independent, non-provider client
    # before concluding the video's regional/IP access is blocked.
    (
        "plain_tv_web_safari",
        ("--extractor-args", "youtube:player_client=tv,web_safari"),
    ),
    ("bgutil_default_clients", ("--extractor-args", PROVIDER_ARG)),
    ("plain_default", ()),
    (
        "bgutil_embedded_android_vr",
        (
            "--extractor-args",
            "youtube:player_client=web_embedded,android_vr",
            "--extractor-args",
            PROVIDER_ARG,
        ),
    ),
)


def source_download_sections(duration_seconds: float) -> list[str]:
    """Isolated Modal worker uses the same full-source limit as the runner.

    This must remain dependency-free: the remote image does not install the
    editor's Python package or its XML dependencies.
    """
    if duration_seconds <= 0:
        raise ValueError("verified YouTube duration must be positive")
    if duration_seconds > 3600:
        raise ValueError("SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT: source exceeds one hour")
    return []


def _yt_command(args: tuple[str, ...], url: str) -> list[str]:
    return [
        "yt-dlp",
        "--ignore-config",
        "--no-warnings",
        "--js-runtimes",
        "node",
        "--no-playlist",
        "--socket-timeout",
        "20",
        "--retries",
        "4",
        "--fragment-retries",
        "4",
        *args,
        url,
    ]


def _transport_error(output: str) -> str:
    lower = output.lower()
    if "sign in to confirm" in lower or "not a bot" in lower:
        return "YOUTUBE_IP_OR_LOGIN_CHALLENGE"
    if "http error 403" in lower or "403 forbidden" in lower:
        return "YOUTUBE_MEDIA_HTTP_403"
    if "private video" in lower:
        return "VIDEO_NOT_PUBLIC"
    if "not available" in lower:
        return "VIDEO_UNAVAILABLE"
    return "UNCLASSIFIED_YOUTUBE_TRANSPORT_FAILURE"


volume = modal.Volume.from_name("clipper-tjr-source-transport", create_if_missing=True)


@app.function(image=image, volumes={"/tjr-media": volume}, timeout=1600, cpu=2, memory=2048)
def inspect_original_youtube(candidates: list[dict[str, str]], run_key: str = "") -> dict[str, Any]:
    """Verify source identity and transfer real HD bytes on one Modal egress.

    Reuses Clipper's successful BgUtils strategy family. Metadata-only
    successes are insufficient: an actual signed HD CDN URL must serve bytes.
    """
    import subprocess
    import tempfile

    approved = {
        "UCGHBUXjDCeiIXNdKR0HUZnA",
        "UCZen39LQJPx04GjPj7FOMcw",
    }
    attempts: list[dict[str, str]] = []
    total_ip_challenges = 0
    blocked_video_count = 0
    for candidate in candidates[:6]:
        ip_challenges = 0
        video_id = candidate["video_id"]
        channel_id = candidate["channel_id"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        if channel_id not in approved or len(video_id) != 11:
            attempts.append({"url": url, "reason": "SOURCE_NOT_ALLOWLISTED"})
            continue
        for strategy_name, strategy_args in ACQUISITION_STRATEGIES:
            try:
                metadata_run = subprocess.run(
                    [
                        *_yt_command(strategy_args, url)[:-1],
                        "--dump-single-json",
                        "--skip-download",
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=100,
                    check=False,
                )
                if metadata_run.returncode:
                    reason = _transport_error(metadata_run.stderr)
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "stage": "metadata",
                            "reason": reason,
                        }
                    )
                    if reason == "YOUTUBE_IP_OR_LOGIN_CHALLENGE":
                        ip_challenges += 1
                        total_ip_challenges += 1
                    if ip_challenges >= 2:
                        # Both a provider-backed mweb request and an independent
                        # plain YouTube client were challenged for this video.
                        # Confirm on another approved video before declaring
                        # this regional/IP route blocked.
                        blocked_video_count += 1
                        if blocked_video_count >= 2:
                            return {
                                "status": "YOUTUBE_EGRESS_BOT_CHALLENGE",
                                "attempts": attempts,
                            }
                        break
                    continue
                metadata = json.loads(metadata_run.stdout)
                if not isinstance(metadata, dict):
                    raise RuntimeError("YouTube metadata is not an object")
                if metadata.get("id") != video_id or metadata.get("channel_id") != channel_id:
                    attempts.append(
                        {"url": url, "strategy": strategy_name, "reason": "VIDEO_OWNER_MISMATCH"}
                    )
                    break
                live_status = str(metadata.get("live_status") or "")
                if (
                    metadata.get("is_live")
                    or metadata.get("is_upcoming")
                    or live_status in {"is_live", "is_upcoming"}
                ):
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "reason": "LIVE_OR_UPCOMING_STREAM",
                        }
                    )
                    break
                duration = float(metadata.get("duration") or 0)
                if duration < 90 or duration > 3600:
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "reason": "SHORT_VIDEO"
                            if duration < 90
                            else "SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT",
                        }
                    )
                    break
                formats = metadata.get("formats") or []
                hd = any(
                    isinstance(item, dict)
                    and int(item.get("height") or 0) >= 720
                    and item.get("vcodec") not in ("none", None)
                    for item in formats
                )
                if not hd:
                    attempts.append(
                        {"url": url, "strategy": strategy_name, "reason": "NO_HD_FORMAT"}
                    )
                    continue
                # Verify actual bytes, not just the extractor's advertised formats.
                with tempfile.TemporaryDirectory(prefix="tjr-modal-real-youtube-") as temp:
                    media_command = [
                        *_yt_command(strategy_args, url)[:-1],
                        "--test",
                        "--no-part",
                        "-f",
                        "bv*[height>=720]/b[height>=720]",
                        "-o",
                        str(Path(temp) / "source.%(ext)s"),
                        url,
                    ]
                    transfer = subprocess.run(
                        media_command, capture_output=True, text=True, timeout=145, check=False
                    )
                    media_bytes = sum(
                        file.stat().st_size
                        for file in Path(temp).glob("source.*")
                        if file.is_file()
                    )
                if transfer.returncode or media_bytes < 1024:
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "stage": "actual_hd_media_transfer",
                            "reason": _transport_error(transfer.stderr)
                            if transfer.returncode
                            else "NO_ORIGINAL_HD_BYTES",
                        }
                    )
                    continue
                result: dict[str, Any] = {
                    "status": "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED",
                    "source_url": url,
                    "source_video_id": video_id,
                    "source_channel_id": channel_id,
                    "duration": float(metadata.get("duration") or 0),
                    "title": str(metadata.get("title") or "")[:160],
                    "transport_strategy": strategy_name,
                    "attempts": attempts,
                }
                # The full original MUST be downloaded in the same Modal
                # worker, with the same verified extractor, before any handoff.
                if run_key:
                    result["staging"] = stage_official_original.local(result, run_key)
                return result
            except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as exc:
                # Preserve the verified staging failure type in durable JSON.
                # A corrupt downloaded file is not a YouTube bot challenge.
                detail = str(exc)[:360] if isinstance(exc, RuntimeError) else ""
                attempts.append(
                    {
                        "url": url,
                        "strategy": strategy_name,
                        "reason": f"{type(exc).__name__}: {detail}"
                        if detail
                        else type(exc).__name__,
                    }
                )
    status = (
        "YOUTUBE_EGRESS_BOT_CHALLENGE"
        if total_ip_challenges and total_ip_challenges >= len(attempts) - 1
        else "NO_ACCESSIBLE_ORIGINAL_YOUTUBE"
    )
    return {"status": status, "attempts": attempts}


def verify_staged_media_probe(
    *, returncode: int, stdout: str, stderr: str, expected_seconds: float
) -> float:
    """Check actual HD bytes before publishing them to the shared Modal volume."""
    if returncode:
        raise RuntimeError(
            "CORRUPT_OR_INCOMPLETE_HD_TRANSFER: ffprobe failed: " + stderr.strip()[-400:]
        )
    try:
        media_info = json.loads(stdout)
        streams = media_info["streams"]
        staged_seconds = float(media_info["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("CORRUPT_OR_INCOMPLETE_HD_TRANSFER: invalid ffprobe metadata") from exc
    if not isinstance(streams, list) or not 0 < staged_seconds < float("inf"):
        raise RuntimeError("CORRUPT_OR_INCOMPLETE_HD_TRANSFER: invalid stream or duration")
    if expected_seconds <= 0 or expected_seconds > 3600:
        raise RuntimeError("SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT")
    if staged_seconds + 30 < expected_seconds:
        raise RuntimeError(
            "SOURCE_DURATION_INCOMPLETE: staged media ends before the verified scan window"
        )
    if not any(
        isinstance(stream, dict)
        and stream.get("codec_type") == "video"
        and int(stream.get("height") or 0) >= 720
        for stream in streams
    ):
        raise RuntimeError("MISSING_HD_VIDEO_STREAM")
    if not any(
        isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams
    ):
        raise RuntimeError("MISSING_AUDIO_STREAM")
    return staged_seconds


@app.function(image=image, volumes={"/tjr-media": volume}, timeout=1600, cpu=2, memory=2048)
def stage_official_original(selected: dict[str, Any], run_key: str) -> dict[str, Any]:
    """Download actual allowlisted original bytes via the verified Modal network."""
    import hashlib
    import re
    import subprocess
    import tempfile
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
    expected_seconds = float(selected.get("duration") or 0)
    folder = Path("/tjr-media") / "runs" / run_key / video_id
    folder.mkdir(parents=True, exist_ok=True)
    verified_strategy = str(selected.get("transport_strategy") or "")
    extractor_args = next(
        (args for label, args in ACQUISITION_STRATEGIES if label == verified_strategy),
        None,
    )
    if extractor_args is None:
        raise RuntimeError("unrecognized previously verified original YouTube transport")
    # Each source/transport attempt receives a fresh private directory.
    # Never let a partial prior download look like a successful new original.
    with tempfile.TemporaryDirectory(prefix="acquire-", dir=folder) as scratch:
        scratch_dir = Path(scratch)
        command = [
            "yt-dlp",
            "--ignore-config",
            "--js-runtimes",
            "node",
            *extractor_args,
            "--retries",
            "10",
            "--fragment-retries",
            "10",
            "--no-playlist",
            "--no-warnings",
            "--merge-output-format",
            "mp4",
            *source_download_sections(expected_seconds),
            "-f",
            "bv*[height>=720][height<=1080]+ba/b[height>=720]/bv*+ba/b",
            "--no-part",
            "-o",
            str(scratch_dir / "original.%(ext)s"),
            url,
        ]
        try:
            outcome = subprocess.run(command, capture_output=True, timeout=1330, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ORIGINAL_TRANSFER_TIMEOUT") from exc
        if outcome.returncode:
            reason = _transport_error(outcome.stderr.decode("utf-8", errors="replace"))
            raise RuntimeError(f"ORIGINAL_TRANSFER_FAILED: exit={outcome.returncode} {reason}")
        source_files = [
            item
            for item in scratch_dir.glob("original.*")
            if item.is_file() and item.suffix.lower() in {".mp4", ".mkv", ".webm"}
        ]
        if len(source_files) != 1 or source_files[0].stat().st_size < 1024:
            raise RuntimeError("ORIGINAL_TRANSFER_MISSING_OR_TOO_SMALL")
        downloaded = source_files[0]
        inspect = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(downloaded),
            ],
            capture_output=True,
            text=True,
            timeout=50,
            check=False,
        )
        staged_seconds = verify_staged_media_probe(
            returncode=inspect.returncode,
            stdout=inspect.stdout,
            stderr=inspect.stderr,
            expected_seconds=expected_seconds,
        )
        with downloaded.open("rb") as media:
            digest = hashlib.file_digest(media, "sha256").hexdigest()
        # Publish only verified media; failed attempts leave no corrupt original.
        original = folder / f"original{downloaded.suffix.lower()}"
        os.replace(downloaded, original)
    volume.commit()
    return {
        "status": "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED",
        "video_id": video_id,
        "channel_id": channel_id,
        "public_video_url": url,
        "source_remote_path": f"runs/{run_key}/{video_id}/{original.name}",
        "source_sha256": digest,
        "size_bytes": original.stat().st_size,
        "title": str(selected.get("title") or ""),
        "duration": expected_seconds,
        "staged_duration_seconds": staged_seconds,
        "source_scan_complete": staged_seconds + 30 >= expected_seconds,
    }


@app.local_entrypoint()
def main() -> None:
    from clipper.brief import load_brief
    from scripts.tjr_youtube_preview import (
        constrain_official_sources,
        discover_official_uploads,
    )

    root = Path("tjr-modal-probe")
    root.mkdir(parents=True, exist_ok=True)
    output = root / "verified-original-egress.json"
    try:
        candidates, discovery_failures = discover_official_uploads()
        channel_id = os.getenv("TJR_MODAL_CHANNEL_ID", "").strip()
        if channel_id and channel_id not in {
            "UCGHBUXjDCeiIXNdKR0HUZnA",
            "UCZen39LQJPx04GjPj7FOMcw",
        }:
            raise RuntimeError(
                "requested channel is not one of the Reach-approved YouTube channels"
            )
        requested_video = os.getenv("TJR_SOURCE_VIDEO_ID", "").strip() or None
        published_after = load_brief(Path("campaigns/reach-tjr-weekly.yaml")).published_after
        official = constrain_official_sources(
            candidates,
            requested_id=requested_video,
            published_after=published_after,
            target_channel_id=channel_id or None,
        )
        excluded = {
            item.strip()
            for item in os.getenv("TJR_MODAL_EXCLUDE_VIDEO_IDS", "").split(",")
            if item.strip()
        }
        if requested_video and requested_video in excluded:
            raise RuntimeError("explicit requested video cannot also be excluded")
        official = [item for item in official if item.video_id not in excluded]
        # The target-channel constraint already orders direct discovery newest-first.
        # Metadata verification below skips live/upcoming/short/inaccessible candidates.
        inputs = [
            {"video_id": item.video_id, "channel_id": item.channel_id} for item in official[:8]
        ]
        if not inputs:
            raise RuntimeError("No feed-confirmed campaign YouTube videos")
        stage_media = os.getenv("TJR_MODAL_STAGE_ORIGINAL") == "1"
        run_key = (
            os.environ["GITHUB_RUN_ID"] + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
            if stage_media
            else ""
        )
        region_attempts: list[dict[str, str]] = []
        result: dict[str, Any] = {
            "status": "NO_ACCESSIBLE_ORIGINAL_YOUTUBE",
            "attempts": [],
        }
        # Bound each job to three distinct egress providers; stop immediately on
        # a real HD transfer. Never treat a metadata-only hit as successful.
        # Reuse the old working pipeline's independent cloud/region
        # selection rather than testing three regions of one network.
        routes = (
            ("cloud:gcp", "cloud", "gcp"),
            ("cloud:aws", "cloud", "aws"),
            ("cloud:oci", "cloud", "oci"),
            ("region:eu", "region", "eu"),
            ("region:ap", "region", "ap"),
            ("region:sa", "region", "sa"),
            ("region:af", "region", "af"),
            ("default", "default", "auto"),
        )
        for label, kind, value in routes[:MAX_MODAL_EGRESS_ATTEMPTS]:
            provider = (
                inspect_original_youtube.with_options(cloud=value)
                if kind == "cloud"
                else inspect_original_youtube.with_options(region=value)
                if kind == "region"
                else inspect_original_youtube
            )
            try:
                candidate = provider.remote(inputs, run_key)
                result = candidate
                if candidate.get("status") == "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED" and (
                    not stage_media
                    or candidate.get("staging", {}).get("status")
                    == "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED"
                ):
                    result["selected_modal_egress"] = label
                    break
                region_attempts.append({"egress": label, "status": str(candidate.get("status"))})
            except Exception as exc:
                region_attempts.append({"egress": label, "error": type(exc).__name__})
        result["region_attempts"] = region_attempts
        result["egress_attempt_limit"] = MAX_MODAL_EGRESS_ATTEMPTS
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
