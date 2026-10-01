#!/usr/bin/env python3
"""Original YouTube acquisition on the user's explicitly selected Modal route.

An optional GitHub viewer-session secret is injected through Modal Secrets.
Cookie files are ephemeral and excluded from source transport and artifacts.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import modal

from scripts.tjr_media_policy import has_production_hd_video_stream, production_hd_format

app = modal.App("clipper-tjr-official-youtube-probe")
MAX_MODAL_EGRESS_ATTEMPTS = 1
# Reuse the original Clipper Modal media image/provider that successfully
# acquired YouTube masters in August, not bare yt-dlp in Debian/Deno.
image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.12")
    .entrypoint([])
    .apt_install("ffmpeg", "git", "ca-certificates")
    .uv_pip_install(
        "yt-dlp[default]>=2026.7.4,<2027",
        "bgutil-ytdlp-pot-provider==2.0.0",
    )
    .run_commands(
        "git clone --depth 1 --branch 2.0.0 "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "
        "/root/bgutil-ytdlp-pot-provider",
        "cd /root/bgutil-ytdlp-pot-provider/server && npm ci && npx tsc",
    )
)

browser_image = image.apt_install("chromium", "xvfb", "xauth").run_commands(
    "python -m venv /opt/youtube-wpc && "
    "/opt/youtube-wpc/bin/pip install 'yt-dlp[default]==2026.8.19' 'yt-dlp-getpot-wpc==1.1.2'"
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


BROWSER_GUEST_STRATEGIES = (
    (
        "browser_guest_mweb_player_tokens",
        (
            "--extractor-args",
            "youtube:player_client=mweb;fetch_pot=always",
            "--extractor-args",
            "youtubepot-wpc:browser_path=/usr/bin/chromium",
        ),
    ),
)


def acquisition_strategies() -> tuple[tuple[str, tuple[str, ...]], ...]:
    mode = os.getenv("TJR_YOUTUBE_SESSION_MODE", "bgutil_guest")
    if mode == "browser_guest":
        return BROWSER_GUEST_STRATEGIES
    if mode not in {"bgutil_guest", "viewer_secret"}:
        raise RuntimeError("INVALID_YOUTUBE_SESSION_MODE")
    return ACQUISITION_STRATEGIES


def acquisition_executable() -> list[str]:
    if os.getenv("TJR_YOUTUBE_SESSION_MODE") == "browser_guest":
        return ["xvfb-run", "--auto-servernum", "/opt/youtube-wpc/bin/yt-dlp"]
    return ["yt-dlp"]


def acquisition_run(
    command: list[str],
    *,
    timeout: int,
    capture_output: bool = True,
    text: bool = False,
    check: bool = False,
) -> Any:
    """Kill the entire browser process group when a guest-token call times out."""
    if os.getenv("TJR_YOUTUBE_SESSION_MODE") != "browser_guest":
        return subprocess.run(
            command, timeout=timeout, capture_output=capture_output, text=text, check=check
        )
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text, start_new_session=True
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result


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


@contextmanager
def viewer_cookie_session() -> Iterator[None]:
    """Materialize an explicitly configured viewer secret only for this call."""
    encoded = os.getenv("TJR_YOUTUBE_COOKIES_B64", "").strip()
    if not encoded:
        yield
        return
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise RuntimeError("INVALID_YOUTUBE_VIEWER_SESSION_SECRET") from None
    if not payload.startswith((b"# Netscape HTTP Cookie File", b"# HTTP Cookie File")):
        raise RuntimeError("INVALID_YOUTUBE_VIEWER_SESSION_COOKIE_FORMAT")
    previous = os.environ.get("YOUTUBE_COOKIES_FILE")
    with tempfile.TemporaryDirectory(prefix="youtube-viewer-") as private:
        path = Path(private) / "cookies.txt"
        path.touch(mode=0o600)
        path.write_bytes(payload)
        os.environ["YOUTUBE_COOKIES_FILE"] = str(path)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("YOUTUBE_COOKIES_FILE", None)
            else:
                os.environ["YOUTUBE_COOKIES_FILE"] = previous


def viewer_cookie_args() -> list[str]:
    path = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
    return ["--cookies", path] if path else []


def _yt_command(args: tuple[str, ...], url: str) -> list[str]:
    return [
        *acquisition_executable(),
        *(["--verbose"] if os.getenv("TJR_YOUTUBE_SESSION_MODE") == "browser_guest" else []),
        "--ignore-config",
        "--no-warnings",
        "--js-runtimes",
        "node",
        "--no-playlist",
        "--socket-timeout",
        "20",
        "--sleep-requests",
        "1",
        "--retries",
        "4",
        "--fragment-retries",
        "4",
        *viewer_cookie_args(),
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
    mode = os.getenv("TJR_YOUTUBE_SESSION_MODE", "bgutil_guest")
    if mode != "viewer_secret" and (
        os.getenv("TJR_YOUTUBE_COOKIES_B64") or os.getenv("YOUTUBE_COOKIES_FILE")
    ):
        raise RuntimeError("GUEST_ACQUISITION_REJECTS_ACCOUNT_COOKIES")
    with viewer_cookie_session() if mode == "viewer_secret" else nullcontext():
        return _inspect_original_youtube(candidates, run_key)


@app.function(image=browser_image, volumes={"/tjr-media": volume}, timeout=1600, cpu=2, memory=3072)
def inspect_browser_guest_youtube(
    candidates: list[dict[str, str]], run_key: str = ""
) -> dict[str, Any]:
    return inspect_original_youtube.local(candidates, run_key)


def _inspect_original_youtube(candidates: list[dict[str, str]], run_key: str) -> dict[str, Any]:
    """Verify source identity and transfer real HD bytes on one Modal egress.

    Reuses Clipper's successful BgUtils strategy family. Metadata-only
    successes are insufficient: an actual signed HD CDN URL must serve bytes.
    """
    import tempfile

    approved = {
        "UCf1q6dhccWr6eQEcFFnJSbA",
    }
    attempts: list[dict[str, Any]] = []
    total_ip_challenges = 0
    blocked_video_count = 0
    strategies = acquisition_strategies()
    for candidate in candidates[:6]:
        ip_challenges = 0
        video_id = candidate["video_id"]
        channel_id = candidate["channel_id"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        if channel_id not in approved or len(video_id) != 11:
            attempts.append({"url": url, "reason": "SOURCE_NOT_ALLOWLISTED"})
            continue
        for strategy_name, strategy_args in strategies:
            try:
                metadata_run = acquisition_run(
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
                token_evidence = (
                    {
                        "browser_launched": "Launching youtube.com in browser"
                        in metadata_run.stderr,
                        "player_token_received": "Retrieved a player PO Token"
                        in metadata_run.stderr,
                        "gvs_token_received": "Retrieved a gvs PO Token" in metadata_run.stderr,
                    }
                    if os.getenv("TJR_YOUTUBE_SESSION_MODE") == "browser_guest"
                    else {}
                )
                if metadata_run.returncode:
                    reason = _transport_error(metadata_run.stderr)
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "stage": "metadata",
                            "reason": reason,
                            "guest_token_evidence": token_evidence,
                        }
                    )
                    if reason == "YOUTUBE_IP_OR_LOGIN_CHALLENGE":
                        ip_challenges += 1
                        total_ip_challenges += 1
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
                hd = any(isinstance(item, dict) and production_hd_format(item) for item in formats)
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
                        "bv*[width>=1280][height>=720]/bv*[width>=720][height>=1280]/b[width>=1280][height>=720]/b[width>=720][height>=1280]",
                        "-o",
                        str(Path(temp) / "source.%(ext)s"),
                        url,
                    ]
                    transfer = acquisition_run(
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
                    "guest_token_evidence": token_evidence,
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
        # Exhaust the bounded strategy set for this exact video. Two clients
        # cannot establish that every configured cookie-free client is blocked.
        if ip_challenges == len(strategies):
            blocked_video_count += 1
            if blocked_video_count >= 2:
                return {"status": "YOUTUBE_EGRESS_BOT_CHALLENGE", "attempts": attempts}
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
    if not has_production_hd_video_stream(streams):
        raise RuntimeError("MISSING_PRODUCTION_HD_VIDEO_STREAM")
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
    import tempfile
    from pathlib import Path

    video_id = str(selected["source_video_id"])
    channel_id = str(selected["source_channel_id"])
    if (
        channel_id not in {"UCf1q6dhccWr6eQEcFFnJSbA"}
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
        (args for label, args in acquisition_strategies() if label == verified_strategy),
        None,
    )
    if extractor_args is None:
        raise RuntimeError("unrecognized previously verified original YouTube transport")
    # Each source/transport attempt receives a fresh private directory.
    # Never let a partial prior download look like a successful new original.
    with tempfile.TemporaryDirectory(prefix="acquire-", dir=folder) as scratch:
        scratch_dir = Path(scratch)
        command = [
            *acquisition_executable(),
            "--ignore-config",
            "--js-runtimes",
            "node",
            "--socket-timeout",
            "20",
            "--sleep-requests",
            "1",
            *viewer_cookie_args(),
            *extractor_args,
            "--retries",
            "10",
            "--fragment-retries",
            "10",
            "--abort-on-unavailable-fragments",
            "--no-playlist",
            "--no-warnings",
            "--merge-output-format",
            "mp4",
            *source_download_sections(expected_seconds),
            "-f",
            "bv*[width>=1280][height>=720][height<=1080]+ba/"
            "bv*[width>=720][height>=1280][width<=1080]+ba/"
            "b[width>=1280][height>=720]/b[width>=720][height>=1280]",
            "--no-part",
            "-o",
            str(scratch_dir / "original.%(ext)s"),
            url,
        ]
        try:
            outcome = acquisition_run(command, capture_output=True, timeout=1330, check=False)
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
        if channel_id and channel_id not in {"UCf1q6dhccWr6eQEcFFnJSbA"}:
            raise RuntimeError(
                "requested channel is not one of the Reach-approved YouTube channels"
            )
        requested_video = os.getenv("TJR_SOURCE_VIDEO_ID", "").strip() or None
        published_after = load_brief(
            Path("campaigns/reach-double-coverage-dedicated.yaml")
        ).published_after
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
        # One explicitly selected Modal route. Do not switch clouds automatically.
        routes = (("cloud:gcp", "cloud", "gcp"),)
        for label, kind, value in routes[:MAX_MODAL_EGRESS_ATTEMPTS]:
            provider = (
                inspect_original_youtube.with_options(cloud=value)
                if kind == "cloud"
                else inspect_original_youtube.with_options(region=value)
                if kind == "region"
                else inspect_original_youtube
            )
            session_mode = os.getenv("TJR_YOUTUBE_SESSION_MODE", "bgutil_guest")
            acquisition_strategies()  # Reject unknown modes before a remote invocation.
            provider = provider.with_options(env={"TJR_YOUTUBE_SESSION_MODE": session_mode})
            if session_mode == "browser_guest":
                provider = inspect_browser_guest_youtube.with_options(
                    cloud="gcp",
                    memory=3072,
                    env={"TJR_YOUTUBE_SESSION_MODE": session_mode},
                )
            viewer_secret = (
                os.getenv("TJR_YOUTUBE_COOKIES_B64", "").strip()
                if session_mode == "viewer_secret"
                else ""
            )
            if session_mode == "viewer_secret" and not viewer_secret:
                raise RuntimeError("VIEWER_SECRET_MODE_REQUIRES_YOUTUBE_SESSION_SECRET")
            if viewer_secret:
                provider = provider.with_options(
                    secrets=[modal.Secret.from_dict({"TJR_YOUTUBE_COOKIES_B64": viewer_secret})]
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
        result["session_mode"] = os.getenv("TJR_YOUTUBE_SESSION_MODE", "bgutil_guest")
        result["authenticated_viewer_session_configured"] = bool(
            os.getenv("TJR_YOUTUBE_COOKIES_B64", "").strip()
        )
        result["region_attempts"] = region_attempts
        result["egress_attempt_limit"] = MAX_MODAL_EGRESS_ATTEMPTS
        result["discovery_failures"] = discovery_failures
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result["status"] != "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED":
            status = str(result["status"])
            if status == "YOUTUBE_EGRESS_BOT_CHALLENGE":
                raise RuntimeError(
                    status + ": selected acquisition mode could not establish playback; "
                    "no source or cloud fallback was attempted"
                )
            raise RuntimeError(status + ": official YouTube HD acquisition failed")
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
