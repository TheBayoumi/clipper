#!/usr/bin/env python3
"""Render review-only HD clips from pinned, official TJR Kick VODs.

Source acquisition is public and provenance-pinned. This command does NOT
verify Whop eligibility, authorize publishing, or certify editorial quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from clipper.brief import load_brief
from clipper.render import FFmpegRenderer
from clipper.scoring import score_transcript, select_diverse_clips
from clipper.transcript import transcribe_with_faster_whisper
from scripts.tjr_quality import check_full_decode, probe_original, probe_video

LOGGER = logging.getLogger("tjr-kick")
# URLs are public VODs on TJR's official Kick channel, an approved Reach source.
PINNED_VODS = (
    "https://kick.com/tjr/videos/01a0aa4e-7680-7082-ae4f-62301f7a038b",
    "https://kick.com/tjr/videos/01a0c414-ca60-7139-8bff-9784e96d40d7",
)
VOD_PATTERN = re.compile(
    r"/tjr/videos/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def assert_official_vod(url: str, *, allow_unpinned: bool = False) -> str:
    parsed = urlparse(url)
    if (
        (not allow_unpinned and url not in PINNED_VODS)
        or parsed.scheme != "https"
        or parsed.hostname != "kick.com"
        or not VOD_PATTERN.fullmatch(parsed.path)
    ):
        raise ValueError("source must be a pinned official TJR Kick VOD")
    return parsed.path.rsplit("/", 1)[-1]


def invoke(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or detail)[-1400:]
        raise RuntimeError(f"{command[0]} failed: {detail}") from exc


def _official_vod_feed() -> list[dict[str, Any]]:
    """Find currently available VODs through TJR's own Kick channel listing.

    The legacy per-VOD Kick API returns 404 after Kick's July 2026 change.
    Channel video listings can expose the current HLS playback source instead.
    """
    from curl_cffi import requests

    headers = {"Referer": "https://kick.com/tjr/videos", "Accept": "application/json"}
    failures: list[str] = []
    for endpoint in (
        "https://kick.com/api/v2/channels/tjr/videos",
        "https://kick.com/api/v2/channels/tjr/videos/latest",
    ):
        try:
            response = requests.get(endpoint, headers=headers, impersonate="chrome", timeout=35)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                raw_vods = payload
            elif isinstance(payload, dict):
                raw_vods = payload.get("data") or payload.get("videos") or []
                if isinstance(raw_vods, dict):
                    raw_vods = raw_vods.get("data") or raw_vods.get("videos") or []
            else:
                raw_vods = []
            if not isinstance(raw_vods, list):
                raise RuntimeError("channel video listing has an unsupported JSON shape")
            found: list[dict[str, Any]] = []
            for item in raw_vods:
                if not isinstance(item, dict):
                    continue
                video = item.get("video") if isinstance(item.get("video"), dict) else item
                livestream = (
                    video.get("livestream") if isinstance(video.get("livestream"), dict) else {}
                )
                channel = (
                    livestream.get("channel") if isinstance(livestream.get("channel"), dict) else {}
                )
                slug = str(channel.get("slug") or "tjr").lower()
                if slug != "tjr":
                    continue
                vod_id = str(video.get("uuid") or video.get("id") or item.get("id") or "")
                vod_url = f"https://kick.com/tjr/videos/{vod_id}"
                try:
                    assert_official_vod(vod_url, allow_unpinned=True)
                except ValueError:
                    continue
                playlist = str(
                    video.get("source")
                    or video.get("playback_url")
                    or item.get("source")
                    or item.get("playback_url")
                    or ""
                )
                if not _trusted_kick_playlist(playlist):
                    continue
                title = str(
                    livestream.get("session_title") or video.get("title") or "TJR official stream"
                )
                found.append(
                    {
                        "url": vod_url,
                        "playlist": playlist,
                        "metadata": {"channel": "tjr", "title": title, "id": vod_id},
                    }
                )
            if found:
                LOGGER.info("Found %d current official TJR VOD sources", len(found))
                return found
            failures.append(f"{endpoint}: no playable current VODs")
        except (ValueError, RuntimeError, OSError) as exc:
            failures.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    raise RuntimeError("official Kick feed unavailable: " + "; ".join(failures))


def _trusted_kick_playlist(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and parsed.path.endswith(".m3u8")
        and (
            host == "stream.kick.com"
            or host.endswith(".kickcdn.com")
            or host.endswith(".cloudfront.net")
            or host.endswith(".playback.live-video.net")
        )
    )


def fetch_official_excerpt(
    url: str,
    work: Path,
    *,
    playlist: str | None = None,
    metadata_override: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    assert_official_vod(url, allow_unpinned=playlist is not None)
    if playlist is not None and not _trusted_kick_playlist(playlist):
        raise RuntimeError("Kick feed provided an untrusted or non-HLS playlist")
    work.mkdir(parents=True, exist_ok=True)
    if metadata_override is not None:
        metadata = metadata_override
    else:
        metadata = json.loads(
            invoke(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--no-warnings",
                    "--dump-single-json",
                    "--skip-download",
                    url,
                ],
                timeout=180,
            ).stdout
        )
    if not isinstance(metadata, dict):
        raise RuntimeError("Kick video metadata was not a JSON object")
    uploader = str(metadata.get("channel") or metadata.get("uploader") or "")
    if uploader and "tjr" not in uploader.lower():
        raise RuntimeError(f"Kick VOD uploader does not match TJR: {uploader}")
    title = str(metadata.get("title") or "")
    if not title:
        raise RuntimeError("Kick video lacks an authentic video title")
    LOGGER.info("Selected official TJR VOD: %s", title)
    invoke(
        [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--no-part",
            "--force-overwrites",
            "--merge-output-format",
            "mp4",
            "--add-header",
            "Referer:https://kick.com/",
            "-f",
            "bv*[height>=720][height<=1080]+ba/b[height>=720]/bv*+ba/b",
            "--download-sections",
            "*00:01:00-00:15:00",
            "-o",
            str(work / "source.%(ext)s"),
            playlist or url,
        ],
        timeout=1800,
    )
    candidates = sorted(
        p
        for p in work.glob("source.*")
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm"}
    )
    if not candidates:
        raise RuntimeError("Kick downloader returned without an actual source video")
    source = candidates[0]
    probe_original(source)
    return source, metadata


def render_preview(root: Path, brief_path: Path) -> Path:
    brief = load_brief(brief_path)
    if not brief.rights_confirmed or brief.watermark_url or brief.watermark_text:
        raise RuntimeError("TJR permissions or no-logo rules are not met")
    run_id = "reach-tjr-kick-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    failures: list[dict[str, str]] = []
    try:
        source: Path | None = None
        metadata: dict[str, Any] = {}
        selected_url = ""
        try:
            current_vods = _official_vod_feed()
        except RuntimeError as exc:
            LOGGER.warning("Kick channel feed failed: %s", exc)
            failures.append({"source_url": "https://kick.com/tjr/videos", "error": str(exc)})
            current_vods = []
        attempts = current_vods[:4] or [
            {"url": url, "playlist": None, "metadata": None} for url in PINNED_VODS
        ]
        for index, item in enumerate(attempts, start=1):
            url = str(item["url"])
            try:
                source, metadata = fetch_official_excerpt(
                    url,
                    run_dir / "work" / str(index),
                    playlist=item["playlist"],
                    metadata_override=item["metadata"],
                )
                selected_url = url
                break
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning("Official VOD candidate failed: %s", exc)
                failures.append({"source_url": url, "error": str(exc)})
        if source is None:
            raise RuntimeError("official Kick VOD acquisition failed; see diagnostic artifact")

        transcript = transcribe_with_faster_whisper(
            source, model_name="small.en", device="cpu", compute_type="int8", language="en"
        )
        if not transcript:
            raise RuntimeError("source has no identifiable English speech")
        source_id = assert_official_vod(selected_url, allow_unpinned=True)
        candidates = score_transcript(brief, source_id, transcript, limit=30)
        chosen = select_diverse_clips(candidates, clip_count=2, max_per_source=2)
        if not chosen:
            raise RuntimeError("no 20-42 second spoken excerpts met campaign timing rules")
        renderer = FFmpegRenderer()
        produced: list[dict[str, Any]] = []
        for index, clip in enumerate(chosen, start=1):
            output = run_dir / "clips" / f"{index:02d}-tjr-kick.mp4"
            renderer.render(source, output, clip, transcript)
            technical = probe_video(output)
            check_full_decode(output)
            preview = output.with_name(output.stem + "-preview.png")
            invoke(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    "3",
                    "-i",
                    str(output),
                    "-frames:v",
                    "1",
                    str(preview),
                ],
                timeout=90,
            )
            produced.append(
                {
                    **technical,
                    "file": str(output.relative_to(run_dir)),
                    "subtitle": str(output.with_suffix(".srt").relative_to(run_dir)),
                    "preview": str(preview.relative_to(run_dir)),
                    "source_url": selected_url,
                    "source_start_in_excerpt": round(clip.start, 3),
                    "source_end_in_excerpt": round(clip.end, 3),
                    "score": clip.score,
                }
            )
            LOGGER.info("Rendered and full-decoded %s", output.name)
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest = {
            "campaign": brief.campaign_id,
            "status": "TECHNICAL_QA_PASSED__HUMAN_REVIEW_REQUIRED",
            "source_platform": "kick",
            "source_channel": "tjr",
            "source_url": selected_url,
            "source_title": metadata.get("title"),
            "source_sha256": digest,
            "source_dimensions": probe_original(source),
            "excerpt_start_in_vod_seconds": 60,
            "clips": produced,
            "source_attempts": failures,
            "manual_checks": [
                "Verify TJR visibly appears in each excerpt.",
                "Verify speech, captions, original context, sound and framing.",
                "Reject embedded logos, negative portrayal, gambling promotions or AI footage.",
                "Confirm live Whop budget, account eligibility and audience before posting.",
                "Post with #TJR and submit to Whop within 30 minutes.",
            ],
        }
        (run_dir / "tjr-qa-report.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Real TJR draft MP4 count: %d", len(produced))
        return run_dir
    except Exception:
        (run_dir / "source-acquisition-errors.json").write_text(
            json.dumps({"source_attempts": failures}, indent=2) + "\n", encoding="utf-8"
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("tjr-artifacts"))
    parser.add_argument("--brief", type=Path, default=Path("campaigns/reach-tjr-weekly.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(render_preview(args.artifact_root, args.brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
