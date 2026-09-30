#!/usr/bin/env python3
"""Review-only clips from Reach's TWO explicitly listed TJR YouTube channels.

No Kick/reposts/search results. Verify the video owner independently before
acquisition, and fail closed if YouTube denies direct media access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET

from clipper.brief import load_brief
from clipper.models import ClipCandidate, TranscriptSegment, WordTiming
from clipper.render import FFmpegRenderer
from clipper.source_fidelity import probe_source_profile
from clipper.tiktok import audit_tiktok_ass
from clipper.transcript import FasterWhisperTranscriber
from scripts.tjr_editorial import (
    MAX_RENDERABLE_CLIPS,
    RUBRIC_VERSION,
    WEIGHTS,
    select_editorial_moments,
)
from scripts.tjr_quality import check_full_decode, probe_original, probe_video
from scripts.tjr_semantic_editor import build_semantic_editorial_candidates
from scripts.tjr_visual_analysis import analyze_candidate_visuals

LOGGER = logging.getLogger("tjr-youtube")
CHANNELS = {
    "UCGHBUXjDCeiIXNdKR0HUZnA": "@TJRTrades",
    "UCZen39LQJPx04GjPj7FOMcw": "@TRichesTrades",
}
ATOM = "{http://www.w3.org/2005/Atom}"
YT = "{http://www.youtube.com/xml/schemas/2015}"
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


@dataclass(frozen=True)
class OfficialVideo:
    video_id: str
    channel_id: str
    title: str
    published: str
    duration_seconds: float | None = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def parse_official_feed(xml: bytes, expected_channel: str) -> list[OfficialVideo]:
    if expected_channel not in CHANNELS:
        raise ValueError("non-campaign YouTube channel")
    feed = ET.fromstring(xml)
    yt_channel = (feed.findtext(f"{YT}channelId") or "").strip()
    atom_id = (feed.findtext(f"{ATOM}id") or "").strip()
    atom_channel = atom_id.removeprefix("yt:channel:") if atom_id else ""
    # Live YouTube feeds observed on 2026-09-26 expose IDs without the UC prefix.
    # Only accept the EXACT suffix of a previously approved, full channel ID.
    trusted_forms = {expected_channel, expected_channel.removeprefix("UC")}
    reported = [owner for owner in (yt_channel, atom_channel) if owner]
    if not reported or any(owner not in trusted_forms for owner in reported):
        raise ValueError(
            "YouTube feed owner mismatch: "
            f"root={feed.tag!r} yt_channel={yt_channel[:40]!r} "
            f"atom_id={atom_id[:60]!r}"
        )
    items: list[OfficialVideo] = []
    for entry in feed.findall(f"{ATOM}entry"):
        video_id = (entry.findtext(f"{YT}videoId") or "").strip()
        channel_id = (entry.findtext(f"{YT}channelId") or expected_channel).strip()
        published = (entry.findtext(f"{ATOM}published") or "").strip()
        if not VIDEO_ID.fullmatch(video_id) or channel_id not in trusted_forms or not published:
            continue
        items.append(
            OfficialVideo(
                video_id=video_id,
                channel_id=expected_channel,
                title=(entry.findtext(f"{ATOM}title") or "").strip(),
                published=published,
            )
        )
    return items


def _flat_channel_playlist(
    channel_id: str, *, section: str = "videos", limit: int = 36
) -> list[OfficialVideo]:
    """Discover older videos/stream replays on the exact allowlisted channel.

    RSS exposes only recent uploads, which can all be sub-90-second Shorts.
    Every discovered playlist entry is still independently owner-verified
    before download. Neither search results nor third-party reposts qualify.
    """
    if channel_id not in CHANNELS or section not in {"videos", "streams"}:
        raise ValueError("not a Reach-listed channel or approved playlist section")
    if not 1 <= limit <= 50:
        raise ValueError("playlist limit must be 1-50")
    url = f"https://www.youtube.com/channel/{channel_id}/{section}"
    result = invoke(
        [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--playlist-end",
            str(limit),
            url,
        ],
        timeout=180,
    )
    items: list[OfficialVideo] = []
    for line in result.stdout.splitlines():
        if not line.startswith("{"):
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id") or "")
        listed_owner = str(entry.get("channel_id") or "")
        if not VIDEO_ID.fullmatch(video_id):
            continue
        if listed_owner and listed_owner != channel_id:
            continue
        timestamp = entry.get("release_timestamp") or entry.get("timestamp")
        published = (
            datetime.fromtimestamp(float(timestamp), UTC).isoformat()
            if timestamp is not None
            else ""
        )
        raw_duration = entry.get("duration")
        duration = (
            float(raw_duration)
            if isinstance(raw_duration, (int, float)) and float(raw_duration) > 0
            else None
        )
        items.append(
            OfficialVideo(
                video_id=video_id,
                channel_id=channel_id,
                title=str(entry.get("title") or ""),
                published=published,
                duration_seconds=duration,
            )
        )
    if not items:
        raise RuntimeError("official channel playlist did not expose any candidate videos")
    LOGGER.info(
        "Found %d candidates in official %s/%s playlist; metadata still unverified",
        len(items),
        CHANNELS[channel_id],
        section,
    )
    return items


def discover_official_uploads() -> tuple[list[OfficialVideo], list[dict[str, str]]]:
    candidates: list[OfficialVideo] = []
    failures: list[dict[str, str]] = []
    for channel_id in CHANNELS:
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        feed_items: list[OfficialVideo] = []
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/atom+xml"}
            )
            try:
                with urllib.request.urlopen(request, timeout=25) as response:  # noqa: S310
                    content = response.read(2_000_000)
            except Exception:
                from curl_cffi import requests

                response = requests.get(url, impersonate="chrome", timeout=25)
                response.raise_for_status()
                content = response.content
            feed_items = parse_official_feed(content, channel_id)
            if not feed_items:
                raise RuntimeError("verified channel feed contains no recent upload entries")
            LOGGER.info(
                "Official %s feed supplied %d videos", CHANNELS[channel_id], len(feed_items)
            )
            candidates.extend(feed_items)
        except Exception as exc:
            failures.append({"source": url, "error": f"{type(exc).__name__}: {exc}"[:500]})
        # An RSS feed populated by recent Shorts must not hide the official
        # channel's longer /videos uploads or completed /streams broadcasts.
        short_only = not feed_items or all(
            (item.duration_seconds is not None and item.duration_seconds < 90)
            or item.title.lower().strip() in {"", "unknown", "#tjr"}
            or ("#" in item.title and len(item.title) < 45)
            for item in feed_items
        )
        sections = ("videos", "streams") if short_only else ("videos",)
        for section in sections:
            playlist_url = f"https://www.youtube.com/channel/{channel_id}/{section}"
            try:
                candidates.extend(_flat_channel_playlist(channel_id, section=section))
            except Exception as exc:
                failures.append(
                    {"source": playlist_url, "error": f"{type(exc).__name__}: {exc}"[-1000:]}
                )
    # Feed timestamps are authoritative; supplement duplicates with duration
    # metadata from the official channel's own playlist when available.
    candidates.sort(key=lambda item: item.published, reverse=True)
    unique: dict[str, OfficialVideo] = {}
    for item in candidates:
        existing = unique.get(item.video_id)
        if existing is None:
            unique[item.video_id] = item
        elif existing.duration_seconds is None and item.duration_seconds is not None:
            unique[item.video_id] = replace(existing, duration_seconds=item.duration_seconds)
    return list(unique.values()), failures


def _auth_args() -> list[str]:
    """Optionally use an encrypted, explicitly supplied dedicated viewer session."""
    value = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
    if not value:
        return []
    path = Path(value)
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError("configured YouTube cookie file is empty or unavailable")
    return ["--cookies", str(path)]


def invoke(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr or exc.stdout or str(exc)
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise RuntimeError(f"{command[0]}: {detail[-1200:]}") from exc


def dynamic_browser_variants() -> tuple[tuple[str, ...], ...]:
    """Mint per-video guest playback tokens using a fresh runner browser.

    Browser-token generation is an automatic public-video transport attempt,
    not a substitute for authenticated access if YouTube blocks runner IPs.
    """
    browser = os.getenv("YT_DLP_WPC_BROWSER_PATH", "").strip()
    if not browser:
        return ()
    path = Path(browser)
    if not path.is_file():
        raise RuntimeError("dynamic YouTube token browser executable is missing")
    plugin = ("--extractor-args", f"youtubepot-wpc:browser_path={path}")
    return (
        ("--extractor-args", "youtube:player_client=mweb", *plugin),
        ("--extractor-args", "youtube:player_client=web_safari", *plugin),
    )


def verified_youtube_metadata(video: OfficialVideo) -> dict[str, Any]:
    """Do not trust a title, channel handle, RSS alone, or search result for provenance."""
    errors: list[str] = []
    # Try browser-minted guest PO tokens first. YouTube may still independently
    # refuse GitHub's public IP before any video metadata can be retrieved.
    client_variants = (
        *dynamic_browser_variants(),
        ("--extractor-args", "youtube:player_client=mweb"),
        ("--extractor-args", "youtube:player_client=tv"),
        ("--extractor-args", "youtube:player_client=web_safari"),
        ("--extractor-args", "youtube:player_client=web_embedded"),
        ("--extractor-args", "youtube:player_client=android_vr"),
        ("--impersonate", "chrome", "--extractor-args", "youtube:player_client=web_safari"),
    )
    for variant in client_variants:
        extra = list(variant)
        try:
            result = invoke(
                [
                    "yt-dlp",
                    *_auth_args(),
                    "--no-warnings",
                    "--no-playlist",
                    "--skip-download",
                    "--dump-single-json",
                    *extra,
                    video.url,
                ],
                timeout=150,
            )
            metadata = json.loads(result.stdout)
            if not isinstance(metadata, dict):
                raise RuntimeError("unexpected YouTube metadata")
            if metadata.get("id") != video.video_id:
                raise RuntimeError("YouTube video ID does not match the official feed")
            if metadata.get("channel_id") != video.channel_id:
                raise RuntimeError("YouTube owner ID does not match Reach's channel")
            live_status = str(metadata.get("live_status") or "")
            if (
                metadata.get("is_live")
                or metadata.get("is_upcoming")
                or live_status in {"is_live", "is_upcoming"}
            ):
                raise RuntimeError("live or upcoming stream is not an eligible source video")
            if float(metadata.get("duration") or 0) < 90:
                raise RuntimeError("video is shorter than the requested clipping workflow")
            metadata["_verified_client_args"] = extra
            return metadata
        except (ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(str(exc)[-600:])
    raise RuntimeError("source metadata unavailable: " + " | ".join(errors))


def youtube_scan_section_args(duration_seconds: float) -> list[str]:
    """Require full originals within the supported one-hour analysis window."""
    if duration_seconds <= 0:
        raise ValueError("verified YouTube duration must be positive")
    if duration_seconds > 3600:
        raise ValueError("SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT: source exceeds one hour")
    return []


def verify_complete_download(path: Path, expected_seconds: float) -> float:
    """Fail closed on valid-looking but incomplete DASH/HLS downloads."""
    if not 90 <= expected_seconds <= 3600:
        raise RuntimeError("SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT")
    measured = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    if measured.returncode:
        raise RuntimeError("SOURCE_DURATION_UNVERIFIABLE")
    try:
        seconds = float(measured.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("SOURCE_DURATION_UNVERIFIABLE") from exc
    if not math.isfinite(seconds) or seconds + 30 < expected_seconds:
        raise RuntimeError("SOURCE_DURATION_INCOMPLETE: original has missing DASH/HLS fragments")
    return seconds


def download_original_excerpt(
    video: OfficialVideo, work: Path, *, metadata: dict[str, Any] | None = None
) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    expected_seconds = float((metadata or {}).get("duration") or video.duration_seconds or 0)
    common = [
        "yt-dlp",
        *_auth_args(),
        "--no-playlist",
        "--no-warnings",
        "--abort-on-unavailable-fragments",
        "--merge-output-format",
        "mp4",
        *youtube_scan_section_args(expected_seconds),
        "-f",
        "bv*[height>=720][height<=1080]+ba/b[height>=720]/bv*+ba/b",
        "-o",
        str(work / "source.%(ext)s"),
    ]
    preferred = tuple((metadata or {}).get("_verified_client_args") or ())
    client_variants = (
        preferred,
        *dynamic_browser_variants(),
        ("--extractor-args", "youtube:player_client=mweb"),
        ("--extractor-args", "youtube:player_client=tv"),
        ("--extractor-args", "youtube:player_client=web_safari"),
        ("--extractor-args", "youtube:player_client=web_embedded"),
        ("--extractor-args", "youtube:player_client=android_vr"),
    )
    errors: list[str] = []
    attempted: set[tuple[str, ...]] = set()
    for variant in client_variants:
        if variant in attempted:
            continue
        attempted.add(variant)
        try:
            invoke([common[0], *variant, *common[1:], video.url], timeout=1500)
            files = sorted(
                p
                for p in work.glob("source.*")
                if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm"}
            )
            if not files:
                raise RuntimeError("YouTube media download produced no source file")
            probe_original(files[0])
            verify_complete_download(files[0], expected_seconds)
            return files[0]
        except RuntimeError as exc:
            errors.append(f"{' '.join(variant) or 'default'}: {str(exc)[-550:]}")
            # Never accept stale bytes left by a previously failed transport.
            for incomplete in work.glob("source.*"):
                if incomplete.is_file():
                    incomplete.unlink(missing_ok=True)
    raise RuntimeError("verified YouTube media inaccessible: " + " | ".join(errors))


def prioritize_campaign_moments(videos: list[OfficialVideo]) -> list[OfficialVideo]:
    """Prefer long-form, TJR-featured campaign moments over newer hashtag Shorts.

    Still inspect and independently validate each actual owner and duration.
    """

    def priority(video: OfficialVideo) -> tuple[int, str]:
        title = video.title.lower().strip()
        if video.duration_seconds is not None and video.duration_seconds < 90:
            return (-1, video.published)
        if any(term in title for term in ("trading", "livestream", "react", "tjr and", "stream")):
            return (
                4 if video.duration_seconds and video.duration_seconds >= 90 else 2,
                video.published,
            )
        if video.duration_seconds is not None and video.duration_seconds >= 90:
            return (3, video.published)
        if title in {"", "unknown", "#tjr"} or ("#" in title and len(title) < 45):
            return (0, video.published)
        return (1, video.published)

    return sorted(videos, key=priority, reverse=True)


def constrain_official_sources(
    candidates: list[OfficialVideo],
    requested_id: str | None,
    *,
    published_after: str | None = None,
    target_channel_id: str | None = None,
) -> list[OfficialVideo]:
    """Require source provenance, one target channel, and the campaign window.

    Missing or malformed publication dates never bypass the brief's cutoff.
    Direct production may discover a video dynamically, but it must never spill
    from the explicitly selected channel into the other Reach-listed channel.
    """
    if target_channel_id:
        if target_channel_id not in CHANNELS:
            raise RuntimeError("target channel is not one of the Reach-listed channels")
        candidates = [video for video in candidates if video.channel_id == target_channel_id]
    if published_after:
        cutoff = datetime.fromisoformat(published_after.replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)
        eligible: list[OfficialVideo] = []
        for video in candidates:
            if not video.published:
                continue
            try:
                published = datetime.fromisoformat(video.published.replace("Z", "+00:00"))
            except ValueError:
                continue
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
            if published.astimezone(UTC) >= cutoff.astimezone(UTC):
                eligible.append(video)
        candidates = eligible
    if not requested_id:
        if target_channel_id:
            return sorted(candidates, key=lambda video: video.published, reverse=True)
        return prioritize_campaign_moments(candidates)
    if not VIDEO_ID.fullmatch(requested_id):
        raise RuntimeError("selected source_video_id must be exactly 11 YouTube ID characters")
    matches = [video for video in candidates if video.video_id == requested_id]
    if len(matches) != 1 or matches[0].channel_id not in CHANNELS:
        scope = (
            "the target Reach-listed channel feed"
            if target_channel_id
            else "either Reach-listed channel feed"
        )
        raise RuntimeError(f"selected source_video_id is not in {scope}")
    return matches


def load_verified_browser_original(
    path: Path, candidates: list[OfficialVideo]
) -> tuple[OfficialVideo, Path, dict[str, Any]] | None:
    """Use Chrome-extracted HD bytes ONLY when exact original and checksum match."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("browser original manifest is not an object")
    selected = next(
        (
            item
            for item in candidates
            if item.video_id == data.get("video_id")
            and item.channel_id == data.get("channel_id")
            and item.url == data.get("public_video_url")
        ),
        None,
    )
    if selected is None:
        raise RuntimeError("browser original is not in the verified official channel feed")
    reported_seconds = int(data.get("duration") or 0)
    if not 90 <= reported_seconds <= 3600:
        raise RuntimeError("SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT: invalid capture duration")
    original = Path(str(data.get("source_path") or ""))
    if not original.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("source_sha256"))):
        raise RuntimeError("browser original is missing or has no valid SHA-256")
    with original.open("rb") as source_bytes:
        actual_digest = hashlib.file_digest(source_bytes, "sha256").hexdigest()
    if actual_digest != data["source_sha256"]:
        raise RuntimeError("browser original SHA-256 does not match Chrome capture manifest")
    probe_original(original)
    duration_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(original),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    try:
        actual_seconds = float(duration_probe.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("captured original has no valid duration") from exc
    if not math.isfinite(actual_seconds) or actual_seconds + 30 < reported_seconds:
        raise RuntimeError("SOURCE_DURATION_INCOMPLETE: captured media is partial")
    metadata: dict[str, Any] = {
        "id": selected.video_id,
        "channel_id": selected.channel_id,
        "title": str(data.get("title") or selected.title),
        "duration": int(data["duration"]),
        "_transport": (
            data.get("source_transport")
            if data.get("source_transport") == "approved_sha256_mirror"
            else "verified_official_youtube_capture"
        ),
    }
    return selected, original, metadata


def select_separate_clips(candidates: list[ClipCandidate], count: int = 2) -> list[ClipCandidate]:
    chosen: list[ClipCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.start)):
        if any(
            candidate.start < existing.end + 2 and existing.start < candidate.end + 2
            for existing in chosen
        ):
            continue
        chosen.append(candidate)
        if len(chosen) >= count:
            break
    return chosen


class NoEditorialMoments(RuntimeError):
    """An approved source had no qualifying draft moments; try another official original."""


def transcribe_source_chunks(
    source: Path,
    run_dir: Path,
    *,
    chunk_seconds: int = 840,
) -> tuple[list[list[TranscriptSegment]], float]:
    """Transcribe bounded 14-minute audio chunks with original-video timestamps.

    The original HD source stays intact so selected clips can be rendered from
    their exact absolute offsets, including moments late in a livestream.
    """
    if not 60 <= chunk_seconds <= 900:
        raise ValueError("audio chunk duration must be between 60 and 900 seconds")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    duration = float(probe.stdout.strip())
    if not math.isfinite(duration) or duration < 1:
        raise RuntimeError("original has no usable audio duration")
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    chunks: list[list[TranscriptSegment]] = []
    transcriber = FasterWhisperTranscriber(
        model_name=os.getenv("TJR_ASR_MODEL", "distil-large-v3").strip() or "distil-large-v3",
        device=os.getenv("TJR_ASR_DEVICE", "cpu").strip() or "cpu",
        compute_type=os.getenv("TJR_ASR_COMPUTE_TYPE", "int8").strip() or "int8",
        language="en",
        word_timestamps=True,
    )
    for index, start in enumerate(range(0, math.ceil(duration), chunk_seconds)):
        length = min(float(chunk_seconds), duration - start)
        if length < 1:
            break
        audio = work / f"audio-chunk-{index:03d}.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    str(source),
                    "-t",
                    str(length),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ],
                check=True,
                capture_output=True,
                timeout=420,
            )
            local = transcriber.transcribe(audio)
            shifted = [
                TranscriptSegment(
                    item.start + start,
                    item.end + start,
                    item.text,
                    tuple(
                        WordTiming(word.start + start, word.end + start, word.text)
                        for word in item.words
                    ),
                )
                for item in local
            ]
            chunks.append(shifted)
        finally:
            audio.unlink(missing_ok=True)
    return chunks, duration


def render_youtube_previews(root: Path, brief_path: Path) -> Path:
    brief = load_brief(brief_path)
    if (
        set(brief.source_channel_ids) != set(CHANNELS)
        or not brief.rights_confirmed
        or brief.watermark_text
        or brief.watermark_url
        or brief.required_hashtags != ["#TJR"]
    ):
        raise RuntimeError("brief differs from Reach's exact approved YouTube channels or policy")
    run_dir = root / ("reach-tjr-youtube-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    errors: list[dict[str, str]] = []
    step = "channel_discovery"
    try:
        candidates, failures = discover_official_uploads()
        errors.extend(failures)
        (run_dir / "official-source-candidates.json").write_text(
            json.dumps(
                [
                    {
                        **asdict(item),
                        "url": item.url,
                        "channel_handle": CHANNELS[item.channel_id],
                    }
                    for item in candidates[:20]
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not candidates:
            raise RuntimeError("both official YouTube channel feeds unavailable")
        chosen_video: OfficialVideo | None = None
        source: Path | None = None
        metadata: dict[str, Any] = {}
        bot_challenges = 0
        # Put full videos before Shorts: a new 15-second hashtag Short is not
        # a suitable 20-42s clip source and must not consume the bot budget.
        requested_id = os.getenv("TJR_SOURCE_VIDEO_ID", "").strip()
        target_channel_id = os.getenv("TJR_TARGET_CHANNEL_ID", "").strip() or None
        official_candidates = constrain_official_sources(
            candidates,
            requested_id,
            published_after=brief.published_after,
            target_channel_id=target_channel_id,
        )
        if requested_id:
            LOGGER.info("Using explicitly requested official YouTube video ID: %s", requested_id)
        browser_capture_file = os.getenv("TJR_BROWSER_CAPTURE_FILE", "").strip()
        if browser_capture_file:
            step = "verified_browser_capture"
            try:
                captured = load_verified_browser_original(
                    Path(browser_capture_file), official_candidates
                )
                if captured is not None:
                    chosen_video, source, metadata = captured
                    LOGGER.info(
                        "Chrome obtained real original HD bytes from official video %s",
                        chosen_video.video_id,
                    )
            except (ValueError, OSError, RuntimeError) as exc:
                if os.getenv("TJR_REQUIRE_STAGED_ORIGINAL") == "1":
                    raise RuntimeError(
                        "required approved staged original failed verification"
                    ) from exc
                errors.append({"source": "verified source capture", "error": str(exc)[:650]})
        if os.getenv("TJR_REQUIRE_STAGED_ORIGINAL") == "1" and source is None:
            raise RuntimeError("required approved staged original was not available")
        for video in official_candidates[:8] if source is None else []:
            step = "official_metadata"
            try:
                metadata = verified_youtube_metadata(video)
                step = "youtube_original_download"
                source = download_original_excerpt(
                    video, run_dir / "work" / video.video_id, metadata=metadata
                )
                chosen_video = video
                break
            except (RuntimeError, ValueError) as exc:
                errors.append({"source": video.url, "error": str(exc)[-1400:]})
                if "Sign in to confirm" in str(exc):
                    bot_challenges += 1
                    if bot_challenges >= 2:
                        raise RuntimeError(
                            "YouTube bot confirmation blocks this GitHub-hosted runner; "
                            "use an authentic original from one of the verified source URLs"
                        ) from exc
        if source is None or chosen_video is None:
            raise RuntimeError("no recent approved-channel YouTube original could be downloaded")
        source_profile = probe_source_profile(source)
        step = "transcription"
        source_chunks, analyzed_seconds = transcribe_source_chunks(source, run_dir)
        segments = [item for chunk in source_chunks for item in chunk]
        (run_dir / "source-analysis-coverage.json").write_text(
            json.dumps(
                {
                    "reported_original_seconds": float(metadata.get("duration") or 0),
                    "analyzed_source_seconds": round(analyzed_seconds, 2),
                    "full_source_analyzed": (
                        analyzed_seconds + 30 >= float(metadata.get("duration") or 0)
                    ),
                    "chunks": [
                        {
                            "index": index,
                            "start": index * 840,
                            "end": round(min((index + 1) * 840, analyzed_seconds), 2),
                            "transcript_segments": len(chunk),
                        }
                        for index, chunk in enumerate(source_chunks)
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not segments:
            raise RuntimeError("the original footage contains no usable English speech")
        step = "clip_selection"
        (run_dir / "transcript.json").write_text(
            json.dumps([s.to_dict() for s in segments], indent=2) + "\n",
            encoding="utf-8",
        )
        render_safety_limit = int(os.getenv("TJR_RENDER_SAFETY_LIMIT", str(MAX_RENDERABLE_CLIPS)))
        ranked, semantic_audit = build_semantic_editorial_candidates(
            brief,
            chosen_video.video_id,
            segments,
        )
        screening_mode = str(semantic_audit["architecture"])
        picks, rejected = select_editorial_moments(
            ranked,
            render_safety_limit=render_safety_limit,
            segments=segments,
            allow_review_only_opening=True,
            review_provider=lambda candidate: analyze_candidate_visuals(source, candidate),
        )
        (run_dir / "ranked-candidates.json").write_text(
            json.dumps([c.to_dict() for c in ranked[:150]], indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "editorial-candidate-audit.json").write_text(
            json.dumps(
                {
                    "rubric_version": RUBRIC_VERSION,
                    "weights": WEIGHTS,
                    "provisional": True,
                    "requires_manual_visual_and_integrity_review": True,
                    "measured_visual_precheck": True,
                    "screening_mode": screening_mode,
                    "semantic_architecture": semantic_audit,
                    "selection_policy": "quality_driven_zero_to_n",
                    "render_safety_limit": render_safety_limit,
                    "selected_count": len(picks),
                    "render_safety_limit_rejections": sum(
                        item.get("reason") == "RENDER_SAFETY_LIMIT" for item in rejected
                    ),
                    "transcript_segment_count": len(segments),
                    "semantic_candidate_count": len(ranked),
                    "selected": [pick.to_dict() for pick in picks],
                    "rejected": rejected[:250],
                    "rejection_breakdown": dict(Counter(str(item["reason"]) for item in rejected)),
                    "source_chunks_analyzed": len(source_chunks),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not picks:
            with source.open("rb") as media:
                digest = hashlib.file_digest(media, "sha256").hexdigest()
            report = {
                "status": "NO_CREATOR_GRADE_MOMENTS",
                "campaign": brief.campaign_id,
                "source_platform": "youtube",
                "source_url": chosen_video.url,
                "source_channel_id": chosen_video.channel_id,
                "source_channel_handle": CHANNELS[chosen_video.channel_id],
                "source_title": metadata.get("title"),
                "source_published_at": chosen_video.published,
                "source_sha256": digest,
                "source_transport": metadata.get("_transport", "verified_official_youtube"),
                "source_dimensions": probe_original(source),
                "source_profile": source_profile.as_dict(),
                "editorial_rubric_version": RUBRIC_VERSION,
                "editorial_weights": WEIGHTS,
                "selection_policy": "quality_driven_zero_to_n",
                "render_safety_limit": render_safety_limit,
                "selected_clip_count": 0,
                "clips": [],
                "source_attempts": errors,
                "manual_checks": [
                    "No clip passed the creator-grade editorial gates for this source.",
                    "Do not manufacture filler or switch sources because of editorial weakness.",
                ],
            }
            (run_dir / "tjr-youtube-qa-report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            LOGGER.info(
                "EDITORIAL_NOOP=NO_CREATOR_GRADE_MOMENTS semantic_candidates=%d",
                len(ranked),
            )
            return run_dir
        LOGGER.info(
            "EDITORIAL_PROVISIONAL_SCREEN_PASSED=%d rubric=%s screening=%s",
            len(picks),
            RUBRIC_VERSION,
            screening_mode,
        )
        step = "render_and_decode"
        renderer = FFmpegRenderer()
        caption_style = os.getenv("TJR_CAPTION_STYLE", "").strip().upper()
        if caption_style != "B2":
            raise RuntimeError(
                "TJR real-source drafts require Style B2 captions and persistent hooks"
            )
        completed: list[dict[str, Any]] = []
        for number, pick in enumerate(picks, start=1):
            clip = pick.clip
            out = run_dir / "clips" / f"{number:02d}-tjr-youtube-{chosen_video.video_id}.mp4"
            # Explicit approved crop for the visually audited 2026-09-25
            # TRiches livestream. New layouts remain review-only until audited.
            dimensions = probe_original(source)
            if dimensions == {"width": 1920, "height": 1080}:
                if chosen_video.video_id == "p2LU37eat70":
                    layout = "tjr-trading-logo-safe"
                elif chosen_video.video_id == "LvnemCfJpQU":
                    layout = "tjr-memecoin-logo-safe"
                else:
                    layout = "default"
            else:
                layout = "default"
            renderer.render(
                source,
                out,
                clip,
                segments,
                editorial_layout=layout,
                tiktok_hook=pick.hook if caption_style in {"B", "B2"} else None,
                source_profile=source_profile if caption_style in {"B", "B2"} else None,
            )
            overlay_acceptance = (
                audit_tiktok_ass(out.with_suffix(".ass"), clip_duration=clip.duration)
                if caption_style == "B2"
                else None
            )
            details = probe_video(out)
            if (
                caption_style in {"B", "B2"}
                and abs(details["fps"] - float(Fraction(source_profile.fps))) > 0.04
            ):
                raise RuntimeError("finished TikTok output changed native source frame rate")
            check_full_decode(out)
            thumbnail = out.with_name(out.stem + "-preview.png")
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
                    str(out),
                    "-frames:v",
                    "1",
                    str(thumbnail),
                ],
                timeout=90,
            )
            sheet = out.with_name(out.stem + "-contact.png")
            invoke(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(out),
                    "-vf",
                    "fps=1/6,scale=270:480,tile=4x1",
                    "-frames:v",
                    "1",
                    str(sheet),
                ],
                timeout=110,
            )
            completed.append(
                {
                    **details,
                    "hook_candidate": pick.hook,
                    "caption_style": caption_style if caption_style else "legacy_srt",
                    "creative_headline": pick.hook if caption_style == "B2" else None,
                    "overlay_acceptance": overlay_acceptance,
                    "burned_in_hook": caption_style in {"B", "B2"},
                    "ass_sidecar": (
                        str(out.with_suffix(".ass").relative_to(run_dir))
                        if caption_style in {"B", "B2"}
                        else None
                    ),
                    "file_megabytes": round(out.stat().st_size / 1_000_000, 2),
                    "source_fidelity": renderer.quality_results.get(str(out.resolve())),
                    "source_matched_quality": (
                        str(out.with_suffix(".quality.json").relative_to(run_dir))
                        if caption_style in {"B", "B2"}
                        else None
                    ),
                    "hook_score": pick.hook_score,
                    "editorial_rubric_version": RUBRIC_VERSION,
                    "editorial_score": pick.editorial_score,
                    "editorial_weighted_points": pick.weighted_points,
                    "editorial_score_coverage": pick.score_coverage,
                    "editorial_criteria": {
                        name: rating.to_dict() for name, rating in pick.criteria.items()
                    },
                    "editorial_integrity_gate": {
                        "status": pick.integrity_status,
                        "evidence": list(pick.integrity_evidence),
                    },
                    "editorial_reasons": list(pick.reasons),
                    "boundary_evidence": [
                        reason
                        for reason in clip.reasons
                        if reason.startswith(("start_boundary=", "end_boundary="))
                    ],
                    "publication_status": "AI_SCREEN_PASSED__VISUAL_REVIEW_REQUIRED",
                    "logo_safe_layout": layout,
                    "logo_compliance_verified": False,
                    "contact_sheet": str(sheet.relative_to(run_dir)),
                    "file": str(out.relative_to(run_dir)),
                    "srt": str(out.with_suffix(".srt").relative_to(run_dir)),
                    "preview": str(thumbnail.relative_to(run_dir)),
                    "source_url": chosen_video.url,
                    "source_start_seconds": clip.start,
                    "source_end_seconds": clip.end,
                    "review_required": True,
                }
            )
        with source.open("rb") as media:
            digest = hashlib.file_digest(media, "sha256").hexdigest()
        report = {
            "status": "TECHNICAL_QA_AND_AI_SCREEN_PASSED__VISUAL_REVIEW_REQUIRED",
            "campaign": brief.campaign_id,
            "source_platform": "youtube",
            "source_url": chosen_video.url,
            "source_channel_id": chosen_video.channel_id,
            "source_channel_handle": CHANNELS[chosen_video.channel_id],
            "source_title": metadata.get("title"),
            "source_published_at": chosen_video.published,
            "source_sha256": digest,
            "source_transport": metadata.get("_transport", "verified_official_youtube"),
            "source_dimensions": probe_original(source),
            "source_profile": source_profile.as_dict(),
            "editorial_rubric_version": RUBRIC_VERSION,
            "editorial_weights": WEIGHTS,
            "selection_policy": "quality_driven_zero_to_n",
            "render_safety_limit": render_safety_limit,
            "selected_clip_count": len(completed),
            "clips": completed,
            "source_attempts": errors,
            "manual_checks": [
                "Watch each draft to verify TJR is actually on screen and portrayed appropriately.",
                "Check spoken words against subtitles, story, retention, framing and hooks.",
                "Assign a verified portrait-visual score and explicitly approve or reject "
                "editorial integrity after watching the full source context.",
                "Reject any source logos, watermarks, synthetic visuals or AI voices.",
                "Verify Reach/Whop account eligibility, remaining budget and audience.",
                "Only publish after human approval; add #TJR and submit within 30 minutes.",
            ],
        }
        (run_dir / "tjr-youtube-qa-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("REAL_VERIFIED_YOUTUBE_MP4_COUNT=%d", len(completed))
        return run_dir
    except Exception as exc:
        (run_dir / "source-acquisition-errors.json").write_text(
            json.dumps({"stage": step, "error": str(exc)[-2000:], "attempts": errors}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, default=Path("campaigns/reach-tjr-weekly.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("tjr-youtube-artifacts"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(render_youtube_previews(args.artifact_root, args.brief))
    except Exception:
        LOGGER.exception("YouTube-only rendering failed; diagnostics were uploaded")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
