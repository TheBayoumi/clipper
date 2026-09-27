#!/usr/bin/env python3
"""Review-only clips from Reach's TWO explicitly listed TJR YouTube channels.

No Kick/reposts/search results. Verify the video owner independently before
acquisition, and fail closed if YouTube denies direct media access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET

from clipper.brief import load_brief
from clipper.models import ClipCandidate
from clipper.render import FFmpegRenderer
from clipper.scoring import score_transcript
from clipper.source_fidelity import probe_source_profile
from clipper.tiktok import audit_tiktok_ass
from clipper.transcript import transcribe_with_faster_whisper
from scripts.tjr_editorial import RUBRIC_VERSION, WEIGHTS, select_editorial_moments
from scripts.tjr_quality import check_full_decode, probe_original, probe_video

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
            LOGGER.info("Official %s feed supplied %d videos", CHANNELS[channel_id], len(feed_items))
            candidates.extend(feed_items)
        except Exception as exc:
            failures.append({"source": url, "error": f"{type(exc).__name__}: {exc}"[:500]})
        # An RSS feed populated by recent Shorts must not hide the official
        # channel's longer /videos uploads or completed /streams broadcasts.
        short_only = not feed_items or all(
            item.duration_seconds is not None and item.duration_seconds < 90
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
            if metadata.get("is_live") or metadata.get("live_status") == "is_live":
                raise RuntimeError("current livestream is not a completed source video")
            if float(metadata.get("duration") or 0) < 90:
                raise RuntimeError("video is shorter than the requested clipping workflow")
            metadata["_verified_client_args"] = extra
            return metadata
        except (ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(str(exc)[-600:])
    raise RuntimeError("source metadata unavailable: " + " | ".join(errors))


def download_original_excerpt(
    video: OfficialVideo, work: Path, *, metadata: dict[str, Any] | None = None
) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    common = [
        "yt-dlp",
        *_auth_args(),
        "--no-playlist",
        "--no-warnings",
        "--merge-output-format",
        "mp4",
        "--download-sections",
        "*00:00:00-00:14:00",
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
            return files[0]
        except RuntimeError as exc:
            errors.append(f"{' '.join(variant) or 'default'}: {str(exc)[-550:]}")
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
            return (4 if video.duration_seconds and video.duration_seconds >= 90 else 2, video.published)
        if video.duration_seconds is not None and video.duration_seconds >= 90:
            return (3, video.published)
        if title in {"", "unknown", "#tjr"} or ("#" in title and len(title) < 45):
            return (0, video.published)
        return (1, video.published)

    return sorted(videos, key=priority, reverse=True)


def constrain_official_sources(
    candidates: list[OfficialVideo], requested_id: str | None
) -> list[OfficialVideo]:
    """Honor an explicit exact YouTube source instead of trying other videos.

    No unverified URL and no fallback to other creators or Kick are accepted.
    """
    if not requested_id:
        return prioritize_campaign_moments(candidates)
    if not VIDEO_ID.fullmatch(requested_id):
        raise RuntimeError("selected source_video_id must be exactly 11 YouTube ID characters")
    matches = [video for video in candidates if video.video_id == requested_id]
    if len(matches) != 1 or matches[0].channel_id not in CHANNELS:
        raise RuntimeError("selected source_video_id is not in either Reach-listed channel feed")
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
    if int(data.get("duration") or 0) < 90:
        raise RuntimeError("browser original is too short for the campaign")
    original = Path(str(data.get("source_path") or ""))
    if not original.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("source_sha256"))):
        raise RuntimeError("browser original is missing or has no valid SHA-256")
    with original.open("rb") as source_bytes:
        actual_digest = hashlib.file_digest(source_bytes, "sha256").hexdigest()
    if actual_digest != data["source_sha256"]:
        raise RuntimeError("browser original SHA-256 does not match Chrome capture manifest")
    probe_original(original)
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
        official_candidates = constrain_official_sources(candidates, requested_id)
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
        segments = transcribe_with_faster_whisper(
            source,
            model_name="small.en",
            device="cpu",
            compute_type="int8",
            language="en",
            word_timestamps=True,
        )
        if not segments:
            raise RuntimeError("the original footage contains no usable English speech")
        step = "clip_selection"
        (run_dir / "transcript.json").write_text(
            json.dumps([s.to_dict() for s in segments], indent=2) + "\n",
            encoding="utf-8",
        )
        batch_limit = int(os.getenv("TJR_EDITORIAL_BATCH_LIMIT", str(brief.clip_count)))
        strict = score_transcript(
            brief, chosen_video.video_id, segments, limit=900, sentence_boundaries=True
        )
        ranked = strict
        picks, rejected = select_editorial_moments(
            ranked, batch_limit=batch_limit, segments=segments
        )
        screening_mode = "strict_sentence_or_700ms_pause"
        relaxed_count = 0
        if not picks:
            # Retry at genuine, shorter ASR-aligned pauses rather than cutting
            # arbitrary mid-sentence transcript windows or lowering the rubric.
            relaxed = score_transcript(
                brief,
                chosen_video.video_id,
                segments,
                limit=900,
                sentence_boundaries=True,
                pause_threshold=0.35,
            )
            relaxed_count = len(relaxed)
            relaxed_picks, relaxed_rejected = select_editorial_moments(
                relaxed, batch_limit=batch_limit, segments=segments
            )
            if relaxed_picks:
                ranked, picks, rejected = relaxed, relaxed_picks, relaxed_rejected
                screening_mode = "aligned_350ms_pause_manual_boundary_review"
            else:
                rejected.extend(relaxed_rejected)
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
                    "screening_mode": screening_mode,
                    "transcript_segment_count": len(segments),
                    "strict_candidate_count": len(strict),
                    "relaxed_candidate_count": relaxed_count,
                    "selected": [pick.to_dict() for pick in picks],
                    "rejected": rejected[:250],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not picks:
            raise NoEditorialMoments(
                "no distinct moments passed the provisional editorial rubric; "
                f"strict={len(strict)} relaxed={relaxed_count} "
                f"transcript_segments={len(segments)}; consult editorial-candidate-audit.json"
            )
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
            layout = (
                "tjr-trading-logo-safe"
                if chosen_video.video_id == "p2LU37eat70"
                and probe_original(source) == {"width": 1920, "height": 1080}
                else "default"
            )
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
