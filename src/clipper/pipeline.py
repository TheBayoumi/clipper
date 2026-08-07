from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import gdown

from .brief import load_brief
from .models import (
    CampaignBrief,
    ClipCandidate,
    PipelineManifest,
    RenderedClip,
    TranscriptSegment,
    VideoCandidate,
)
from .render import FFmpegRenderer
from .rights import RightsError, assert_campaign_authorized, assert_video_allowed
from .scoring import score_transcript, select_diverse_clips
from .transcript import load_vtt, transcribe_with_faster_whisper
from .youtube import YouTubeClient

LOGGER = logging.getLogger("clipper")


class SourceClient(Protocol):
    def discover(self, brief: CampaignBrief) -> list[VideoCandidate]: ...

    def download_subtitles(
        self,
        video: VideoCandidate,
        work_dir: Path,
        language: str,
    ) -> Path | None: ...

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path: ...


class Renderer(Protocol):
    def render(
        self,
        source_path: Path,
        output_path: Path,
        clip: ClipCandidate,
        segments: Sequence[TranscriptSegment],
        watermark_path: Path | None = None,
    ) -> Path: ...


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    artifact_root: Path = Path("artifacts")
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    speaker_focus: bool = True
    speaker_zoom: float = 1.12
    speaker_sample_fps: float = 4.0
    speaker_switch_margin: float = 1.35
    speaker_transition_seconds: float = 0.22
    speaker_window_seconds: float = 0.8
    speaker_min_detection_coverage: float = 0.35

    @classmethod
    def from_env(cls) -> PipelineSettings:
        return cls(
            artifact_root=Path(os.getenv("CLIPPER_ARTIFACT_ROOT", "artifacts")),
            whisper_model=os.getenv("CLIPPER_WHISPER_MODEL", "small"),
            whisper_device=os.getenv("CLIPPER_WHISPER_DEVICE", "auto"),
            whisper_compute_type=os.getenv("CLIPPER_WHISPER_COMPUTE_TYPE", "int8"),
            speaker_focus=_env_bool(
                "CLIPPER_SPEAKER_FOCUS", _env_bool("CLIPPER_FACE_TRACKING", True)
            ),
            speaker_zoom=float(
                os.getenv("CLIPPER_SPEAKER_ZOOM", os.getenv("CLIPPER_FACE_ZOOM", "1.12"))
            ),
            speaker_sample_fps=float(
                os.getenv("CLIPPER_SPEAKER_SAMPLE_FPS", os.getenv("CLIPPER_FACE_SAMPLE_FPS", "4.0"))
            ),
            speaker_switch_margin=float(os.getenv("CLIPPER_SPEAKER_SWITCH_MARGIN", "1.35")),
            speaker_transition_seconds=float(
                os.getenv("CLIPPER_SPEAKER_TRANSITION_SECONDS", "0.22")
            ),
            speaker_window_seconds=float(os.getenv("CLIPPER_SPEAKER_WINDOW_SECONDS", "0.8")),
            speaker_min_detection_coverage=float(
                os.getenv("CLIPPER_SPEAKER_MIN_DETECTION_COVERAGE", "0.35")
            ),
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_id(campaign_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{campaign_id}-{timestamp}"


def _normalize_asset_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("campaign assets must use https")
    if parsed.netloc == "drive.google.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "file" and parts[1] == "d":
            query = urlencode({"id": parts[2], "export": "download", "confirm": "t"})
            return f"https://drive.usercontent.google.com/download?{query}"
        file_id = parse_qs(parsed.query).get("id", [None])[0]
        if file_id:
            query = urlencode({"id": file_id, "export": "download", "confirm": "t"})
            return f"https://drive.usercontent.google.com/download?{query}"
    return url


def _download_google_drive_media(url: str, output_path: Path, *, max_bytes: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    try:
        downloaded = gdown.download(  # type: ignore[attr-defined]
            url=url, output=str(temporary), quiet=True
        )
        if not downloaded or not temporary.is_file():
            raise RuntimeError("Google Drive media download did not create a file")
        size = temporary.stat().st_size
        if size == 0:
            raise RuntimeError("media asset is empty")
        if size > max_bytes:
            raise RuntimeError("media asset exceeds size limit")
        temporary.replace(output_path)
        return output_path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _download_asset(
    url: str,
    output_path: Path,
    *,
    max_bytes: int = 10_000_000,
    expected_kind: str = "image",
) -> Path:
    if expected_kind == "media" and urlparse(url).netloc == "drive.google.com":
        return _download_google_drive_media(url, output_path, max_bytes=max_bytes)
    normalized = _normalize_asset_url(url)
    request = Request(  # noqa: S310 -- _normalize_asset_url enforces HTTPS.
        normalized, headers={"User-Agent": "whop-clipper/0.1"}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    size = 0
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:  # noqa: S310
            content_type = response.headers.get_content_type()
            if expected_kind == "image" and not content_type.startswith("image/"):
                raise RuntimeError(f"watermark response is not an image: {content_type}")
            if expected_kind == "media" and content_type.startswith("text/"):
                raise RuntimeError(f"media response is not binary media: {content_type}")
            while chunk := response.read(64 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise RuntimeError(f"{expected_kind} asset exceeds size limit")
                handle.write(chunk)
        if size == 0:
            raise RuntimeError(f"{expected_kind} asset is empty")
        temporary.replace(output_path)
        return output_path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _campaign_media_candidates(brief: CampaignBrief) -> list[VideoCandidate]:
    channel_id = brief.source_channel_ids[0] if brief.source_channel_ids else ""
    return [
        VideoCandidate(
            video_id=video_id,
            title=f"{brief.title} campaign source",
            channel_id=channel_id,
            channel_title="Campaign-provided source",
            url=f"https://www.youtube.com/watch?v={video_id}",
        )
        for video_id in brief.allowed_video_ids
        if video_id in brief.source_media_urls
    ]


def run_pipeline(
    brief_path: str | Path,
    *,
    settings: PipelineSettings | None = None,
    source_client: SourceClient | None = None,
    renderer: Renderer | None = None,
    render: bool = True,
) -> Path:
    brief = load_brief(brief_path)
    assert_campaign_authorized(brief)
    cfg = settings or PipelineSettings.from_env()
    source = source_client or YouTubeClient()
    active_renderer = renderer or (
        FFmpegRenderer(
            speaker_focus=cfg.speaker_focus,
            zoom_factor=cfg.speaker_zoom,
            speaker_sample_fps=cfg.speaker_sample_fps,
            speaker_switch_margin=cfg.speaker_switch_margin,
            speaker_transition_seconds=cfg.speaker_transition_seconds,
            speaker_window_seconds=cfg.speaker_window_seconds,
            speaker_min_detection_coverage=cfg.speaker_min_detection_coverage,
        )
        if render
        else None
    )

    run_dir = cfg.artifact_root / _run_id(brief.campaign_id)
    work_dir = run_dir / "work"
    clips_dir = run_dir / "clips"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = PipelineManifest(campaign_id=brief.campaign_id)
    _write_json(run_dir / "brief.normalized.json", brief.to_dict())
    watermark_path: Path | None = None
    if brief.watermark_url:
        watermark_path = _download_asset(brief.watermark_url, run_dir / "assets" / "watermark.png")

    direct_candidates = _campaign_media_candidates(brief)
    direct_ids = {video.video_id for video in direct_candidates}
    if brief.allowed_video_ids and set(brief.allowed_video_ids).issubset(direct_ids):
        discovered = direct_candidates
    else:
        discovered = source.discover(brief)
        discovered_ids = {video.video_id for video in discovered}
        discovered.extend(
            video for video in direct_candidates if video.video_id not in discovered_ids
        )
    allowed: list[VideoCandidate] = []
    for video in discovered:
        try:
            assert_video_allowed(brief, video)
        except RightsError as exc:
            manifest.errors.append({"video_id": video.video_id, "error": str(exc)})
            continue
        allowed.append(video)
    allowed = allowed[: brief.source_limit]
    manifest.discovered_videos = [video.to_dict() for video in allowed]

    transcripts: dict[str, list[TranscriptSegment]] = {}
    media_paths: dict[str, Path] = {}
    all_candidates: list[ClipCandidate] = []
    video_index = {video.video_id: video for video in allowed}

    for video in allowed:
        video_work = work_dir / video.video_id
        try:
            direct_media_url = brief.source_media_urls.get(video.video_id)
            if direct_media_url:
                media_path = _download_asset(
                    direct_media_url,
                    video_work / "source.mp4",
                    max_bytes=4_000_000_000,
                    expected_kind="media",
                )
                media_paths[video.video_id] = media_path
                segments = transcribe_with_faster_whisper(
                    media_path,
                    model_name=cfg.whisper_model,
                    device=cfg.whisper_device,
                    compute_type=cfg.whisper_compute_type,
                    language=brief.language,
                )
            else:
                subtitle_path = source.download_subtitles(video, video_work, brief.language)
                if subtitle_path:
                    segments = load_vtt(subtitle_path)
                else:
                    media_path = source.download_media(video, video_work)
                    media_paths[video.video_id] = media_path
                    segments = transcribe_with_faster_whisper(
                        media_path,
                        model_name=cfg.whisper_model,
                        device=cfg.whisper_device,
                        compute_type=cfg.whisper_compute_type,
                        language=brief.language,
                    )
            if not segments:
                raise RuntimeError("transcription produced no timestamped segments")
            transcripts[video.video_id] = segments
            candidates = score_transcript(brief, video.video_id, segments)
            all_candidates.extend(candidates)
            _write_json(video_work / "transcript.json", [segment.to_dict() for segment in segments])
            _write_json(
                video_work / "clip-candidates.json",
                [item.to_dict() for item in candidates],
            )
        except Exception as exc:
            LOGGER.exception("source processing failed", extra={"video_id": video.video_id})
            manifest.errors.append({"video_id": video.video_id, "error": str(exc)})

    selected = select_diverse_clips(
        all_candidates,
        clip_count=brief.clip_count,
        max_per_source=brief.max_clips_per_source,
    )
    manifest.planned_clips = [candidate.to_dict() for candidate in selected]

    if render and active_renderer:
        for index, candidate in enumerate(selected, start=1):
            video = video_index[candidate.video_id]
            try:
                media_path = media_paths.get(video.video_id) or source.download_media(
                    video, work_dir / video.video_id
                )
                output_path = clips_dir / f"{index:02d}-{video.video_id}.mp4"
                rendered_path = active_renderer.render(
                    media_path,
                    output_path,
                    candidate,
                    transcripts[video.video_id],
                    watermark_path,
                )
                rendered = RenderedClip(
                    video_id=video.video_id,
                    output_path=str(rendered_path),
                    start=candidate.start,
                    end=candidate.end,
                    score=candidate.score,
                    source_url=video.url,
                )
                manifest.rendered_clips.append(rendered.to_dict())
            except Exception as exc:
                manifest.errors.append({"video_id": video.video_id, "error": str(exc)})

    _write_json(run_dir / "manifest.json", manifest.to_dict())
    return run_dir
