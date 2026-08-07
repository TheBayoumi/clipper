from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

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
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    artifact_root: Path = Path("artifacts")
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        return cls(
            artifact_root=Path(os.getenv("CLIPPER_ARTIFACT_ROOT", "artifacts")),
            whisper_model=os.getenv("CLIPPER_WHISPER_MODEL", "small"),
            whisper_device=os.getenv("CLIPPER_WHISPER_DEVICE", "auto"),
            whisper_compute_type=os.getenv("CLIPPER_WHISPER_COMPUTE_TYPE", "int8"),
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_id(campaign_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{campaign_id}-{timestamp}"


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
    active_renderer = renderer or (FFmpegRenderer() if render else None)

    run_dir = cfg.artifact_root / _run_id(brief.campaign_id)
    work_dir = run_dir / "work"
    clips_dir = run_dir / "clips"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = PipelineManifest(campaign_id=brief.campaign_id)
    _write_json(run_dir / "brief.normalized.json", brief.to_dict())

    discovered = source.discover(brief)
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
