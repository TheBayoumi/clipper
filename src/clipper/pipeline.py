from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import gdown

from .brief import load_brief
from .cache import (
    CACHE_SCHEMA_VERSION,
    FileCache,
    analysis_cache_key,
    clip_concepts_from_payload,
    file_sha256,
    stable_hash,
    story_moments_from_payload,
    transcript_cache_key,
    transcript_segments_from_payload,
)
from .editorial import (
    build_edit_plan,
    discover_story_moments,
    generate_hook_variants,
    mine_clip_concepts,
    select_distinct_concepts,
    select_render_plans,
    select_submission_shortlist,
)
from .models import (
    CampaignBrief,
    ClipCandidate,
    ClipConcept,
    EditPlan,
    PipelineManifest,
    RenderedClip,
    StoryMoment,
    TranscriptSegment,
    VideoCandidate,
)
from .performance import RunTelemetry
from .qc import run_technical_qc
from .render import FFmpegRenderer
from .rights import RightsError, assert_campaign_authorized, assert_video_allowed
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
        edit_plan: EditPlan | None = None,
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
    source_max_height: int = 1080
    render_profile: str = "production"
    speaker_focus: bool = True
    speaker_zoom: float = 1.0
    speaker_sample_fps: float = 4.0
    speaker_switch_margin: float = 1.35
    speaker_min_reframe_seconds: float = 0.35
    speaker_max_reframe_seconds: float = 0.9
    speaker_seconds_per_crop: float = 0.75
    speaker_hold_threshold: float = 0.28
    speaker_reversal_guard_seconds: float = 1.25
    speaker_window_seconds: float = 0.8
    speaker_min_detection_coverage: float = 0.35
    cache_root: Path | None = None

    @classmethod
    def from_env(cls) -> PipelineSettings:
        return cls(
            artifact_root=Path(os.getenv("CLIPPER_ARTIFACT_ROOT", "artifacts")),
            whisper_model=os.getenv("CLIPPER_WHISPER_MODEL", "small"),
            whisper_device=os.getenv("CLIPPER_WHISPER_DEVICE", "auto"),
            whisper_compute_type=os.getenv("CLIPPER_WHISPER_COMPUTE_TYPE", "int8"),
            source_max_height=int(os.getenv("CLIPPER_SOURCE_MAX_HEIGHT", "1080")),
            render_profile=os.getenv("CLIPPER_RENDER_PROFILE", "production").strip().lower(),
            speaker_focus=_env_bool(
                "CLIPPER_SPEAKER_FOCUS", _env_bool("CLIPPER_FACE_TRACKING", True)
            ),
            speaker_zoom=float(
                os.getenv("CLIPPER_SPEAKER_ZOOM", os.getenv("CLIPPER_FACE_ZOOM", "1.0"))
            ),
            speaker_sample_fps=float(
                os.getenv("CLIPPER_SPEAKER_SAMPLE_FPS", os.getenv("CLIPPER_FACE_SAMPLE_FPS", "4.0"))
            ),
            speaker_switch_margin=float(os.getenv("CLIPPER_SPEAKER_SWITCH_MARGIN", "1.35")),
            speaker_min_reframe_seconds=float(
                os.getenv("CLIPPER_SPEAKER_MIN_REFRAME_SECONDS", "0.35")
            ),
            speaker_max_reframe_seconds=float(
                os.getenv("CLIPPER_SPEAKER_MAX_REFRAME_SECONDS", "0.9")
            ),
            speaker_seconds_per_crop=float(os.getenv("CLIPPER_SPEAKER_SECONDS_PER_CROP", "0.75")),
            speaker_hold_threshold=float(os.getenv("CLIPPER_SPEAKER_HOLD_THRESHOLD", "0.28")),
            speaker_reversal_guard_seconds=float(
                os.getenv("CLIPPER_SPEAKER_REVERSAL_GUARD_SECONDS", "1.25")
            ),
            speaker_window_seconds=float(os.getenv("CLIPPER_SPEAKER_WINDOW_SECONDS", "0.8")),
            speaker_min_detection_coverage=float(
                os.getenv("CLIPPER_SPEAKER_MIN_DETECTION_COVERAGE", "0.35")
            ),
            cache_root=(
                Path(os.environ["CLIPPER_CACHE_ROOT"]) if os.getenv("CLIPPER_CACHE_ROOT") else None
            ),
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return file_sha256(path)


def _record_source_media_metadata(
    manifest: PipelineManifest, video_id: str, media_path: Path
) -> None:
    metadata_path = media_path.with_suffix(".source.json")
    if not metadata_path.is_file():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    manifest.run_metadata.setdefault("source_media", {})[video_id] = payload


def _git_sha() -> str | None:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    value = process.stdout.strip()
    return value if process.returncode == 0 and value else None


def _cache_event(manifest: PipelineManifest, stage: str, key: str, hit: bool) -> None:
    manifest.cache.setdefault("hits", 0)
    manifest.cache.setdefault("misses", 0)
    manifest.cache.setdefault("events", [])
    counter = "hits" if hit else "misses"
    manifest.cache[counter] = int(manifest.cache[counter]) + 1
    events = manifest.cache["events"]
    if isinstance(events, list):
        events.append({"stage": stage, "key": key, "hit": hit})


def _cached_vtt_transcript(
    cache: FileCache,
    manifest: PipelineManifest,
    video_id: str,
    subtitle_path: Path,
    language: str,
) -> tuple[list[TranscriptSegment], str]:
    source_hash = file_sha256(subtitle_path)
    key = transcript_cache_key(
        video_id, source_hash, engine="youtube-vtt-word-parser-v2", language=language
    )
    cached = cache.read(key, "transcript")
    if cached is not None:
        try:
            segments = transcript_segments_from_payload(cached)
        except (KeyError, TypeError, ValueError):
            segments = []
        if segments:
            _cache_event(manifest, "transcript", key, True)
            return segments, source_hash
    segments = load_vtt(subtitle_path)
    cache.write(key, "transcript", [segment.to_dict() for segment in segments])
    _cache_event(manifest, "transcript", key, False)
    return segments, source_hash


def _cached_asr_transcript(
    cache: FileCache,
    manifest: PipelineManifest,
    video_id: str,
    media_path: Path,
    brief: CampaignBrief,
    cfg: PipelineSettings,
) -> tuple[list[TranscriptSegment], str]:
    source_hash = file_sha256(media_path)
    model_identity = f"{cfg.whisper_model}:{cfg.whisper_device}:{cfg.whisper_compute_type}"
    key = transcript_cache_key(
        video_id,
        source_hash,
        engine="faster-whisper-word-v1",
        model=model_identity,
        language=brief.language,
    )
    cached = cache.read(key, "transcript")
    if cached is not None:
        try:
            segments = transcript_segments_from_payload(cached)
        except (KeyError, TypeError, ValueError):
            segments = []
        if segments:
            _cache_event(manifest, "transcript", key, True)
            return segments, source_hash
    segments = transcribe_with_faster_whisper(
        media_path,
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        language=brief.language,
    )
    cache.write(key, "transcript", [segment.to_dict() for segment in segments])
    _cache_event(manifest, "transcript", key, False)
    return segments, source_hash


def _cached_editorial_analysis(
    cache: FileCache,
    manifest: PipelineManifest,
    brief: CampaignBrief,
    video_id: str,
    segments: list[TranscriptSegment],
) -> tuple[list[StoryMoment], list[ClipConcept]]:
    key = analysis_cache_key(video_id, segments, brief)
    cached_moments = cache.read(key, "story-moments")
    cached_concepts = cache.read(key, "clip-candidates")
    if cached_moments is not None and cached_concepts is not None:
        try:
            moments = story_moments_from_payload(cached_moments)
            concepts = clip_concepts_from_payload(cached_concepts)
        except (KeyError, TypeError, ValueError):
            moments, concepts = [], []
        if moments and concepts:
            _cache_event(manifest, "editorial-analysis", key, True)
            return moments, concepts
    moments = discover_story_moments(brief, video_id, segments)
    concepts = mine_clip_concepts(brief, video_id, segments, moments)
    cache.write(key, "story-moments", [item.to_dict() for item in moments])
    cache.write(key, "clip-candidates", [item.to_dict() for item in concepts])
    _cache_event(manifest, "editorial-analysis", key, False)
    return moments, concepts


def _safe_slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part)[:48] or "clip"


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
    telemetry = RunTelemetry()
    cache = FileCache(cfg.cache_root or (cfg.artifact_root / "_cache"))
    source = source_client or YouTubeClient(max_height=cfg.source_max_height)
    active_renderer = renderer or (
        FFmpegRenderer(
            speaker_focus=cfg.speaker_focus,
            zoom_factor=cfg.speaker_zoom,
            speaker_sample_fps=cfg.speaker_sample_fps,
            speaker_switch_margin=cfg.speaker_switch_margin,
            speaker_min_reframe_seconds=cfg.speaker_min_reframe_seconds,
            speaker_max_reframe_seconds=cfg.speaker_max_reframe_seconds,
            speaker_seconds_per_crop=cfg.speaker_seconds_per_crop,
            speaker_hold_threshold=cfg.speaker_hold_threshold,
            speaker_reversal_guard_seconds=cfg.speaker_reversal_guard_seconds,
            speaker_window_seconds=cfg.speaker_window_seconds,
            speaker_min_detection_coverage=cfg.speaker_min_detection_coverage,
            profile=cfg.render_profile,
        )
        if render
        else None
    )

    run_dir = cfg.artifact_root / _run_id(brief.campaign_id)
    work_dir = run_dir / "work"
    clips_dir = run_dir / "clips"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = PipelineManifest(campaign_id=brief.campaign_id)
    manifest.run_metadata = {
        "git_commit_sha": _git_sha(),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "transcription": {
            "engine": "faster-whisper-or-youtube-vtt",
            "model": cfg.whisper_model,
            "device": cfg.whisper_device,
            "compute_type": cfg.whisper_compute_type,
        },
        "source_hashes": {},
        "source_media": {},
        "transcript_hashes": {},
        "transcript_sources": {},
        "render": {
            "profile": cfg.render_profile,
            "source_max_height": cfg.source_max_height,
            "speaker_zoom": cfg.speaker_zoom,
        },
    }
    manifest.cache = {"root": str(cache.root), "hits": 0, "misses": 0, "events": []}
    _write_json(run_dir / "brief.normalized.json", brief.to_dict())
    watermark_path: Path | None = None
    if brief.watermark_url:
        telemetry.start("watermark_download")
        watermark_path = _download_asset(brief.watermark_url, run_dir / "assets" / "watermark.png")
        telemetry.stop("watermark_download")

    telemetry.start("source_discovery")
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
    telemetry.stop("source_discovery")
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
    all_moments: list[StoryMoment] = []
    all_concepts: list[ClipConcept] = []
    video_index = {video.video_id: video for video in allowed}

    for video in allowed:
        video_work = work_dir / video.video_id
        try:
            direct_media_url = brief.source_media_urls.get(video.video_id)
            if direct_media_url:
                telemetry.start(f"source_acquisition:{video.video_id}")
                media_path = _download_asset(
                    direct_media_url,
                    video_work / "source.mp4",
                    max_bytes=4_000_000_000,
                    expected_kind="media",
                )
                telemetry.stop(f"source_acquisition:{video.video_id}")
                media_paths[video.video_id] = media_path
                telemetry.start(f"transcription:{video.video_id}")
                segments, source_hash = _cached_asr_transcript(
                    cache, manifest, video.video_id, media_path, brief, cfg
                )
                telemetry.stop(f"transcription:{video.video_id}")
                manifest.run_metadata["source_hashes"][video.video_id] = source_hash
                manifest.run_metadata["transcript_sources"][video.video_id] = {
                    "kind": "faster-whisper",
                    "source_sha256": source_hash,
                }
            else:
                telemetry.start(f"subtitle_acquisition:{video.video_id}")
                subtitle_path = source.download_subtitles(video, video_work, brief.language)
                telemetry.stop(f"subtitle_acquisition:{video.video_id}")
                if subtitle_path:
                    telemetry.start(f"transcription:{video.video_id}")
                    segments, subtitle_hash = _cached_vtt_transcript(
                        cache, manifest, video.video_id, subtitle_path, brief.language
                    )
                    telemetry.stop(f"transcription:{video.video_id}")
                    manifest.run_metadata["transcript_sources"][video.video_id] = {
                        "kind": "youtube-vtt",
                        "source_sha256": subtitle_hash,
                    }
                else:
                    telemetry.start(f"source_acquisition:{video.video_id}")
                    media_path = source.download_media(video, video_work)
                    telemetry.stop(f"source_acquisition:{video.video_id}")
                    media_paths[video.video_id] = media_path
                    _record_source_media_metadata(manifest, video.video_id, media_path)
                    telemetry.start(f"transcription:{video.video_id}")
                    segments, source_hash = _cached_asr_transcript(
                        cache, manifest, video.video_id, media_path, brief, cfg
                    )
                    telemetry.stop(f"transcription:{video.video_id}")
                    manifest.run_metadata["source_hashes"][video.video_id] = source_hash
                    manifest.run_metadata["transcript_sources"][video.video_id] = {
                        "kind": "faster-whisper",
                        "source_sha256": source_hash,
                    }
            if not segments:
                raise RuntimeError("transcription produced no timestamped segments")
            transcripts[video.video_id] = segments
            manifest.run_metadata["transcript_hashes"][video.video_id] = stable_hash(
                [segment.to_dict() for segment in segments]
            )
            telemetry.start(f"editorial_analysis:{video.video_id}")
            moments, concepts = _cached_editorial_analysis(
                cache, manifest, brief, video.video_id, segments
            )
            telemetry.stop(f"editorial_analysis:{video.video_id}")
            all_moments.extend(moments)
            all_concepts.extend(concepts)
            _write_json(video_work / "transcript.json", [segment.to_dict() for segment in segments])
            _write_json(video_work / "story-moments.json", [item.to_dict() for item in moments])
            _write_json(video_work / "clip-candidates.json", [item.to_dict() for item in concepts])
        except Exception as exc:
            LOGGER.exception("source processing failed", extra={"video_id": video.video_id})
            manifest.errors.append({"video_id": video.video_id, "error": str(exc)})

    selected_concepts = select_distinct_concepts(brief, all_concepts)
    concept_index = {concept.concept_id: concept for concept in selected_concepts}
    variants = []
    plans: list[EditPlan] = []
    for concept in selected_concepts:
        concept_variants = generate_hook_variants(brief, concept, transcripts[concept.video_id])
        variants.extend(concept_variants)
        plans.extend(
            build_edit_plan(brief, concept, variant, transcripts[concept.video_id])
            for variant in concept_variants
        )
    render_plans = select_render_plans(plans, budget=brief.production.final_render_budget)
    submission_plans = select_submission_shortlist(
        render_plans, clip_count=brief.clip_count, max_per_source=brief.max_clips_per_source
    )

    manifest.story_moments = [item.to_dict() for item in all_moments]
    manifest.clip_concepts = [item.to_dict() for item in selected_concepts]
    manifest.hook_variants = [item.to_dict() for item in variants]
    manifest.edit_plans = [item.to_dict() for item in plans]
    manifest.planned_clips = [item.to_dict() for item in render_plans]
    manifest.submission_shortlist = [item.to_dict() for item in submission_plans]
    _write_json(run_dir / "story-moments.json", manifest.story_moments)
    _write_json(run_dir / "concept-ranking.json", manifest.clip_concepts)
    _write_json(run_dir / "hook-variants.json", manifest.hook_variants)
    for plan in plans:
        _write_json(run_dir / "edit-plans" / f"{_safe_slug(plan.plan_id)}.json", plan.to_dict())

    if render and active_renderer:
        for index, plan in enumerate(render_plans, start=1):
            concept = concept_index[plan.concept_id]
            video = video_index[plan.video_id]
            clip = plan.to_clip_candidate(concept.text)
            try:
                render_media_path = media_paths.get(video.video_id)
                if render_media_path is None:
                    telemetry.start(f"source_acquisition:{video.video_id}")
                    render_media_path = source.download_media(video, work_dir / video.video_id)
                    telemetry.stop(f"source_acquisition:{video.video_id}")
                    media_paths[video.video_id] = render_media_path
                    _record_source_media_metadata(manifest, video.video_id, render_media_path)
                    manifest.run_metadata["source_hashes"][video.video_id] = file_sha256(
                        render_media_path
                    )
                filename = (
                    f"{index:02d}-{_safe_slug(concept.topic)}-{_safe_slug(plan.hook_mode)}.mp4"
                )
                output_path = clips_dir / filename
                telemetry.start(f"render:{plan.plan_id}")
                rendered_path = active_renderer.render(
                    render_media_path,
                    output_path,
                    clip,
                    transcripts[video.video_id],
                    watermark_path,
                    plan,
                )
                telemetry.stop(f"render:{plan.plan_id}")
                telemetry.sample_gpu()
                rendered = RenderedClip(
                    video_id=video.video_id,
                    output_path=str(rendered_path),
                    start=clip.start,
                    end=clip.end,
                    score=plan.score,
                    source_url=video.url,
                    plan_id=plan.plan_id,
                    hook_mode=plan.hook_mode,
                    render_sha256=_sha256_file(rendered_path),
                )
                manifest.rendered_clips.append(rendered.to_dict())
                telemetry.start(f"technical_qc:{plan.plan_id}")
                qc_report = run_technical_qc(
                    rendered_path,
                    expected_duration=clip.duration,
                    caption_path=rendered_path.with_suffix(".ass"),
                    tracking_path=rendered_path.with_suffix(".tracking.json"),
                    caption_platform=plan.caption_platform,
                    watermark_required=bool(brief.watermark_url),
                    watermark_present=watermark_path is not None and watermark_path.is_file(),
                )
                telemetry.stop(f"technical_qc:{plan.plan_id}")
                qc_report["plan_id"] = plan.plan_id
                manifest.technical_qc.append(qc_report)
                _write_json(run_dir / "qc" / f"{rendered_path.stem}.json", qc_report)
            except Exception as exc:
                manifest.errors.append(
                    {"video_id": video.video_id, "plan_id": plan.plan_id, "error": str(exc)}
                )

    manifest.performance = telemetry.finish(run_dir)
    _write_json(run_dir / "manifest.json", manifest.to_dict())
    return run_dir
