from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import gdown

from .autonomous_editor import AutonomousEditorialPlanner, OpenVideoAnalysis
from .brief import load_brief
from .cache import (
    CACHE_SCHEMA_VERSION,
    FileCache,
    analysis_cache_key,
    clip_concepts_from_payload,
    file_sha256,
    model_stage_cache_key,
    stable_hash,
    story_moments_from_payload,
    transcript_cache_key,
    transcript_segments_from_payload,
)
from .canonical import (
    CanonicalTimeline,
    canonical_timeline_from_segments,
    transcript_segments_from_canonical,
)
from .editorial import (
    build_edit_plan,
    discover_story_moments,
    generate_hook_variants,
    mine_clip_concepts,
    select_distinct_concepts,
    select_render_plan_queue,
    select_submission_shortlist,
)
from .fixture import FixtureSourceClient, SpanMedia
from .models import (
    CampaignBrief,
    ClipCandidate,
    ClipConcept,
    EditPlan,
    PipelineManifest,
    RenderedClip,
    SourceSpan,
    StoryMoment,
    TranscriptSegment,
    TranscriptWord,
    VideoCandidate,
)
from .performance import RunTelemetry
from .providers.base import (
    AlignmentProvider,
    DiarizationProvider,
    EditorialProvider,
    EmbeddingProvider,
    TranscriptionProvider,
    VisionProvider,
)
from .providers.factory import (
    editorial_and_embedding_providers,
    speech_providers,
)
from .providers.factory import (
    vision_provider as build_vision_provider,
)
from .qc import run_technical_qc
from .render import FFmpegRenderer
from .rights import RightsError, assert_campaign_authorized, assert_video_allowed
from .runtime import ComputeBudget, StageJournal
from .transcript import load_vtt, transcribe_with_faster_whisper
from .visual import VisualTimeline
from .visual_ai import repair_stage, review_rendered_clip, scout_visual_timeline
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


def _span_media_for_plan(
    source: SourceClient,
    video: VideoCandidate,
    plan: EditPlan,
    work_dir: Path,
) -> SpanMedia | None:
    acquire = getattr(source, "download_media_span", None)
    if not callable(acquire) or not plan.source_spans:
        return None
    start = min(span.start for span in plan.source_spans)
    end = max(span.end for span in plan.source_spans)
    result = acquire(video, start, end, work_dir)
    if not isinstance(result, SpanMedia):
        raise RuntimeError("span-aware source returned an invalid media descriptor")
    return result


def _localize_render_inputs(
    clip: ClipCandidate,
    plan: EditPlan,
    segments: Sequence[TranscriptSegment],
    span_media: SpanMedia,
) -> tuple[ClipCandidate, EditPlan, list[TranscriptSegment]]:
    origin = span_media.source_origin
    end = span_media.source_end
    if clip.start < origin - 1e-6 or clip.end > end + 1e-6:
        raise RuntimeError(
            "source span "
            f"{origin:.3f}-{end:.3f} does not cover clip {clip.start:.3f}-{clip.end:.3f}"
        )
    local_clip = replace(clip, start=clip.start - origin, end=clip.end - origin)
    local_spans = tuple(
        SourceSpan(span.start - origin, span.end - origin) for span in plan.source_spans
    )
    local_anchor = (
        plan.caption_start_source_time - origin
        if plan.caption_start_source_time is not None
        else None
    )
    local_plan = replace(plan, source_spans=local_spans, caption_start_source_time=local_anchor)
    localized: list[TranscriptSegment] = []
    for segment in segments:
        if segment.end <= origin or segment.start >= end:
            continue
        if segment.words:
            words = tuple(
                TranscriptWord(word.start - origin, word.end - origin, word.text)
                for word in segment.words
                if word.start >= origin and word.end <= end
            )
            if not words:
                continue
            localized.append(
                TranscriptSegment(
                    max(0.0, max(segment.start, origin) - origin),
                    min(segment.end, end) - origin,
                    segment.text,
                    words,
                )
            )
            continue
        if segment.start < origin or segment.end > end:
            continue
        localized.append(
            TranscriptSegment(segment.start - origin, segment.end - origin, segment.text)
        )
    if not localized:
        raise RuntimeError("span-aware source produced no transcript context for the render")
    return local_clip, local_plan, localized


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
    editorial_engine: str = "heuristic"
    grounding_engine: str = "legacy"
    compute_profile: str = "balanced"
    editorial_chunk_words: int = 500
    editorial_chunk_overlap_words: int = 80
    semantic_duplicate_threshold: float = 0.9
    visual_scout_enabled: bool = False
    visual_review_enabled: bool = False
    visual_escalation_enabled: bool = True
    visual_escalation_threshold: float = 0.75
    compute_budget_usd: float = 1.0
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
            editorial_engine=os.getenv("CLIPPER_EDITORIAL_ENGINE", "heuristic").strip().lower(),
            grounding_engine=os.getenv("CLIPPER_GROUNDING_ENGINE", "legacy").strip().lower(),
            compute_profile=os.getenv("CLIPPER_COMPUTE_PROFILE", "balanced").strip().lower(),
            editorial_chunk_words=int(os.getenv("CLIPPER_EDITORIAL_CHUNK_WORDS", "500")),
            editorial_chunk_overlap_words=int(
                os.getenv("CLIPPER_EDITORIAL_CHUNK_OVERLAP_WORDS", "80")
            ),
            semantic_duplicate_threshold=float(
                os.getenv("CLIPPER_SEMANTIC_DUPLICATE_THRESHOLD", "0.9")
            ),
            visual_scout_enabled=_env_bool("CLIPPER_VISUAL_SCOUT", False),
            visual_review_enabled=_env_bool("CLIPPER_VISUAL_REVIEW", False),
            visual_escalation_enabled=_env_bool("CLIPPER_VISUAL_ESCALATION", True),
            visual_escalation_threshold=float(
                os.getenv("CLIPPER_VISUAL_ESCALATION_THRESHOLD", "0.75")
            ),
            compute_budget_usd=float(os.getenv("CLIPPER_COMPUTE_BUDGET_USD", "1.0")),
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


def _grounding_cache_key(stage: str, source_hash: str, provider: object, payload: object) -> str:
    identity = getattr(provider, "identity", None)
    if identity is None:
        raise ValueError("grounding provider has no model identity")
    return model_stage_cache_key(
        stage,
        source_hash=source_hash,
        campaign={},
        model=identity,
        payload=payload,
    )


def _cached_open_transcription(
    cache: FileCache,
    manifest: PipelineManifest,
    provider: TranscriptionProvider,
    media_path: Path,
    video_id: str,
    source_hash: str,
) -> tuple[CanonicalTimeline, dict[str, object]]:
    key = _grounding_cache_key(
        "canonical-transcription", source_hash, provider, {"video_id": video_id}
    )
    cached = cache.read(key, "canonical")
    if isinstance(cached, dict):
        try:
            value = CanonicalTimeline.from_dict(cached)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            _cache_event(manifest, "canonical-transcription", key, True)
            return value, {"model": provider.identity.to_dict(), "cache_hit": True}
    result = provider.transcribe(media_path, video_id=video_id, source_hash=source_hash)
    cache.write(key, "canonical", result.value.to_dict())
    _cache_event(manifest, "canonical-transcription", key, False)
    return result.value, {
        "model": result.model.to_dict(),
        "usage": asdict(result.usage),
        "degraded": result.degraded,
        "cache_hit": False,
    }


def _cached_open_alignment(
    cache: FileCache,
    manifest: PipelineManifest,
    provider: AlignmentProvider,
    media_path: Path,
    timeline: CanonicalTimeline,
) -> tuple[CanonicalTimeline, dict[str, object]]:
    key = _grounding_cache_key(
        "canonical-alignment",
        timeline.source_hash,
        provider,
        {"timeline_sha256": stable_hash(timeline.to_dict())},
    )
    cached = cache.read(key, "canonical")
    if isinstance(cached, dict):
        try:
            value = CanonicalTimeline.from_dict(cached)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            _cache_event(manifest, "canonical-alignment", key, True)
            return value, {"model": provider.identity.to_dict(), "cache_hit": True}
    result = provider.align(media_path, timeline)
    cache.write(key, "canonical", result.value.to_dict())
    _cache_event(manifest, "canonical-alignment", key, False)
    return result.value, {
        "model": result.model.to_dict(),
        "usage": asdict(result.usage),
        "degraded": result.degraded,
        "cache_hit": False,
    }


def _cached_open_diarization(
    cache: FileCache,
    manifest: PipelineManifest,
    provider: DiarizationProvider,
    media_path: Path,
    timeline: CanonicalTimeline,
) -> tuple[CanonicalTimeline, dict[str, object]]:
    key = _grounding_cache_key(
        "canonical-diarization",
        timeline.source_hash,
        provider,
        {"timeline_sha256": stable_hash(timeline.to_dict())},
    )
    cached = cache.read(key, "canonical")
    if isinstance(cached, dict):
        try:
            value = CanonicalTimeline.from_dict(cached)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            _cache_event(manifest, "canonical-diarization", key, True)
            return value, {"model": provider.identity.to_dict(), "cache_hit": True}
    result = provider.diarize(media_path, timeline)
    cache.write(key, "canonical", result.value.to_dict())
    _cache_event(manifest, "canonical-diarization", key, False)
    return result.value, {
        "model": result.model.to_dict(),
        "usage": asdict(result.usage),
        "degraded": result.degraded,
        "cache_hit": False,
    }


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
) -> tuple[list[StoryMoment], list[ClipConcept], list[dict[str, object]], dict[str, int]]:
    key = analysis_cache_key(video_id, segments, brief)
    cached_moments = cache.read(key, "story-moments")
    cached_concepts = cache.read(key, "clip-candidates")
    cached_rejections = cache.read(key, "editorial-rejections")
    cached_stats = cache.read(key, "editorial-stats")
    if cached_moments is not None and cached_concepts is not None:
        try:
            moments = story_moments_from_payload(cached_moments)
            concepts = clip_concepts_from_payload(cached_concepts)
            cached_rejection_items = (
                [item for item in cached_rejections if isinstance(item, dict)]
                if isinstance(cached_rejections, list)
                else []
            )
            cached_analysis_stats = (
                {str(k): int(v) for k, v in cached_stats.items() if isinstance(v, int)}
                if isinstance(cached_stats, dict)
                else {}
            )
        except (KeyError, TypeError, ValueError):
            moments, concepts, cached_rejection_items, cached_analysis_stats = [], [], [], {}
        if moments and concepts:
            _cache_event(manifest, "editorial-analysis", key, True)
            return moments, concepts, cached_rejection_items, cached_analysis_stats
    rejections: list[dict[str, object]] = []
    analysis_stats: dict[str, int] = {}
    moments = discover_story_moments(brief, video_id, segments)
    concepts = mine_clip_concepts(
        brief, video_id, segments, moments, rejections=rejections, stats=analysis_stats
    )
    cache.write(key, "story-moments", [item.to_dict() for item in moments])
    cache.write(key, "clip-candidates", [item.to_dict() for item in concepts])
    cache.write(key, "editorial-rejections", rejections)
    cache.write(key, "editorial-stats", analysis_stats)
    _cache_event(manifest, "editorial-analysis", key, False)
    return moments, concepts, rejections, analysis_stats


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
    editorial_provider: EditorialProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    visual_scout_provider: VisionProvider | None = None,
    visual_review_provider: VisionProvider | None = None,
    visual_escalation_provider: VisionProvider | None = None,
    transcription_provider: TranscriptionProvider | None = None,
    alignment_provider: AlignmentProvider | None = None,
    diarization_provider: DiarizationProvider | None = None,
    render: bool = True,
) -> Path:
    brief = load_brief(brief_path)
    assert_campaign_authorized(brief)
    cfg = settings or PipelineSettings.from_env()
    telemetry = RunTelemetry()
    cache = FileCache(cfg.cache_root or (cfg.artifact_root / "_cache"))
    if cfg.editorial_engine not in {"heuristic", "open"}:
        raise ValueError("CLIPPER_EDITORIAL_ENGINE must be heuristic or open")
    if cfg.grounding_engine not in {"legacy", "open"}:
        raise ValueError("CLIPPER_GROUNDING_ENGINE must be legacy or open")
    grounding_providers = (transcription_provider, alignment_provider, diarization_provider)
    if cfg.grounding_engine == "open":
        supplied = [provider is not None for provider in grounding_providers]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "open grounding requires transcription, alignment, and diarization providers"
            )
        if not all(supplied):
            transcription_provider, alignment_provider, diarization_provider = speech_providers(
                cfg.compute_profile
            )
    open_planner: AutonomousEditorialPlanner | None = None
    if cfg.editorial_engine == "open":
        if (editorial_provider is None) != (embedding_provider is None):
            raise ValueError("open editorial mode requires both editorial and embedding providers")
        if editorial_provider is None or embedding_provider is None:
            editorial_provider, embedding_provider = editorial_and_embedding_providers(
                cfg.compute_profile
            )
        open_planner = AutonomousEditorialPlanner(
            editorial_provider,
            embedding_provider,
            cache,
            max_words_per_chunk=cfg.editorial_chunk_words,
            chunk_overlap_words=cfg.editorial_chunk_overlap_words,
            semantic_duplicate_threshold=cfg.semantic_duplicate_threshold,
        )
    if cfg.visual_scout_enabled and visual_scout_provider is None:
        visual_scout_provider = build_vision_provider(cfg.compute_profile)
    if cfg.visual_review_enabled and visual_review_provider is None:
        visual_review_provider = build_vision_provider(cfg.compute_profile)
    if (
        cfg.visual_review_enabled
        and cfg.visual_escalation_enabled
        and cfg.compute_profile == "quality"
        and visual_escalation_provider is None
    ):
        visual_escalation_provider = build_vision_provider(cfg.compute_profile, large=True)
    if source_client is not None:
        source: SourceClient = source_client
    elif os.getenv("CLIPPER_SOURCE_FIXTURE_DIR"):
        source = FixtureSourceClient(os.environ["CLIPPER_SOURCE_FIXTURE_DIR"])
    else:
        source = YouTubeClient(max_height=cfg.source_max_height)
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
    journal = StageJournal(run_dir / "progress.json")
    journal.start("pipeline", message="pipeline initialized")
    compute_budget = ComputeBudget(cfg.compute_budget_usd)
    model_progress_count = [0]

    def _model_progress(stage: str, event: str) -> None:
        if event in {"success", "cache_hit"}:
            model_progress_count[0] += 1
        journal.start("model_inference", checkpoint=stage, message=f"{event}:{stage}")
        journal.progress(
            "model_inference",
            model_progress_count[0],
            checkpoint=stage,
            message=f"{event}:{stage}",
        )

    if open_planner is not None:
        open_planner.progress_callback = _model_progress
    manifest = PipelineManifest(campaign_id=brief.campaign_id)
    manifest.run_metadata = {
        "git_commit_sha": _git_sha(),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "transcription": {
            "engine": (
                "canonical-open-asr-alignment-diarization"
                if cfg.grounding_engine == "open"
                else "faster-whisper-or-youtube-vtt"
            ),
            "model": cfg.whisper_model,
            "device": cfg.whisper_device,
            "compute_type": cfg.whisper_compute_type,
        },
        "source_hashes": {},
        "source_span_hashes": {},
        "source_media": {},
        "source_mode": "private-fixture" if isinstance(source, FixtureSourceClient) else "live",
        "transcript_hashes": {},
        "transcript_sources": {},
        "canonical_timelines": {},
        "grounding_inference": {
            "engine": cfg.grounding_engine,
            "compute_profile": cfg.compute_profile,
            "models": [],
        },
        "editorial_inference": {
            "engine": cfg.editorial_engine,
            "compute_profile": cfg.compute_profile,
            "degraded": cfg.editorial_engine == "heuristic",
        },
        "visual_inference": {
            "scout_enabled": cfg.visual_scout_enabled,
            "review_enabled": cfg.visual_review_enabled,
            "escalation_enabled": cfg.visual_escalation_enabled,
            "escalation_threshold": cfg.visual_escalation_threshold,
            "scout_model": (
                visual_scout_provider.identity.to_dict()
                if visual_scout_provider is not None
                else None
            ),
            "primary_model": (
                visual_review_provider.identity.to_dict()
                if visual_review_provider is not None
                else None
            ),
            "escalation_model": (
                visual_escalation_provider.identity.to_dict()
                if visual_escalation_provider is not None
                else None
            ),
        },
        "compute_budget": compute_budget.to_dict(),
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
        fixture_watermark = getattr(source, "campaign_watermark", None)
        telemetry.start("watermark_download")
        if callable(fixture_watermark):
            watermark_path = fixture_watermark(brief)
        else:
            watermark_path = _download_asset(
                brief.watermark_url, run_dir / "assets" / "watermark.png"
            )
        telemetry.stop("watermark_download")

    journal.start("source_discovery", message="discovering authorized sources")
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
    journal.complete("source_discovery", message=f"{len(allowed)} authorized sources")
    journal.start(
        "source_processing", total=len(allowed), message="building canonical source evidence"
    )

    transcripts: dict[str, list[TranscriptSegment]] = {}
    canonical_timelines: dict[str, CanonicalTimeline] = {}
    visual_timelines: dict[str, VisualTimeline] = {}
    open_analyses: list[OpenVideoAnalysis] = []
    media_paths: dict[str, Path] = {}
    all_moments: list[StoryMoment] = []
    all_concepts: list[ClipConcept] = []
    mining_stats: Counter[str] = Counter()
    video_index = {video.video_id: video for video in allowed}

    for source_index, video in enumerate(allowed, start=1):
        journal.progress(
            "source_processing",
            source_index - 1,
            checkpoint=video.video_id,
            message=f"processing {video.video_id}",
        )
        video_work = work_dir / video.video_id
        try:
            direct_media_url = brief.source_media_urls.get(video.video_id)
            if cfg.grounding_engine == "open":
                if (
                    transcription_provider is None
                    or alignment_provider is None
                    or diarization_provider is None
                ):
                    raise RuntimeError("open grounding providers are not configured")
                telemetry.start(f"source_acquisition:{video.video_id}")
                if direct_media_url:
                    media_path = _download_asset(
                        direct_media_url,
                        video_work / "source.mp4",
                        max_bytes=4_000_000_000,
                        expected_kind="media",
                    )
                else:
                    media_path = source.download_media(video, video_work)
                telemetry.stop(f"source_acquisition:{video.video_id}")
                media_paths[video.video_id] = media_path
                _record_source_media_metadata(manifest, video.video_id, media_path)
                source_hash = file_sha256(media_path)
                manifest.run_metadata["source_hashes"][video.video_id] = source_hash

                telemetry.start(f"canonical_transcription:{video.video_id}")
                canonical, transcription_evidence = _cached_open_transcription(
                    cache,
                    manifest,
                    transcription_provider,
                    media_path,
                    video.video_id,
                    source_hash,
                )
                compute_budget.record_mapping(transcription_evidence.get("usage"))
                telemetry.stop(f"canonical_transcription:{video.video_id}")
                telemetry.start(f"canonical_alignment:{video.video_id}")
                canonical, alignment_evidence = _cached_open_alignment(
                    cache, manifest, alignment_provider, media_path, canonical
                )
                compute_budget.record_mapping(alignment_evidence.get("usage"))
                telemetry.stop(f"canonical_alignment:{video.video_id}")
                telemetry.start(f"canonical_diarization:{video.video_id}")
                canonical, diarization_evidence = _cached_open_diarization(
                    cache, manifest, diarization_provider, media_path, canonical
                )
                compute_budget.record_mapping(diarization_evidence.get("usage"))
                telemetry.stop(f"canonical_diarization:{video.video_id}")
                segments = transcript_segments_from_canonical(canonical)
                manifest.run_metadata["transcript_sources"][video.video_id] = {
                    "kind": "canonical-open",
                    "source_sha256": source_hash,
                }
                grounding_inference = manifest.run_metadata.get("grounding_inference")
                if isinstance(grounding_inference, dict):
                    models = grounding_inference.get("models")
                    if isinstance(models, list):
                        models.append(
                            {
                                "video_id": video.video_id,
                                "transcription": transcription_evidence,
                                "alignment": alignment_evidence,
                                "diarization": diarization_evidence,
                            }
                        )
            else:
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
                transcript_hash = stable_hash([segment.to_dict() for segment in segments])
                transcript_source = manifest.run_metadata["transcript_sources"][video.video_id]
                grounding_hash = str(
                    manifest.run_metadata["source_hashes"].get(video.video_id)
                    or transcript_source.get("source_sha256")
                    or transcript_hash
                )
                canonical = canonical_timeline_from_segments(
                    video.video_id,
                    grounding_hash,
                    segments,
                    transcript_source=str(transcript_source.get("kind") or "unknown"),
                )

            if not segments:
                raise RuntimeError("transcription produced no timestamped segments")
            transcripts[video.video_id] = segments
            transcript_hash = stable_hash([segment.to_dict() for segment in segments])
            manifest.run_metadata["transcript_hashes"][video.video_id] = transcript_hash
            canonical_payload = canonical.to_dict()
            canonical_hash = stable_hash(canonical_payload)
            _write_json(run_dir / "canonical" / f"{video.video_id}.json", canonical_payload)
            manifest.run_metadata["canonical_timelines"][video.video_id] = {
                "schema_version": canonical.schema_version,
                "sha256": canonical_hash,
                "source_hash": canonical.source_hash,
                "word_count": len(canonical.words),
                "timing_modes": sorted({word.timing_mode for word in canonical.words}),
                "speaker_count": len(
                    {word.speaker_id for word in canonical.words if word.speaker_id is not None}
                ),
            }
            canonical_timelines[video.video_id] = canonical
            visual_timeline: VisualTimeline | None = None
            media_for_visual = media_paths.get(video.video_id)
            if cfg.visual_scout_enabled and media_for_visual is None:
                try:
                    telemetry.start(f"visual_source_acquisition:{video.video_id}")
                    media_for_visual = source.download_media(video, video_work / "visual-source")
                    telemetry.stop(f"visual_source_acquisition:{video.video_id}")
                    media_paths[video.video_id] = media_for_visual
                    _record_source_media_metadata(manifest, video.video_id, media_for_visual)
                except Exception as exc:
                    LOGGER.warning(
                        "visual scouting media acquisition unavailable",
                        extra={"video_id": video.video_id, "error": str(exc)},
                    )
                    media_for_visual = None
            if cfg.visual_scout_enabled and visual_scout_provider is not None and media_for_visual:
                telemetry.start(f"visual_scout:{video.video_id}")
                visual_media_hash = file_sha256(media_for_visual)
                visual_cache_key = stable_hash(
                    {
                        "schema": CACHE_SCHEMA_VERSION,
                        "stage": "visual-timeline-scout-v1",
                        "video_id": video.video_id,
                        "media_sha256": visual_media_hash,
                        "duration": round(max(canonical.end, 0.05), 3),
                        "model": visual_scout_provider.identity.to_dict(),
                    }
                )
                try:
                    cached_visual = cache.read(visual_cache_key, "visual-timeline")
                    if isinstance(cached_visual, dict):
                        visual_timeline = VisualTimeline.from_dict(cached_visual)
                        visual_result = None
                    else:
                        visual_timeline, visual_result = scout_visual_timeline(
                            media_for_visual,
                            visual_scout_provider,
                            video_id=video.video_id,
                            source_hash=visual_media_hash,
                            duration=max(canonical.end, 0.05),
                            output_dir=video_work / "visual-scout-frames",
                        )
                        cache.write(visual_cache_key, "visual-timeline", visual_timeline.to_dict())
                except Exception as exc:
                    LOGGER.warning(
                        "visual scouting failed; continuing with canonical text evidence",
                        extra={"video_id": video.video_id, "error": str(exc)},
                    )
                    visual_meta = manifest.run_metadata.get("visual_inference")
                    if isinstance(visual_meta, dict):
                        visual_meta.setdefault("scout_errors", []).append(
                            {"video_id": video.video_id, "error": str(exc)}
                        )
                else:
                    if visual_result is not None:
                        compute_budget.record(visual_result.usage)
                    visual_timelines[video.video_id] = visual_timeline
                    _write_json(video_work / "visual-timeline.json", visual_timeline.to_dict())
                    _write_json(
                        run_dir / "visual" / f"{video.video_id}.json", visual_timeline.to_dict()
                    )
                    visual_meta = manifest.run_metadata.get("visual_inference")
                    if isinstance(visual_meta, dict):
                        run_entry: dict[str, object] = {
                            "video_id": video.video_id,
                            "model": visual_scout_provider.identity.to_dict(),
                            "event_count": len(visual_timeline.events),
                            "cache_key": visual_cache_key,
                            "cache_hit": visual_result is None,
                        }
                        if visual_result is not None:
                            run_entry["usage"] = asdict(visual_result.usage)
                            run_entry["degraded"] = visual_result.degraded
                        visual_meta.setdefault("scout_runs", []).append(run_entry)
                finally:
                    telemetry.stop(f"visual_scout:{video.video_id}")
            elif cfg.visual_scout_enabled:
                visual_meta = manifest.run_metadata.get("visual_inference")
                if isinstance(visual_meta, dict):
                    visual_meta.setdefault("scout_skipped", []).append(
                        {"video_id": video.video_id, "reason": "full_visual_media_unavailable"}
                    )
            telemetry.start(f"editorial_analysis:{video.video_id}")
            if open_planner is not None:
                analysis = open_planner.analyze_video(brief, canonical, visual_timeline)
                open_analyses.append(analysis)
                moments = analysis.moments
                concepts = analysis.concepts
                source_rejections = analysis.rejections
                source_stats = {
                    "candidate_starts": 0,
                    "eligible_endpoints": 0,
                    "concepts_after_quality": len(concepts),
                    "concepts_after_moment_dedup": len(moments),
                    "semantic_representatives": len(concepts),
                }
            else:
                moments, concepts, source_rejections, source_stats = _cached_editorial_analysis(
                    cache, manifest, brief, video.video_id, segments
                )
            telemetry.stop(f"editorial_analysis:{video.video_id}")
            all_moments.extend(moments)
            all_concepts.extend(concepts)
            manifest.rejections.extend(source_rejections)
            mining_stats.update(source_stats)
            _write_json(video_work / "transcript.json", [segment.to_dict() for segment in segments])
            _write_json(video_work / "story-moments.json", [item.to_dict() for item in moments])
            _write_json(video_work / "clip-candidates.json", [item.to_dict() for item in concepts])
        except Exception as exc:
            LOGGER.exception("source processing failed", extra={"video_id": video.video_id})
            manifest.errors.append({"video_id": video.video_id, "error": str(exc)})
        journal.progress(
            "source_processing",
            source_index,
            checkpoint=video.video_id,
            message=f"completed {video.video_id}",
        )

    journal.complete("source_processing", message=f"processed {len(allowed)} sources")
    _write_json(
        run_dir / "transcript.json",
        {
            video_id: [segment.to_dict() for segment in segments]
            for video_id, segments in transcripts.items()
        },
    )
    plans: list[EditPlan]
    journal.start("editorial_planning", message="constructing diverse source-grounded plans")
    if open_planner is not None:
        open_batch = open_planner.plan_batch(brief, canonical_timelines, open_analyses)
        all_moments = open_batch.discovered_moments
        all_concepts = open_batch.discovered_concepts
        selected_concepts = open_batch.selected_concepts
        variants = open_batch.variants
        plans = open_batch.plans
        manifest.rejections.extend(open_batch.rejections)
        manifest.run_metadata["editorial_inference"]["model_invocations"] = (
            open_batch.model_invocations
        )
        for invocation in open_batch.model_invocations:
            compute_budget.record_mapping(invocation.get("usage"))
        _write_json(run_dir / "open-model" / "model-invocations.json", open_batch.model_invocations)
        _write_json(
            run_dir / "open-model" / "discovered-concepts.json",
            [item.to_dict() for item in all_concepts],
        )
    else:
        selected_concepts = select_distinct_concepts(
            brief, all_concepts, rejections=manifest.rejections
        )
        variants = []
        plans = []
        for concept in selected_concepts:
            concept_variants = generate_hook_variants(brief, concept, transcripts[concept.video_id])
            if not concept_variants:
                manifest.rejections.append(
                    {
                        "concept_id": concept.concept_id,
                        "video_id": concept.video_id,
                        "stage": "hook_generation",
                        "decision": "REJECT",
                        "reasons": ["no_legitimate_hook_variants"],
                        "scores": concept.scores.to_dict(),
                    }
                )
                continue
            variants.extend(concept_variants)
            plans.extend(
                build_edit_plan(brief, concept, variant, transcripts[concept.video_id])
                for variant in concept_variants
            )
    journal.complete("editorial_planning", message=f"planned {len(plans)} edit plans")
    if "model_inference" in journal.states:
        journal.complete(
            "model_inference", message=f"completed {model_progress_count[0]} model stages"
        )
    concept_index = {concept.concept_id: concept for concept in selected_concepts}
    target_finalists = brief.production.final_render_budget
    primary_plans, reserve_plans = select_render_plan_queue(plans, budget=target_finalists)

    manifest.targets = {
        "rendered_finalists": target_finalists,
        "submission_shortlist": brief.clip_count,
        "distinct_finalist_concepts": brief.production.minimum_distinct_finalist_concepts,
        "distinct_shortlist_concepts": brief.clip_count,
    }
    manifest.story_moments = [item.to_dict() for item in all_moments]
    manifest.clip_concepts = [item.to_dict() for item in selected_concepts]
    manifest.hook_variants = [item.to_dict() for item in variants]
    manifest.edit_plans = [item.to_dict() for item in plans]
    manifest.planned_clips = [item.to_dict() for item in primary_plans]
    manifest.reserve_plans = [item.to_dict() for item in reserve_plans]
    manifest.submission_shortlist = []
    transcript_segment_count = sum(len(items) for items in transcripts.values())
    raw_count = len(all_concepts)
    selected_count = len(selected_concepts)
    manifest.funnel = {
        "transcript_segments": transcript_segment_count,
        "story_moments": len(all_moments),
        "candidate_starts": mining_stats["candidate_starts"],
        "eligible_endpoints": mining_stats["eligible_endpoints"],
        "concepts_after_quality": mining_stats["concepts_after_quality"],
        "concepts_after_moment_dedup": mining_stats["concepts_after_moment_dedup"],
        "concepts_after_semantic_dedupe": mining_stats["semantic_representatives"],
        "raw_concepts": raw_count,
        "selected_concepts": selected_count,
        "hook_variants": len(variants),
        "edit_plans": len(plans),
        "render_plans": len(primary_plans),
        "reserve_plans": len(reserve_plans),
        "render_attempts": 0,
        "render_failures": 0,
        "replacement_attempts": 0,
        "tracking_preflight_pass": 0,
        "tracking_preflight_repaired": 0,
        "tracking_preflight_fail": 0,
        "technical_qc_pass": 0,
        "technical_qc_fail": 0,
        "editorial_qc_pass": 0,
        "editorial_qc_fail": 0,
        "visual_review_escalations": 0,
        "render_success": 0,
        "distinct_finalist_concepts": 0,
        "submission_shortlist": 0,
        "distinct_shortlist_concepts": 0,
        "concept_selection_retention_ratio": (
            round(selected_count / raw_count, 4) if raw_count else 0.0
        ),
        "attrition_flag": bool(raw_count >= 8 and selected_count / max(raw_count, 1) < 0.2),
    }
    source_coverage: dict[str, object] = {}
    for video_id, source_segments in transcripts.items():
        duration = max((segment.end for segment in source_segments), default=0.0)
        third = duration / 3 if duration else 0.0

        def _period(start: float, period_size: float = third) -> str:
            if not period_size or start < period_size:
                return "early"
            if start < period_size * 2:
                return "middle"
            return "late"

        raw_periods = Counter(
            _period(item.source_start) for item in all_concepts if item.video_id == video_id
        )
        selected_periods = Counter(
            _period(item.source_start) for item in selected_concepts if item.video_id == video_id
        )
        render_periods = Counter(
            _period(plan.source_spans[0].start)
            for plan in primary_plans
            if plan.video_id == video_id and plan.source_spans
        )
        render_count = sum(render_periods.values())
        source_coverage[video_id] = {
            "duration_seconds": duration,
            "raw_concepts_by_period": dict(raw_periods),
            "selected_concepts_by_period": dict(selected_periods),
            "render_plans_by_period": dict(render_periods),
            "suspicious_concentration": bool(
                len(raw_periods) >= 2 and len(selected_periods) == 1 and len(selected_concepts) >= 4
            ),
            "render_suspicious_concentration": bool(
                render_count >= 4
                and render_periods
                and max(render_periods.values()) > render_count / 2
            ),
        }
    manifest.run_metadata["source_coverage"] = source_coverage
    rejection_counts = Counter(
        str(reason) for item in manifest.rejections for reason in (item.get("reasons") or [])
    )
    manifest.run_metadata["rejection_reason_counts"] = dict(rejection_counts)
    _write_json(run_dir / "coverage.json", source_coverage)
    _write_json(run_dir / "story-moments.json", manifest.story_moments)
    _write_json(run_dir / "concept-ranking.json", manifest.clip_concepts)
    _write_json(run_dir / "hook-variants.json", manifest.hook_variants)
    for plan in plans:
        _write_json(run_dir / "edit-plans" / f"{_safe_slug(plan.plan_id)}.json", plan.to_dict())

    if render and active_renderer:
        queue = [("primary", plan) for plan in primary_plans] + [
            ("reserve", plan) for plan in reserve_plans
        ]
        accepted_plans: list[EditPlan] = []
        journal.start("render", total=len(queue), message="rendering preflight-approved finalists")
        for queue_index, (queue_kind, plan) in enumerate(queue, start=1):
            journal.progress(
                "render",
                queue_index - 1,
                checkpoint=plan.plan_id,
                message=f"attempting {queue_kind} plan {plan.plan_id}",
            )
            if len(accepted_plans) >= target_finalists:
                break
            concept = concept_index[plan.concept_id]
            video = video_index[plan.video_id]
            clip = plan.to_clip_candidate(concept.text)
            attempt = {
                "attempt": queue_index,
                "plan_id": plan.plan_id,
                "concept_id": plan.concept_id,
                "queue": queue_kind,
                "status": "STARTED",
            }
            manifest.render_attempts.append(attempt)
            manifest.funnel["render_attempts"] = int(manifest.funnel["render_attempts"]) + 1
            if queue_kind == "reserve":
                manifest.funnel["replacement_attempts"] = (
                    int(manifest.funnel["replacement_attempts"]) + 1
                )
            try:
                telemetry.start(f"source_acquisition:{video.video_id}")
                span_media = _span_media_for_plan(source, video, plan, work_dir / video.video_id)
                if span_media is not None:
                    render_media_path = span_media.path
                    render_clip, render_plan, render_segments = _localize_render_inputs(
                        clip, plan, transcripts[video.video_id], span_media
                    )
                    span_hashes = manifest.run_metadata["source_span_hashes"].setdefault(
                        video.video_id, {}
                    )
                    span_hashes[plan.plan_id] = {
                        "sha256": span_media.sha256,
                        "source_origin": span_media.source_origin,
                        "source_end": span_media.source_end,
                    }
                else:
                    cached_media_path = media_paths.get(video.video_id)
                    if cached_media_path is None:
                        cached_media_path = source.download_media(video, work_dir / video.video_id)
                        media_paths[video.video_id] = cached_media_path
                        _record_source_media_metadata(manifest, video.video_id, cached_media_path)
                        manifest.run_metadata["source_hashes"][video.video_id] = file_sha256(
                            cached_media_path
                        )
                    render_media_path = cached_media_path
                    render_clip, render_plan, render_segments = (
                        clip,
                        plan,
                        transcripts[video.video_id],
                    )
                telemetry.stop(f"source_acquisition:{video.video_id}")
                if render_media_path is None:
                    raise RuntimeError("source acquisition returned no media path")
                filename = (
                    f"attempt-{queue_index:02d}-{_safe_slug(concept.topic)}-"
                    f"{_safe_slug(plan.hook_mode)}.mp4"
                )
                output_path = clips_dir / filename
                telemetry.start(f"render:{plan.plan_id}")
                rendered_path = active_renderer.render(
                    render_media_path,
                    output_path,
                    render_clip,
                    render_segments,
                    watermark_path,
                    render_plan,
                )
                telemetry.stop(f"render:{plan.plan_id}")
                telemetry.sample_gpu()
                preflight_path = rendered_path.with_suffix(".tracking-preflight.json")
                if preflight_path.is_file():
                    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
                    attempt["tracking_preflight"] = preflight
                    if preflight.get("status") == "PASS":
                        manifest.funnel["tracking_preflight_pass"] = (
                            int(manifest.funnel["tracking_preflight_pass"]) + 1
                        )
                    else:
                        manifest.funnel["tracking_preflight_fail"] = (
                            int(manifest.funnel["tracking_preflight_fail"]) + 1
                        )
                    if preflight.get("repaired_with_stable_fallback"):
                        manifest.funnel["tracking_preflight_repaired"] = (
                            int(manifest.funnel["tracking_preflight_repaired"]) + 1
                        )
                telemetry.start(f"technical_qc:{plan.plan_id}")
                qc_report = run_technical_qc(
                    rendered_path,
                    expected_duration=clip.duration,
                    caption_path=rendered_path.with_suffix(".ass"),
                    tracking_path=rendered_path.with_suffix(".tracking.json"),
                    caption_platform=plan.caption_platform,
                    watermark_required=bool(brief.watermark_url),
                    watermark_present=watermark_path is not None and watermark_path.is_file(),
                    caption_audit_path=rendered_path.with_suffix(".caption-audit.json"),
                )
                telemetry.stop(f"technical_qc:{plan.plan_id}")
                qc_report["plan_id"] = plan.plan_id
                manifest.technical_qc.append(qc_report)
                _write_json(run_dir / "qc" / f"{rendered_path.stem}.json", qc_report)
                if qc_report.get("status") != "PASS":
                    manifest.funnel["technical_qc_fail"] = (
                        int(manifest.funnel["technical_qc_fail"]) + 1
                    )
                    attempt["status"] = "QC_FAILED"
                    attempt["issues"] = list(qc_report.get("issues") or [])
                    manifest.rejections.append(
                        {
                            "concept_id": plan.concept_id,
                            "video_id": plan.video_id,
                            "stage": "technical_qc",
                            "decision": "REJECT",
                            "reasons": list(qc_report.get("issues") or ["technical_qc_failed"]),
                            "scores": {"plan_score": plan.score},
                            "plan_id": plan.plan_id,
                        }
                    )
                    rejected_dir = run_dir / "rejected"
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    for candidate_path in [
                        rendered_path,
                        rendered_path.with_suffix(".ass"),
                        rendered_path.with_suffix(".tracking.json"),
                        rendered_path.with_suffix(".tracking-preflight.json"),
                        rendered_path.with_suffix(".render.json"),
                        rendered_path.with_suffix(".caption-audit.json"),
                    ]:
                        if candidate_path.exists():
                            candidate_path.replace(rejected_dir / candidate_path.name)
                    continue
                manifest.funnel["technical_qc_pass"] = int(manifest.funnel["technical_qc_pass"]) + 1
                if cfg.visual_review_enabled:
                    if visual_review_provider is None:
                        raise RuntimeError("visual review is enabled without a VisionProvider")
                    tracking_payload: dict[str, object] = {}
                    tracking_file = rendered_path.with_suffix(".tracking.json")
                    if tracking_file.is_file():
                        loaded_tracking = json.loads(tracking_file.read_text(encoding="utf-8"))
                        if isinstance(loaded_tracking, dict):
                            tracking_payload = loaded_tracking
                    raw_transitions = tracking_payload.get("transitions", [])
                    transition_items = raw_transitions if isinstance(raw_transitions, list) else []
                    transition_times = tuple(
                        float(item.get("start_time") or 0.0)
                        for item in transition_items
                        if isinstance(item, dict)
                    )
                    telemetry.start(f"editorial_qc:{plan.plan_id}")
                    review, review_results = review_rendered_clip(
                        rendered_path,
                        visual_review_provider,
                        duration=clip.duration,
                        output_dir=run_dir / "visual-review" / rendered_path.stem / "frames",
                        context={
                            "plan_id": plan.plan_id,
                            "concept_id": plan.concept_id,
                            "source_start": clip.start,
                            "source_end": clip.end,
                            "hook_mode": plan.hook_mode,
                            "technical_qc": qc_report,
                        },
                        transitions=transition_times,
                        escalation=(
                            visual_escalation_provider if compute_budget.allow_large_vlm() else None
                        ),
                        escalation_threshold=cfg.visual_escalation_threshold,
                    )
                    telemetry.stop(f"editorial_qc:{plan.plan_id}")
                    for result in review_results:
                        compute_budget.record(result.usage)
                    review_payload = review.to_dict()
                    review_payload["plan_id"] = plan.plan_id
                    review_payload["models"] = [result.model.to_dict() for result in review_results]
                    review_payload["usage"] = [asdict(result.usage) for result in review_results]
                    manifest.editorial_qc.append(review_payload)
                    _write_json(
                        run_dir / "visual-review" / f"{rendered_path.stem}.json",
                        review_payload,
                    )
                    if review.escalated:
                        manifest.funnel["visual_review_escalations"] = (
                            int(manifest.funnel["visual_review_escalations"]) + 1
                        )
                    if review.decision != "PASS":
                        manifest.funnel["editorial_qc_fail"] = (
                            int(manifest.funnel["editorial_qc_fail"]) + 1
                        )
                        attempt["status"] = "EDITORIAL_QC_FAILED"
                        attempt["editorial_qc"] = review_payload
                        issue_types = [issue.issue_type for issue in review.issues]
                        manifest.rejections.append(
                            {
                                "concept_id": plan.concept_id,
                                "video_id": plan.video_id,
                                "stage": "editorial_qc",
                                "decision": "REJECT",
                                "reasons": issue_types or ["open_vlm_editorial_qc_failed"],
                                "repair_stages": sorted(
                                    {repair_stage(issue) for issue in issue_types}
                                ),
                                "scores": {"plan_score": plan.score},
                                "plan_id": plan.plan_id,
                            }
                        )
                        rejected_dir = run_dir / "rejected"
                        rejected_dir.mkdir(parents=True, exist_ok=True)
                        for candidate_path in [
                            rendered_path,
                            rendered_path.with_suffix(".ass"),
                            rendered_path.with_suffix(".tracking.json"),
                            rendered_path.with_suffix(".tracking-preflight.json"),
                            rendered_path.with_suffix(".render.json"),
                            rendered_path.with_suffix(".caption-audit.json"),
                        ]:
                            if candidate_path.exists():
                                candidate_path.replace(rejected_dir / candidate_path.name)
                        continue
                    manifest.funnel["editorial_qc_pass"] = (
                        int(manifest.funnel["editorial_qc_pass"]) + 1
                    )
                    attempt["editorial_qc"] = review_payload
                rendered = RenderedClip(
                    video_id=video.video_id,
                    output_path=str(rendered_path),
                    start=clip.start,
                    end=clip.end,
                    score=plan.score,
                    source_url=video.url,
                    concept_id=plan.concept_id,
                    plan_id=plan.plan_id,
                    hook_mode=plan.hook_mode,
                    render_sha256=_sha256_file(rendered_path),
                )
                manifest.rendered_clips.append(rendered.to_dict())
                accepted_plans.append(plan)
                for evidence_dir, suffix in (
                    ("captions", ".ass"),
                    ("captions", ".caption-audit.json"),
                    ("tracking", ".tracking.json"),
                    ("tracking", ".tracking-preflight.json"),
                ):
                    source_evidence = rendered_path.with_suffix(suffix)
                    if source_evidence.is_file():
                        destination = run_dir / evidence_dir / source_evidence.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_evidence, destination)
                attempt["status"] = "ACCEPTED"
            except Exception as exc:
                manifest.funnel["render_failures"] = int(manifest.funnel["render_failures"]) + 1
                attempt["status"] = "RENDER_FAILED"
                attempt["error"] = str(exc)
                manifest.errors.append(
                    {"video_id": video.video_id, "plan_id": plan.plan_id, "error": str(exc)}
                )
                manifest.rejections.append(
                    {
                        "concept_id": plan.concept_id,
                        "video_id": plan.video_id,
                        "stage": "render",
                        "decision": "REJECT",
                        "reasons": ["render_exception"],
                        "error": str(exc),
                        "scores": {"plan_score": plan.score},
                        "plan_id": plan.plan_id,
                    }
                )

        journal.complete(
            "render",
            message=f"accepted {len(accepted_plans)} of {target_finalists} target finalists",
        )
        shortlist_plans = select_submission_shortlist(
            accepted_plans,
            clip_count=brief.clip_count,
            max_per_source=brief.max_clips_per_source,
        )
        rendered_by_plan = {
            str(item.get("plan_id")): item
            for item in manifest.rendered_clips
            if item.get("plan_id")
        }
        manifest.submission_shortlist = [
            rendered_by_plan[plan.plan_id]
            for plan in shortlist_plans
            if plan.plan_id in rendered_by_plan
        ]
        finalist_concepts = {
            str(item.get("concept_id"))
            for item in manifest.rendered_clips
            if item.get("concept_id")
        }
        shortlist_concepts = {
            str(item.get("concept_id"))
            for item in manifest.submission_shortlist
            if item.get("concept_id")
        }
        manifest.funnel["render_success"] = len(manifest.rendered_clips)
        manifest.funnel["distinct_finalist_concepts"] = len(finalist_concepts)
        manifest.funnel["submission_shortlist"] = len(manifest.submission_shortlist)
        manifest.funnel["distinct_shortlist_concepts"] = len(shortlist_concepts)
        manifest.actual = {
            "rendered_finalists": len(manifest.rendered_clips),
            "submission_shortlist": len(manifest.submission_shortlist),
            "distinct_finalist_concepts": len(finalist_concepts),
            "distinct_shortlist_concepts": len(shortlist_concepts),
        }
        if len(manifest.rendered_clips) < target_finalists:
            manifest.status = "FAILED"
            manifest.status_reason = "render_yield_below_required_target"
        elif len(finalist_concepts) < brief.production.minimum_distinct_finalist_concepts:
            manifest.status = "FAILED"
            manifest.status_reason = "distinct_finalist_concepts_below_required_target"
        elif len(manifest.submission_shortlist) < brief.clip_count:
            manifest.status = "FAILED"
            manifest.status_reason = "submission_shortlist_below_required_target"
        elif len(shortlist_concepts) < brief.clip_count:
            manifest.status = "FAILED"
            manifest.status_reason = "distinct_shortlist_concepts_below_required_target"
        elif any(item.get("status") != "ACCEPTED" for item in manifest.render_attempts):
            manifest.status = "DEGRADED"
            manifest.status_reason = "recovered_with_replacement_candidates"
        else:
            manifest.status = "SUCCESS"
            manifest.status_reason = None
    else:
        manifest.actual = {
            "rendered_finalists": 0,
            "submission_shortlist": 0,
            "distinct_finalist_concepts": 0,
            "distinct_shortlist_concepts": 0,
        }
        manifest.status = "SUCCESS"
        manifest.status_reason = "planning_only"

    manifest.run_metadata["rejection_reason_counts"] = dict(
        Counter(
            str(reason) for item in manifest.rejections for reason in (item.get("reasons") or [])
        )
    )
    _write_json(
        run_dir / "editorial-review.json",
        {
            "status": "PENDING_HUMAN_REVIEW" if render else "NOT_APPLICABLE_PLANNING_ONLY",
            "required": bool(render),
            "clips": [
                {
                    "output_path": item.get("output_path"),
                    "plan_id": item.get("plan_id"),
                    "concept_id": item.get("concept_id"),
                    "technical_qc": "PASS",
                    "human_review": "PENDING",
                }
                for item in manifest.rendered_clips
            ],
        },
    )
    manifest.run_metadata["compute_budget"] = compute_budget.to_dict()
    journal.complete("pipeline", checkpoint="manifest.json", message=manifest.status)
    manifest.performance = telemetry.finish(run_dir)
    _write_json(run_dir / "funnel.json", manifest.funnel)
    _write_json(run_dir / "rejections.json", manifest.rejections)
    _write_json(run_dir / "manifest.json", manifest.to_dict())
    return run_dir
