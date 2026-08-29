from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import gdown

from .brief import load_brief, load_explicit_targets
from .cache import FileCache, file_sha256, model_stage_cache_key, stable_hash
from .dag import DagStore, StageResult
from .canonical import CanonicalTimeline, transcript_segments_from_canonical
from .fixture import FixtureSourceClient
from .models import (
    CampaignBrief,
    ClipCandidate,
    EditPlan,
    PipelineManifest,
    RenderedClip,
    TranscriptSegment,
    VideoCandidate,
)
from .multimodal_timeline import EvidenceProvenance, build_multimodal_timeline
from .providers.base import (
    AlignmentProvider,
    DiarizationProvider,
    EditorialProvider,
    ProviderResult,
    TranscriptionProvider,
    VisionProvider,
)
from .providers.factory import editorial_provider as build_editorial_provider
from .providers.factory import speech_providers, vision_provider
from .qc import run_technical_qc
from .quality_batch import QualityBatchResult, plan_quality_batch
from .render import FFmpegRenderer
from .rights import assert_campaign_authorized, assert_video_allowed
from .runtime import StageJournal
from .stage_contracts import StageContract, StageIdentity, stage_identity
from .visual import VisualTimeline
from .visual_ai import (
    review_rendered_clip,
    scout_visual_timeline,
    tracking_transition_sample_times,
)
from .visual_strategy import derive_visual_strategy
from .youtube import YouTubeClient
from .yield_policy import accepted_quality_plans, group_quality_plans, quality_render_queue

LOGGER = logging.getLogger("clipper")
_VISUAL_CHECKPOINT_DIR: ContextVar[Path | None] = ContextVar(
    "clipper_visual_checkpoint_dir", default=None
)
_VISUAL_CHECKPOINT_COMMIT: ContextVar[Callable[[], None] | None] = ContextVar(
    "clipper_visual_checkpoint_commit", default=None
)


class SourceClient(Protocol):
    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path: ...


class Renderer(Protocol):
    def render(
        self,
        source_path: Path,
        output_path: Path,
        clip: ClipCandidate,
        segments: list[TranscriptSegment],
        watermark_path: Path | None = None,
        edit_plan: EditPlan | None = None,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Technical execution settings only; editorial policy belongs to the campaign/model graph."""

    artifact_root: Path = Path("artifacts")
    cache_root: Path | None = None
    compute_profile: str = "balanced"
    source_max_height: int = 2160
    render_profile: str = "production"
    speaker_focus_override: bool | None = None
    speaker_zoom: float = 1.0
    speaker_sample_fps: float = 4.0
    speaker_switch_margin: float = 1.35
    speaker_min_reframe_seconds: float = 0.35
    speaker_max_reframe_seconds: float = 0.9
    speaker_seconds_per_crop: float = 0.75
    speaker_hold_threshold: float = 0.28
    speaker_reversal_guard_seconds: float = 2.0
    speaker_window_seconds: float = 0.8
    speaker_min_detection_coverage: float = 0.35
    visual_escalation_enabled: bool = False
    visual_escalation_threshold: float = 0.75

    @classmethod
    def from_env(cls) -> PipelineSettings:
        artifact_root = Path(os.getenv("CLIPPER_ARTIFACT_ROOT", "artifacts"))
        cache_root_value = os.getenv("CLIPPER_CACHE_ROOT")
        raw_focus = os.getenv("CLIPPER_SPEAKER_FOCUS")
        speaker_focus_override = (
            None if raw_focus is None else raw_focus.strip().lower() in {"1", "true", "yes"}
        )
        return cls(
            artifact_root=artifact_root,
            cache_root=Path(cache_root_value) if cache_root_value else artifact_root / "_cache",
            compute_profile=os.getenv("CLIPPER_COMPUTE_PROFILE", "balanced").strip().lower(),
            source_max_height=int(os.getenv("CLIPPER_SOURCE_MAX_HEIGHT", "2160")),
            render_profile=os.getenv("CLIPPER_RENDER_PROFILE", "production").strip().lower(),
            speaker_focus_override=speaker_focus_override,
            speaker_zoom=float(os.getenv("CLIPPER_SPEAKER_ZOOM", "1.0")),
            speaker_sample_fps=float(os.getenv("CLIPPER_SPEAKER_SAMPLE_FPS", "4.0")),
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
                os.getenv("CLIPPER_SPEAKER_REVERSAL_GUARD_SECONDS", "2.0")
            ),
            speaker_window_seconds=float(os.getenv("CLIPPER_SPEAKER_WINDOW_SECONDS", "0.8")),
            speaker_min_detection_coverage=float(
                os.getenv("CLIPPER_SPEAKER_MIN_DETECTION_COVERAGE", "0.35")
            ),
            visual_escalation_enabled=os.getenv("CLIPPER_VISUAL_ESCALATION", "false")
            .strip()
            .lower()
            in {"1", "true", "yes"},
            visual_escalation_threshold=float(
                os.getenv("CLIPPER_VISUAL_ESCALATION_THRESHOLD", "0.75")
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceRuntime:
    video: VideoCandidate
    media_path: Path
    source_hash: str
    timeline: CanonicalTimeline
    segments: tuple[TranscriptSegment, ...]
    visual_timeline: VisualTimeline


def _client(cfg: PipelineSettings) -> SourceClient:
    fixture_dir = os.getenv("CLIPPER_SOURCE_FIXTURE_DIR")
    if fixture_dir:
        return FixtureSourceClient(fixture_dir)
    return YouTubeClient(max_height=cfg.source_max_height)


def _run_id(campaign_id: str, execution_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{campaign_id}-{timestamp}-{execution_id}"


def _safe_slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part)[:64] or "clip"


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


def _grounding_stage_identity(stage: str, key: str) -> StageIdentity:
    return stage_identity(
        StageContract(
            stage,
            {
                "grounding_cache_key": key,
                "output_contract": "canonical-timeline-with-model-evidence",
            },
        ),
        source_hash=key,
    )


def _grounding_stage_output(
    result: ProviderResult[CanonicalTimeline],
) -> dict[str, object]:
    return {
        "canonical": result.value.to_dict(),
        "model": result.model.to_dict(),
        "usage": asdict(result.usage),
        "degraded": result.degraded,
    }


def _grounding_stage_value(
    *,
    stage: str,
    key: str,
    cache: FileCache,
    dag: DagStore | None,
    operation: Callable[[], ProviderResult[CanonicalTimeline]],
) -> tuple[dict[str, object], bool]:
    if dag is None:
        result = operation()
        payload = _grounding_stage_output(result)
        cache.write(key, "canonical", payload["canonical"])
        return payload, False

    identity = _grounding_stage_identity(stage, key)

    def execute() -> StageResult:
        result = operation()
        payload = _grounding_stage_output(result)
        cache.write(key, "canonical", payload["canonical"])
        usage = payload.get("usage")
        usage_dict = dict(usage) if isinstance(usage, dict) else {}
        cost = float(usage_dict.get("estimated_cost_usd") or 0.0)
        return StageResult(payload, usage=usage_dict, cost_usd=cost)

    output, cached = dag.execute(identity, execute)
    if not isinstance(output, dict):
        raise RuntimeError(f"{stage} grounding DAG returned invalid output")
    return {str(key): value for key, value in output.items()}, cached


def _grounding_payload(
    payload: dict[str, object],
    provider: TranscriptionProvider | AlignmentProvider | DiarizationProvider,
) -> tuple[CanonicalTimeline, dict[str, object]]:
    raw_canonical = payload.get("canonical")
    if not isinstance(raw_canonical, dict):
        raise RuntimeError("grounding DAG output is missing canonical timeline")
    timeline = CanonicalTimeline.from_dict(raw_canonical)
    raw_model = payload.get("model")
    model = dict(raw_model) if isinstance(raw_model, dict) else provider.identity.to_dict()
    raw_usage = payload.get("usage")
    usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
    return timeline, {
        "model": model,
        "usage": usage,
        "degraded": bool(payload.get("degraded", False)),
    }


def _cached_transcription(
    cache: FileCache,
    manifest: PipelineManifest,
    provider: TranscriptionProvider,
    media_path: Path,
    video_id: str,
    source_hash: str,
    *,
    dag: DagStore | None = None,
) -> tuple[CanonicalTimeline, dict[str, object]]:
    stage = "canonical-transcription"
    key = _grounding_cache_key(stage, source_hash, provider, {"video_id": video_id})
    cached_value = cache.read(key, "canonical")
    if isinstance(cached_value, dict):
        try:
            value = CanonicalTimeline.from_dict(cached_value)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            _cache_event(manifest, stage, key, True)
            return value, {"model": provider.identity.to_dict(), "cache_hit": True}

    payload, dag_cached = _grounding_stage_value(
        stage=stage,
        key=key,
        cache=cache,
        dag=dag,
        operation=lambda: provider.transcribe(
            media_path,
            video_id=video_id,
            source_hash=source_hash,
        ),
    )
    value, metadata = _grounding_payload(payload, provider)
    metadata["cache_hit"] = dag_cached
    _cache_event(manifest, stage, key, dag_cached)
    return value, metadata

def _cached_alignment(
    cache: FileCache,
    manifest: PipelineManifest,
    provider: AlignmentProvider,
    media_path: Path,
    timeline: CanonicalTimeline,
    *,
    dag: DagStore | None = None,
) -> tuple[CanonicalTimeline, dict[str, object]]:
    stage = "canonical-alignment"
    key = _grounding_cache_key(
        stage,
        timeline.source_hash,
        provider,
        {"timeline_sha256": stable_hash(timeline.to_dict())},
    )
    cached_value = cache.read(key, "canonical")
    if isinstance(cached_value, dict):
        try:
            value = CanonicalTimeline.from_dict(cached_value)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            _cache_event(manifest, stage, key, True)
            return value, {"model": provider.identity.to_dict(), "cache_hit": True}

    payload, dag_cached = _grounding_stage_value(
        stage=stage,
        key=key,
        cache=cache,
        dag=dag,
        operation=lambda: provider.align(media_path, timeline),
    )
    value, metadata = _grounding_payload(payload, provider)
    metadata["cache_hit"] = dag_cached
    _cache_event(manifest, stage, key, dag_cached)
    return value, metadata

def _cached_diarization(
    cache: FileCache,
    manifest: PipelineManifest,
    provider: DiarizationProvider,
    media_path: Path,
    timeline: CanonicalTimeline,
    *,
    dag: DagStore | None = None,
) -> tuple[CanonicalTimeline, dict[str, object]]:
    stage = "canonical-diarization"
    key = _grounding_cache_key(
        stage,
        timeline.source_hash,
        provider,
        {"timeline_sha256": stable_hash(timeline.to_dict())},
    )
    cached_value = cache.read(key, "canonical")
    if isinstance(cached_value, dict):
        try:
            value = CanonicalTimeline.from_dict(cached_value)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            _cache_event(manifest, stage, key, True)
            return value, {"model": provider.identity.to_dict(), "cache_hit": True}

    payload, dag_cached = _grounding_stage_value(
        stage=stage,
        key=key,
        cache=cache,
        dag=dag,
        operation=lambda: provider.diarize(media_path, timeline),
    )
    value, metadata = _grounding_payload(payload, provider)
    metadata["cache_hit"] = dag_cached
    _cache_event(manifest, stage, key, dag_cached)
    return value, metadata

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
        downloaded = gdown.download(url=url, output=str(temporary), quiet=True)  # type: ignore[attr-defined]
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
    request = Request(normalized, headers={"User-Agent": "clipper/production"})  # noqa: S310
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


def _target_candidates(brief_path: str | Path, brief: CampaignBrief) -> list[VideoCandidate]:
    specs = load_explicit_targets(brief_path)
    candidates = [
        VideoCandidate(
            video_id=spec.video_id,
            title=f"{brief.title} explicit target",
            channel_id=spec.channel_id,
            channel_title="Authorized explicit target",
            url=spec.url,
        )
        for spec in specs
    ]
    for candidate in candidates:
        assert_video_allowed(brief, candidate)
    return candidates


def _campaign_watermark(
    brief: CampaignBrief,
    source: SourceClient,
    run_dir: Path,
) -> Path | None:
    if not brief.watermark_url:
        return None
    fixture_watermark = getattr(source, "campaign_watermark", None)
    output = run_dir / "assets" / "campaign-watermark.png"
    if callable(fixture_watermark):
        supplied = fixture_watermark(brief)
        if supplied is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(supplied, output)
            return output
    return _download_asset(brief.watermark_url, output, expected_kind="image")


def _source_media(
    brief: CampaignBrief,
    source: SourceClient,
    video: VideoCandidate,
    work_dir: Path,
    *,
    source_is_authoritative: bool = False,
) -> Path:
    if source_is_authoritative:
        return source.download_media(video, work_dir)
    direct_url = brief.source_media_urls.get(video.video_id)
    if direct_url:
        return _download_asset(
            direct_url,
            work_dir / f"{video.video_id}.source",
            max_bytes=10_000_000_000,
            expected_kind="media",
        )
    return source.download_media(video, work_dir)


def _visual_timeline(
    media_path: Path,
    video: VideoCandidate,
    timeline: CanonicalTimeline,
    provider: VisionProvider,
    run_dir: Path,
) -> tuple[VisualTimeline, dict[str, object]]:
    if not timeline.words:
        raise RuntimeError("canonical grounding produced no source words")
    duration = max(timeline.end, float(video.duration_seconds or 0.0))
    visual, result = scout_visual_timeline(
        media_path,
        provider,
        video_id=video.video_id,
        source_hash=timeline.source_hash,
        duration=duration,
        output_dir=run_dir / "visual-scout" / video.video_id / "frames",
        checkpoint_dir=_VISUAL_CHECKPOINT_DIR.get(),
        checkpoint_commit=_VISUAL_CHECKPOINT_COMMIT.get(),
    )
    _write_json(
        run_dir / "visual-scout" / f"{video.video_id}.json",
        visual.to_dict(),
    )
    return visual, {
        "model": result.model.to_dict(),
        "usage": asdict(result.usage),
        "degraded": result.degraded,
    }


def _speaker_focus_for_source(
    cfg: PipelineSettings,
    quality: QualityBatchResult,
    video_id: str,
) -> bool:
    if cfg.speaker_focus_override is not None:
        return cfg.speaker_focus_override
    evidence = quality.source_evidence.get(video_id)
    profile = evidence.get("modality_profile") if isinstance(evidence, dict) else None
    return bool(profile.get("requires_speaker_identity")) if isinstance(profile, dict) else False


def _renderer_for_source(
    cfg: PipelineSettings,
    quality: QualityBatchResult,
    video_id: str,
) -> FFmpegRenderer:
    return FFmpegRenderer(
        speaker_focus=_speaker_focus_for_source(cfg, quality, video_id),
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


def _remove_render_attempt_files(rendered: Path) -> None:
    for path in (
        rendered,
        rendered.with_suffix(".ass"),
        rendered.with_suffix(".caption-audit.json"),
        rendered.with_suffix(".tracking.json"),
    ):
        path.unlink(missing_ok=True)


def _copy_render_sidecars(rendered: Path, run_dir: Path, plan: EditPlan) -> None:
    slug = _safe_slug(plan.plan_id)
    copies = (
        (rendered.with_suffix(".ass"), run_dir / "captions" / f"{slug}.ass"),
        (
            rendered.with_suffix(".caption-audit.json"),
            run_dir / "captions" / f"{slug}.caption-audit.json",
        ),
        (
            rendered.with_suffix(".tracking.json"),
            run_dir / "tracking" / f"{slug}.tracking.json",
        ),
    )
    for source, target in copies:
        if not source.is_file():
            raise RuntimeError(f"renderer omitted required evidence: {source.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _tracking_transitions(rendered: Path) -> tuple[float, ...]:
    path = rendered.with_suffix(".tracking.json")
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    transitions = payload.get("transitions") if isinstance(payload, dict) else ()
    return tracking_transition_sample_times(transitions)


def _rendered_clip(
    plan: EditPlan,
    rendered: Path,
    video: VideoCandidate,
) -> dict[str, object]:
    span = plan.source_spans[0]
    return RenderedClip(
        video_id=plan.video_id,
        output_path=str(rendered),
        start=span.start,
        end=span.end,
        score=plan.score,
        source_url=video.url,
        concept_id=plan.concept_id,
        plan_id=plan.plan_id,
        hook_mode=plan.hook_mode,
        render_sha256=file_sha256(rendered),
    ).to_dict()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _funnel_template() -> dict[str, int | float | bool]:
    return {
        "transcript_segments": 0,
        "story_moments": 0,
        "raw_concepts": 0,
        "selected_concepts": 0,
        "quality_moments": 0,
        "hook_variants": 0,
        "edit_plans": 0,
        "render_plans": 0,
        "render_attempts": 0,
        "technical_qc_pass": 0,
        "boundary_reject_count": 0,
        "boundary_repair_count": 0,
        "policy_reject_count": 0,
        "hazard_reject_count": 0,
        "editorial_qc_pass": 0,
        "editorial_review_reject_count": 0,
        "reserve_promotions": 0,
        "render_success": 0,
        "submission_shortlist": 0,
    }


def run_pipeline(
    brief_path: str | Path,
    *,
    settings: PipelineSettings | None = None,
    source_client: SourceClient | None = None,
    renderer: Renderer | None = None,
    editorial_provider: EditorialProvider | None = None,
    visual_scout_provider: VisionProvider | None = None,
    visual_review_provider: VisionProvider | None = None,
    visual_escalation_provider: VisionProvider | None = None,
    transcription_provider: TranscriptionProvider | None = None,
    alignment_provider: AlignmentProvider | None = None,
    diarization_provider: DiarizationProvider | None = None,
    render: bool = True,
    checkpoint_commit: Callable[[], None] | None = None,
    dag_store_factory: Callable[[Path], DagStore] | None = None,
    execution_id: str | None = None,
) -> Path:
    """Execute the single supported production architecture over exact campaign targets."""
    brief = load_brief(brief_path)
    assert_campaign_authorized(brief)
    if (
        render
        and brief.watermark_url
        and not brief.acceptance_policy.branding.supplied_campaign_assets_allowed
    ):
        raise ValueError(
            "campaign watermark_url is prohibited by "
            "acceptance_policy.branding.supplied_campaign_assets_allowed"
        )
    cfg = settings or PipelineSettings.from_env()
    source = source_client or _client(cfg)
    source_is_authoritative = source_client is not None
    editor = editorial_provider or build_editorial_provider(cfg.compute_profile)
    scout = visual_scout_provider or vision_provider(cfg.compute_profile)
    reviewer = visual_review_provider or scout
    escalation = visual_escalation_provider
    if cfg.visual_escalation_enabled and escalation is None:
        try:
            escalation = vision_provider(cfg.compute_profile, large=True)
        except ValueError:
            escalation = None

    grounding = (transcription_provider, alignment_provider, diarization_provider)
    supplied = tuple(provider is not None for provider in grounding)
    if any(supplied) and not all(supplied):
        raise ValueError("canonical grounding requires transcription, alignment, and diarization")
    if not all(supplied):
        transcription_provider, alignment_provider, diarization_provider = speech_providers(
            cfg.compute_profile
        )
    if transcription_provider is None or alignment_provider is None or diarization_provider is None:
        raise RuntimeError("canonical grounding provider resolution failed")

    run_execution_id = (execution_id or uuid.uuid4().hex).strip().lower()
    if len(run_execution_id) != 32 or any(
        character not in "0123456789abcdef" for character in run_execution_id
    ):
        raise ValueError("pipeline execution_id must be a 32-character hexadecimal ID")
    run_dir = cfg.artifact_root / _run_id(brief.campaign_id, run_execution_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    journal = StageJournal(run_dir / "progress.json")
    cache_root = cfg.cache_root or (cfg.artifact_root / "_cache")
    cache = FileCache(cache_root)
    grounding_dag = (
        dag_store_factory(cache_root / "grounding") if dag_store_factory is not None else None
    )
    manifest = PipelineManifest(brief.campaign_id)
    manifest.funnel = _funnel_template()
    manifest.run_metadata = {
        "architecture": "autonomous-multimodal-quality-graph",
        "execution_id": run_execution_id,
        "git_sha": _git_sha(),
        "compute_profile": cfg.compute_profile,
        "source_hashes": {},
        "grounding_inference": {"models": []},
        "editorial_inference": {
            "model": editor.identity.to_dict(),
            "model_invocations": [],
        },
        "visual_inference": {"scout": [], "review": []},
    }
    _write_json(run_dir / "brief.normalized.json", brief.to_dict())

    started = time.perf_counter()
    targets = _target_candidates(brief_path, brief)
    manifest.discovered_videos = [video.to_dict() for video in targets]
    source_runtimes: dict[str, SourceRuntime] = {}
    canonical_timelines: dict[str, CanonicalTimeline] = {}
    visual_timelines: dict[str, VisualTimeline] = {}

    journal.start("source_grounding", total=len(targets))
    for index, video in enumerate(targets, start=1):
        work_dir = run_dir / "work" / video.video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            media_path = _source_media(
                brief,
                source,
                video,
                work_dir,
                source_is_authoritative=source_is_authoritative,
            )
            if not media_path.is_file() or media_path.stat().st_size <= 0:
                raise RuntimeError("source client returned an invalid media master")
            source_hash = file_sha256(media_path)
            manifest.run_metadata["source_hashes"][video.video_id] = source_hash
            _record_source_media_metadata(manifest, video.video_id, media_path)

            transcribed, transcription_meta = _cached_transcription(
                cache,
                manifest,
                transcription_provider,
                media_path,
                video.video_id,
                source_hash,
                dag=grounding_dag,
            )
            aligned, alignment_meta = _cached_alignment(
                cache,
                manifest,
                alignment_provider,
                media_path,
                transcribed,
                dag=grounding_dag,
            )
            timeline, diarization_meta = _cached_diarization(
                cache,
                manifest,
                diarization_provider,
                media_path,
                aligned,
                dag=grounding_dag,
            )
            if not timeline.words:
                raise RuntimeError("canonical grounding produced no timestamped source evidence")
            segments = tuple(transcript_segments_from_canonical(timeline))
            if not segments:
                raise RuntimeError("canonical timeline produced no transcript segments")
            checkpoint_dir_token = _VISUAL_CHECKPOINT_DIR.set(cache_root / "source-policy-vision")
            checkpoint_commit_token = _VISUAL_CHECKPOINT_COMMIT.set(checkpoint_commit)
            try:
                visual, visual_meta = _visual_timeline(
                    media_path,
                    video,
                    timeline,
                    scout,
                    run_dir,
                )
            finally:
                _VISUAL_CHECKPOINT_COMMIT.reset(checkpoint_commit_token)
                _VISUAL_CHECKPOINT_DIR.reset(checkpoint_dir_token)

            source_runtimes[video.video_id] = SourceRuntime(
                video=video,
                media_path=media_path,
                source_hash=source_hash,
                timeline=timeline,
                segments=segments,
                visual_timeline=visual,
            )
            canonical_timelines[video.video_id] = timeline
            visual_timelines[video.video_id] = visual
            manifest.run_metadata["grounding_inference"]["models"].append(
                {
                    "video_id": video.video_id,
                    "transcription": transcription_meta,
                    "alignment": alignment_meta,
                    "diarization": diarization_meta,
                }
            )
            manifest.run_metadata["visual_inference"]["scout"].append(
                {"video_id": video.video_id, **visual_meta}
            )
            _write_json(run_dir / "canonical" / f"{video.video_id}.json", timeline.to_dict())
            journal.progress("source_grounding", index, total=len(targets))
        except Exception as exc:
            manifest.errors.append(
                {
                    "stage": "source_grounding",
                    "video_id": video.video_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            journal.fail("source_grounding", str(exc), checkpoint=video.video_id)
            manifest.status = "FAILED"
            manifest.status_reason = "explicit_target_grounding_failed"
            manifest.publication_state = "TECHNICALLY_INCOMPLETE"
            _write_json(run_dir / "manifest.json", manifest.to_dict())
            return run_dir
    journal.complete("source_grounding")

    manifest.funnel["transcript_segments"] = sum(
        len(runtime.segments) for runtime in source_runtimes.values()
    )
    _write_json(
        run_dir / "transcript.json",
        {
            video_id: [segment.to_dict() for segment in runtime.segments]
            for video_id, runtime in source_runtimes.items()
        },
    )

    journal.start("quality_graph")
    try:
        if dag_store_factory is None:
            quality = plan_quality_batch(
                brief,
                canonical_timelines,
                editorial=editor,
                dag_root=cfg.cache_root or (cfg.artifact_root / "_cache"),
                visual_timelines=visual_timelines,
            )
        else:
            quality = plan_quality_batch(
                brief,
                canonical_timelines,
                editorial=editor,
                dag_root=cfg.cache_root or (cfg.artifact_root / "_cache"),
                visual_timelines=visual_timelines,
                dag_store_factory=dag_store_factory,
            )
    except Exception as exc:
        journal.fail("quality_graph", str(exc))
        manifest.status = "FAILED"
        manifest.status_reason = "autonomous_quality_graph_failed"
        manifest.errors.append(
            {
                "stage": "quality_graph",
                "video_id": "*",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        manifest.publication_state = "TECHNICALLY_INCOMPLETE"
        _write_json(run_dir / "manifest.json", manifest.to_dict())
        return run_dir
    journal.complete("quality_graph")

    manifest.story_moments = [item.to_dict() for item in quality.story_moments]
    manifest.clip_concepts = [item.to_dict() for item in quality.concepts]
    manifest.hook_variants = [item.to_dict() for item in quality.variants]
    manifest.edit_plans = [item.to_dict() for item in quality.plans]
    manifest.rejections.extend(quality.rejections)
    quality_groups = group_quality_plans(quality.plans)
    eligible = len(quality_groups)
    manifest.targets = {"eligible_quality_moments": eligible}
    manifest.funnel.update(
        {
            "story_moments": len(quality.story_moments),
            "raw_concepts": len(quality.concepts),
            "selected_concepts": len(quality.concepts),
            "quality_moments": eligible,
            "hook_variants": len(quality.variants),
            "edit_plans": len(quality.plans),
            "render_plans": eligible,
            "hazard_reject_count": sum(
                1
                for item in quality.rejections
                if isinstance(item, dict) and item.get("stage") == "source_hazards"
            ),
        }
    )
    manifest.run_metadata["quality_yield"] = {
        "semantic_cores_discovered": sum(
            value
            for item in quality.source_evidence.values()
            if isinstance(item, dict)
            for value in (item.get("semantic_cores"),)
            if isinstance(value, int) and not isinstance(value, bool)
        ),
        "eligible_quality_moments": eligible,
        "primary_plans": eligible,
        "reserve_variants": max(0, len(quality.plans) - eligible),
        "rendered": 0,
        "accepted": 0,
        "unrendered_or_rejected": eligible,
    }
    manifest.run_metadata["editorial_inference"]["model_invocations"] = list(
        quality.model_invocations
    )
    manifest.run_metadata["editorial_inference"]["cache_summary"] = {
        "stage_cache_hits": quality.stage_cache_hits,
        "stage_executions": quality.stage_executions,
        "editorial_model_fingerprint": editor.identity.cache_fingerprint(),
        "editorial_model": editor.identity.to_dict(),
    }
    manifest.cache["quality_graph"] = {
        "stage_cache_hits": quality.stage_cache_hits,
        "stage_executions": quality.stage_executions,
        "dag_root": str(quality.stage_dag_root),
    }

    _write_json(run_dir / "story-moments.json", manifest.story_moments)
    _write_json(run_dir / "concept-ranking.json", manifest.clip_concepts)
    _write_json(run_dir / "hook-variants.json", manifest.hook_variants)
    _write_json(run_dir / "edit-plans.json", manifest.edit_plans)

    visual_strategy_payloads: list[dict[str, object]] = []
    for moment in quality.quality_moments:
        runtime = source_runtimes[moment.core.video_id]
        multimodal = build_multimodal_timeline(
            runtime.timeline,
            runtime.visual_timeline,
            visual_provenance=EvidenceProvenance(
                provider="vision",
                model_id=scout.identity.model_id,
                revision=scout.identity.revision,
                contract=scout.identity.schema_version,
            ),
        )
        strategy = derive_visual_strategy(moment, multimodal)
        payload = strategy.to_dict()
        visual_strategy_payloads.append(payload)
        _write_json(
            run_dir / "visual-strategy" / f"{_safe_slug(moment.quality_moment_id)}.json",
            payload,
        )
    manifest.run_metadata["visual_strategies"] = visual_strategy_payloads

    concept_text = {concept.concept_id: concept.text for concept in quality.concepts}
    video_index = {video.video_id: video for video in targets}
    manifest.planned_clips = [
        plan.to_clip_candidate(concept_text.get(plan.concept_id, "")).to_dict()
        for plan in quality.plans
    ]

    if not render:
        manifest.status = "SUCCESS"
        manifest.status_reason = "planning_complete"
        manifest.publication_state = "PLANNED_NOT_RENDERED"
        manifest.actual = {
            "eligible_quality_moments": eligible,
            "rendered_finalists": 0,
            "submission_shortlist": 0,
            "distinct_finalist_concepts": 0,
            "distinct_shortlist_concepts": 0,
        }
        manifest.performance = {"elapsed_seconds": round(time.perf_counter() - started, 3)}
        _finalize_run_artifacts(run_dir, manifest)
        return run_dir

    watermark_path = _campaign_watermark(brief, source, run_dir)
    accepted_plans: list[EditPlan] = []
    accepted_concepts: set[str] = set()
    render_queue = quality_render_queue(quality_groups)
    journal.start("render_and_review", total=len(render_queue))

    for attempt_number, (queue_type, plan) in enumerate(render_queue, start=1):
        if plan.concept_id in accepted_concepts:
            continue
        runtime = source_runtimes[plan.video_id]
        clip = plan.to_clip_candidate(concept_text.get(plan.concept_id, ""))
        active_renderer = renderer or _renderer_for_source(cfg, quality, plan.video_id)
        output = (
            run_dir
            / "clips"
            / (
                f"attempt-{attempt_number:03d}-{_safe_slug(plan.concept_id)}-"
                f"{_safe_slug(plan.plan_id)}.mp4"
            )
        )
        attempt: dict[str, object] = {
            "attempt": attempt_number,
            "queue": queue_type,
            "concept_id": plan.concept_id,
            "plan_id": plan.plan_id,
        }
        rendered: Path | None = None
        try:
            rendered = active_renderer.render(
                runtime.media_path,
                output,
                clip,
                list(runtime.segments),
                watermark_path,
                plan,
            )
            manifest.funnel["render_attempts"] += 1
            qc = run_technical_qc(
                rendered,
                expected_duration=clip.duration,
                caption_path=rendered.with_suffix(".ass"),
                tracking_path=rendered.with_suffix(".tracking.json"),
                caption_platform=plan.caption_platform,
                watermark_required=bool(brief.watermark_url),
                watermark_present=watermark_path is not None,
                caption_audit_path=rendered.with_suffix(".caption-audit.json"),
            )
            qc["concept_id"] = plan.concept_id
            qc["plan_id"] = plan.plan_id
            manifest.technical_qc.append(qc)
            if qc.get("status") != "PASS":
                _remove_render_attempt_files(rendered)
                attempt.update({"status": "REJECTED", "reason": "technical_qc_failed"})
                manifest.render_attempts.append(attempt)
                continue
            manifest.funnel["technical_qc_pass"] += 1

            boundary = dict(plan.boundary_audit or {})
            policy = dict(plan.campaign_policy_audit or {})
            boundary.update({"concept_id": plan.concept_id, "plan_id": plan.plan_id})
            policy.update({"concept_id": plan.concept_id, "plan_id": plan.plan_id})

            review, review_results = review_rendered_clip(
                rendered,
                reviewer,
                duration=clip.duration,
                output_dir=run_dir / "visual-review" / rendered.stem / "frames",
                context={
                    "plan_id": plan.plan_id,
                    "concept_id": plan.concept_id,
                    "opening_strategy": plan.hook_mode,
                    "source_start": clip.start,
                    "source_end": clip.end,
                    "canonical_text": concept_text.get(plan.concept_id, ""),
                    "technical_qc": qc,
                    "boundary_audit": boundary,
                    "campaign_policy_audit": policy,
                    "visual_strategy": next(
                        (
                            item
                            for item in visual_strategy_payloads
                            if item.get("quality_moment_id") == plan.concept_id
                        ),
                        None,
                    ),
                },
                transitions=_tracking_transitions(rendered),
                escalation=escalation,
                escalation_threshold=cfg.visual_escalation_threshold,
            )
            review_payload = review.to_dict()
            review_payload.update(
                {
                    "concept_id": plan.concept_id,
                    "plan_id": plan.plan_id,
                    "models": [result.model.to_dict() for result in review_results],
                    "usage": [asdict(result.usage) for result in review_results],
                }
            )
            manifest.editorial_qc.append(review_payload)
            manifest.run_metadata["visual_inference"]["review"].append(review_payload)
            boundary["multimodal_editorial_review_decision"] = review.decision
            policy["multimodal_policy_review_decision"] = review.decision
            manifest.boundary_qc.append(boundary)
            manifest.campaign_policy_qc.append(policy)
            if review.decision != "PASS" or review.issues:
                manifest.funnel["editorial_review_reject_count"] += 1
                _remove_render_attempt_files(rendered)
                attempt.update({"status": "REJECTED", "reason": "multimodal_review_failed"})
                manifest.render_attempts.append(attempt)
                continue

            _copy_render_sidecars(rendered, run_dir, plan)
            slug = _safe_slug(plan.plan_id)
            _write_json(run_dir / "boundary" / f"{slug}.boundary-audit.json", boundary)
            _write_json(run_dir / "policy" / f"{slug}.policy-audit.json", policy)
            rendered_payload = _rendered_clip(plan, rendered, video_index[plan.video_id])
            manifest.rendered_clips.append(rendered_payload)
            accepted_plans.append(plan)
            accepted_concepts.add(plan.concept_id)
            manifest.funnel["render_success"] += 1
            manifest.funnel["editorial_qc_pass"] += 1
            if queue_type == "reserve":
                manifest.funnel["reserve_promotions"] += 1
            attempt.update({"status": "ACCEPTED"})
            manifest.render_attempts.append(attempt)
        except Exception as exc:
            _remove_render_attempt_files(output)
            if rendered is not None and rendered != output:
                _remove_render_attempt_files(rendered)
            manifest.errors.append(
                {
                    "stage": "render",
                    "video_id": plan.video_id,
                    "concept_id": plan.concept_id,
                    "plan_id": plan.plan_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            attempt.update({"status": "FAILED", "reason": str(exc)})
            manifest.render_attempts.append(attempt)
        journal.progress("render_and_review", attempt_number, total=len(render_queue))
    journal.complete("render_and_review")

    accepted = accepted_quality_plans(accepted_plans)
    accepted_plan_ids = {item.plan_id for item in accepted}
    manifest.submission_shortlist = [
        item
        for item in manifest.rendered_clips
        if isinstance(item, dict) and str(item.get("plan_id")) in accepted_plan_ids
    ]
    rendered_count = len(manifest.submission_shortlist)
    manifest.funnel["submission_shortlist"] = rendered_count
    manifest.actual = {
        "eligible_quality_moments": eligible,
        "rendered_finalists": rendered_count,
        "submission_shortlist": rendered_count,
        "distinct_finalist_concepts": rendered_count,
        "distinct_shortlist_concepts": rendered_count,
    }
    quality_yield = manifest.run_metadata["quality_yield"]
    if isinstance(quality_yield, dict):
        quality_yield.update(
            {
                "rendered": rendered_count,
                "accepted": rendered_count,
                "unrendered_or_rejected": eligible - rendered_count,
            }
        )

    if eligible == 0:
        manifest.status = "SUCCESS"
        manifest.status_reason = "no_quality_moments"
        manifest.publication_state = "COMPLETED_NO_ELIGIBLE_MOMENTS"
    elif rendered_count == eligible:
        manifest.status = "SUCCESS"
        manifest.status_reason = None
        manifest.publication_state = "READY_FOR_HUMAN_REVIEW"
    elif rendered_count > 0:
        manifest.status = "DEGRADED"
        manifest.status_reason = "partial_quality_yield"
        manifest.publication_state = "READY_FOR_HUMAN_REVIEW"
    else:
        manifest.status = "FAILED"
        manifest.status_reason = "eligible_quality_moments_not_rendered"
        manifest.publication_state = "TECHNICALLY_INCOMPLETE"

    manifest.performance = {"elapsed_seconds": round(time.perf_counter() - started, 3)}
    _finalize_run_artifacts(run_dir, manifest)
    return run_dir


def _finalize_run_artifacts(run_dir: Path, manifest: PipelineManifest) -> None:
    _write_json(run_dir / "manifest.json", manifest.to_dict())
    _write_json(run_dir / "funnel.json", manifest.funnel)
    _write_json(run_dir / "rejections.json", manifest.rejections)
    _write_json(
        run_dir / "coverage.json",
        {
            "explicit_targets": len(manifest.discovered_videos),
            "grounded_targets": len(manifest.run_metadata.get("source_hashes", {})),
            "eligible_quality_moments": int(manifest.targets.get("eligible_quality_moments", 0)),
            "accepted_quality_moments": len(manifest.submission_shortlist),
        },
    )
    _write_json(
        run_dir / "editorial-review.json",
        {
            "status": (
                "PENDING_HUMAN_REVIEW" if manifest.submission_shortlist else "NO_ELIGIBLE_OUTPUTS"
            ),
            "required": bool(manifest.submission_shortlist),
            "clips": [
                {
                    "output_path": item.get("output_path"),
                    "plan_id": item.get("plan_id"),
                    "concept_id": item.get("concept_id"),
                    "human_review": "PENDING",
                }
                for item in manifest.submission_shortlist
                if isinstance(item, dict)
            ],
        },
    )
    _write_json(run_dir / "performance.json", manifest.performance)
