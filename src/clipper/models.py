from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast


class BriefValidationError(ValueError):
    """Raised when a campaign brief is incomplete or unsafe."""


def _string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BriefValidationError(f"{key} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _string_map(data: dict[str, Any], key: str) -> dict[str, str]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(item_key, str) and isinstance(item_value, str)
        for item_key, item_value in value.items()
    ):
        raise BriefValidationError(f"{key} must be an object mapping strings to strings")
    return {
        item_key.strip(): item_value.strip()
        for item_key, item_value in value.items()
        if item_key.strip() and item_value.strip()
    }


HookMode = Literal[
    "direct",
    "payoff_first",
    "curiosity_text",
    "question",
    "number",
    "conflict",
    "strong_opinion",
]
BeatType = Literal[
    "hold",
    "punch_in",
    "punch_out",
    "speaker_switch",
    "reaction",
    "source_cut",
    "text_emphasis",
    "broll",
    "screenshot",
    "graphic",
    "payoff_hold",
]


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    candidate_pool_size: int = 36
    concept_count: int = 10
    variants_per_concept: int = 1
    final_render_budget: int = 1
    minimum_distinct_finalist_concepts: int = 1

    @classmethod
    def from_dict(cls, value: object) -> ProductionConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("production must be an object")
        config = cls(
            candidate_pool_size=int(value.get("candidate_pool_size", 36)),
            concept_count=int(value.get("concept_count", 10)),
            variants_per_concept=int(value.get("variants_per_concept", 1)),
            final_render_budget=int(value.get("final_render_budget", 1)),
            minimum_distinct_finalist_concepts=int(
                value.get("minimum_distinct_finalist_concepts", 1)
            ),
        )
        if not 10 <= config.candidate_pool_size <= 100:
            raise BriefValidationError("production.candidate_pool_size must be between 10 and 100")
        if not 1 <= config.concept_count <= 30:
            raise BriefValidationError("production.concept_count must be between 1 and 30")
        if not 1 <= config.variants_per_concept <= 6:
            raise BriefValidationError("production.variants_per_concept must be between 1 and 6")
        if not 1 <= config.final_render_budget <= 24:
            raise BriefValidationError("production.final_render_budget must be between 1 and 24")
        if not 1 <= config.minimum_distinct_finalist_concepts <= config.final_render_budget:
            raise BriefValidationError(
                "production.minimum_distinct_finalist_concepts must be between 1 "
                "and final_render_budget"
            )
        if config.minimum_distinct_finalist_concepts > config.concept_count:
            raise BriefValidationError(
                "production.minimum_distinct_finalist_concepts cannot exceed concept_count"
            )
        return config


@dataclass(frozen=True, slots=True)
class DiversityConfig:
    semantic_similarity_threshold: float = 0.72
    max_concepts_per_topic: int = 2

    @classmethod
    def from_dict(cls, value: object) -> DiversityConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("diversity must be an object")
        config = cls(
            semantic_similarity_threshold=float(value.get("semantic_similarity_threshold", 0.72)),
            max_concepts_per_topic=int(value.get("max_concepts_per_topic", 2)),
        )
        if not 0.3 <= config.semantic_similarity_threshold <= 0.95:
            raise BriefValidationError(
                "diversity.semantic_similarity_threshold must be between 0.3 and 0.95"
            )
        if not 1 <= config.max_concepts_per_topic <= 5:
            raise BriefValidationError("diversity.max_concepts_per_topic must be between 1 and 5")
        return config


@dataclass(frozen=True, slots=True)
class HooksConfig:
    enabled: tuple[HookMode, ...] = (
        "direct",
        "curiosity_text",
        "question",
        "number",
        "conflict",
        "strong_opinion",
    )

    @classmethod
    def from_dict(cls, value: object) -> HooksConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("hooks must be an object")
        raw = value.get("enabled", list(cls().enabled))
        if not isinstance(raw, (list, tuple)) or not all(isinstance(item, str) for item in raw):
            raise BriefValidationError("hooks.enabled must be a list or tuple of strings")
        allowed = set(cls().enabled) | {"payoff_first"}
        enabled: list[HookMode] = []
        for item in raw:
            normalized = item.strip().lower()
            if normalized not in allowed:
                raise BriefValidationError(f"unsupported hook mode: {normalized}")
            enabled.append(cast(HookMode, normalized))
        if not enabled:
            raise BriefValidationError("hooks.enabled must contain at least one mode")
        return cls(tuple(dict.fromkeys(enabled)))


@dataclass(frozen=True, slots=True)
class EditorialScoreWeights:
    hook_strength: float = 1.2
    curiosity: float = 0.95
    payoff_strength: float = 1.15
    standalone_clarity: float = 1.05
    emotional_energy: float = 0.55
    information_value: float = 0.75
    controversy_or_tension: float = 0.45
    quoteability: float = 0.65
    specificity: float = 0.7
    campaign_relevance: float = 1.25
    story_completeness: float = 1.1
    retention_potential: float = 1.2

    @classmethod
    def from_dict(cls, value: object) -> EditorialScoreWeights:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("editorial.score_weights must be an object")
        defaults = asdict(cls())
        unknown = set(value) - set(defaults)
        if unknown:
            raise BriefValidationError(f"unsupported editorial score weight: {sorted(unknown)[0]}")
        values = {key: float(value.get(key, default)) for key, default in defaults.items()}
        if any(weight <= 0 or weight > 5 for weight in values.values()):
            raise BriefValidationError("editorial score weights must be between 0 and 5")
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EditorialConfig:
    platform: str = "tiktok"
    punch_ins_enabled: bool = False
    max_punch_ins_per_clip: int = 0
    semantic_endings: bool = True
    post_speech_tail_seconds: float = 0.25
    caption_max_lines: int = 2
    score_weights: EditorialScoreWeights = field(default_factory=EditorialScoreWeights)

    @classmethod
    def from_dict(cls, value: object) -> EditorialConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("editorial must be an object")
        config = cls(
            platform=str(value.get("platform", "tiktok")).strip().lower(),
            punch_ins_enabled=bool(value.get("punch_ins_enabled", False)),
            max_punch_ins_per_clip=int(value.get("max_punch_ins_per_clip", 0)),
            semantic_endings=bool(value.get("semantic_endings", True)),
            post_speech_tail_seconds=float(value.get("post_speech_tail_seconds", 0.25)),
            caption_max_lines=int(value.get("caption_max_lines", 2)),
            score_weights=EditorialScoreWeights.from_dict(value.get("score_weights")),
        )
        if config.platform not in {
            "tiktok",
            "instagram_reels",
            "youtube_shorts",
            "generic_vertical",
        }:
            raise BriefValidationError("editorial.platform is unsupported")
        if not 0 <= config.max_punch_ins_per_clip <= 3:
            raise BriefValidationError("editorial.max_punch_ins_per_clip must be between 0 and 3")
        if not 0.0 <= config.post_speech_tail_seconds <= 1.0:
            raise BriefValidationError("editorial.post_speech_tail_seconds must be between 0 and 1")
        if config.caption_max_lines not in {1, 2}:
            raise BriefValidationError("editorial.caption_max_lines must be 1 or 2")
        return config


@dataclass(frozen=True, slots=True)
class CampaignBrief:
    campaign_id: str
    title: str
    objective: str
    keywords: list[str]
    source_channel_ids: list[str] = field(default_factory=list)
    allowed_video_ids: list[str] = field(default_factory=list)
    source_media_urls: dict[str, str] = field(default_factory=dict)
    negative_keywords: list[str] = field(default_factory=list)
    required_phrases: list[str] = field(default_factory=list)
    language: str = "en"
    region_code: str = "US"
    clip_count: int = 3
    min_clip_seconds: float = 20.0
    max_clip_seconds: float = 45.0
    source_limit: int = 8
    max_clips_per_source: int = 1
    published_after: str | None = None
    rights_confirmed: bool = False
    attribution_required: bool = True
    watermark_text: str | None = None
    watermark_url: str | None = None
    required_hashtags: list[str] = field(default_factory=list)
    posting_requirements: list[str] = field(default_factory=list)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    editorial: EditorialConfig = field(default_factory=EditorialConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignBrief:
        if not isinstance(data, dict):
            raise BriefValidationError("brief root must be an object")
        required = ("campaign_id", "title", "objective", "keywords")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise BriefValidationError(f"missing required fields: {', '.join(missing)}")
        brief = cls(
            campaign_id=str(data["campaign_id"]).strip(),
            title=str(data["title"]).strip(),
            objective=str(data["objective"]).strip(),
            keywords=_string_list(data, "keywords"),
            source_channel_ids=_string_list(data, "source_channel_ids"),
            allowed_video_ids=_string_list(data, "allowed_video_ids"),
            source_media_urls=_string_map(data, "source_media_urls"),
            negative_keywords=_string_list(data, "negative_keywords"),
            required_phrases=_string_list(data, "required_phrases"),
            language=str(data.get("language", "en")).strip() or "en",
            region_code=str(data.get("region_code", "US")).strip().upper() or "US",
            clip_count=int(data.get("clip_count", 3)),
            min_clip_seconds=float(data.get("min_clip_seconds", 20.0)),
            max_clip_seconds=float(data.get("max_clip_seconds", 45.0)),
            source_limit=int(data.get("source_limit", 8)),
            max_clips_per_source=int(data.get("max_clips_per_source", 1)),
            published_after=(
                str(data["published_after"]).strip() if data.get("published_after") else None
            ),
            rights_confirmed=bool(data.get("rights_confirmed", False)),
            attribution_required=bool(data.get("attribution_required", True)),
            watermark_text=(
                str(data["watermark_text"]).strip() if data.get("watermark_text") else None
            ),
            watermark_url=(
                str(data["watermark_url"]).strip() if data.get("watermark_url") else None
            ),
            required_hashtags=_string_list(data, "required_hashtags"),
            posting_requirements=_string_list(data, "posting_requirements"),
            production=ProductionConfig.from_dict(data.get("production")),
            diversity=DiversityConfig.from_dict(data.get("diversity")),
            hooks=HooksConfig.from_dict(data.get("hooks")),
            editorial=EditorialConfig.from_dict(data.get("editorial")),
        )
        brief.validate()
        return brief

    def validate(self) -> None:
        if not self.keywords:
            raise BriefValidationError("keywords must contain at least one value")
        if len(self.region_code) != 2:
            raise BriefValidationError("region_code must be a two-letter ISO country code")
        if self.clip_count < 1 or self.clip_count > 20:
            raise BriefValidationError("clip_count must be between 1 and 20")
        if self.source_limit < 1 or self.source_limit > 50:
            raise BriefValidationError("source_limit must be between 1 and 50")
        if self.max_clips_per_source < 1:
            raise BriefValidationError("max_clips_per_source must be at least 1")
        if self.min_clip_seconds < 8:
            raise BriefValidationError("min_clip_seconds must be at least 8")
        if self.max_clip_seconds > 180:
            raise BriefValidationError("max_clip_seconds must be at most 180")
        if self.min_clip_seconds >= self.max_clip_seconds:
            raise BriefValidationError("min_clip_seconds must be less than max_clip_seconds")
        if self.watermark_url and not self.watermark_url.startswith("https://"):
            raise BriefValidationError("watermark_url must use https")
        for video_id, media_url in self.source_media_urls.items():
            if video_id not in self.allowed_video_ids:
                raise BriefValidationError(
                    "source_media_urls keys must also be listed in allowed_video_ids"
                )
            if not media_url.startswith("https://"):
                raise BriefValidationError("source_media_urls values must use https")
        if not self.source_channel_ids and not self.allowed_video_ids:
            raise BriefValidationError(
                "provide source_channel_ids or allowed_video_ids; unrestricted scraping is disabled"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def search_query(self) -> str:
        return " ".join([self.title, *self.keywords]).strip()


@dataclass(frozen=True, slots=True)
class VideoCandidate:
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    url: str
    description: str = ""
    published_at: str | None = None
    duration_seconds: float | None = None
    view_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("transcript word timestamps are invalid")
        if not self.text.strip():
            raise ValueError("transcript word text cannot be empty")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("transcript segment timestamps are invalid")
        if not self.text.strip():
            raise ValueError("transcript segment text cannot be empty")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClipCandidate:
    video_id: str
    start: float
    end: float
    text: str
    score: float
    reasons: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


@dataclass(frozen=True, slots=True)
class EditorialScores:
    hook_strength: float
    curiosity: float
    payoff_strength: float
    standalone_clarity: float
    emotional_energy: float
    information_value: float
    controversy_or_tension: float
    quoteability: float
    specificity: float
    campaign_relevance: float
    story_completeness: float
    retention_potential: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StoryMoment:
    moment_id: str
    video_id: str
    start: float
    end: float
    text: str
    moment_type: str
    topic: str
    setup: str
    payoff: str
    scores: EditorialScores
    score: float
    transcript_fingerprint: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClipConcept:
    concept_id: str
    video_id: str
    source_start: float
    source_end: float
    text: str
    topic: str
    setup: str
    payoff: str
    moment_type: str
    recommended_duration: float
    scores: EditorialScores
    score: float
    semantic_cluster: str
    transcript_fingerprint: str

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source span timestamps are invalid")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class HookVariant:
    variant_id: str
    concept_id: str
    mode: HookMode
    source_spans: tuple[SourceSpan, ...]
    overlay_text: str | None
    score: float
    rationale: str
    fingerprint: str
    caption_start_source_time: float | None = None
    caption_start_word: str | None = None

    @property
    def duration(self) -> float:
        return sum(span.duration for span in self.source_spans)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_spans"] = [asdict(span) for span in self.source_spans]
        return data


@dataclass(frozen=True, slots=True)
class EditorialBeat:
    start: float
    end: float
    beat_type: BeatType
    strength: float = 0.0
    text: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("editorial beat timestamps are invalid")
        if not 0.0 <= self.strength <= 0.2:
            raise ValueError("editorial beat strength must be between 0 and 0.2")


@dataclass(frozen=True, slots=True)
class EditPlan:
    plan_id: str
    video_id: str
    concept_id: str
    variant_id: str
    hook_mode: HookMode
    source_spans: tuple[SourceSpan, ...]
    hook_text: str | None
    beats: tuple[EditorialBeat, ...]
    caption_platform: str
    score: float
    transcript_fingerprint: str
    caption_start_source_time: float | None = None
    caption_start_word: str | None = None

    @property
    def duration(self) -> float:
        return sum(span.duration for span in self.source_spans)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_spans"] = [asdict(span) for span in self.source_spans]
        data["beats"] = [asdict(beat) for beat in self.beats]
        return data

    def to_clip_candidate(self, text: str) -> ClipCandidate:
        if len(self.source_spans) != 1:
            raise ValueError("current renderer requires a contiguous source span")
        span = self.source_spans[0]
        return ClipCandidate(
            video_id=self.video_id,
            start=span.start,
            end=span.end,
            text=text,
            score=self.score,
            reasons=(f"concept={self.concept_id}", f"hook={self.hook_mode}"),
        )


@dataclass(frozen=True, slots=True)
class RenderedClip:
    video_id: str
    output_path: str
    start: float
    end: float
    score: float
    source_url: str
    concept_id: str | None = None
    plan_id: str | None = None
    hook_mode: str | None = None
    render_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineManifest:
    campaign_id: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "SUCCESS"
    status_reason: str | None = None
    targets: dict[str, int] = field(default_factory=dict)
    actual: dict[str, int] = field(default_factory=dict)
    funnel: dict[str, int | float | bool] = field(default_factory=dict)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    render_attempts: list[dict[str, Any]] = field(default_factory=list)
    reserve_plans: list[dict[str, Any]] = field(default_factory=list)
    discovered_videos: list[dict[str, Any]] = field(default_factory=list)
    story_moments: list[dict[str, Any]] = field(default_factory=list)
    clip_concepts: list[dict[str, Any]] = field(default_factory=list)
    hook_variants: list[dict[str, Any]] = field(default_factory=list)
    edit_plans: list[dict[str, Any]] = field(default_factory=list)
    planned_clips: list[dict[str, Any]] = field(default_factory=list)
    submission_shortlist: list[dict[str, Any]] = field(default_factory=list)
    rendered_clips: list[dict[str, Any]] = field(default_factory=list)
    technical_qc: list[dict[str, Any]] = field(default_factory=list)
    editorial_qc: list[dict[str, Any]] = field(default_factory=list)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
