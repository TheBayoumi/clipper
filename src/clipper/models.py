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


# Editorial/opening/beat labels are intentionally free-form. Their meaning is inferred
# from source evidence by the model/VLM; Python never enumerates desirable hook categories.
HookMode = str
BeatType = str
PolicyAction = Literal["allow", "forbid", "escalate"]
ForeignLogoPolicy = Literal["allow", "forbid", "escalate"]

SOURCE_HAZARD_CLASSIFICATIONS = frozenset(
    {
        "editorial_content",
        "advertisement",
        "sponsor_read",
        "promo",
        "intro",
        "outro",
        "housekeeping",
        "graphic_heavy",
        "unknown",
    }
)


def _policy_action(value: object, field_name: str) -> PolicyAction:
    normalized = str(value).strip().lower()
    if normalized not in {"allow", "forbid", "escalate"}:
        raise BriefValidationError(f"{field_name} must be allow, forbid, or escalate")
    return cast(PolicyAction, normalized)


@dataclass(frozen=True, slots=True)
class SourceSegmentPolicy:
    allow: tuple[str, ...] = ("editorial_content",)
    forbid: tuple[str, ...] = (
        "advertisement",
        "sponsor_read",
        "promo",
        "intro",
        "outro",
        "housekeeping",
    )
    unknown: PolicyAction = "escalate"
    safety_buffer_seconds: float = 0.0

    @classmethod
    def from_dict(cls, value: object) -> SourceSegmentPolicy:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("acceptance_policy.source_segments must be an object")
        unknown_fields = set(value) - {
            "allow",
            "forbid",
            "unknown",
            "safety_buffer_seconds",
        }
        if unknown_fields:
            raise BriefValidationError(
                f"unsupported acceptance_policy.source_segments rule: {sorted(unknown_fields)[0]}"
            )

        def classifications(field_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(field_name, list(default))
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise BriefValidationError(
                    f"acceptance_policy.source_segments.{field_name} must be a list of strings"
                )
            normalized = tuple(dict.fromkeys(item.strip().lower() for item in raw if item.strip()))
            unsupported = set(normalized) - SOURCE_HAZARD_CLASSIFICATIONS
            if unsupported:
                raise BriefValidationError(
                    f"unsupported source hazard classification: {sorted(unsupported)[0]}"
                )
            return normalized

        allow = classifications("allow", cls().allow)
        forbid = classifications("forbid", cls().forbid)
        if set(allow) & set(forbid):
            raise BriefValidationError(
                "acceptance_policy source classifications cannot be both allowed and forbidden"
            )
        safety_buffer = float(value.get("safety_buffer_seconds", 0.0))
        if not 0.0 <= safety_buffer <= 5.0:
            raise BriefValidationError(
                "acceptance_policy.source_segments.safety_buffer_seconds must be between 0 and 5"
            )
        return cls(
            allow=allow,
            forbid=forbid,
            unknown=_policy_action(
                value.get("unknown", "escalate"), "acceptance_policy.source_segments.unknown"
            ),
            safety_buffer_seconds=safety_buffer,
        )


@dataclass(frozen=True, slots=True)
class BrandingPolicy:
    supplied_campaign_assets_allowed: bool = True
    foreign_logos: ForeignLogoPolicy = "escalate"
    minimum_confidence: float = 0.75

    @classmethod
    def from_dict(cls, value: object) -> BrandingPolicy:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("acceptance_policy.branding must be an object")
        unknown_fields = set(value) - {
            "supplied_campaign_assets_allowed",
            "foreign_logos",
            "minimum_confidence",
        }
        if unknown_fields:
            raise BriefValidationError(
                f"unsupported acceptance_policy.branding rule: {sorted(unknown_fields)[0]}"
            )
        action = _policy_action(
            value.get("foreign_logos", "escalate"),
            "acceptance_policy.branding.foreign_logos",
        )
        confidence = float(value.get("minimum_confidence", 0.75))
        if not 0.0 <= confidence <= 1.0:
            raise BriefValidationError(
                "acceptance_policy.branding.minimum_confidence must be between 0 and 1"
            )
        return cls(
            supplied_campaign_assets_allowed=bool(
                value.get("supplied_campaign_assets_allowed", True)
            ),
            foreign_logos=action,
            minimum_confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class EditorialAcceptancePolicy:
    require_standalone_context: bool = True
    require_resolved_ending: bool = True
    minimum_boundary_confidence: float = 0.75

    @classmethod
    def from_dict(cls, value: object) -> EditorialAcceptancePolicy:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("acceptance_policy.editorial must be an object")
        unknown_fields = set(value) - {
            "require_standalone_context",
            "require_resolved_ending",
            "minimum_boundary_confidence",
        }
        if unknown_fields:
            raise BriefValidationError(
                f"unsupported acceptance_policy.editorial rule: {sorted(unknown_fields)[0]}"
            )
        confidence = float(value.get("minimum_boundary_confidence", 0.75))
        if not 0.0 <= confidence <= 1.0:
            raise BriefValidationError(
                "acceptance_policy.editorial.minimum_boundary_confidence must be between 0 and 1"
            )
        return cls(
            require_standalone_context=bool(value.get("require_standalone_context", True)),
            require_resolved_ending=bool(value.get("require_resolved_ending", True)),
            minimum_boundary_confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    enabled: bool = False
    source_segments: SourceSegmentPolicy = field(default_factory=SourceSegmentPolicy)
    branding: BrandingPolicy = field(default_factory=BrandingPolicy)
    ai_generated_source_video: PolicyAction = "escalate"
    negative_creator_portrayal: PolicyAction = "escalate"
    on_screen_text_language: str | None = None
    editorial: EditorialAcceptancePolicy = field(default_factory=EditorialAcceptancePolicy)

    @classmethod
    def from_dict(cls, value: object) -> AcceptancePolicy:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("acceptance_policy must be an object")
        unknown_fields = set(value) - {
            "enabled",
            "source_segments",
            "branding",
            "generated_media",
            "portrayal",
            "language",
            "editorial",
        }
        if unknown_fields:
            raise BriefValidationError(
                f"unsupported acceptance_policy rule: {sorted(unknown_fields)[0]}"
            )

        generated = value.get("generated_media", {})
        if not isinstance(generated, dict):
            raise BriefValidationError("acceptance_policy.generated_media must be an object")
        generated_unknown = set(generated) - {"ai_generated_source_video"}
        if generated_unknown:
            raise BriefValidationError(
                "unsupported acceptance_policy.generated_media rule: "
                f"{sorted(generated_unknown)[0]}"
            )

        portrayal = value.get("portrayal", {})
        if not isinstance(portrayal, dict):
            raise BriefValidationError("acceptance_policy.portrayal must be an object")
        portrayal_unknown = set(portrayal) - {"negative_creator_portrayal"}
        if portrayal_unknown:
            raise BriefValidationError(
                f"unsupported acceptance_policy.portrayal rule: {sorted(portrayal_unknown)[0]}"
            )

        language = value.get("language", {})
        if not isinstance(language, dict):
            raise BriefValidationError("acceptance_policy.language must be an object")
        language_unknown = set(language) - {"on_screen_text"}
        if language_unknown:
            raise BriefValidationError(
                f"unsupported acceptance_policy.language rule: {sorted(language_unknown)[0]}"
            )
        language_value = str(language.get("on_screen_text") or "").strip().lower() or None
        if language_value is not None and len(language_value) not in {2, 3}:
            raise BriefValidationError(
                "acceptance_policy.language.on_screen_text must be an ISO language code"
            )

        return cls(
            enabled=bool(value.get("enabled", True)),
            source_segments=SourceSegmentPolicy.from_dict(value.get("source_segments")),
            branding=BrandingPolicy.from_dict(value.get("branding")),
            ai_generated_source_video=_policy_action(
                generated.get("ai_generated_source_video", "escalate"),
                "acceptance_policy.generated_media.ai_generated_source_video",
            ),
            negative_creator_portrayal=_policy_action(
                portrayal.get("negative_creator_portrayal", "escalate"),
                "acceptance_policy.portrayal.negative_creator_portrayal",
            ),
            on_screen_text_language=language_value,
            editorial=EditorialAcceptancePolicy.from_dict(value.get("editorial")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_segments": {
                "allow": list(self.source_segments.allow),
                "forbid": list(self.source_segments.forbid),
                "unknown": self.source_segments.unknown,
                "safety_buffer_seconds": self.source_segments.safety_buffer_seconds,
            },
            "branding": asdict(self.branding),
            "generated_media": {
                "ai_generated_source_video": self.ai_generated_source_video,
            },
            "portrayal": {
                "negative_creator_portrayal": self.negative_creator_portrayal,
            },
            "language": {"on_screen_text": self.on_screen_text_language},
            "editorial": asdict(self.editorial),
        }


@dataclass(frozen=True, slots=True)
class CampaignBrief:
    """Runtime campaign policy. Editorial algorithms and output quotas do not belong here."""

    campaign_id: str
    title: str
    objective: str
    allowed_video_ids: list[str] = field(default_factory=list)
    source_channel_ids: list[str] = field(default_factory=list)
    source_media_urls: dict[str, str] = field(default_factory=dict)
    language: str = "en"
    region_code: str = "US"
    min_clip_seconds: float = 20.0
    max_clip_seconds: float = 45.0
    rights_confirmed: bool = False
    attribution_required: bool = True
    watermark_text: str | None = None
    watermark_url: str | None = None
    required_hashtags: list[str] = field(default_factory=list)
    posting_requirements: list[str] = field(default_factory=list)
    acceptance_policy: AcceptancePolicy = field(default_factory=AcceptancePolicy)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignBrief:
        if not isinstance(data, dict):
            raise BriefValidationError("brief root must be an object")
        required = ("campaign_id", "title", "objective")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise BriefValidationError(f"missing required fields: {', '.join(missing)}")
        obsolete = {
            "keywords",
            "negative_keywords",
            "required_phrases",
            "clip_count",
            "source_limit",
            "max_clips_per_source",
            "published_after",
            "production",
            "diversity",
            "hooks",
            "editorial",
        }
        present = sorted(obsolete & set(data))
        if present:
            raise BriefValidationError(
                "obsolete editorial/discovery fields are not accepted by the production brief: "
                + ", ".join(present)
            )
        brief = cls(
            campaign_id=str(data["campaign_id"]).strip(),
            title=str(data["title"]).strip(),
            objective=str(data["objective"]).strip(),
            source_channel_ids=_string_list(data, "source_channel_ids"),
            allowed_video_ids=_string_list(data, "allowed_video_ids"),
            source_media_urls=_string_map(data, "source_media_urls"),
            language=str(data.get("language", "en")).strip() or "en",
            region_code=str(data.get("region_code", "US")).strip().upper() or "US",
            min_clip_seconds=float(data.get("min_clip_seconds", 20.0)),
            max_clip_seconds=float(data.get("max_clip_seconds", 45.0)),
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
            acceptance_policy=AcceptancePolicy.from_dict(data.get("acceptance_policy")),
        )
        brief.validate()
        return brief

    def validate(self) -> None:
        if len(self.region_code) != 2:
            raise BriefValidationError("region_code must be a two-letter ISO country code")
        if self.min_clip_seconds <= 0:
            raise BriefValidationError("min_clip_seconds must be positive")
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
        if not self.allowed_video_ids:
            raise BriefValidationError("production requires at least one explicit target video")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["acceptance_policy"] = self.acceptance_policy.to_dict()
        return data


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
    speaker_id: str | None = None

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
    """Compatibility quality evidence with no predefined editorial dimensions."""

    quality: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 10.0:
            raise ValueError("editorial quality must be between 0 and 10")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("editorial confidence must be between 0 and 1")

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
    """Renderer compatibility object; mode is a free-form model-derived opening rationale."""

    variant_id: str
    concept_id: str
    mode: str
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
    beat_type: str
    strength: float = 0.0
    text: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("editorial beat timestamps are invalid")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("editorial beat strength must be between 0 and 1")
        if not self.beat_type.strip():
            raise ValueError("editorial beat requires a model-derived type/rationale")


@dataclass(frozen=True, slots=True)
class EditPlan:
    plan_id: str
    video_id: str
    concept_id: str
    variant_id: str
    hook_mode: str
    source_spans: tuple[SourceSpan, ...]
    hook_text: str | None
    beats: tuple[EditorialBeat, ...]
    caption_platform: str
    score: float
    transcript_fingerprint: str
    caption_start_source_time: float | None = None
    caption_start_word: str | None = None
    boundary_audit: dict[str, Any] | None = None
    campaign_policy_audit: dict[str, Any] | None = None
    pre_render_eligibility: dict[str, Any] | None = None

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
            reasons=(f"concept={self.concept_id}", f"opening={self.hook_mode}"),
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
    boundary_qc: list[dict[str, Any]] = field(default_factory=list)
    campaign_policy_qc: list[dict[str, Any]] = field(default_factory=list)
    editorial_qc: list[dict[str, Any]] = field(default_factory=list)
    publication_state: str = "TECHNICALLY_INCOMPLETE"
    run_metadata: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
