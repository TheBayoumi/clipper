from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


class BriefValidationError(ValueError):
    """Raised when a campaign brief is incomplete or unsafe."""


def _string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BriefValidationError(f"{key} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


@dataclass(frozen=True, slots=True)
class CampaignBrief:
    campaign_id: str
    title: str
    objective: str
    keywords: list[str]
    source_channel_ids: list[str] = field(default_factory=list)
    allowed_video_ids: list[str] = field(default_factory=list)
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
class TranscriptSegment:
    start: float
    end: float
    text: str

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
class RenderedClip:
    video_id: str
    output_path: str
    start: float
    end: float
    score: float
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineManifest:
    campaign_id: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    discovered_videos: list[dict[str, Any]] = field(default_factory=list)
    planned_clips: list[dict[str, Any]] = field(default_factory=list)
    rendered_clips: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
