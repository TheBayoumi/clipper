"""Immutable, content-fingerprinted book pipeline stage boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from clipper.stage_contracts import (
    StageContract,
    content_fingerprint,
    structural_contract_fingerprint,
)

SourceKind = Literal["owner_file", "public_domain", "archive_candidate", "unresolved"]


@dataclass(frozen=True, slots=True)
class BookSource:
    title: str
    author: str
    edition_identity: str | None
    source_uri: str
    source_kind: SourceKind
    declared_by: str
    declared_at: str
    content_sha256: str
    format: str
    authorization_status: str = "unverified"
    authorization_basis: str | None = None

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class BookSection:
    chapter_id: str
    heading: str
    paragraphs: tuple[str, ...]



@dataclass(frozen=True, slots=True)
class SourceBlock:
    anchor: str
    section_id: str
    source_path: str
    kind: str
    text: str
    asset_sha256: str | None = None
    method: str = "source"
    review_status: str = "source_unverified"


@dataclass(frozen=True, slots=True)
class ImageOccurrence:
    anchor: str
    source_path: str
    alt_text: str


@dataclass(frozen=True, slots=True)
class ImageEvidence:
    asset_sha256: str
    media_type: str
    byte_count: int
    occurrences: tuple[ImageOccurrence, ...]
    original_transcription: str | None = None
    normalized_transcription: str | None = None
    normalization_notes: str | None = None
    method: str = "none"
    uncertainty: str = "image words not recovered"
    review_status: str = "needs_review"


@dataclass(frozen=True, slots=True)
class ExcludedPage:
    page_number: int
    reason: str
    evidence: str
    observed_text_sha256: str


@dataclass(frozen=True, slots=True)
class BookText:
    source: BookSource
    edition_identity: str | None
    sections: tuple[BookSection, ...]
    total_word_count: int
    content_sha256: str
    source_fingerprint: str
    blocks: tuple[SourceBlock, ...] = ()
    images: tuple[ImageEvidence, ...] = ()
    excluded_pages: tuple[ExcludedPage, ...] = ()
    warnings: tuple[str, ...] = ()
    recovery_status: str = "legacy_unassessed"
    parser_version: str = "legacy-v1"
    parser_config_sha256: str = ""

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class SourceUnresolved:
    title: str
    author: str
    required_owner_decision: str
    reason: str


@dataclass(frozen=True, slots=True)
class BookRef:
    book_id: str
    title: str
    author: str
    pillar: str
    source_url: str

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class LessonSelection:
    book: BookRef
    lesson_id: str
    angle_id: str
    episode_id: str
    topic: str
    claim: str
    claim_source_url: str
    hook_angle: str
    visual_action: str
    script_version: str
    legacy_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class ScriptScene:
    headline: str
    narration: str


@dataclass(frozen=True, slots=True)
class EpisodeScript:
    episode_id: str
    selection_fingerprint: str
    script_cache_key: str
    title: str
    scenes: tuple[ScriptScene, ...]

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class NarrationTrack:
    episode_id: str
    voice_id: str
    audio_sha256: str
    sample_rate: int

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class ShotPlan:
    episode_id: str
    plan_fingerprint: str
    shot_ids: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class ShotSet:
    episode_id: str
    plan_fingerprint: str
    shot_sha256: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class EpisodeDraft:
    episode_id: str
    audio_sha256: str
    video_sha256: str

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class ReviewDossier:
    episode_id: str
    video_sha256: str
    review_sha256: str
    human_approved: bool

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


BOOK_STAGE_CONTRACTS: dict[str, StageContract] = {
    "ingest": StageContract(
        "books.ingest",
        {
            "structure": structural_contract_fingerprint(
                "books.ingest", BookSource, BookSection, BookText
            )
        },
    ),
    "mining": StageContract(
        "books.mining",
        {
            "schema": "clipper-book-editorial-catalog-v1",
            "structure": structural_contract_fingerprint("books.mining", BookText),
        },
    ),
    "selection": StageContract(
        "books.selection",
        {
            "schema": "clipper-book-selection-v1",
            "structure": structural_contract_fingerprint(
                "books.selection", BookRef, LessonSelection
            ),
        },
    ),
    "scripting": StageContract(
        "books.scripting",
        {
            "schema": "niche-scenes-open-v3",
            "structure": structural_contract_fingerprint(
                "books.scripting", EpisodeScript, ScriptScene
            ),
        },
    ),
    "narration": StageContract(
        "books.narration",
        {"structure": structural_contract_fingerprint("books.narration", NarrationTrack)},
    ),
    "direction": StageContract(
        "books.direction",
        {"structure": structural_contract_fingerprint("books.direction", ShotPlan)},
    ),
    "visuals": StageContract(
        "books.visuals",
        {"structure": structural_contract_fingerprint("books.visuals", ShotSet)},
    ),
    "assembly": StageContract(
        "books.assembly",
        {"structure": structural_contract_fingerprint("books.assembly", EpisodeDraft)},
    ),
    "review": StageContract(
        "books.review",
        {"structure": structural_contract_fingerprint("books.review", ReviewDossier)},
    ),
}
