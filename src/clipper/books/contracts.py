"""Immutable, content-fingerprinted book pipeline stage boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from clipper.stage_contracts import (
    StageContract,
    content_fingerprint,
    structural_contract_fingerprint,
)


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
