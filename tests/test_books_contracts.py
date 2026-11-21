"""Pinned immutable book-domain boundaries use existing stage-contract hashing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict

import pytest

from clipper.books.contracts import (
    BOOK_STAGE_CONTRACTS,
    BookRef,
    BookSection,
    BookSource,
    BookText,
    SourceUnresolved,
    EpisodeDraft,
    EpisodeScript,
    LessonSelection,
    NarrationTrack,
    ReviewDossier,
    ScriptScene,
    ShotPlan,
    ShotSet,
)
from clipper.stage_contracts import content_fingerprint, stage_identity


def test_book_contracts_are_immutable_and_hash_their_serialized_content() -> None:
    book = BookRef(
        "atomic-habits",
        "Atomic Habits",
        "James Clear",
        "self_improvement",
        "https://jamesclear.com/",
    )
    lesson = LessonSelection(
        book,
        "two-minute-rule",
        "primary",
        "episode",
        "Start small",
        "Use simple steps",
        "https://jamesclear.com/rule",
        "Hook",
        "Move a notebook",
        "book-lesson-complete-sentences-v2",
        "f" * 64,
    )
    episode = EpisodeScript(
        "episode",
        lesson.fingerprint,
        "a" * 64,
        "Small starts",
        (ScriptScene("Hook", "An original introduction."),),
    )
    stages = [
        book,
        lesson,
        episode,
        NarrationTrack("episode", "am_adam", "b" * 64, 24_000),
        ShotPlan("episode", "c" * 64, ("shot-1",)),
        ShotSet("episode", "c" * 64, ("d" * 64,)),
        EpisodeDraft("episode", "b" * 64, "e" * 64),
        ReviewDossier("episode", "e" * 64, "f" * 64, False),
    ]
    for item in stages:
        assert item.fingerprint == content_fingerprint(asdict(item))
        assert len(item.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        book.title = "Invented book"  # type: ignore[misc]


def test_every_book_stage_is_registered_with_versioned_identity() -> None:
    assert tuple(BOOK_STAGE_CONTRACTS) == (
        "ingest",
        "mining",
        "selection",
        "scripting",
        "narration",
        "direction",
        "visuals",
        "assembly",
        "review",
    )
    for key, contract in BOOK_STAGE_CONTRACTS.items():
        identity = stage_identity(contract, source_hash="f" * 64)
        assert identity.stage_name == f"books.{key}"
        assert len(identity.cache_key) == 64
        assert contract.contract_hash == BOOK_STAGE_CONTRACTS[key].contract_hash


def test_book_ingest_contracts_are_immutable_and_unverified() -> None:
    source = BookSource(
        "A Book", "A Writer", None, "file:///owner/book.txt", "owner_file",
        "owner", "", "a" * 64, "txt",
    )
    assert source.authorization_status == "unverified"
    section = BookSection("chapter-01", "Beginning", ("One first paragraph.",))
    text = BookText(source, None, (section,), 3, source.content_sha256, source.fingerprint)
    assert text.fingerprint == content_fingerprint(asdict(text))
    unresolved = SourceUnresolved("A Book", "A Writer", "supply source file", "none supplied")
    assert unresolved.required_owner_decision == "supply source file"
    with pytest.raises(FrozenInstanceError):
        source.title = "changed"  # type: ignore[misc]
