"""Creative shot acceptance is explicit, content-bound and human-gated."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import book_shot_review as creative  # noqa: E402


def _fixture(tmp_path: Path, *, previous: bool = False) -> tuple[Path, Path, list[Path], dict]:
    video, reference = tmp_path / "shot.mp4", tmp_path / "approved.png"
    video.write_bytes(b"generated-candidate-video")
    reference.write_bytes(b"source-character-reference")
    frames = [tmp_path / f"frame-{index}.png" for index in range(3)]
    for index, frame in enumerate(frames):
        frame.write_bytes(f"frame {index}".encode())
    shot = {
        "id": "book_intro",
        "narration_text": "Read the exact original author and title",
        "character_action": "Host holds an Atomic Habits book",
        "setting": "Warm office with consistent lamp",
        "required_on_screen_text": ["ATOMIC HABITS", "JAMES CLEAR"],
    }
    return video, reference, frames, shot


def _provider(
    frames: list[Path],
    *,
    failed: str | None = None,
    confidence: float = 0.99,
    text: list[str] | None = None,
) -> SimpleNamespace:
    model = ModelIdentity("fake-vision", "revision-1", "none", "test")
    raw = {
        name: {
            "status": "FAIL" if name == failed else "PASS",
            "confidence": confidence,
            "evidence": f"checked {name} against narration, prior shot and reference",
            "frame_sha256": creative.file_sha(frames[0]),
            "prior_shot_compared": True,
            "observed_text": ["ATOMIC HABITS", "JAMES CLEAR"] if text is None else text,
        }
        for name in creative.CRITERION_VERSIONS
    }
    result = ProviderResult(
        value={"schema": creative.OBSERVATION_SCHEMA, "criteria": raw},
        model=model,
        usage=InferenceUsage("fixture-vision", "now", 0.1),
    )
    return SimpleNamespace(identity=model, inspect=Mock(return_value=result))


def _review(tmp_path: Path, *, failed: str | None = None) -> tuple[Path, Path, dict, dict]:
    video, ref, frames, shot = _fixture(tmp_path)
    record = creative.review_candidate(
        provider=_provider(frames, failed=failed),
        video=video,
        reference=ref,
        frames=frames,
        shot=shot,
        plan_fingerprint="c" * 64,
    )
    review = tmp_path / "creative-review.json"
    review.write_text(json.dumps(record))
    return review, video, shot, record


def _approve(record: dict) -> dict:
    result = copy.deepcopy(record)
    result["human_approval"] = {
        "status": "APPROVED",
        "reviewer": "manual-qa-fixture",
        "approved_at": "2026-11-21T00:00:00Z",
        "video_sha256": record["video_sha256"],
        "shot_fingerprint": record["shot_fingerprint"],
        "criterion_versions": record["criterion_versions"],
        "observation_sha256": creative.digest(
            {key: value for key, value in record.items() if key != "human_approval"}
        ),
        "criteria": {
            name: {"status": "PASS", "evidence": "reviewed frame, timing and action"}
            for name in creative.CRITERION_VERSIONS
        },
    }
    return result


def _require(path: Path, video: Path, shot: dict) -> None:
    creative.verify_review(
        path,
        video=video,
        reference_sha256=_reference_sha(path),
        shot=shot,
        plan_fingerprint="c" * 64,
        previous_video_sha256=None,
    )


def _reference_sha(path: Path) -> str:
    return creative.file_sha(path.with_name("approved.png"))


@pytest.mark.parametrize(
    "failed",
    [
        "character_identity",
        "setting_continuity",
        "action_to_narration",
        "speech_synchronization",
        "on_screen_titles",
    ],
)
def test_creative_failures_block_even_with_human_approval(tmp_path: Path, failed: str) -> None:
    path, video, shot, record = _review(tmp_path, failed=failed)
    assert record["automated_criteria"][failed]["status"] == "FAIL"
    path.write_text(json.dumps(_approve(record)))
    with pytest.raises(creative.CreativeReviewError, match="failed"):
        _require(path, video, shot)


def test_unreviewed_or_low_confidence_never_passes(tmp_path: Path) -> None:
    path, video, shot, record = _review(tmp_path)
    assert record["automated_criteria"]["speech_synchronization"]["status"] == "PENDING_HUMAN"
    with pytest.raises(creative.CreativeReviewError, match="mandatory human"):
        _require(path, video, shot)
    frames = [tmp_path / f"frame-{index}.png" for index in range(3)]
    rechecked = creative.review_candidate(
        provider=_provider(frames, confidence=0.3),
        video=video,
        reference=tmp_path / "approved.png",
        frames=frames,
        shot=shot,
        plan_fingerprint="c" * 64,
    )
    assert all(x["status"] == "PENDING_HUMAN" for x in rechecked["automated_criteria"].values())


def test_human_approval_binds_video_fingerprint_versions_and_evidence(tmp_path: Path) -> None:
    path, video, shot, record = _review(tmp_path)
    path.write_text(json.dumps(_approve(record)))
    _require(path, video, shot)
    bad = json.loads(path.read_text())
    bad["human_approval"]["criterion_versions"]["on_screen_titles"] = "forged"
    path.write_text(json.dumps(bad))
    with pytest.raises(creative.CreativeReviewError, match="does not bind"):
        _require(path, video, shot)


@pytest.mark.parametrize("tamper", ["video", "fingerprint", "review_hash", "model_revision"])
def test_stale_or_malicious_approval_is_rejected(tmp_path: Path, tamper: str) -> None:
    path, video, shot, record = _review(tmp_path)
    approved = _approve(record)
    path.write_text(json.dumps(approved))
    if tamper == "video":
        video.write_bytes(b"replaced")
    elif tamper == "fingerprint":
        shot["character_action"] = "An unrelated action"
    elif tamper == "model_revision":
        approved["provider"]["model_revision"] = ""
        path.write_text(json.dumps(approved))
    with pytest.raises(creative.CreativeReviewError):
        creative.verify_review(
            path,
            video=video,
            reference_sha256=_reference_sha(path),
            shot=shot,
            plan_fingerprint="c" * 64,
            previous_video_sha256=None,
            review_sha256="0" * 64 if tamper == "review_hash" else None,
        )


def test_required_book_title_is_not_exempt_and_no_text_is_explicit(tmp_path: Path) -> None:
    video, reference, frames, shot = _fixture(tmp_path)
    inspected = creative.review_candidate(
        provider=_provider(frames, text=["ATOMIC HAEITS"]),
        video=video,
        reference=reference,
        frames=frames,
        shot=shot,
        plan_fingerprint="c" * 64,
    )
    assert inspected["automated_criteria"]["on_screen_titles"]["status"] == "FAIL"
    shot["required_on_screen_text"] = []
    no_text = creative.review_candidate(
        provider=_provider(frames, text=[]),
        video=video,
        reference=reference,
        frames=frames,
        shot=shot,
        plan_fingerprint="c" * 64,
    )
    assert no_text["automated_criteria"]["on_screen_titles"]["status"] == "PASS"
    with_text = creative.review_candidate(
        provider=_provider(frames, text=["floating gibberish"]),
        video=video,
        reference=reference,
        frames=frames,
        shot=shot,
        plan_fingerprint="c" * 64,
    )
    assert with_text["automated_criteria"]["on_screen_titles"]["status"] == "FAIL"


@pytest.mark.parametrize(
    "book,expected",
    [
        ({"title": "Atomic Habits", "author": "James Clear"}, ["ATOMIC HABITS", "JAMES CLEAR"]),
        ({"title": "Deep Work", "author": "Cal Newport"}, ["DEEP WORK", "CAL NEWPORT"]),
    ],
)
def test_two_books_receive_only_their_own_declared_intro_text(
    book: dict[str, str], expected: list[str]
) -> None:
    shot = {
        "id": "scene-02",
        "keyframe_name": "book_intro",
        "required_on_screen_text": creative.intro_text_from_book_metadata(book),
    }
    assert creative.expected_text(shot) == expected


@pytest.mark.parametrize("explicit_none", [False, True])
def test_book_intro_without_text_declaration_fails_closed(explicit_none: bool) -> None:
    shot: dict = {"id": "scene-02", "keyframe_name": "book_intro"}
    if explicit_none:
        shot["required_on_screen_text"] = None
    with pytest.raises(creative.CreativeReviewError, match=r"scene-02.*plan must declare"):
        creative.expected_text(shot)


@pytest.mark.parametrize(
    "value",
    ["not a list", [""], ["   "], [None], [5], ["TITLE", " TITLE "], ["one", " one"]],
)
def test_book_intro_rejects_blank_wrong_type_and_duplicate_contracts(value: object) -> None:
    with pytest.raises(creative.CreativeReviewError, match="invalid required on-screen"):
        creative.expected_text(
            {"id": "scene-02", "keyframe_name": "book_intro", "required_on_screen_text": value}
        )


def test_explicit_no_text_intro_and_undeclared_other_shot() -> None:
    assert (
        creative.expected_text(
            {"id": "scene-02", "keyframe_name": "book_intro", "required_on_screen_text": []}
        )
        == []
    )
    assert creative.expected_text({"id": "other", "role": "hook"}) == []


def test_bound_review_rejects_changed_plan_text_even_when_human_approved(tmp_path: Path) -> None:
    path, video, shot, record = _review(tmp_path)
    path.write_text(json.dumps(_approve(record)))
    _require(path, video, shot)
    shot["required_on_screen_text"] = ["OTHER BOOK", "OTHER AUTHOR"]
    with pytest.raises(creative.CreativeReviewError, match="stale or contradictory"):
        _require(path, video, shot)


def test_creative_review_gate_contains_no_hardcoded_book_literature() -> None:
    source = (ROOT / "scripts/book_shot_review.py").read_text(encoding="utf-8")
    assert "ATOMIC HABITS" not in source
    assert "JAMES CLEAR" not in source
    assert "DEEP WORK" not in source
    assert "CAL NEWPORT" not in source
