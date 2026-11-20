from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from clipper.hilp import (
    HilpSimulationError,
    _delayed_anchor,
    _first_source_word,
    _load_object,
    _plan,
    _qc,
    _review_caption_anchor,
    _segment,
    _sha256,
    _word,
    validate_hilp_evidence,
)
from clipper.models import TranscriptSegment, TranscriptWord


def _plan_payload() -> dict[str, object]:
    return {
        "plan_id": "plan",
        "video_id": "video",
        "concept_id": "concept",
        "variant_id": "variant",
        "hook_mode": "direct",
        "source_spans": [{"start": 0.0, "end": 2.0}],
        "hook_text": None,
        "beats": [],
        "caption_platform": "tiktok",
        "score": 0.9,
        "transcript_fingerprint": "fp",
        "caption_start_source_time": None,
        "caption_start_word": None,
    }


def _segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            0.0,
            2.0,
            "one two three",
            (
                TranscriptWord(0.0, 0.2, "one"),
                TranscriptWord(0.9, 1.1, "two"),
                TranscriptWord(1.5, 1.8, "three"),
            ),
        )
    ]


def _valid_hilp_payload() -> dict[str, Any]:
    return {
        "status": "PASS",
        "events": [
            {"decision": "APPROVE"},
            {"decision": "REJECT"},
            {"decision": "REVISE"},
        ],
        "final_shortlist": [
            {"concept_id": "a"},
            {"concept_id": "b"},
            {"concept_id": "c"},
        ],
        "revision_proof": {
            "before_sha256": "before",
            "after_sha256": "after",
            "after_qc": "PASS",
        },
    }


def test_hilp_parsers_and_hashing_are_fail_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(HilpSimulationError, match="expected JSON object"):
        _load_object(invalid)

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    assert _sha256(payload) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    with pytest.raises(HilpSimulationError, match="word is not an object"):
        _word("bad")
    with pytest.raises(HilpSimulationError, match="segment is not an object"):
        _segment("bad")
    with pytest.raises(HilpSimulationError, match="edit plan is not an object"):
        _plan("bad")

    word = _word({"start": 0.0, "end": 0.2, "text": "one"})
    assert word.text == "one"
    segment = _segment(
        {
            "start": 0.0,
            "end": 1.0,
            "text": "one",
            "speaker_id": "speaker",
            "words": [{"start": 0.0, "end": 0.2, "text": "one"}],
        }
    )
    assert segment.speaker_id == "speaker"
    plan = _plan(_plan_payload())
    assert plan.plan_id == "plan"


def test_hilp_caption_anchor_helpers_reject_missing_source_evidence() -> None:
    plan = _plan(_plan_payload())
    segments = _segments()
    first = _first_source_word(plan, segments)
    assert first.text == "one"
    expected, delayed = _delayed_anchor(plan, segments)
    assert expected.text == "one"
    assert delayed == pytest.approx(0.9)

    approve = _review_caption_anchor(expected, plan)
    assert approve["decision"] == "APPROVE"
    revise = _review_caption_anchor(expected, replace(plan, caption_start_source_time=1.0))
    assert revise["decision"] == "REVISE"
    assert revise["issue"] == "caption_start_delayed"

    with pytest.raises(HilpSimulationError, match="no source span"):
        _first_source_word(replace(plan, source_spans=()), segments)
    with pytest.raises(HilpSimulationError, match="no word-level source evidence"):
        _first_source_word(plan, [TranscriptSegment(0.0, 2.0, "text")])

    short_plan = replace(plan, source_spans=(replace(plan.source_spans[0], end=0.5),))
    with pytest.raises(HilpSimulationError, match="too short"):
        _delayed_anchor(short_plan, segments)


def test_hilp_qc_wrapper_rejects_failed_repaired_render(tmp_path: Path) -> None:
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"clip")
    plan = _plan(_plan_payload())

    def failed_qc(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"status": "FAIL", "issues": ["decode"]}

    with pytest.raises(HilpSimulationError, match="failed technical QC"):
        _qc(
            failed_qc,
            output,
            plan,
            watermark_required=True,
            watermark_present=False,
        )

    def passed_qc(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"status": "PASS"}

    assert (
        _qc(
            passed_qc,
            output,
            plan,
            watermark_required=False,
            watermark_present=False,
        )["status"]
        == "PASS"
    )


def test_hilp_evidence_validation_exercises_every_review_branch() -> None:
    valid = _valid_hilp_payload()
    validate_hilp_evidence(valid)

    cases = (
        ({**valid, "status": "FAIL"}, "status is not PASS"),
        ({**valid, "events": None}, "events are missing"),
        ({**valid, "events": [{"decision": "APPROVE"}]}, "all decision branches"),
        ({**valid, "final_shortlist": [{"concept_id": "a"}]}, "shortlist is incomplete"),
        (
            {
                **valid,
                "final_shortlist": [
                    {"concept_id": "a"},
                    {"concept_id": "a"},
                    {"concept_id": "c"},
                ],
            },
            "lost concept diversity",
        ),
        ({**valid, "revision_proof": None}, "revision proof is missing"),
        (
            {
                **valid,
                "revision_proof": {
                    "before_sha256": "same",
                    "after_sha256": "same",
                    "after_qc": "PASS",
                },
            },
            "did not produce a changed render",
        ),
        (
            {
                **valid,
                "revision_proof": {
                    "before_sha256": "before",
                    "after_sha256": "after",
                    "after_qc": "FAIL",
                },
            },
            "did not pass technical QC",
        ),
    )
    for payload, match in cases:
        with pytest.raises(HilpSimulationError, match=match):
            validate_hilp_evidence(payload)
