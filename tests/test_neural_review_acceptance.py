"""Review-gold qualification never accepts a checklist or VLM assertion as evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from clipper.cli import main
from clipper_engine import qualification
from clipper_engine.profiles import load_profile

CONFIG = Path("campaigns/campaign-moments-mw4.json")
REVIEW = Path("acceptance/mw4-neural-source-review.json")


def _gold() -> dict[str, Any]:
    seed = json.loads(REVIEW.read_text(encoding="utf-8"))
    seed["independent_review_complete"] = True
    seed["gold_qualification"] = "PASS"
    seed["type"] = "independently_reviewed_source_gold"
    for source in seed["sources"]:
        source["review_status"] = "INDEPENDENTLY_REVIEWED"
        source["certified_proxy_sha256"] = "a" * 64
        source["nominal_frame_rate"] = "60000/1001"
        source["playback_review"] = {
            "reviewer_id": "independent-reviewer",
            "reviewer_role": "human",
            "full_video_and_audio_playback": True,
            "reviewed_at_utc": "2026-10-22T16:00:00Z",
        }
        for cinematic in source["prior_campaign_verified_cinematic_anchors"]:
            cinematic["independent_re_review"] = "VERIFIED"
        source["adjudicated_source_frame_annotations"] = [
            {
                "category": "cinematic_confrontation",
                "adjudication": "PRESENT",
                "source_frame_index": 60,
                "source_time_seconds": 1.001,
                "reviewer_id": "independent-reviewer",
                "source_evidence": "independent direct-frame review test fixture",
            },
            {
                "category": "negative_visual_distractor",
                "adjudication": "ABSENT",
                "source_frame_index": 120,
                "source_time_seconds": 2.002,
                "reviewer_id": "independent-reviewer",
                "source_evidence": "independent negative playback test fixture",
            },
        ]
    return seed


def _qualify(payload: dict[str, Any]) -> dict[str, Any]:
    return qualification.validate_neural_source_review(payload, load_profile("mw4", CONFIG).config)


def test_existing_six_source_seed_is_not_accepted_as_independent_gold() -> None:
    with pytest.raises(AssertionError, match="annotation_seed_is_not_verified_gold"):
        _qualify(json.loads(REVIEW.read_text(encoding="utf-8")))


def test_synthetic_complete_review_contract_accepts_all_six_sources() -> None:
    result = _qualify(_gold())
    assert result["status"] == "PASS"
    assert result["source_count"] == 6
    assert result["adjudications"]["PRESENT:cinematic_confrontation"] == 6
    assert result["adjudications"]["ABSENT:negative_visual_distractor"] == 6


@pytest.mark.parametrize(
    "edit,expected",
    [
        (lambda d: d["sources"].pop(), "incomplete_or_duplicate_six_source_review"),
        (lambda d: d["sources"][0].update(source_title="wrong"), "campaign_source_title_mismatch"),
        (
            lambda d: d["sources"][0]["playback_review"].update(reviewer_role="model"),
            "independent_full_audio_video_playback_evidence_missing",
        ),
        (
            lambda d: d["sources"][0].update(certified_proxy_sha256="short"),
            "certified_proxy_sha256_missing",
        ),
        (
            lambda d: d["sources"][0].update(nominal_frame_rate="0/0"),
            "certified_source_frame_rate_missing",
        ),
        (
            lambda d: d["sources"][0]["adjudicated_source_frame_annotations"][0].update(
                source_frame_index=9000
            ),
            "invalid_source_frame_adjudication",
        ),
        (
            lambda d: d["sources"][0]["adjudicated_source_frame_annotations"][0].update(
                reviewer_id="different"
            ),
            "invalid_source_frame_adjudication",
        ),
        (
            lambda d: d["sources"][1]["prior_campaign_verified_cinematic_anchors"][0].update(
                independent_re_review="PENDING"
            ),
            "previously_verified_cinematic_anchor_not_re_reviewed",
        ),
    ],
)
def test_gold_rejects_missing_human_source_evidence(edit: Any, expected: str) -> None:
    payload = copy.deepcopy(_gold())
    edit(payload)
    with pytest.raises(AssertionError, match=expected):
        _qualify(payload)


def test_public_mw4_review_contract_writes_machine_readable_blockers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "review-result.json"
    code = main(
        [
            "mw4",
            "review-contract",
            "--config",
            str(CONFIG),
            "--review",
            str(REVIEW),
            "--output",
            str(output),
        ]
    )
    assert code == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "REJECTED"
    assert result["source_count"] == 6
    assert all(item["reason"] for item in result["blockers"])
