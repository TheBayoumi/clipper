"""Validation-only batch acceptance rejects hidden source/semantic QA gaps."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from clipper_engine import build_identity, qualification
from clipper_engine.profiles import load_profile


def _fixtures() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_profile("mw4").config
    config["source_profile"] = {
        "analysis_derivative": "proxy",
        "assets": [{"source_key": "ghost_fight"}],
    }
    allocation = {
        "execution_sha": build_identity.current_execution_sha(),
        "source_order": ["ghost_fight"],
        "target_count": 1,
        "selected_count": 1,
        "source_allocations": {
            "ghost_fight": {
                "count": 1,
                "plan_keys": ["verified-plan"],
                "selected_pair_conflict_count": 0,
                "required_anchor_count": 2,
                "qualified_anchor_count": 2,
                "anchor_dispositions": [
                    {
                        "anchor": 51.0,
                        "disposition": "covered_by_selected_clip",
                        "selected_plan_keys": ["verified-plan"],
                    },
                    {
                        "anchor": "cinematic:ghost_fight:rooftop",
                        "disposition": "covered_by_selected_clip",
                        "selected_plan_keys": ["verified-plan"],
                    },
                ],
                "outcome_dispositions": [
                    {
                        "anchor": 51.0,
                        "disposition": "covered_by_selected_clip",
                        "selected_plan_keys": ["verified-plan"],
                    },
                ],
            }
        },
    }
    summary = {
        "source_key": "ghost_fight",
        "execution_sha": build_identity.current_execution_sha(),
        "selected_count": 1,
        "rendered_count": 1,
        "technical_qa_passed": True,
        "ffv1_nut_transport_contract_passed": True,
        "analysis_alignment_verified": True,
        "unplanned_source_cut_count": 0,
        "verified_finishing_move_count": 0,
        "selected_finishing_move_count": 0,
        "failure": None,
    }
    return config, allocation, summary


def _check(
    tmp_path: Path, config: dict[str, Any], alloc: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    (tmp_path / "ghost_fight_pipeline_summary.json").write_text(json.dumps(summary))
    return qualification.validate_batch(tmp_path, config, alloc)


def test_batch_allows_complete_mixed_semantic_outcome_dispositions(tmp_path: Path) -> None:
    config, alloc, summary = _fixtures()
    assert _check(tmp_path, config, alloc, summary)["status"] == "PASS"


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            lambda c, a, s: a["source_allocations"]["ghost_fight"].update(outcome_dispositions=[]),
            "incomplete explicit outcome",
        ),
        (
            lambda c, a, s: a["source_allocations"]["ghost_fight"]["anchor_dispositions"].pop(),
            "required semantic anchor dispositions",
        ),
        (
            lambda c, a, s: a["source_allocations"]["ghost_fight"]["anchor_dispositions"][0].update(
                selected_plan_keys=[]
            ),
            "covered anchor lacks",
        ),
        (
            lambda c, a, s: a["source_allocations"]["ghost_fight"]["anchor_dispositions"][0].update(
                disposition="PENDING"
            ),
            "unresolved semantic-anchor",
        ),
        (lambda c, a, s: s.update(ffv1_nut_transport_contract_passed=False), "FFV1/NUT"),
        (lambda c, a, s: s.update(analysis_alignment_verified=False), "original-to-proxy"),
        (
            lambda c, a, s: c["source_profile"]["assets"].append({"source_key": "missing"}),
            "every configured source",
        ),
    ],
)
def test_batch_rejects_incomplete_semantic_and_fidelity_acceptance(
    tmp_path: Path, change: Any, reason: str
) -> None:
    config, alloc, summary = (copy.deepcopy(x) for x in _fixtures())
    change(config, alloc, summary)
    with pytest.raises(AssertionError, match=reason):
        _check(tmp_path, config, alloc, summary)
