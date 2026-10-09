"""The saved complete relation path is identity-bound and scored without approval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clipper import editorial_answer_comparison_probe as probe
from clipper.editorial_benchmark import FrozenRelation

QUESTION = "Who paced?"
UNITS = ("Bobby paced.",)


def relation(index: int) -> FrozenRelation:
    return FrozenRelation(
        case_id=f"case_{index}",
        fixture_index=index % 12,
        headline="Bobby paced at the fighter meeting",
        dimension="actor_action",
        question=QUESTION,
        answer="Bobby",
        expected_supported=index != 0,
        annotation_reason="Pinned test control",
        evidence_first_unit=0,
        evidence_last_unit=0,
        evidence_quote="Bobby paced",
        source_units=UNITS,
    )


def inputs(tmp_path: Path) -> tuple[Path, ...]:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "source_sha256": "a" * 64,
                "transcript_sha256": "b" * 64,
                "annotation_status": "transcript_derived_not_audio_gold",
            }
        ),
        encoding="utf-8",
    )
    proof = tmp_path / "proof.json"
    proof.write_text("{}", encoding="utf-8")
    gold = tmp_path / "scope-gold.json"
    gold.write_text(
        json.dumps(
            {
                "schema": "clipper-source-answer-scope-gold-v1",
                "source_answer_report_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    citation = {"first_unit": 0, "last_unit": 0, "text": UNITS[0]}
    cases = []
    for index in range(15):
        answer = {
            "question": QUESTION,
            "status": "answered",
            "answer_quote": "Bobby",
            "citation": citation,
            "diagnostic_only": True,
            "production_approved": False,
        }
        scope = {
            "question": QUESTION,
            "answer_quote": "Bobby",
            "citation": citation,
            "responsiveness": "answers_question",
            "scope": "actual_event_or_state",
            "narrative_role": "recounted_event",
            "diagnostic_only": True,
            "production_approved": False,
        }
        if index == 1:
            answer.update(status="unknown", answer_quote="", citation=None)
            scope = {
                "question": QUESTION,
                "responsiveness": "uncertain",
                "scope": "unknown",
                "narrative_role": "unknown",
                "diagnostic_only": True,
                "production_approved": False,
            }
        elif index == 2:
            scope["responsiveness"] = "does_not_answer"
        cases.append(
            {
                "case_id": f"case_{index}",
                "question": QUESTION,
                "expected_headline_answer": "Bobby",
                "expected_headline_supported": index != 0,
                "source_answer": answer,
                "scope_review": scope,
            }
        )
    saved = tmp_path / "scope-report.json"
    saved.write_text(
        json.dumps(
            {
                "experiment": "independent_source_answer_scope_v2_narrative",
                "experiment_complete": True,
                "production_approved": False,
                "source_sha256": "a" * 64,
                "transcript_sha256": "b" * 64,
                "baseline_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
                "scope_gold_sha256": hashlib.sha256(gold.read_bytes()).hexdigest(),
                "source_answer_report_sha256": "c" * 64,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return (
        fixture,
        proof,
        tmp_path / "transcript.json",
        tmp_path / "provenance.json",
        saved,
        tmp_path / "comparison-report.json",
        gold,
    )


def run(args: tuple[Path, ...], completion):
    return probe.run_answer_comparison_probe(
        *args[:6],
        completion=completion,
        comparison_model_profile={"model": "contract-test"},
        request_metrics=lambda: {"model_calls": 13},
        scope_gold_path=args[6],
    )


def test_scores_complete_pipeline_and_never_approves(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    calls = []

    def completion(prompt, payload, schema, tokens):
        calls.append(payload)
        assert "proposed_answer" in payload
        assert payload["source_units"][0]["text"] == UNITS[0]
        return {
            "relation": "equivalent",
            "claimed_narrative_role": "recounted_event",
            "source_narrative_role": "recounted_event",
            "event_link": "same_event",
            "event_first_unit": 0,
            "event_last_unit": 0,
        }

    assert run(args, completion) == 1
    report = json.loads(args[5].read_text(encoding="utf-8"))
    assert report["experiment_complete"] is True
    assert report["production_approved"] is False
    assert report["benchmark"]["comparison_calls"] == 13
    assert report["benchmark"]["false_approval_case_ids"] == ["case_0"]
    assert report["benchmark"]["false_rejection_case_ids"] == ["case_1", "case_2"]
    assert report["benchmark"]["qualified_for_production"] is False
    assert len(calls) == 13


@pytest.mark.parametrize("tamper", ["source", "gold", "answer", "citation", "duplicate"])
def test_rejects_tampered_saved_evidence_before_model_call(tmp_path, monkeypatch, tamper):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    report = json.loads(args[4].read_text(encoding="utf-8"))
    if tamper == "source":
        report["source_sha256"] = "f" * 64
    elif tamper == "gold":
        report["scope_gold_sha256"] = "f" * 64
    elif tamper == "answer":
        report["cases"][0]["expected_headline_answer"] = "Sean"
    elif tamper == "citation":
        report["cases"][0]["source_answer"]["citation"]["text"] = "Not in source"
    else:
        report["cases"][1]["case_id"] = "case_0"
    args[4].write_text(json.dumps(report), encoding="utf-8")

    def completion(*_):
        pytest.fail("model called on tampered saved evidence")

    with pytest.raises(ValueError):
        run(args, completion)


def test_workflow_routes_saved_scope_evidence_to_comparison_only():
    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/tjr-weekly-hd.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["editorial_preflight"]
    assert "validate_only" in job["if"]
    download = next(
        step
        for step in job["steps"]
        if step.get("with", {}).get("path") == "reviewer-source-answer"
    )
    assert "answer_comparison" in download["if"]
    assess = next(
        step for step in job["steps"] if step.get("name", "").startswith("Assess reviewer evidence")
    )["run"]
    assert "--answer-comparison-probe" in assess
    assert '--source-scope-report "$SOURCE_SCOPE"' in assess
    assert "--frozen-relation-proof" in assess


def test_v1_source_scope_cannot_qualify_narrative_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    previous = json.loads(args[4].read_text())
    previous["experiment"] = "independent_source_answer_scope_v1"
    args[4].write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="complete pinned source-scope"):
        run(args, lambda *_: pytest.fail("v1 evidence must not reach comparison model"))
