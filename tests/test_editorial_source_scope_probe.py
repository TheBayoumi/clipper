"""Saved source answers are identity-checked before scope calls are made."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper import editorial_source_scope_probe as probe
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
        expected_supported=True,
        annotation_reason="Bobby is named",
        evidence_first_unit=0,
        evidence_last_unit=0,
        evidence_quote="Bobby paced",
        source_units=UNITS,
    )


def saved_answer():
    return {
        "question": QUESTION,
        "status": "answered",
        "answer_quote": "Bobby",
        "citation": {"first_unit": 0, "last_unit": 0, "text": UNITS[0]},
    }


def inputs(tmp_path: Path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "annotation_status": "provisional",
                "source_sha256": "a" * 64,
                "transcript_sha256": "b" * 64,
            }
        )
    )
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    saved = tmp_path / "source-answer.json"
    saved.write_text(
        json.dumps(
            {
                "experiment": "claim_blind_source_answer_v1",
                "experiment_complete": True,
                "production_approved": False,
                "source_sha256": "a" * 64,
                "transcript_sha256": "b" * 64,
                "cases": [
                    {
                        "case_id": f"case_{i}",
                        "question": QUESTION,
                        "source_units": list(UNITS),
                        "source_answer": saved_answer(),
                    }
                    for i in range(15)
                ],
            }
        )
    )
    return (
        fixture,
        proof,
        tmp_path / "transcript.json",
        tmp_path / "provenance.json",
        saved,
        tmp_path / "scope-report.json",
    )


def test_scope_probe_uses_saved_answers_and_checkpoints_every_case(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    calls = []

    def completion(prompt, payload, schema, tokens):
        calls.append(payload)
        return {"responsiveness": "answers_question", "scope": "actual_event_or_state"}

    assert (
        probe.run_source_scope_probe(
            *args,
            completion=completion,
            scope_model_profile={"model": "injected"},
            request_metrics=lambda: {"model_calls": len(calls)},
        )
        == 1
    )
    report = json.loads(args[-1].read_text())
    assert report["experiment_complete"] is True
    assert report["production_approved"] is False
    assert len(report["cases"]) == len(calls) == 15
    assert report["request_cache_metrics"] == {"model_calls": 15}
    assert all(case["scope_review"]["production_approved"] is False for case in report["cases"])


def test_scope_probe_rejects_incomplete_report(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    saved = json.loads(args[4].read_text())
    saved["cases"].pop()
    args[4].write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="complete pinned"):
        probe.run_source_scope_probe(
            *args, completion=lambda *_: {}, scope_model_profile={}, request_metrics=lambda: {}
        )


@pytest.mark.parametrize("mutation", ["identity", "question", "units"])
def test_scope_probe_rejects_changed_case_before_model_call(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    saved = json.loads(args[4].read_text())
    if mutation == "identity":
        saved["cases"][0]["case_id"] = "changed"
    elif mutation == "question":
        saved["cases"][0]["question"] = "What changed?"
    else:
        saved["cases"][0]["source_units"] = ["Changed."]
    args[4].write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=r"identity|differs"):
        probe.run_source_scope_probe(
            *args, completion=lambda *_: {}, scope_model_profile={}, request_metrics=lambda: {}
        )


def test_scope_probe_records_contract_errors_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    assert (
        probe.run_source_scope_probe(
            *args,
            completion=lambda *_: {},
            scope_model_profile={},
            request_metrics=lambda: {},
        )
        == 1
    )
    report = json.loads(args[-1].read_text())
    assert len(report["cases"]) == 15
    assert all("missing or unknown fields" in case["error"] for case in report["cases"])
