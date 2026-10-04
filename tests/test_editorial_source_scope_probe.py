"""Saved source answers are identity-checked before scope calls are made."""

from __future__ import annotations

import hashlib
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


def gold_for(args, *, wrong_first=False):
    path = args[-1].with_name("scope-gold.json")
    cases = [
        {
            "case_id": f"case_{i}",
            "responsiveness": "answers_question",
            "scope": "actual_event_or_state",
        }
        for i in range(15)
    ]
    if wrong_first:
        cases[0]["responsiveness"] = "does_not_answer"
    path.write_text(
        json.dumps(
            {
                "schema": "clipper-source-answer-scope-gold-v1",
                "annotation_status": "transcript_derived_not_independent_audio_gold",
                "source_answer_report_sha256": hashlib.sha256(args[4].read_bytes()).hexdigest(),
                "cases": cases,
            }
        )
    )
    return path


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
        saved["cases"][-1]["question"] = "What changed?"
    else:
        saved["cases"][-1]["source_units"] = ["Changed."]
    args[4].write_text(json.dumps(saved))
    calls = []
    with pytest.raises(ValueError, match=r"identity|differs"):
        probe.run_source_scope_probe(
            *args,
            completion=lambda *payload: calls.append(payload),
            scope_model_profile={},
            request_metrics=lambda: {},
        )
    assert calls == []


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


def test_scope_probe_scores_sealed_labels_after_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    gold = gold_for(args, wrong_first=True)
    payloads = []

    def completion(prompt, payload, schema, tokens):
        payloads.append(payload)
        assert "expected" not in str(payload).lower()
        return {"responsiveness": "answers_question", "scope": "actual_event_or_state"}

    assert (
        probe.run_source_scope_probe(
            *args,
            completion=completion,
            scope_model_profile={},
            request_metrics=lambda: {},
            scope_gold_path=gold,
        )
        == 1
    )
    report = json.loads(args[-1].read_text())
    score = report["scope_benchmark"]
    assert len(payloads) == 15
    assert score["exact_pair_matches"] == 14
    assert score["answered_exact_pair_matches"] == 14
    assert score["answered_total"] == 15
    assert score["source_answer_abstentions"] == 0
    assert score["false_responsive_case_ids"] == ["case_0"]
    assert score["failures"][0]["case_id"] == "case_0"
    assert score["qualified_for_production"] is False


def test_scope_score_separates_model_judgments_from_source_abstentions(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    saved = json.loads(args[4].read_text())
    saved["cases"][0]["source_answer"] = {
        "question": QUESTION,
        "status": "unknown",
        "answer_quote": "",
        "citation": None,
    }
    args[4].write_text(json.dumps(saved))
    path = gold_for(args)
    gold = json.loads(path.read_text())
    gold["cases"][0].update(responsiveness="uncertain", scope="unknown")
    path.write_text(json.dumps(gold))
    calls = []

    def completion(prompt, payload, schema, tokens):
        calls.append(payload)
        return {"responsiveness": "answers_question", "scope": "actual_event_or_state"}

    assert (
        probe.run_source_scope_probe(
            *args,
            completion=completion,
            scope_model_profile={},
            request_metrics=lambda: {},
            scope_gold_path=path,
        )
        == 1
    )
    score = json.loads(args[-1].read_text())["scope_benchmark"]
    assert len(calls) == 14
    assert score["exact_pair_matches"] == 15
    assert score["answered_exact_pair_matches"] == 14
    assert score["answered_total"] == 14
    assert score["source_answer_abstentions"] == 1


def test_scope_probe_rejects_gold_bound_to_another_report(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    gold = gold_for(args)
    saved = json.loads(args[4].read_text())
    saved["extra"] = "changed bytes"
    args[4].write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="scope gold does not match"):
        probe.run_source_scope_probe(
            *args,
            completion=lambda *_: {},
            scope_model_profile={},
            request_metrics=lambda: {},
            scope_gold_path=gold,
        )


@pytest.mark.parametrize("mutation", ["duplicate", "different_identity", "bad_label"])
def test_scope_probe_rejects_invalid_gold_before_model_call(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: [relation(i) for i in range(15)])
    args = inputs(tmp_path)
    path = gold_for(args)
    gold = json.loads(path.read_text())
    if mutation == "duplicate":
        gold["cases"][-1]["case_id"] = "case_0"
    elif mutation == "different_identity":
        gold["cases"][-1]["case_id"] = "not_a_frozen_case"
    else:
        gold["cases"][-1]["scope"] = ["not_a_label"]
    path.write_text(json.dumps(gold))
    calls = []
    with pytest.raises(ValueError, match=r"scope gold has|scope gold case identities"):
        probe.run_source_scope_probe(
            *args,
            completion=lambda *payload: calls.append(payload),
            scope_model_profile={},
            request_metrics=lambda: {},
            scope_gold_path=path,
        )
    assert calls == []
