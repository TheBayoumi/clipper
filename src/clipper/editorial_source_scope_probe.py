"""Pinned source-answer scope evaluation, preserving every reviewed case."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clipper.editorial_benchmark import load_frozen_relations
from clipper.editorial_source_scope import Completion, assess_source_answer_scope

_RESPONSIVENESS = {"answers_question", "does_not_answer", "uncertain"}
_SCOPES = {
    "actual_event_or_state",
    "actual_report_of_opinion",
    "quoted_instruction",
    "hypothetical_or_conditional",
    "negated_actual_state",
    "unknown",
}


def _load_scope_gold(
    path: Path, *, source_answer_sha256: str, case_ids: set[str]
) -> dict[str, dict[str, str]]:
    """Validate sealed evaluation labels before inference; never send them to the model."""
    gold = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(gold, dict)
        or gold.get("schema") != "clipper-source-answer-scope-gold-v1"
        or gold.get("annotation_status") != "transcript_derived_not_independent_audio_gold"
        or gold.get("source_answer_report_sha256") != source_answer_sha256
        or not isinstance(gold.get("cases"), list)
        or len(gold["cases"]) != len(case_ids)
    ):
        raise ValueError("scope gold does not match the pinned source-answer report")
    labels: dict[str, dict[str, str]] = {}
    for item in gold["cases"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"case_id", "responsiveness", "scope"}
            or not isinstance(item["case_id"], str)
            or item["case_id"] in labels
            or not isinstance(item["responsiveness"], str)
            or not isinstance(item["scope"], str)
            or item["responsiveness"] not in _RESPONSIVENESS
            or item["scope"] not in _SCOPES
        ):
            raise ValueError("scope gold has an invalid or duplicate case label")
        labels[item["case_id"]] = item
    if set(labels) != case_ids:
        raise ValueError("scope gold case identities differ from frozen relations")
    return labels


def _score_scope_cases(
    cases: list[dict[str, Any]], labels: dict[str, dict[str, str]]
) -> dict[str, Any]:
    failures = []
    for case in cases:
        expected = labels[case["case_id"]]
        actual = case.get("scope_review", {})
        if (
            actual.get("responsiveness") != expected["responsiveness"]
            or actual.get("scope") != expected["scope"]
        ):
            failures.append(
                {
                    "case_id": case["case_id"],
                    "expected": {
                        "responsiveness": expected["responsiveness"],
                        "scope": expected["scope"],
                    },
                    "actual": {
                        "responsiveness": actual.get("responsiveness"),
                        "scope": actual.get("scope"),
                    },
                }
            )
    return {
        "annotation_status": "transcript_derived_not_independent_audio_gold",
        "exact_pair_matches": len(cases) - len(failures),
        "total": len(cases),
        "failures": failures,
        "qualified_for_production": False,
    }


def run_source_scope_probe(
    fixture_path: Path,
    proof_path: Path,
    transcript_path: Path,
    provenance_path: Path,
    source_answer_report_path: Path,
    output_path: Path,
    *,
    completion: Completion,
    scope_model_profile: dict[str, Any],
    request_metrics: Callable[[], dict[str, Any]],
    scope_gold_path: Path | None = None,
) -> int:
    """Assess saved, exact-source blind answers without repeating their model calls."""
    relations = load_frozen_relations(fixture_path, proof_path, transcript_path, provenance_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    saved = json.loads(source_answer_report_path.read_text(encoding="utf-8"))
    if (
        len(relations) != 15
        or not isinstance(saved, dict)
        or saved.get("experiment") != "claim_blind_source_answer_v1"
        or saved.get("experiment_complete") is not True
        or saved.get("production_approved") is not False
        or saved.get("source_sha256") != fixture["source_sha256"]
        or saved.get("transcript_sha256") != fixture["transcript_sha256"]
        or not isinstance(saved.get("cases"), list)
        or len(saved["cases"]) != len(relations)
    ):
        raise ValueError("source-scope probe needs the complete pinned source-answer report")
    by_id = {
        item["case_id"]: item
        for item in saved["cases"]
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    if len(by_id) != len(relations) or set(by_id) != {row.case_id for row in relations}:
        raise ValueError("source-scope probe case identity differs from frozen relations")
    for relation in relations:
        saved_case = by_id[relation.case_id]
        if (
            saved_case.get("question") != relation.question
            or saved_case.get("source_units") != list(relation.source_units)
            or not isinstance(saved_case.get("source_answer"), dict)
        ):
            raise ValueError("source-scope probe source answer differs from pinned relation")
    source_answer_sha256 = hashlib.sha256(source_answer_report_path.read_bytes()).hexdigest()
    gold = (
        _load_scope_gold(
            scope_gold_path,
            source_answer_sha256=source_answer_sha256,
            case_ids=set(by_id),
        )
        if scope_gold_path is not None
        else None
    )
    report: dict[str, Any] = {
        "experiment": "independent_source_answer_scope_v1",
        "diagnostic_only": True,
        "production_approved": False,
        "experiment_complete": False,
        "annotation_status": fixture["annotation_status"],
        "source_sha256": fixture["source_sha256"],
        "transcript_sha256": fixture["transcript_sha256"],
        "baseline_proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "source_answer_report_sha256": source_answer_sha256,
        "scope_gold_sha256": (
            hashlib.sha256(scope_gold_path.read_bytes()).hexdigest()
            if scope_gold_path is not None
            else None
        ),
        "scope_model_profile": scope_model_profile,
        "cases": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    for relation in relations:
        saved_case = by_id[relation.case_id]
        case: dict[str, Any] = {
            "case_id": relation.case_id,
            "question": relation.question,
            "expected_headline_answer": relation.answer,
            "expected_headline_supported": relation.expected_supported,
            "source_answer": saved_case["source_answer"],
        }
        began = time.monotonic()
        try:
            case["scope_review"] = assess_source_answer_scope(
                question=relation.question,
                source_answer=saved_case["source_answer"],
                delivered_units=relation.source_units,
                completion=completion,
            )
        except (RuntimeError, ValueError) as error:
            case["error"] = f"{type(error).__name__}: {error}"
        case["seconds"] = round(time.monotonic() - began, 3)
        report["cases"].append(case)
        checkpoint()
    report["request_cache_metrics"] = request_metrics()
    if gold is not None:
        report["scope_benchmark"] = _score_scope_cases(report["cases"], gold)
    report["experiment_complete"] = True
    report["qualification_rule"] = (
        "A model scope label cannot certify source entailment. Compare every cited "
        "response against the original speech and independent audio-reviewed held-out "
        "controls before integrating any fully automated factual approval gate."
    )
    checkpoint()
    return 1
