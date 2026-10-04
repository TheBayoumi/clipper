"""Evaluate the complete saved source-first relation path, without media or approval.

The source answer and scope decisions are replayed from a pinned hosted report.
Only the final comparison is inferred here, so failed upstream labels remain
visible and cannot be erased by repeating or changing their model requests.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clipper.editorial_answer_comparison import compare_claim_answer
from clipper.editorial_benchmark import load_frozen_relations
from clipper.editorial_source_scope import Completion


def run_answer_comparison_probe(
    fixture_path: Path,
    proof_path: Path,
    transcript_path: Path,
    provenance_path: Path,
    scope_report_path: Path,
    output_path: Path,
    *,
    completion: Completion,
    comparison_model_profile: dict[str, Any],
    request_metrics: Callable[[], dict[str, Any]],
    scope_gold_path: Path,
) -> int:
    """Measure end-to-end relation verdicts, including upstream abstentions."""
    relations = load_frozen_relations(fixture_path, proof_path, transcript_path, provenance_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    saved = json.loads(scope_report_path.read_text(encoding="utf-8"))
    gold = json.loads(scope_gold_path.read_text(encoding="utf-8"))
    proof_sha256 = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    gold_sha256 = hashlib.sha256(scope_gold_path.read_bytes()).hexdigest()
    if (
        len(relations) != 15
        or not isinstance(saved, dict)
        or saved.get("experiment") != "independent_source_answer_scope_v1"
        or saved.get("experiment_complete") is not True
        or saved.get("production_approved") is not False
        or saved.get("source_sha256") != fixture["source_sha256"]
        or saved.get("transcript_sha256") != fixture["transcript_sha256"]
        or saved.get("baseline_proof_sha256") != proof_sha256
        or saved.get("scope_gold_sha256") != gold_sha256
        or not isinstance(gold, dict)
        or gold.get("schema") != "clipper-source-answer-scope-gold-v1"
        or gold.get("source_answer_report_sha256") != saved.get("source_answer_report_sha256")
        or not isinstance(saved.get("cases"), list)
        or len(saved["cases"]) != len(relations)
    ):
        raise ValueError("answer comparison needs the complete pinned source-scope report")
    by_id = {
        item["case_id"]: item
        for item in saved["cases"]
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    if len(by_id) != len(relations) or set(by_id) != {row.case_id for row in relations}:
        raise ValueError("source-scope report has missing or duplicate relation cases")
    for relation in relations:
        saved_case = by_id[relation.case_id]
        answer = saved_case.get("source_answer")
        scope = saved_case.get("scope_review")
        if (
            saved_case.get("question") != relation.question
            or saved_case.get("expected_headline_answer") != relation.answer
            or saved_case.get("expected_headline_supported") is not relation.expected_supported
            or not isinstance(answer, dict)
            or not isinstance(scope, dict)
            or answer.get("question") != relation.question
            or scope.get("question") != relation.question
            or answer.get("diagnostic_only") is not True
            or answer.get("production_approved") is not False
            or scope.get("diagnostic_only") is not True
            or scope.get("production_approved") is not False
            or saved_case.get("error")
        ):
            raise ValueError("source-scope case differs from pinned relation or is incomplete")
        if answer.get("status") == "answered":
            citation = answer.get("citation")
            if not isinstance(citation, dict):
                raise ValueError("source-scope answer lacks a citation")
            first, last = citation.get("first_unit"), citation.get("last_unit")
            if (
                type(first) is not int
                or type(last) is not int
                or not 0 <= first <= last < len(relation.source_units)
                or citation.get("text") != " ".join(relation.source_units[first : last + 1])
                or not isinstance(answer.get("answer_quote"), str)
                or answer["answer_quote"] not in citation["text"]
                or scope.get("citation") != citation
                or scope.get("answer_quote") != answer["answer_quote"]
            ):
                raise ValueError("source-scope answer citation differs from pinned speech")
        elif (
            answer.get("status") not in {"unknown", "uncertain"}
            or answer.get("citation") is not None
            or answer.get("answer_quote") != ""
            or scope.get("responsiveness") != "uncertain"
            or scope.get("scope") != "unknown"
        ):
            raise ValueError("source-scope abstention has an invalid evidence contract")
    report: dict[str, Any] = {
        "experiment": "source_first_atomic_answer_comparison_v1",
        "experiment_complete": False,
        "diagnostic_only": True,
        "production_approved": False,
        "annotation_status": fixture["annotation_status"],
        "source_sha256": fixture["source_sha256"],
        "transcript_sha256": fixture["transcript_sha256"],
        "baseline_proof_sha256": proof_sha256,
        "source_scope_report_sha256": hashlib.sha256(scope_report_path.read_bytes()).hexdigest(),
        "scope_gold_sha256": gold_sha256,
        "comparison_model_profile": comparison_model_profile,
        "cases": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    for relation in relations:
        upstream = by_id[relation.case_id]
        case: dict[str, Any] = {
            "case_id": relation.case_id,
            "question": relation.question,
            "proposed_answer": relation.answer,
            "expected_supported": relation.expected_supported,
            "source_answer": upstream["source_answer"],
            "scope_review": upstream["scope_review"],
            "comparison": None,
            "verdict": "uncertain",
        }
        began = time.monotonic()
        if (
            case["source_answer"].get("status") == "answered"
            and case["scope_review"].get("responsiveness") == "answers_question"
            and case["scope_review"].get("scope") != "unknown"
        ):
            try:
                case["comparison"] = compare_claim_answer(
                    question=relation.question,
                    proposed_answer=relation.answer,
                    source_answer=case["source_answer"],
                    scope_review=case["scope_review"],
                    delivered_units=relation.source_units,
                    completion=completion,
                )
                case["verdict"] = case["comparison"]["relation"]
            except (RuntimeError, ValueError) as error:
                case["error"] = f"{type(error).__name__}: {error}"
        case["seconds"] = round(time.monotonic() - began, 3)
        report["cases"].append(case)
        checkpoint()
    false_approvals = [
        case["case_id"]
        for case in report["cases"]
        if case["verdict"] == "equivalent" and case["expected_supported"] is False
    ]
    false_rejections = [
        case["case_id"]
        for case in report["cases"]
        if case["verdict"] != "equivalent" and case["expected_supported"] is True
    ]
    report["benchmark"] = {
        "total": len(report["cases"]),
        "exact_binary_matches": len(report["cases"]) - len(false_approvals) - len(false_rejections),
        "false_approval_case_ids": false_approvals,
        "false_rejection_case_ids": false_rejections,
        "comparison_calls": sum(case["comparison"] is not None for case in report["cases"]),
        "case_errors": [case["case_id"] for case in report["cases"] if "error" in case],
        "qualified_for_production": False,
    }
    report["request_cache_metrics"] = request_metrics()
    report["experiment_complete"] = True
    report["qualification_rule"] = (
        "This measures frozen transcript-derived relations only. Source-answer and scope "
        "false positives, incomplete headline relation inventory, and unreviewed held-out audio "
        "prevent automatic production factual approval regardless of this binary score."
    )
    checkpoint()
    return 1
