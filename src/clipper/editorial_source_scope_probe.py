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
    by_id = {item.get("case_id"): item for item in saved["cases"] if isinstance(item, dict)}
    if len(by_id) != len(relations) or set(by_id) != {row.case_id for row in relations}:
        raise ValueError("source-scope probe case identity differs from frozen relations")
    report: dict[str, Any] = {
        "experiment": "independent_source_answer_scope_v1",
        "diagnostic_only": True,
        "production_approved": False,
        "experiment_complete": False,
        "annotation_status": fixture["annotation_status"],
        "source_sha256": fixture["source_sha256"],
        "transcript_sha256": fixture["transcript_sha256"],
        "baseline_proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "source_answer_report_sha256": hashlib.sha256(
            source_answer_report_path.read_bytes()
        ).hexdigest(),
        "scope_model_profile": scope_model_profile,
        "cases": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    for relation in relations:
        saved_case = by_id[relation.case_id]
        if (
            saved_case.get("question") != relation.question
            or saved_case.get("source_units") != list(relation.source_units)
            or not isinstance(saved_case.get("source_answer"), dict)
        ):
            raise ValueError("source-scope probe source answer differs from pinned relation")
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
    report["experiment_complete"] = True
    report["qualification_rule"] = (
        "A model scope label cannot certify source entailment. Compare every cited "
        "response against the original speech and independent audio-reviewed held-out "
        "controls before integrating any fully automated factual approval gate."
    )
    checkpoint()
    return 1
