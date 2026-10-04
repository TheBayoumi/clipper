"""Pinned, validation-only headline syntax inventory on frozen and held-out controls."""

from __future__ import annotations

import hashlib
import json
import time
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

from clipper.editorial_benchmark import load_frozen_relations, load_heldout_claims
from clipper.editorial_obligations import build_syntax_obligations
from clipper.editorial_syntax_inventory import inventory_headline_syntax

SPACY_VERSION = "3.8.7"
MODEL_PACKAGE = "en-core-web-sm"
MODEL_VERSION = "3.8.0"
MODEL_WHEEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"


def run_syntax_inventory_probe(
    fixture_path: Path,
    proof_path: Path,
    transcript_path: Path,
    provenance_path: Path,
    heldout_path: Path,
    output_path: Path,
) -> int:
    """Report parsed obligations; never substitute syntax for source entailment."""
    relations = load_frozen_relations(fixture_path, proof_path, transcript_path, provenance_path)
    heldout = load_heldout_claims(heldout_path, transcript_path, provenance_path)
    if len(relations) != 15 or len({row.fixture_index for row in relations}) != 12:
        raise ValueError("syntax inventory requires all frozen relation controls")
    if len(heldout) != 12:
        raise ValueError("syntax inventory requires all provisional held-out headlines")
    if version("spacy") != SPACY_VERSION or version(MODEL_PACKAGE) != MODEL_VERSION:
        raise RuntimeError("syntax inventory parser or model version is not pinned")
    spacy = import_module("spacy")
    parser = spacy.load("en_core_web_sm")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "experiment": "parser_owned_headline_inventory_v1",
        "diagnostic_only": True,
        "production_approved": False,
        "experiment_complete": False,
        "source_sha256": fixture["source_sha256"],
        "transcript_sha256": fixture["transcript_sha256"],
        "baseline_proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "parser": {
            "spacy_version": SPACY_VERSION,
            "model_package": MODEL_PACKAGE,
            "model_version": MODEL_VERSION,
            "model_wheel_sha256_expected": MODEL_WHEEL_SHA256,
            "installed_wheel_hash_independently_verified": False,
            "trained_genre": "written_web_text_not_podcast_transcript",
        },
        "frozen": [],
        "heldout": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    by_index: dict[int, list[Any]] = {}
    for relation in relations:
        by_index.setdefault(relation.fixture_index, []).append(relation)
    checkpoint()
    began = time.monotonic()
    for index, gold in sorted(by_index.items()):
        inventory = inventory_headline_syntax(gold[0].headline, parser)
        obligations = build_syntax_obligations(inventory)
        roles = [role for frame in inventory["frames"] for role in frame["roles"]]
        report["frozen"].append(
            {
                "fixture_index": index,
                "headline": gold[0].headline,
                "inventory": inventory,
                "obligations": obligations,
                "anchor_checks": [
                    {
                        "case_id": row.case_id,
                        "answer": row.answer,
                        "anchor_present": any(
                            row.answer.casefold() in role["text"].casefold() for role in roles
                        ),
                    }
                    for row in gold
                ],
            }
        )
        checkpoint()
    for case in heldout:
        inventory = inventory_headline_syntax(case.headline, parser)
        report["heldout"].append(
            {
                "case_id": case.case_id,
                "headline": case.headline,
                "provisional_expected_supported": case.expected_supported,
                "inventory": inventory,
                "obligations": build_syntax_obligations(inventory),
            }
        )
        checkpoint()
    checks = [check for row in report["frozen"] for check in row["anchor_checks"]]
    report["syntax_anchor_presence"] = {
        "found": sum(check["anchor_present"] for check in checks),
        "total": len(checks),
    }
    report["parse_warning_count"] = sum(
        len(row["inventory"]["parse_warnings"]) for row in (*report["frozen"], *report["heldout"])
    )
    report["syntax_obligation_summary"] = {
        group: {
            "headlines": len(report[group]),
            "ready_for_source_review": sum(
                row["obligations"]["ready_for_source_review"] for row in report[group]
            ),
            "obligations": sum(len(row["obligations"]["obligations"]) for row in report[group]),
        }
        for group in ("frozen", "heldout")
    }
    report["seconds"] = round(time.monotonic() - began, 3)
    report["experiment_complete"] = True
    report["qualification_rule"] = (
        "Syntax anchor presence is not semantic relation recall or source truth. "
        "Any parse warning is unresolved; even a warning-free parse needs independent "
        "source entailment, audio-reviewed held-out qualification and integrated "
        "automatic factual approval before render."
    )
    checkpoint()
    return 1
