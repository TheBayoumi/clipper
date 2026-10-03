"""The relation probe is diagnostic even with complete, pinned controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper import cli
from clipper import editorial_relation_probe as probe
from clipper.editorial_benchmark import FrozenRelation


class AlwaysSupport:
    def score(self, document: str, claim: str) -> tuple[float, int]:
        assert document == "Speaker talks."
        assert claim
        return 0.9, 20


def _inputs(tmp_path, monkeypatch):
    relations = [
        FrozenRelation(
            case_id=f"relation_{number}",
            fixture_index=number % 12,
            headline="Speaker talks",
            dimension="actor_action",
            question="Who talks?",
            answer="Speaker",
            expected_supported=number % 2 == 0,
            annotation_reason="Provisional source reading.",
            evidence_first_unit=0,
            evidence_last_unit=0,
            evidence_quote="Speaker talks.",
            source_units=("Speaker talks.",),
        )
        for number in range(15)
    ]
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *args: relations)
    proof = {
        "model_profile": {"source_sha256": probe.SOURCE_SHA256},
        "transcript_sha256": "b" * 64,
        "annotated_fixtures": [
            {"headline": "Speaker talks", "expected_supported": index % 2 == 0}
            for index in range(12)
        ],
    }
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    fixture_path = tmp_path / "relations.json"
    fixture_path.write_text("{}", encoding="utf-8")
    return fixture_path, proof_path, tmp_path / "transcript.json", tmp_path / "cache.json"


def test_relation_probe_records_raw_scores_but_never_approves_production(tmp_path, monkeypatch):
    fixture, proof, transcript, provenance = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "report.json"
    assert (
        probe.run_relation_probe(
            fixture, proof, transcript, provenance, output, scorer=AlwaysSupport()
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["experiment_complete"] is True
    assert report["production_approved"] is False
    assert report["semantic_pass"] is False
    assert report["scores"]["relations"] == {
        "correct": 8,
        "total": 15,
        "false_approvals": 7,
        "false_rejections": 0,
    }
    assert report["scores"]["whole_headlines"]["total"] == 12
    assert report["relation_rows"][0]["claim"] == "Who talks? Speaker"
    assert report["model"]["runtime"] == "injected_test_scorer"


def test_relation_probe_rejects_invalid_scorer_contract(tmp_path, monkeypatch):
    fixture, proof, transcript, provenance = _inputs(tmp_path, monkeypatch)

    class InvalidScorer:
        def score(self, document: str, claim: str) -> tuple[float, int]:
            return 1.5, 20

    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="invalid or truncated"):
        probe.run_relation_probe(
            fixture, proof, transcript, provenance, output, scorer=InvalidScorer()
        )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["experiment_complete"] is False
    assert report["relation_rows"] == []


def test_cli_routes_relation_benchmark_to_clipper_package(tmp_path, monkeypatch):
    observed = {}

    def run(*args):
        observed["paths"] = args
        return 1

    monkeypatch.setattr(probe, "run_relation_probe", run)
    paths = [tmp_path / name for name in ("fixture", "proof", "transcript", "cache", "output")]
    assert (
        cli.main(
            [
                "relation-benchmark",
                "--fixture",
                str(paths[0]),
                "--proof",
                str(paths[1]),
                "--transcript",
                str(paths[2]),
                "--provenance",
                str(paths[3]),
                "--output",
                str(paths[4]),
            ]
        )
        == 1
    )
    assert observed["paths"] == tuple(paths)


def test_workflow_relation_mode_uses_pinned_cpu_model_and_saved_evidence():
    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/tjr-weekly-hd.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["editorial_preflight"]
    assert "validate_only" in job["if"]
    steps = job["steps"]
    install = next(
        step for step in steps if step.get("name") == "Install pinned CPU relation verifier"
    )
    assert install["if"] == "inputs.reviewer_model_probe == 'relation_minicheck'"
    assert '"torch==2.8.0"' in install["run"]
    assert '"transformers==4.55.4"' in install["run"]
    assert '"huggingface-hub==0.34.4"' in install["run"]
    assess = next(
        step for step in steps if step.get("name", "").startswith("Assess reviewer evidence")
    )["run"]
    assert "baseline_4b/cold/proof.json" in assess
    assert "--fixture tests/fixtures/issue8_frozen_relations.json" in assess
    assert '--provenance "$PROVENANCE"' in assess
    assert "--output reviewer-preflight.json" in assess
