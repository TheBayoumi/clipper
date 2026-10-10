"""Diagnostic GPU transport is successful separately from strict semantic qualification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from clipper.editorial_gpu_reporting import (
    narrative_acceptance_issues,
    narrative_execution_issues,
    narrative_summary,
)


def report(*, scope_matches=12, claim_matches=13, semantic_pass=False):
    scope = {
        "total": 15,
        "exact_pair_matches": scope_matches,
        "false_responsive_case_ids": [],
        "failures": [
            {"case_id": "instagram_millions_earned_asserted"},
            {"case_id": "instagram_not_income"},
            {"case_id": "others_listened_to_sean"},
        ]
        if scope_matches != 15
        else [],
    }
    comparison = {
        "total": 15,
        "exact_binary_matches": claim_matches,
        "false_approval_case_ids": [],
        "false_rejection_case_ids": ["instagram_not_income", "others_listened_to_sean"]
        if claim_matches != 15
        else [],
        "case_errors": [],
    }
    return {
        "experiment": "podcast_narrative_gpu_qualification",
        "experiment_complete": True,
        "gpu_execution_succeeded": True,
        "semantic_pass": semantic_pass,
        "production_approved": False,
        "diagnostic_only": True,
        "worker": {
            "experiment_complete": True,
            "semantic_pass": semantic_pass,
            "diagnostic_only": True,
            "production_approved": False,
            "warm_replay_same_verdicts": True,
            "scoring": {
                "cold": {"scope": scope, "comparison": comparison},
                "warm": {
                    "scope": scope,
                    "comparison": comparison,
                    "scope_cache": {"model_calls": 0},
                    "comparison_cache": {"model_calls": 0},
                },
            },
        },
    }


def test_gpu_success_does_not_approve_failed_editorial_qualification():
    evidence = report()
    assert narrative_execution_issues(evidence) == []
    blockers = narrative_acceptance_issues(evidence)
    assert "Source-scope labels do not match all 15 frozen controls" in blockers
    assert "Claim comparison does not match all 15 frozen controls" in blockers
    summary = narrative_summary(evidence, require_semantic=False)
    assert "**GPU execution and evidence:** SUCCESS" in summary
    assert "**Editorial semantic gate:** FAIL" in summary
    assert "12/15" in summary and "13/15" in summary
    assert "`others_listened_to_sean`" in summary
    assert "**Production approval:** BLOCKED" in summary


def test_perfect_frozen_case_passes_semantic_but_never_approves_production():
    evidence = report(scope_matches=15, claim_matches=15, semantic_pass=True)
    assert narrative_execution_issues(evidence) == []
    assert narrative_acceptance_issues(evidence) == []
    assert "**Production approval:** BLOCKED" in narrative_summary(evidence, require_semantic=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(experiment_complete=False),
        lambda value: value.update(gpu_execution_succeeded=False),
        lambda value: value["worker"].update(error="GPU failed"),
        lambda value: value.update(production_approved=True),
        lambda value: value["worker"]["scoring"].pop("warm"),
    ],
)
def test_execution_failure_is_not_misreported_as_completed(mutation):
    evidence = report()
    mutation(evidence)
    assert narrative_execution_issues(evidence)
    assert narrative_acceptance_issues(evidence)
    assert "GPU execution and evidence:** FAILED" in narrative_summary(
        evidence, require_semantic=False
    )


def test_warm_replay_is_required_for_acceptance():
    evidence = report(scope_matches=15, claim_matches=15, semantic_pass=True)
    evidence["worker"]["scoring"]["warm"]["comparison_cache"]["model_calls"] = 1
    assert narrative_execution_issues(evidence) == []
    assert any("new model calls" in x for x in narrative_acceptance_issues(evidence))


def test_diagnostic_runner_cli_returns_zero_but_gate_cli_fails(tmp_path):
    evidence = report()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(evidence))
    summary_path = tmp_path / "summary.md"
    base = [
        sys.executable,
        "-m",
        "clipper.editorial_gpu_reporting",
        "--report",
        str(report_path),
        "--summary",
        str(summary_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    execution = subprocess.run(base, env=environment, capture_output=True, text=True, check=False)
    assert execution.returncode == 0
    acceptance = subprocess.run(
        [*base, "--require-semantic"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert acceptance.returncode == 1
    assert summary_path.read_text().count("## 30B podcast narrative GPU qualification") == 2


def test_narrative_quality_has_separate_github_actions_gate():
    workflow = yaml.safe_load(Path(".github/workflows/tjr-weekly-hd.yml").read_text())
    jobs = workflow["jobs"]
    probe = jobs["editorial_preflight"]
    reporter = next(
        step
        for step in probe["steps"]
        if step.get("name") == "Report 30B GPU execution and semantic status separately"
    )
    assert "--require-semantic" not in reporter["run"]
    gate = jobs["editorial_semantic_gate"]
    assert gate["needs"] == "editorial_preflight"
    assert "inputs.reviewer_model_probe == 'narrative_gpu'" in gate["if"]
    assert "needs.editorial_preflight.result == 'success'" in gate["if"]
    assert "--require-semantic" in gate["steps"][-1]["run"]
    assert "--report reviewer-report/reviewer-preflight.json" in gate["steps"][-1]["run"]
    assert "inputs.source_mode == 'validate_only'" in gate["if"]


def test_missing_report_is_not_counted_as_gpu_success(tmp_path):
    assert narrative_execution_issues({"error": "missing"})
    assert narrative_acceptance_issues(None)
    assert "FAILED" in narrative_summary(None, require_semantic=False)


def test_main_writes_execution_and_quality_summaries(tmp_path, monkeypatch, capsys):
    from clipper.editorial_gpu_reporting import main

    path = tmp_path / "gpu.json"
    path.write_text(json.dumps(report()), encoding="utf-8")
    summary = tmp_path / "summary.md"
    base = ["clipper.editorial_gpu_reporting", "--report", str(path), "--summary", str(summary)]
    monkeypatch.setattr(sys, "argv", base)
    assert main() == 0
    assert "**GPU execution and evidence:** SUCCESS" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*base, "--require-semantic"])
    assert main() == 1
    assert "**Editorial semantic gate:** FAIL" in capsys.readouterr().out
    path.write_text(json.dumps(report(scope_matches=15, claim_matches=15, semantic_pass=True)))
    assert main() == 0
    assert (
        summary.read_text(encoding="utf-8").count("## 30B podcast narrative GPU qualification") == 3
    )


def test_main_missing_or_malformed_report_is_execution_failure(tmp_path, monkeypatch, capsys):
    from clipper.editorial_gpu_reporting import main

    path = tmp_path / "gpu.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", ["cli", "--report", str(path), "--summary", str(summary)])
    assert main() == 1
    assert "FAILED" in capsys.readouterr().out
    path.write_text("{not valid json")
    assert main() == 1
    assert "FAILED" in capsys.readouterr().out


def test_worker_safety_and_incomplete_scores_fail_closed():
    evidence = report()
    del evidence["worker"]
    assert "Missing GPU worker report" in narrative_execution_issues(evidence)
    evidence = report()
    evidence["worker"]["production_approved"] = True
    assert narrative_execution_issues(evidence)
    evidence = report()
    del evidence["worker"]["scoring"]["cold"]["scope"]
    assert "Missing cold scope/comparison diagnostic evidence" in narrative_execution_issues(
        evidence
    )
    evidence = report()
    evidence["worker"]["warm_replay_same_verdicts"] = False
    assert narrative_execution_issues(evidence) == []
    assert narrative_acceptance_issues(evidence)
