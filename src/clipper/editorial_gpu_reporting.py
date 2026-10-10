"""Separate GPU execution integrity from editorial semantic acceptance.

A diagnostic may execute successfully while failing to qualify. This distinction
never authorizes production; qualification is enforced in a separate CI job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def narrative_execution_issues(report: object) -> list[str]:
    if not isinstance(report, dict):
        return ["Missing or invalid narrative diagnostic report"]
    issues = []
    if report.get("experiment") != "podcast_narrative_gpu_qualification":
        issues.append("Unexpected GPU qualification experiment identity")
    if report.get("experiment_complete") is not True:
        issues.append("GPU qualification experiment incomplete")
    if report.get("gpu_execution_succeeded") is not True:
        issues.append("GPU execution or evidence collection incomplete")
    if report.get("diagnostic_only") is not True or report.get("production_approved") is not False:
        issues.append("Diagnostic-only production safety contract violated")
    if report.get("error"):
        issues.append("GPU orchestrator error recorded")
    worker = report.get("worker")
    if not isinstance(worker, dict):
        issues.append("Missing GPU worker report")
    else:
        if worker.get("experiment_complete") is not True or worker.get("error"):
            issues.append("GPU worker experiment incomplete or errored")
        if (
            worker.get("diagnostic_only") is not True
            or worker.get("production_approved") is not False
        ):
            issues.append("Worker production safety contract violated")
        scores = worker.get("scoring")
        if not isinstance(scores, dict) or not all(
            isinstance(scores.get(mode), dict) for mode in ("cold", "warm")
        ):
            issues.append("Missing cold/warm model evidence scores")
        else:
            for mode in ("cold", "warm"):
                if not isinstance(scores[mode].get("scope"), dict) or not isinstance(
                    scores[mode].get("comparison"), dict
                ):
                    issues.append(f"Missing {mode} scope/comparison diagnostic evidence")
    return issues


def narrative_acceptance_issues(report: object) -> list[str]:
    issues = narrative_execution_issues(report)
    if issues:
        return issues
    if not isinstance(report, dict):
        return ["Missing or invalid report"]
    worker = report.get("worker")
    if not isinstance(worker, dict):
        return ["Missing or invalid worker"]
    cold = worker["scoring"]["cold"]
    scope = cold["scope"]
    comparison = cold["comparison"]
    if report.get("semantic_pass") is not True or worker.get("semantic_pass") is not True:
        issues.append("Semantic qualification result is not passing")
    if scope.get("exact_pair_matches") != 15 or scope.get("total") != 15:
        issues.append("Source-scope labels do not match all 15 frozen controls")
    if comparison.get("exact_binary_matches") != 15 or comparison.get("total") != 15:
        issues.append("Claim comparison does not match all 15 frozen controls")
    if scope.get("false_responsive_case_ids"):
        issues.append("Source-scope reviewer falsely treated nonresponsive evidence as responsive")
    if comparison.get("false_approval_case_ids"):
        issues.append("False factual approvals remain")
    if comparison.get("case_errors"):
        issues.append("Comparison contract errors remain")
    if worker.get("warm_replay_same_verdicts") is not True:
        issues.append("Warm replay did not reproduce cold verdicts without new inference")
    warm = worker["scoring"]["warm"]
    for key in ("scope_cache", "comparison_cache"):
        metric = warm.get(key)
        if not isinstance(metric, dict) or metric.get("model_calls") != 0:
            issues.append(f"Warm replay {key} made new model calls or lacks evidence")
    return issues


def _case_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _display_ids(values: list[str]) -> str:
    return ", ".join(f"`{case}`" for case in values) if values else "None"


def narrative_summary(report: object, *, require_semantic: bool) -> str:
    execution_issues = narrative_execution_issues(report)
    acceptance_issues = narrative_acceptance_issues(report)
    lines = [
        "## 30B podcast narrative GPU qualification",
        "",
        f"**GPU execution and evidence:** {'SUCCESS' if not execution_issues else 'FAILED'}",
        f"**Editorial semantic gate:** {'PASS' if not acceptance_issues else 'FAIL'}",
        "**Production approval:** BLOCKED — diagnostic-only experiment",
        "",
    ]
    if isinstance(report, dict) and isinstance(report.get("worker"), dict):
        worker = report["worker"]
        scores = worker.get("scoring")
        if isinstance(scores, dict) and isinstance(scores.get("cold"), dict):
            cold = scores["cold"]
            scope = cold.get("scope")
            comparison = cold.get("comparison")
            if isinstance(scope, dict) and isinstance(comparison, dict):
                scope_errors = _case_ids(scope.get("false_responsive_case_ids"))
                mismatches = scope.get("failures", [])
                mismatch_ids = (
                    _case_ids([x.get("case_id") for x in mismatches if isinstance(x, dict)])
                    if isinstance(mismatches, list)
                    else []
                )
                false_approvals = _case_ids(comparison.get("false_approval_case_ids"))
                false_rejections = _case_ids(comparison.get("false_rejection_case_ids"))
                case_errors = _case_ids(comparison.get("case_errors"))
                lines.extend(
                    [
                        "| Quality metric | Result |",
                        "| --- | --- |",
                        "| Source-scope exact pairs | "
                        f"{scope.get('exact_pair_matches', '?')}/{scope.get('total', '?')} |",
                        "| Final claim comparisons | "
                        f"{comparison.get('exact_binary_matches', '?')}/"
                        f"{comparison.get('total', '?')} |",
                        f"| False factual approvals | {len(false_approvals)} |",
                        f"| False factual rejections | {len(false_rejections)} |",
                        "| Warm-cache replay identical | "
                        f"{worker.get('warm_replay_same_verdicts') is True} |",
                        "",
                        f"**Source-scope mismatches:** {_display_ids(mismatch_ids)}",
                        f"**False-responsive cases:** {_display_ids(scope_errors)}",
                        f"**False approvals:** {_display_ids(false_approvals)}",
                        f"**False rejections:** {_display_ids(false_rejections)}",
                        f"**Comparator errors:** {_display_ids(case_errors)}",
                        "",
                    ]
                )
    if execution_issues:
        lines.extend(
            ["**Execution integrity issues:**", *[f"- {issue}" for issue in execution_issues], ""]
        )
    if acceptance_issues:
        lines.extend(
            ["**Editorial gate blockers:**", *[f"- {issue}" for issue in acceptance_issues], ""]
        )
    if not require_semantic:
        lines.append(
            "This job verifies execution and complete evidence only. "
            "The separate editorial gate enforces semantic acceptance."
        )
    else:
        lines.append(
            "This semantic gate must fail closed until every frozen control passes; "
            "a passing diagnostic never approves publication."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--require-semantic", action="store_true")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report = {"error": f"Cannot read qualification report: {type(exc).__name__}: {exc}"}
    rendered = narrative_summary(report, require_semantic=args.require_semantic)
    with args.summary.open("a", encoding="utf-8") as sink:
        sink.write(rendered)
    print(rendered)
    issues = (
        narrative_acceptance_issues(report)
        if args.require_semantic
        else narrative_execution_issues(report)
    )
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
