"""Apply review-only recovery evidence to a new manifest without changing the original."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_plan(plans: list[object], *, plan_id: object, concept_id: object) -> dict[str, Any]:
    matches = [
        item
        for item in plans
        if isinstance(item, dict)
        and item.get("plan_id") == plan_id
        and item.get("concept_id") == concept_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one edit plan for concept={concept_id} plan={plan_id}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    args = _parser().parse_args()
    run_dir = args.run_dir.resolve()
    recovery_path = args.recovery or run_dir / "visual-review-recovery.json"
    output_path = args.output or run_dir / "manifest.visual-recovered.json"
    original = _read_object(run_dir / "manifest.json")
    recovery = _read_object(recovery_path)
    manifest = deepcopy(original)
    attempts = manifest.get("render_attempts")
    plans = manifest.get("edit_plans")
    videos = manifest.get("discovered_videos")
    results = recovery.get("results")
    if not isinstance(attempts, list):
        raise ValueError("manifest is missing render_attempts")
    if not isinstance(plans, list):
        raise ValueError("manifest is missing edit_plans")
    if not isinstance(videos, list):
        raise ValueError("manifest is missing discovered_videos")
    if not isinstance(results, list):
        raise ValueError("recovery evidence is missing results")

    rendered: list[dict[str, Any]] = []
    editorial_qc: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict) or result.get("status") != "REVIEWED":
            raise RuntimeError("all visual recovery results must be REVIEWED")
        report = result.get("report")
        if not isinstance(report, dict) or report.get("decision") != "PASS":
            raise RuntimeError("all visual recovery reports must pass before application")
        attempt_number = int(result["attempt"])
        attempt = next(
            (
                item
                for item in attempts
                if isinstance(item, dict) and item.get("attempt") == attempt_number
            ),
            None,
        )
        if not isinstance(attempt, dict):
            raise RuntimeError(f"missing render attempt {attempt_number}")
        plan = _matching_plan(
            plans,
            plan_id=attempt.get("plan_id"),
            concept_id=attempt.get("concept_id"),
        )
        clip_path = run_dir / "clips" / str(result["clip"])
        if not clip_path.is_file():
            raise FileNotFoundError(clip_path)
        spans = plan.get("source_spans")
        span = spans[0] if isinstance(spans, list) and spans else None
        if not isinstance(span, dict):
            raise RuntimeError(f"plan {plan.get('plan_id')} has no source span")
        video = next(
            (
                item
                for item in videos
                if isinstance(item, dict) and item.get("video_id") == plan.get("video_id")
            ),
            None,
        )
        if not isinstance(video, dict):
            raise RuntimeError(f"plan {plan.get('plan_id')} has no discovered video")
        review_payload = {
            **report,
            "plan_id": plan.get("plan_id"),
            "models": [result.get("model")],
            "usage": [result.get("usage")],
            "recovered": True,
        }
        editorial_qc.append(review_payload)
        attempt["status"] = "ACCEPTED"
        attempt.pop("error", None)
        attempt["editorial_qc"] = review_payload
        rendered.append(
            {
                "video_id": plan.get("video_id"),
                "output_path": str(clip_path),
                "start": float(span["start"]),
                "end": float(span["end"]),
                "score": float(plan.get("score") or 0.0),
                "source_url": video.get("url"),
                "concept_id": plan.get("concept_id"),
                "plan_id": plan.get("plan_id"),
                "hook_mode": plan.get("hook_mode"),
                "render_sha256": _sha256(clip_path),
            }
        )

    recovery_keys = {(item.get("concept_id"), item.get("plan_id")) for item in rendered}
    for rejection in manifest.get("rejections", []):
        if (
            isinstance(rejection, dict)
            and rejection.get("stage") == "render"
            and (rejection.get("concept_id"), rejection.get("plan_id")) in recovery_keys
        ):
            rejection["resolved_by"] = "clipper-visual-review-recovery-v1"

    funnel = manifest.get("funnel")
    if not isinstance(funnel, dict):
        raise ValueError("manifest funnel must be an object")
    original_funnel = deepcopy(funnel)
    distinct_concepts = {str(item["concept_id"]) for item in rendered}
    funnel.update(
        {
            "render_failures": 0,
            "editorial_qc_pass": len(rendered),
            "editorial_qc_fail": 0,
            "render_success": len(rendered),
            "distinct_finalist_concepts": len(distinct_concepts),
            "submission_shortlist": 1 if rendered else 0,
            "distinct_shortlist_concepts": 1 if rendered else 0,
        }
    )
    manifest["rendered_clips"] = rendered
    manifest["editorial_qc"] = editorial_qc
    manifest["submission_shortlist"] = rendered[:1]
    manifest["actual"] = {
        "rendered_finalists": len(rendered),
        "submission_shortlist": 1 if rendered else 0,
        "distinct_finalist_concepts": len(distinct_concepts),
        "distinct_shortlist_concepts": 1 if rendered else 0,
    }
    targets = manifest.get("targets")
    target_finalists = int(targets.get("rendered_finalists", 0)) if isinstance(targets, dict) else 0
    manifest["status"] = "FAILED"
    manifest["status_reason"] = (
        "render_yield_below_required_target"
        if len(rendered) < target_finalists
        else "submission_shortlist_below_required_target"
    )
    metadata = manifest.setdefault("run_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("manifest run_metadata must be an object")
    metadata["visual_review_recovery"] = {
        "schema_version": "clipper-visual-review-recovery-v1",
        "applied_at": datetime.now(UTC).isoformat(),
        "evidence_path": str(recovery_path),
        "source_manifest": str(run_dir / "manifest.json"),
        "original_funnel": original_funnel,
        "passed_reviews": len(rendered),
        "remaining_finalists": max(0, target_finalists - len(rendered)),
        "remaining_configuration_gate": ("original normalized brief limits one clip per source"),
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps({"status": manifest["status"], "actual": manifest["actual"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
