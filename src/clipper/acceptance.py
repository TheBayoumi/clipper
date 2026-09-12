from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def validate_live_run(
    run_dir: str | Path,
    *,
    expected_finalists: int | None = None,
    expected_shortlist: int | None = None,
    expected_distinct_finalists: int | None = None,
) -> dict[str, Any]:
    """Validate one completed production run from its own evidence-derived yield.

    ``expected_*`` remains accepted only as a migration shim for older Modal callers.
    Those arguments are intentionally ignored: production correctness must never depend
    on a caller-provided clip quota.
    """

    del expected_finalists, expected_shortlist, expected_distinct_finalists

    root = Path(run_dir)
    manifest = _load_object(root / "manifest.json")
    status = manifest.get("status")
    if status not in {"SUCCESS", "DEGRADED"}:
        raise ValueError(f"production status is {status}: {manifest.get('status_reason')}")

    targets = manifest.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("production targets must be an object")
    if set(targets) != {"eligible_quality_moments"}:
        raise ValueError(f"unexpected production targets: {targets}")
    eligible = _non_negative_int(
        targets.get("eligible_quality_moments"), label="eligible_quality_moments"
    )

    actual = manifest.get("actual")
    if not isinstance(actual, dict):
        raise ValueError("production actual yield must be an object")
    actual_eligible = _non_negative_int(
        actual.get("eligible_quality_moments"), label="actual.eligible_quality_moments"
    )
    rendered_count = _non_negative_int(
        actual.get("rendered_finalists"), label="actual.rendered_finalists"
    )
    shortlist_count = _non_negative_int(
        actual.get("submission_shortlist"), label="actual.submission_shortlist"
    )
    distinct_finalist_count = _non_negative_int(
        actual.get("distinct_finalist_concepts"), label="actual.distinct_finalist_concepts"
    )
    distinct_shortlist_count = _non_negative_int(
        actual.get("distinct_shortlist_concepts"), label="actual.distinct_shortlist_concepts"
    )
    if actual_eligible != eligible:
        raise ValueError("actual eligible quality yield does not match evidence-derived targets")
    if rendered_count > eligible:
        raise ValueError("rendered quality yield exceeds eligible quality moments")

    rendered = manifest.get("rendered_clips") or []
    shortlist = manifest.get("submission_shortlist") or []
    qc = manifest.get("technical_qc") or []
    boundary_qc = manifest.get("boundary_qc") or []
    campaign_policy_qc = manifest.get("campaign_policy_qc") or []
    editorial_qc = manifest.get("editorial_qc") or []
    for label, value in (
        ("rendered_clips", rendered),
        ("submission_shortlist", shortlist),
        ("technical_qc", qc),
        ("boundary_qc", boundary_qc),
        ("campaign_policy_qc", campaign_policy_qc),
        ("editorial_qc", editorial_qc),
    ):
        if not isinstance(value, list):
            raise ValueError(f"{label} evidence must be a list")

    if len(rendered) != rendered_count:
        raise ValueError("manifest rendered_clips disagrees with actual dynamic yield")
    if len(shortlist) != shortlist_count:
        raise ValueError("manifest submission_shortlist disagrees with actual dynamic yield")

    rendered_plans = {
        str(item.get("plan_id"))
        for item in rendered
        if isinstance(item, dict) and item.get("plan_id")
    }
    rendered_concepts = {
        str(item.get("concept_id"))
        for item in rendered
        if isinstance(item, dict) and item.get("concept_id")
    }
    shortlist_plans = {
        str(item.get("plan_id"))
        for item in shortlist
        if isinstance(item, dict) and item.get("plan_id")
    }
    shortlist_concepts = {
        str(item.get("concept_id"))
        for item in shortlist
        if isinstance(item, dict) and item.get("concept_id")
    }
    if len(rendered_plans) != rendered_count:
        raise ValueError("every accepted MP4 must have one unique plan_id")
    if len(rendered_concepts) != rendered_count:
        raise ValueError(
            "dynamic yield must contain at most one accepted output per quality concept"
        )
    if shortlist_plans != rendered_plans:
        raise ValueError(
            "review set must contain every accepted rendered quality moment exactly once"
        )
    if shortlist_concepts != rendered_concepts:
        raise ValueError("review set concept coverage does not match accepted quality moments")
    if shortlist_count != rendered_count:
        raise ValueError(
            "all accepted quality moments must be reviewable; fixed shortlists are forbidden"
        )
    if distinct_finalist_count != rendered_count or distinct_shortlist_count != rendered_count:
        raise ValueError("actual distinct-concept counts do not match one-output-per-concept yield")

    run_metadata = manifest.get("run_metadata") or {}
    if not isinstance(run_metadata, dict):
        raise ValueError("run_metadata must be an object")
    quality_yield = run_metadata.get("quality_yield")
    if not isinstance(quality_yield, dict):
        raise ValueError("quality_yield evidence is missing")
    if (
        _non_negative_int(
            quality_yield.get("eligible_quality_moments"),
            label="quality_yield.eligible_quality_moments",
        )
        != eligible
    ):
        raise ValueError("quality_yield eligible count disagrees with targets")
    primary_count = _non_negative_int(
        quality_yield.get("primary_plans"), label="quality_yield.primary_plans"
    )
    _non_negative_int(quality_yield.get("reserve_variants"), label="quality_yield.reserve_variants")
    if primary_count != eligible:
        raise ValueError("every eligible quality moment must have exactly one primary plan")
    rendered_yield = _non_negative_int(
        quality_yield.get("rendered"), label="quality_yield.rendered"
    )
    if rendered_yield != rendered_count:
        raise ValueError("quality_yield rendered count disagrees with accepted MP4s")
    accepted_yield = _non_negative_int(
        quality_yield.get("accepted"), label="quality_yield.accepted"
    )
    if accepted_yield != rendered_count:
        raise ValueError("quality_yield accepted count disagrees with accepted MP4s")
    if (
        _non_negative_int(
            quality_yield.get("unrendered_or_rejected"),
            label="quality_yield.unrendered_or_rejected",
        )
        != eligible - rendered_count
    ):
        raise ValueError("quality_yield attrition ledger is inconsistent")

    publication_state = manifest.get("publication_state")
    errors = manifest.get("errors") or []
    if not isinstance(errors, list):
        raise ValueError("pipeline errors evidence must be a list")

    if eligible == 0:
        if status != "SUCCESS":
            raise ValueError("zero-quality yield must complete successfully, not as a failure")
        if publication_state != "COMPLETED_NO_ELIGIBLE_MOMENTS":
            raise ValueError("zero-quality yield is missing its explicit completion state")
        if rendered_count or shortlist_count:
            raise ValueError("zero-quality yield must not manufacture rendered outputs")
        if errors:
            raise ValueError("zero-quality yield cannot hide pipeline errors")
    else:
        if rendered_count == 0:
            raise ValueError("eligible quality moments exist but none were accepted")
        if publication_state not in {"READY_FOR_HUMAN_REVIEW", "READY_TO_PUBLISH"}:
            raise ValueError(
                "production completed without publish-readiness evidence: "
                f"{publication_state or 'UNKNOWN'}"
            )
        if status == "SUCCESS" and rendered_count != eligible:
            raise ValueError("SUCCESS requires every eligible quality moment to be accepted")
        for item in errors:
            if not isinstance(item, dict) or not item.get("plan_id"):
                raise ValueError(f"unscoped pipeline error remains: {item}")
            if str(item.get("plan_id")) in rendered_plans:
                raise ValueError(f"accepted plan still has a pipeline error: {item}")

    def require_one_pass(
        reports: list[object],
        *,
        decision_field: str,
        label: str,
    ) -> None:
        matching = [
            item
            for item in reports
            if isinstance(item, dict) and str(item.get("plan_id") or "") in rendered_plans
        ]
        report_plans = [str(item.get("plan_id")) for item in matching]
        if (
            len(matching) != rendered_count
            or set(report_plans) != rendered_plans
            or any(item.get(decision_field) != "PASS" for item in matching)
        ):
            raise ValueError(
                f"every accepted quality moment must have exactly one PASS {label} report"
            )

    require_one_pass(qc, decision_field="status", label="technical-QC")
    require_one_pass(boundary_qc, decision_field="decision", label="boundary-QC")
    require_one_pass(
        campaign_policy_qc,
        decision_field="decision",
        label="campaign-policy-QC",
    )
    require_one_pass(editorial_qc, decision_field="decision", label="multimodal-editorial-QC")
    if any(
        item.get("multimodal_editorial_review_decision") != "PASS"
        for item in boundary_qc
        if isinstance(item, dict) and str(item.get("plan_id") or "") in rendered_plans
    ):
        raise ValueError("boundary-QC is missing final multimodal confirmation")
    if any(
        item.get("multimodal_policy_review_decision") != "PASS"
        for item in campaign_policy_qc
        if isinstance(item, dict) and str(item.get("plan_id") or "") in rendered_plans
    ):
        raise ValueError("campaign-policy-QC is missing final multimodal confirmation")

    clips = sorted((root / "clips").glob("*.mp4"))
    captions = sorted((root / "captions").glob("*.ass"))
    audits = sorted((root / "captions").glob("*.caption-audit.json"))
    tracking = sorted((root / "tracking").glob("*.tracking.json"))
    boundary_audits = sorted((root / "boundary").glob("*.boundary-audit.json"))
    policy_audits = sorted((root / "policy").glob("*.policy-audit.json"))
    counts = {
        "clips": len(clips),
        "captions": len(captions),
        "caption_audits": len(audits),
        "tracking": len(tracking),
        "boundary_audits": len(boundary_audits),
        "policy_audits": len(policy_audits),
    }
    if any(value != rendered_count for value in counts.values()):
        raise ValueError(f"incomplete dynamic-yield artifact inventory: {counts}")

    for audit_path in audits:
        audit = _load_object(audit_path)
        if audit.get("alignment") != "PASS":
            raise ValueError(f"first-caption alignment failed: {audit_path.name}")
        narrative_layers = audit.get("simultaneous_narrative_layers_max")
        if not isinstance(narrative_layers, int) or isinstance(narrative_layers, bool):
            raise ValueError(f"caption concurrency evidence is missing: {audit_path.name}")
        if narrative_layers > 1:
            raise ValueError(f"overlapping narrative captions detected: {audit_path.name}")
        hook_rendered = audit.get("hook_overlay_rendered")
        if not isinstance(hook_rendered, bool):
            raise ValueError(f"hook-overlay evidence is missing: {audit_path.name}")
        hook_overlap = audit.get("potential_hook_caption_overlap_seconds")
        if not isinstance(hook_overlap, int | float) or isinstance(hook_overlap, bool):
            raise ValueError(f"hook-overlap evidence is missing: {audit_path.name}")
        if hook_rendered and float(hook_overlap) > 0.0:
            raise ValueError(f"hook and spoken captions overlap: {audit_path.name}")

    for item in qc:
        if not isinstance(item, dict) or str(item.get("plan_id") or "") not in rendered_plans:
            continue
        caption_qc = item.get("captions") or {}
        framing_qc = item.get("framing") or {}
        watermark_qc = item.get("watermark") or {}
        if caption_qc.get("alignment") != "PASS":
            raise ValueError(f"caption alignment QC failed for {item.get('plan_id')}")
        if caption_qc.get("simultaneous_narrative_layers_max") != 1:
            raise ValueError(f"caption concurrency QC failed for {item.get('plan_id')}")
        if not isinstance(caption_qc.get("hook_overlay_rendered"), bool):
            raise ValueError(f"hook-overlay QC evidence is missing for {item.get('plan_id')}")
        if not framing_qc.get("transition_qc_pass"):
            raise ValueError(f"transition QC failed for {item.get('plan_id')}")
        if watermark_qc.get("required") and not watermark_qc.get("renderer_asset_present"):
            raise ValueError(f"required watermark missing for {item.get('plan_id')}")

    funnel = manifest.get("funnel") or {}
    if not isinstance(funnel, dict):
        raise ValueError("funnel ledger must be an object")
    required_funnel = {
        "transcript_segments",
        "story_moments",
        "raw_concepts",
        "selected_concepts",
        "quality_moments",
        "hook_variants",
        "edit_plans",
        "render_plans",
        "render_attempts",
        "technical_qc_pass",
        "boundary_reject_count",
        "boundary_repair_count",
        "policy_reject_count",
        "hazard_reject_count",
        "editorial_qc_pass",
        "editorial_review_reject_count",
        "reserve_promotions",
        "render_success",
        "submission_shortlist",
    }
    missing = sorted(required_funnel - set(funnel))
    if missing:
        raise ValueError(f"funnel ledger is missing fields: {missing}")
    quality_moments = _non_negative_int(funnel["quality_moments"], label="funnel.quality_moments")
    if quality_moments != eligible:
        raise ValueError("funnel quality_moments does not match evidence-derived yield")
    yield_fields = (
        "render_success",
        "technical_qc_pass",
        "editorial_qc_pass",
        "submission_shortlist",
    )
    for field in yield_fields:
        if _non_negative_int(funnel[field], label=f"funnel.{field}") != rendered_count:
            raise ValueError(f"funnel {field} does not match accepted dynamic yield")

    for required in (
        "funnel.json",
        "rejections.json",
        "coverage.json",
        "transcript.json",
        "editorial-review.json",
    ):
        if not (root / required).is_file():
            raise ValueError(f"required live acceptance artifact is missing: {required}")
    return manifest
