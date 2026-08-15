from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_live_run(
    run_dir: str | Path,
    *,
    expected_finalists: int,
    expected_shortlist: int,
    expected_distinct_finalists: int | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _load_object(root / "manifest.json")
    if manifest.get("status") not in {"SUCCESS", "DEGRADED"}:
        raise ValueError(
            f"production status is {manifest.get('status')}: {manifest.get('status_reason')}"
        )
    if manifest.get("errors"):
        raise ValueError(f"pipeline errors remain: {manifest['errors']}")
    if manifest.get("publication_state") not in {
        "READY_FOR_HUMAN_REVIEW",
        "READY_TO_PUBLISH",
    }:
        raise ValueError(
            "production completed without publish-readiness evidence: "
            f"{manifest.get('publication_state') or 'UNKNOWN'}"
        )
    targets = manifest.get("targets") or {}
    actual = manifest.get("actual") or {}
    distinct_target = expected_distinct_finalists or min(expected_shortlist, expected_finalists)
    expected_targets = {
        "rendered_finalists": expected_finalists,
        "submission_shortlist": expected_shortlist,
        "distinct_finalist_concepts": distinct_target,
        "distinct_shortlist_concepts": expected_shortlist,
    }
    if targets != expected_targets:
        raise ValueError(f"unexpected production targets: {targets}")
    if (
        int(actual.get("rendered_finalists") or 0) != expected_finalists
        or int(actual.get("submission_shortlist") or 0) != expected_shortlist
        or int(actual.get("distinct_finalist_concepts") or 0) < distinct_target
        or int(actual.get("distinct_shortlist_concepts") or 0) < expected_shortlist
    ):
        raise ValueError(
            f"production yield does not meet target: target={targets}, actual={actual}"
        )

    rendered = manifest.get("rendered_clips") or []
    shortlist = manifest.get("submission_shortlist") or []
    qc = manifest.get("technical_qc") or []
    boundary_qc = manifest.get("boundary_qc") or []
    campaign_policy_qc = manifest.get("campaign_policy_qc") or []
    editorial_qc = manifest.get("editorial_qc") or []
    if len(rendered) != expected_finalists:
        raise ValueError(f"expected {expected_finalists} rendered finalists, found {len(rendered)}")
    if len(shortlist) != expected_shortlist:
        raise ValueError(f"expected {expected_shortlist} shortlist clips, found {len(shortlist)}")
    if len(qc) != expected_finalists or any(item.get("status") != "PASS" for item in qc):
        raise ValueError("every finalist must have exactly one PASS technical-QC report")

    rendered_plans = {item.get("plan_id") for item in rendered}
    shortlist_plans = {item.get("plan_id") for item in shortlist}
    if None in rendered_plans or None in shortlist_plans or not shortlist_plans <= rendered_plans:
        raise ValueError("shortlist references a plan without an accepted rendered MP4")

    def require_one_pass(
        reports: object,
        *,
        decision_field: str,
        label: str,
    ) -> None:
        if not isinstance(reports, list):
            raise ValueError(f"{label} evidence must be a list")
        matching = [
            item
            for item in reports
            if isinstance(item, dict) and item.get("plan_id") in rendered_plans
        ]
        report_plans = [item.get("plan_id") for item in matching]
        if (
            len(matching) != expected_finalists
            or set(report_plans) != rendered_plans
            or any(item.get(decision_field) != "PASS" for item in matching)
        ):
            raise ValueError(f"every finalist must have exactly one PASS {label} report")

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
        if isinstance(item, dict) and item.get("plan_id") in rendered_plans
    ):
        raise ValueError("boundary-QC is missing final multimodal confirmation")
    if any(
        item.get("multimodal_policy_review_decision") != "PASS"
        for item in campaign_policy_qc
        if isinstance(item, dict) and item.get("plan_id") in rendered_plans
    ):
        raise ValueError("campaign-policy-QC is missing final multimodal confirmation")
    concepts = {item.get("concept_id") for item in rendered if item.get("concept_id")}
    if len(concepts) < distinct_target:
        raise ValueError("finalist batch lacks minimum concept diversity")
    shortlist_concepts = {item.get("concept_id") for item in shortlist if item.get("concept_id")}
    if len(shortlist_concepts) != expected_shortlist:
        raise ValueError("submission shortlist lacks required distinct concepts")

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
    if any(value != expected_finalists for value in counts.values()):
        raise ValueError(f"incomplete finalist artifact inventory: {counts}")
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
        if not isinstance(hook_overlap, (int, float)) or isinstance(hook_overlap, bool):
            raise ValueError(f"hook-overlap evidence is missing: {audit_path.name}")
        if hook_rendered and float(hook_overlap) > 0.0:
            raise ValueError(f"hook and spoken captions overlap: {audit_path.name}")
    for item in qc:
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
    required_funnel = {
        "transcript_segments",
        "story_moments",
        "raw_concepts",
        "selected_concepts",
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
    if int(funnel["render_success"]) != expected_finalists:
        raise ValueError("funnel render_success does not match target")
    if int(funnel["technical_qc_pass"]) != expected_finalists:
        raise ValueError("funnel technical_qc_pass does not match target")
    if int(funnel["editorial_qc_pass"]) != expected_finalists:
        raise ValueError("funnel editorial_qc_pass does not match target")
    if int(funnel["submission_shortlist"]) != expected_shortlist:
        raise ValueError("funnel submission_shortlist does not match target")

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
