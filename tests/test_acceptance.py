import json
from pathlib import Path

import pytest

from clipper.acceptance import validate_live_run


def _write_live_run(
    root: Path,
    *,
    eligible: int,
    accepted: int | None = None,
    status: str | None = None,
) -> None:
    accepted = eligible if accepted is None else accepted
    if accepted > eligible:
        raise ValueError("accepted cannot exceed eligible")
    for directory in ("clips", "captions", "tracking", "boundary", "policy"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    rendered = []
    qc = []
    boundary_qc = []
    campaign_policy_qc = []
    editorial_qc = []
    for index in range(accepted):
        plan_id = f"plan-{index}"
        concept_id = f"concept-{index}"
        clip = root / "clips" / f"{index:02d}.mp4"
        clip.write_bytes(b"mp4")
        (root / "captions" / f"{index:02d}.ass").write_text("captions")
        (root / "captions" / f"{index:02d}.caption-audit.json").write_text(
            json.dumps(
                {
                    "alignment": "PASS",
                    "hook_overlay_rendered": False,
                    "potential_hook_caption_overlap_seconds": 0.0,
                    "simultaneous_narrative_layers_max": 1,
                }
            )
        )
        (root / "tracking" / f"{index:02d}.tracking.json").write_text("{}")
        (root / "boundary" / f"{index:02d}.boundary-audit.json").write_text(
            json.dumps({"plan_id": plan_id, "decision": "PASS"})
        )
        (root / "policy" / f"{index:02d}.policy-audit.json").write_text(
            json.dumps({"plan_id": plan_id, "decision": "PASS"})
        )
        rendered.append({"plan_id": plan_id, "concept_id": concept_id, "output_path": str(clip)})
        qc.append(
            {
                "plan_id": plan_id,
                "status": "PASS",
                "captions": {
                    "alignment": "PASS",
                    "hook_overlay_rendered": False,
                    "simultaneous_narrative_layers_max": 1,
                },
                "framing": {"transition_qc_pass": True},
                "watermark": {"required": True, "renderer_asset_present": True},
            }
        )
        boundary_qc.append(
            {
                "plan_id": plan_id,
                "decision": "PASS",
                "multimodal_editorial_review_decision": "PASS",
            }
        )
        campaign_policy_qc.append(
            {
                "plan_id": plan_id,
                "decision": "PASS",
                "multimodal_policy_review_decision": "PASS",
            }
        )
        editorial_qc.append({"plan_id": plan_id, "decision": "PASS"})

    if status is None:
        status = "SUCCESS" if accepted == eligible else "DEGRADED"
    publication_state = (
        "COMPLETED_NO_ELIGIBLE_MOMENTS" if eligible == 0 else "READY_FOR_HUMAN_REVIEW"
    )
    errors = []
    if 0 < accepted < eligible:
        errors = [
            {
                "plan_id": f"failed-plan-{index}",
                "error": "candidate render failed",
            }
            for index in range(eligible - accepted)
        ]

    manifest = {
        "status": status,
        "status_reason": None,
        "publication_state": publication_state,
        "errors": errors,
        "targets": {"eligible_quality_moments": eligible},
        "actual": {
            "eligible_quality_moments": eligible,
            "rendered_finalists": accepted,
            "submission_shortlist": accepted,
            "distinct_finalist_concepts": accepted,
            "distinct_shortlist_concepts": accepted,
        },
        "run_metadata": {
            "quality_yield": {
                "eligible_quality_moments": eligible,
                "primary_plans": eligible,
                "reserve_variants": max(0, eligible - 1),
                "rendered": accepted,
                "accepted": accepted,
                "unrendered_or_rejected": eligible - accepted,
            }
        },
        "rendered_clips": rendered,
        "submission_shortlist": list(rendered),
        "technical_qc": qc,
        "boundary_qc": boundary_qc,
        "campaign_policy_qc": campaign_policy_qc,
        "editorial_qc": editorial_qc,
        "funnel": {
            "transcript_segments": 100,
            "story_moments": eligible * 2,
            "raw_concepts": eligible,
            "selected_concepts": eligible,
            "quality_moments": eligible,
            "hook_variants": eligible * 2,
            "edit_plans": eligible * 2,
            "render_plans": eligible,
            "render_attempts": eligible,
            "technical_qc_pass": accepted,
            "boundary_reject_count": 0,
            "boundary_repair_count": 0,
            "policy_reject_count": 0,
            "hazard_reject_count": 0,
            "editorial_qc_pass": accepted,
            "editorial_review_reject_count": 0,
            "reserve_promotions": 0,
            "render_success": accepted,
            "submission_shortlist": accepted,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for required in (
        "funnel.json",
        "rejections.json",
        "coverage.json",
        "transcript.json",
        "editorial-review.json",
    ):
        (root / required).write_text("{}")


def _manifest(root: Path) -> dict[str, object]:
    value = json.loads((root / "manifest.json").read_text())
    assert isinstance(value, dict)
    return value


def _save_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("eligible", [0, 1, 2, 7])
def test_live_validator_accepts_evidence_derived_zero_to_many_yield(
    tmp_path: Path, eligible: int
) -> None:
    _write_live_run(tmp_path, eligible=eligible)
    manifest = validate_live_run(tmp_path)
    assert manifest["actual"]["rendered_finalists"] == eligible


def test_legacy_expected_count_arguments_are_inert_migration_shims(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=2)
    manifest = validate_live_run(
        tmp_path,
        expected_finalists=999,
        expected_shortlist=999,
        expected_distinct_finalists=999,
    )
    assert manifest["targets"] == {"eligible_quality_moments": 2}


def test_partial_quality_yield_can_be_degraded_and_reviewable(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=3, accepted=2)
    manifest = validate_live_run(tmp_path)
    assert manifest["status"] == "DEGRADED"
    assert manifest["run_metadata"]["quality_yield"]["unrendered_or_rejected"] == 1


def test_success_cannot_hide_partial_quality_yield(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=3, accepted=2, status="SUCCESS")
    with pytest.raises(ValueError, match="SUCCESS requires"):
        validate_live_run(tmp_path)


def test_zero_quality_yield_cannot_manufacture_output_or_hide_errors(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=0)
    manifest = _manifest(tmp_path)
    manifest["actual"]["rendered_finalists"] = 1
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="exceeds eligible"):
        validate_live_run(tmp_path)

    _write_live_run(tmp_path, eligible=0)
    manifest = _manifest(tmp_path)
    manifest["errors"] = [{"error": "model failed"}]
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="cannot hide pipeline errors"):
        validate_live_run(tmp_path)


def test_dynamic_targets_reject_legacy_quota_shape(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=2)
    manifest = _manifest(tmp_path)
    manifest["targets"] = {"rendered_finalists": 6, "submission_shortlist": 3}
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="unexpected production targets"):
        validate_live_run(tmp_path)


def test_one_output_per_quality_concept_is_enforced(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=2)
    manifest = _manifest(tmp_path)
    manifest["rendered_clips"][1]["concept_id"] = "concept-0"
    manifest["submission_shortlist"][1]["concept_id"] = "concept-0"
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="one accepted output per quality concept"):
        validate_live_run(tmp_path)


def test_all_accepted_quality_moments_must_be_in_review_set(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=2)
    manifest = _manifest(tmp_path)
    manifest["submission_shortlist"] = manifest["submission_shortlist"][:1]
    manifest["actual"]["submission_shortlist"] = 1
    manifest["actual"]["distinct_shortlist_concepts"] = 1
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="review set"):
        validate_live_run(tmp_path)


def test_quality_yield_ledger_must_reconcile(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=3, accepted=2)
    manifest = _manifest(tmp_path)
    manifest["run_metadata"]["quality_yield"]["unrendered_or_rejected"] = 0
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="attrition ledger"):
        validate_live_run(tmp_path)

    _write_live_run(tmp_path, eligible=2)
    manifest = _manifest(tmp_path)
    manifest["run_metadata"]["quality_yield"]["primary_plans"] = 1
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="primary plan"):
        validate_live_run(tmp_path)


def test_accepted_plan_cannot_retain_pipeline_error(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=1)
    manifest = _manifest(tmp_path)
    manifest["errors"] = [{"plan_id": "plan-0", "error": "render failed"}]
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="accepted plan"):
        validate_live_run(tmp_path)


def test_unscoped_pipeline_error_is_not_publish_ready(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=1)
    manifest = _manifest(tmp_path)
    manifest["errors"] = [{"video_id": "v", "error": "source failed"}]
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="unscoped pipeline error"):
        validate_live_run(tmp_path)


def test_qc_reports_must_exactly_cover_dynamic_rendered_set(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=2)
    manifest = _manifest(tmp_path)
    manifest["technical_qc"] = manifest["technical_qc"][:1]
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="technical-QC"):
        validate_live_run(tmp_path)

    _write_live_run(tmp_path, eligible=2)
    manifest = _manifest(tmp_path)
    manifest["editorial_qc"][0]["decision"] = "REJECT"
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="multimodal-editorial-QC"):
        validate_live_run(tmp_path)


def test_final_multimodal_boundary_and_policy_confirmation_are_required(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=1)
    manifest = _manifest(tmp_path)
    manifest["boundary_qc"][0]["multimodal_editorial_review_decision"] = "SKIPPED"
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="boundary-QC is missing"):
        validate_live_run(tmp_path)

    manifest["boundary_qc"][0]["multimodal_editorial_review_decision"] = "PASS"
    manifest["campaign_policy_qc"][0]["multimodal_policy_review_decision"] = "UNKNOWN"
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="campaign-policy-QC is missing"):
        validate_live_run(tmp_path)


def test_caption_audit_rejects_missing_concurrency_and_hook_overlap(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=1)
    audit_path = tmp_path / "captions" / "00.caption-audit.json"
    audit = json.loads(audit_path.read_text())
    audit.pop("simultaneous_narrative_layers_max")
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="concurrency evidence is missing"):
        validate_live_run(tmp_path)

    audit["simultaneous_narrative_layers_max"] = 2
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="overlapping narrative captions"):
        validate_live_run(tmp_path)

    audit["simultaneous_narrative_layers_max"] = 1
    audit["hook_overlay_rendered"] = True
    audit["potential_hook_caption_overlap_seconds"] = 0.2
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="hook and spoken captions overlap"):
        validate_live_run(tmp_path)


def test_technical_qc_rejects_transition_caption_and_watermark_failures(tmp_path: Path) -> None:
    cases = (
        ("captions", "alignment", "FAIL", "caption alignment QC"),
        ("framing", "transition_qc_pass", False, "transition QC"),
        ("watermark", "renderer_asset_present", False, "watermark"),
    )
    for index, (section, field, value, message) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        _write_live_run(root, eligible=1)
        manifest = _manifest(root)
        manifest["technical_qc"][0][section][field] = value
        _save_manifest(root, manifest)
        with pytest.raises(ValueError, match=message):
            validate_live_run(root)


def test_dynamic_artifact_inventory_and_required_ledgers_are_fail_closed(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=2)
    (tmp_path / "tracking" / "00.tracking.json").unlink()
    with pytest.raises(ValueError, match="artifact inventory"):
        validate_live_run(tmp_path)

    root = tmp_path / "missing-funnel"
    root.mkdir()
    _write_live_run(root, eligible=1)
    manifest = _manifest(root)
    del manifest["funnel"]["quality_moments"]
    _save_manifest(root, manifest)
    with pytest.raises(ValueError, match="funnel ledger"):
        validate_live_run(root)

    root = tmp_path / "missing-required"
    root.mkdir()
    _write_live_run(root, eligible=1)
    (root / "coverage.json").unlink()
    with pytest.raises(ValueError, match="required live acceptance artifact"):
        validate_live_run(root)


def test_publication_state_and_zero_yield_state_are_strict(tmp_path: Path) -> None:
    _write_live_run(tmp_path, eligible=1)
    manifest = _manifest(tmp_path)
    manifest["publication_state"] = "REVIEW_REQUIRED"
    _save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="publish-readiness"):
        validate_live_run(tmp_path)

    root = tmp_path / "zero"
    root.mkdir()
    _write_live_run(root, eligible=0)
    manifest = _manifest(root)
    manifest["publication_state"] = "READY_FOR_HUMAN_REVIEW"
    _save_manifest(root, manifest)
    with pytest.raises(ValueError, match="zero-quality yield"):
        validate_live_run(root)


def test_non_object_and_invalid_count_evidence_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        validate_live_run(tmp_path)

    root = tmp_path / "bad-count"
    root.mkdir()
    _write_live_run(root, eligible=1)
    manifest = _manifest(root)
    manifest["targets"]["eligible_quality_moments"] = True
    _save_manifest(root, manifest)
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_live_run(root)


def test_production_workflow_is_dynamic_yield_and_reviews_every_mp4() -> None:
    workflow = Path(".github/workflows/production-pipeline.yml").read_text()
    assert "feat/word-reveal-face-tracking" not in workflow
    assert 'int(result["rendered_finalists"]) >= 6' not in workflow
    assert 'int(result["initial_shortlist"]) >= 3' not in workflow
    assert 'targets["eligible_quality_moments"]' in workflow
    assert "for item in rendered:" in workflow
    assert '"ffmpeg"' in workflow
    assert "manual-review-queue.json" in workflow
    assert "PENDING_ACTUAL_REVIEW" in workflow
    assert '"automated_hilp_allowed": False' in workflow
    assert "production-review-${{ github.sha }}" in workflow
