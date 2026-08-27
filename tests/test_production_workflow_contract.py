from pathlib import Path


def _workflow() -> str:
    return Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")


def test_production_workflow_is_single_pass_resumable_and_exact_head() -> None:
    workflow = _workflow()

    deploy_models = workflow.index("Deploy exact-head open-model workers")
    deploy_pipeline = workflow.index("Deploy exact-head production pipeline worker")
    rendering = workflow.index("Run current-model production render with explicit inference mode")
    validation = workflow.index(
        "Validate dynamic yield, resumable inference, cost bounds, and actual media"
    )

    assert deploy_models < deploy_pipeline < rendering < validation
    assert '"render": False' not in workflow
    assert '"render": True' in workflow
    assert "fresh_inference:" in workflow
    assert '"fresh_inference": True' not in workflow
    assert '"fresh_inference": os.environ["CLIPPER_FRESH_INFERENCE"] == "true"' in workflow
    assert '"resume_from_run_id": os.environ.get("CLIPPER_RESUME_FROM_RUN_ID") or None' in workflow
    assert '"sources": [source]' in workflow
    assert '"git_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"]' in workflow
    assert "content-addressed-resume" in workflow
    assert "content-addressed-stage-resume" in workflow
    assert "PENDING_ACTUAL_MP4_REVIEW" in workflow
    assert "READY_FOR_HUMAN_REVIEW" in workflow
    assert "READY_TO_PUBLISH" in workflow
    assert "cycle-evidence" in workflow
    assert "hilp-review" in workflow


def test_production_workflow_resolves_campaign_and_target_from_request_data() -> None:
    workflow = _workflow()

    assert "campaign_brief:" in workflow
    assert "target_video_id:" in workflow
    assert '"acceptance/production-run-request.json"' in workflow
    assert 'marker.get("campaign_brief")' in workflow
    assert 'marker.get("target_video_id")' in workflow
    assert 'targets.get("videos")' in workflow
    assert 'rights.get("authorized_channels")' in workflow
    assert 'os.environ["CLIPPER_TARGET_VIDEO_ID"]' in workflow
    assert 'os.environ["CLIPPER_TARGET_VIDEO_URL"]' in workflow
    assert 'os.environ["CLIPPER_TARGET_CHANNEL_ID"]' in workflow
    assert 'Path(os.environ["CLIPPER_CAMPAIGN_BRIEF"]).read_text(encoding="utf-8")' in workflow


def test_production_acceptance_contains_no_campaign_specific_identity() -> None:
    workflow = _workflow()

    forbidden = (
        "reach-double-coverage",
        "Double Coverage",
        "2Y4LP85PTak",
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "#DoubleCoverage",
    )
    assert all(value not in workflow for value in forbidden)


def test_production_workflow_is_dynamic_yield_and_human_review_gated() -> None:
    workflow = _workflow()

    assert 'int(result["rendered_finalists"]) >= 6' not in workflow
    assert 'int(result["initial_shortlist"]) >= 3' not in workflow
    assert "eligible_quality_moments" in workflow
    assert 'rendered = int(result.get("rendered") or 0)' in workflow
    assert 'reviewable = int(result.get("reviewable") or 0)' in workflow
    assert "if reviewable != rendered:" in workflow
    assert "contract permits zero quality yield" in workflow
    assert "PENDING_ACTUAL_MP4_REVIEW" in workflow
    assert '"human_review": "PENDING_ACTUAL_REVIEW"' in workflow
    assert "modal deploy scripts/modal_open_models.py" in workflow
    assert "modal deploy scripts/modal_pipeline.py" in workflow


def test_production_workflow_enforces_current_model_and_budget_evidence() -> None:
    workflow = _workflow()

    assert 'int(schema.get("task_families", 0)) != 4' in workflow
    assert 'int(cache.get("hits") or 0) != 0' in workflow
    assert 'int(cache_summary.get("stage_cache_hits") or 0) != 0' in workflow
    assert 'editorial.get("model_invocations")' in workflow
    assert '"semantic_cores", "narrative_envelope", "quality_windows"' in workflow
    assert "gpu_seconds > gpu_limit" in workflow
    assert "estimated_usd > cost_limit" in workflow
