from pathlib import Path


def _workflow() -> str:
    return Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")


def test_production_workflow_is_planning_first_and_cache_preserving() -> None:
    workflow = _workflow()

    planning = workflow.index("Planning-only pass using persistent paid cache")
    validation = workflow.index("Download and validate planning evidence before rendering")
    rendering = workflow.index("Render only after planning gate passes using the same cache identity")

    assert planning < validation < rendering
    assert '"render": False' in workflow
    assert '"render": True' in workflow
    assert "fresh_inference" not in workflow
    assert "resume_from_run_id" not in workflow
    assert "clipper-production-artifacts" in workflow
    assert "planning-evidence/" in workflow
    assert "cycle-evidence/" in workflow


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
    assert 'Path(os.environ["CLIPPER_CAMPAIGN_BRIEF"]).read_text()' in workflow


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
    assert "reviewable == rendered" in workflow
    assert "eligible_quality_moments" in workflow
    assert '"automated_hilp_allowed": False' in workflow
    assert "PENDING_ACTUAL_REVIEW" in workflow
