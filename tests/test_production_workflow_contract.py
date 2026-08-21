from pathlib import Path


def test_production_workflow_is_planning_first_and_cache_preserving() -> None:
    workflow = Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")

    planning = workflow.index("Planning-only pass using persistent paid cache")
    validation = workflow.index("Download and validate planning evidence before rendering")
    rendering = workflow.index("Render only after planning gate passes")

    assert planning < validation < rendering
    assert '"render": False' in workflow
    assert '"render": True' in workflow
    assert "fresh_inference" not in workflow
    assert "clipper-production-artifacts" in workflow
    assert "planning-evidence/" in workflow
    assert "cycle-evidence/" in workflow


def test_production_workflow_is_explicit_marker_triggered_and_quota_free() -> None:
    workflow = Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")

    assert '"acceptance/production-run-request.json"' in workflow
    assert 'rendered_finalists"]) >= 6' not in workflow
    assert 'initial_shortlist"]) >= 3' not in workflow
    assert "reviewable == rendered" in workflow
    assert "eligible_quality_moments" in workflow
    assert '"automated_hilp_allowed": False' in workflow
