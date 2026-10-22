"""Never mix qualification manifests from different source commits."""

from __future__ import annotations

from typing import Any

import pytest

from clipper_engine import build_identity
from clipper_engine.gameplay import allocation, candidates

SHA = "a" * 40
OTHER = "b" * 40


def test_no_runner_sha_is_permitted_only_for_local_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)
    assert build_identity.current_execution_sha() == ""
    assert build_identity.validate_artifact_sha("", stage="offline") == ""


def test_production_runner_requires_same_artifact_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)
    assert build_identity.current_execution_sha() == SHA
    assert build_identity.validate_artifact_sha(SHA, stage="analysis") == SHA
    with pytest.raises(RuntimeError, match="execution SHA mismatch"):
        build_identity.validate_artifact_sha(OTHER, stage="analysis")
    with pytest.raises(RuntimeError, match="execution SHA mismatch"):
        build_identity.validate_artifact_sha(None, stage="allocation")


@pytest.mark.parametrize("value", ["short", "not-a-git-sha", "1" * 41])
def test_invalid_execution_sha_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GITHUB_SHA", value)
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)
    with pytest.raises(RuntimeError, match="exactly 40"):
        build_identity.current_execution_sha()


def test_conflicting_git_authorities_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("CLIPPER_ACCEPTANCE_SHA", OTHER)
    with pytest.raises(RuntimeError, match="conflicts"):
        build_identity.current_execution_sha()


def test_allocated_render_cannot_read_other_commit(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)
    alloc = tmp_path / "allocation.json"
    alloc.write_text(
        json.dumps(
            {"execution_sha": OTHER, "source_allocations": {"test": {"plan_keys": [], "plans": []}}}
        )
    )
    with pytest.raises(RuntimeError, match="allocation: execution SHA mismatch"):
        candidates._select_from_allocation("test", alloc)


def test_allocator_rejects_cross_commit_analysis_before_coverage(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from clipper_engine.profiles import load_profile

    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)
    path = tmp_path / "test_analysis.json"
    path.write_text(
        json.dumps(
            {
                "source_key": "test",
                "execution_sha": OTHER,
                "semantic_engine": "deterministic-gameplay",
                "editorial_planner": "semantic-editor",
                "candidate_mode": "engagement_driven_hardened_source_integrity",
            }
        )
    )
    with pytest.raises(RuntimeError, match="analysis: execution SHA mismatch"):
        allocation.allocate(tmp_path, load_profile("mw4").config)
