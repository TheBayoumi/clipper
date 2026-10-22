from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper.providers.base import InferenceUsage
from clipper.runtime import ComputeBudget, StageJournal


def test_stage_journal_is_atomic_resumable_and_tracks_failure(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    journal = StageJournal(path)
    started = journal.start("editorial", total=4, timeout_seconds=120, message="starting")
    assert started.status == "RUNNING"
    journal.progress("editorial", 2, checkpoint="chunk-2", message="halfway")
    resumed = StageJournal(path)
    assert resumed.states["editorial"].completed == 2
    assert resumed.states["editorial"].checkpoint == "chunk-2"
    completed = resumed.complete("editorial", checkpoint="done")
    assert completed.status == "SUCCESS"
    assert completed.completed == 4
    payload = json.loads(path.read_text())
    assert len(payload["contract_fingerprint"]) == 64
    assert "schema_version" not in payload
    assert payload["stages"]["editorial"]["status"] == "SUCCESS"
    failed = resumed.fail("render", "encoder crashed", checkpoint="attempt-2")
    assert failed.failure_reason == "encoder crashed"
    assert not list(tmp_path.glob(".*.tmp"))


def test_stage_journal_rejects_invalid_state_and_counts(tmp_path: Path) -> None:
    journal = StageJournal(tmp_path / "progress.json")
    with pytest.raises(ValueError, match="cannot be empty"):
        journal.start("")
    journal.start("render", total=2)
    with pytest.raises(ValueError, match="completed count"):
        journal.progress("render", 3)
    with pytest.raises(ValueError, match="failure reason"):
        journal.fail("render", "")
    bad = tmp_path / "bad.json"
    bad.write_text('{"stages":{"x":{"status":"UNKNOWN"}}}')
    with pytest.raises(ValueError, match="invalid progress"):
        StageJournal(bad)


def test_compute_budget_accounts_usage_and_disables_large_vlm_first() -> None:
    budget = ComputeBudget(1.0, large_vlm_fraction=0.25)
    budget.record(
        InferenceUsage(
            "modal",
            "now",
            100,
            gpu_type="L4",
            gpu_seconds=100,
            estimated_cost_usd=0.20,
        )
    )
    assert budget.estimated_cost_usd == 0.2
    assert budget.gpu_seconds == 100
    assert budget.allow_large_vlm(estimated_next_cost_usd=0.54)
    assert not budget.allow_large_vlm(estimated_next_cost_usd=0.56)
    budget.record_mapping(
        {
            "provider": "modal",
            "started_at": "later",
            "duration_seconds": 10,
            "gpu_type": "L4",
            "gpu_seconds": 10,
            "estimated_cost_usd": 0.01,
        }
    )
    assert budget.estimated_cost_usd == 0.21
    budget.record_mapping("ignored")
    budget.record_mapping({"duration_seconds": object()})
    assert budget.to_dict()["large_vlm_allowed"] is True
    with pytest.raises(ValueError, match="positive"):
        ComputeBudget(0)
    with pytest.raises(ValueError, match="fraction"):
        ComputeBudget(1, large_vlm_fraction=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        budget.allow_large_vlm(estimated_next_cost_usd=-1)
