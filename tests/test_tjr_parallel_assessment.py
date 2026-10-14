"""Small, model-free checks for parallel checkpoint and merge invariants."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.tjr_parallel_assessment import assessment_data, merge, plan


def _json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _archive(root: Path) -> tuple[dict, list[dict]]:
    from scripts.tjr_parallel_assessment import digest

    proposals = [
        {
            "video_id": "3S-fL9rNPp0",
            "start": float(i * 45),
            "end": float((i + 1) * 45),
            "text": f"Podcast segment {i}",
            "score": 50,
            "reasons": [],
        }
        for i in range(5)
    ]
    transcript = [{"start": 0, "end": 5, "text": "hello", "words": []}]
    identity = {
        "version": "podcast_structured_editor_v7",
        "source_sha256": "b" * 64,
        "proposal_sha256": digest(proposals),
        "transcript_sha256": digest(transcript),
        "editor_code_sha256": "a" * 64,
        "model_sha256": "c" * 64,
        "model_revision": "revision",
        "duration_bounds": [20, 45],
        "seed": 0,
        "temperature": 0,
        "selector_contract_version": "source-exchange-selection-v1",
        "selector_sha256": "d" * 64,
        "boundary_contract_version": "source-unit-boundary-v1",
        "boundary_sha256": "e" * 64,
        "reviewer_sha256": "f" * 64,
        "reviewer_model_profile": {"model": "test"},
    }
    baseline = {
        "identity": identity,
        "complete": False,
        "processed_count": 2,
        "decisions": [
            {"proposal_start": c["start"], "proposal_end": c["end"], "decision": {"keep": False}}
            for c in proposals[:2]
        ],
        "candidates": [],
    }
    _json(root / "case" / "editorial-cache.json", baseline)
    _json(root / "case" / "proposal-cache.json", {"complete": True, "candidates": proposals})
    _json(root / "case" / "transcript.json", transcript)
    _json(
        root / "tjr-modal-probe" / "staged-original.json",
        {"video_id": "3S-fL9rNPp0", "source_sha256": "b" * 64},
    )
    _json(
        root / "case" / "official-source-candidates.json",
        [
            {
                "video_id": "3S-fL9rNPp0",
                "channel_id": "UCf1q6dhccWr6eQEcFFnJSbA",
                "url": "https://www.youtube.com/watch?v=3S-fL9rNPp0",
                "published": "2026-10-09T20:27:11+00:00",
            }
        ],
    )
    return identity, proposals


def _shard(root: Path, identity: dict, proposals: list[dict], start: int, stop: int) -> None:
    folder = root / f"editorial-shard-{start}-{stop}"
    decisions = [
        {
            "proposal_start": candidate["start"],
            "proposal_end": candidate["end"],
            **(
                {"decision": {"keep": False}}
                if start <= index < stop
                else {"rejection": "SHARD_NOT_ASSIGNED"}
            ),
        }
        for index, candidate in enumerate(proposals)
    ]
    _json(
        folder / "editorial-cache.json",
        {
            "identity": {**identity, "editor_code_sha256": "1" * 64},
            "complete": False,
            "audit": {"assessments": decisions},
            "candidates": [],
        },
    )
    _json(
        folder / "shard-manifest.json",
        {
            "start": start,
            "stop": stop,
            "total": len(proposals),
            "source_sha256": identity["source_sha256"],
            "proposal_sha256": identity["proposal_sha256"],
            "assessment_count": stop - start,
            "complete": True,
        },
    )


def test_plan_reuses_completed_assessments_and_shards_only_remaining(tmp_path: Path) -> None:
    _archive(tmp_path / "baseline")
    rows = plan(tmp_path / "baseline", tmp_path / "plan.json", width=2)
    assert rows == [{"start": 2, "stop": 4}, {"start": 4, "stop": 5}]
    assert json.loads((tmp_path / "plan.json").read_text())["cached"] == 2


def test_aggregation_preserves_cached_prefix_and_reconciles_every_candidate(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    identity, proposals = _archive(baseline)
    parts = tmp_path / "parts"
    _shard(parts, identity, proposals, 2, 4)
    _shard(parts, identity, proposals, 4, 5)
    output = tmp_path / "merged"
    merge(baseline, parts, output)
    saved = json.loads((output / "editorial-cache.json").read_text())
    assert saved["complete"] is True
    assert len(saved["audit"]["assessments"]) == len(proposals)
    assert saved["audit"]["resumed_assessments"] == 2
    assert json.loads((output / "merge-proof.json").read_text())["shard_count"] == 2
    assert (output / "proposal-cache.json").is_file()


def test_aggregation_rejects_missing_overlap_and_identity_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    identity, proposals = _archive(baseline)
    parts = tmp_path / "parts"
    _shard(parts, identity, proposals, 2, 4)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        merge(baseline, parts, tmp_path / "not-complete")
    _shard(parts, identity, proposals, 3, 5)
    with pytest.raises(RuntimeError, match="OVERLAP"):
        merge(baseline, parts, tmp_path / "overlap")
    shutil.rmtree(parts / "editorial-shard-3-5")
    _shard(parts, identity, proposals, 4, 5)
    shard_path = parts / "editorial-shard-4-5" / "editorial-cache.json"
    edited = json.loads(shard_path.read_text())
    edited["identity"]["reviewer_sha256"] = "0" * 64
    _json(shard_path, edited)
    with pytest.raises(RuntimeError, match="IDENTITY_DRIFT"):
        merge(baseline, parts, tmp_path / "drift")


def test_archive_rejects_mismatched_transcript(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    _archive(baseline)
    _json(
        baseline / "case" / "transcript.json",
        [{"start": 0, "end": 5, "text": "tampered", "words": []}],
    )
    with pytest.raises(RuntimeError, match="SOURCE_IDENTITY_MISMATCH"):
        assessment_data(baseline)
