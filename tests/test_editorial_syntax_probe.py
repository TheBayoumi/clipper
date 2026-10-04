"""The pinned parser benchmark checkpoints both control sets and never approves."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper import editorial_syntax_probe as probe
from clipper.editorial_benchmark import FrozenRelation, HeldoutClaim


def frozen(index: int) -> FrozenRelation:
    return FrozenRelation(
        case_id=f"case_{index}",
        fixture_index=index,
        headline="Bobby was the speaker's opponent",
        dimension="relationship_role",
        question="Who was the opponent?",
        answer="Bobby",
        expected_supported=False,
        annotation_reason="diagnostic",
        evidence_first_unit=0,
        evidence_last_unit=0,
        evidence_quote="Bobby",
        source_units=("Bobby was present.",),
    )


def heldout(index: int) -> HeldoutClaim:
    return HeldoutClaim(
        case_id=f"heldout_{index}",
        headline="Bobby was the speaker's opponent",
        expected_supported=False,
        annotation_reason="provisional",
        annotation_status="transcript_only_pending_audio_review",
        source_start=0.0,
        source_end=10.0,
        source_clip="clip.mp4",
        source_units=("Bobby was present.",),
    )


def paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"source_sha256": "a" * 64, "transcript_sha256": "b" * 64}))
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    return (
        fixture,
        proof,
        tmp_path / "transcript.json",
        tmp_path / "provenance.json",
        tmp_path / "heldout.json",
        tmp_path / "report.json",
    )


def test_syntax_probe_checkpoints_frozen_and_heldout_without_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe,
        "load_frozen_relations",
        lambda *_: [*(frozen(index) for index in range(12)), frozen(0), frozen(1), frozen(2)],
    )
    monkeypatch.setattr(
        probe, "load_heldout_claims", lambda *_: [heldout(index) for index in range(12)]
    )
    monkeypatch.setattr(
        probe,
        "version",
        lambda name: probe.SPACY_VERSION if name == "spacy" else probe.MODEL_VERSION,
    )
    monkeypatch.setattr(probe, "import_module", lambda _: SimpleNamespace(load=lambda _: object()))
    seen = []

    def inventory(headline, parser):
        seen.append((headline, parser))
        return {
            "frames": [{"roles": [{"text": "Bobby"}]}],
            "parse_warnings": [{"reason": "unresolved"}],
            "production_approved": False,
        }

    monkeypatch.setattr(probe, "inventory_headline_syntax", inventory)
    args = paths(tmp_path)
    assert probe.run_syntax_inventory_probe(*args) == 1
    report = json.loads(args[-1].read_text())
    assert report["experiment_complete"] is True
    assert report["production_approved"] is False
    assert report["syntax_anchor_presence"] == {"found": 15, "total": 15}
    assert report["parse_warning_count"] == 24
    assert len(report["frozen"]) == len(report["heldout"]) == 12
    assert len(seen) == 24


@pytest.mark.parametrize("frozen_count,heldout_count", [(14, 12), (15, 11)])
def test_syntax_probe_rejects_incomplete_controls(
    tmp_path, monkeypatch, frozen_count, heldout_count
):
    rows = [frozen(index % 12) for index in range(frozen_count)]
    monkeypatch.setattr(probe, "load_frozen_relations", lambda *_: rows)
    monkeypatch.setattr(
        probe, "load_heldout_claims", lambda *_: [heldout(i) for i in range(heldout_count)]
    )
    with pytest.raises(ValueError, match="requires all"):
        probe.run_syntax_inventory_probe(*paths(tmp_path))


def test_syntax_probe_rejects_unpinned_parser(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe,
        "load_frozen_relations",
        lambda *_: [*(frozen(index) for index in range(12)), frozen(0), frozen(1), frozen(2)],
    )
    monkeypatch.setattr(
        probe, "load_heldout_claims", lambda *_: [heldout(index) for index in range(12)]
    )
    monkeypatch.setattr(probe, "version", lambda _: "unversioned")
    with pytest.raises(RuntimeError, match="not pinned"):
        probe.run_syntax_inventory_probe(*paths(tmp_path))
