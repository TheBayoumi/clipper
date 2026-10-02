"""Identity/shape checks do not pretend to verify semantic annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clipper.editorial_benchmark import load_heldout_claims

FIXTURE = Path(__file__).resolve().parent / "fixtures/issue8_heldout_claims.json"


def test_committed_heldout_controls_are_explicitly_provisional():
    saved = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = saved["cases"]
    assert saved["annotation_status"] == "transcript_only_pending_audio_review"
    assert len(cases) == 12
    assert sum(item["expected_supported"] for item in cases) == 6
    assert len({item["source_clip"] for item in cases}) == 3
    assert len({item["id"] for item in cases}) == len(cases)


def test_loader_binds_claims_to_exact_transcript_and_provenance(tmp_path):
    transcript = [{"start": 10.0, "end": 12.0, "text": "The advice was to spar less."}]
    transcript_hash = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
    fixture = {
        "schema": "clipper-heldout-headlines-v1",
        "annotation_status": "transcript_only_pending_audio_review",
        "source_video_id": "_kDrxucOx9g",
        "source_sha256": "a" * 64,
        "transcript_sha256": transcript_hash,
        "source_artifact_run_id": "12345",
        "cases": [
            {
                "id": "spar_less",
                "source_start": 10.0,
                "source_end": 12.0,
                "source_clip": "05-double-coverage-_kDrxucOx9g.mp4",
                "headline": "The advice was to spar less",
                "expected_supported": True,
                "annotation_reason": "The source states the advice.",
            }
        ],
    }
    fixture_path = tmp_path / "claims.json"
    transcript_path = tmp_path / "transcript.json"
    provenance_path = tmp_path / "editorial-cache.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    provenance_path.write_text(
        json.dumps({"identity": {"source_sha256": "a" * 64, "transcript_sha256": transcript_hash}}),
        encoding="utf-8",
    )
    claims = load_heldout_claims(fixture_path, transcript_path, provenance_path)
    assert len(claims) == 1 and claims[0].source_units == ("The advice was to spar less.",)
    fixture["cases"][0]["source_start"] = 13.0
    fixture["cases"][0]["source_end"] = 15.0
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="no exact transcript window"):
        load_heldout_claims(fixture_path, transcript_path, provenance_path)
    fixture["cases"][0]["source_start"] = 10.0
    fixture["cases"][0]["source_end"] = 12.0
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    transcript_path.write_text(json.dumps([dict(transcript[0], text="Different speech")]))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_heldout_claims(fixture_path, transcript_path, provenance_path)
