"""Narrative GPU diagnostic is provenance-bound and never a production gate."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import yaml


def test_narrative_gpu_workflow_is_validation_only_and_cannot_start_media():
    workflow = yaml.safe_load(Path(".github/workflows/tjr-weekly-hd.yml").read_text())
    event = workflow.get("on", workflow.get(True))
    assert (
        "narrative_gpu" in event["workflow_dispatch"]["inputs"]["reviewer_model_probe"]["options"]
    )
    job = workflow["jobs"]["editorial_preflight"]
    assert "inputs.source_mode == 'validate_only'" in job["if"]
    steps = job["steps"]
    gpu = next(s for s in steps if s.get("name") == "Install private GPU orchestration client")
    assert "inputs.reviewer_model_probe == 'narrative_gpu'" in gpu["if"]
    cpu = next(s for s in steps if s.get("name") == "Install pinned CPU reviewer")
    assert "inputs.reviewer_model_probe != 'narrative_gpu'" in cpu["if"]
    diagnostic = next(s for s in steps if s.get("name", "").startswith("Assess reviewer evidence"))
    assert "MODE_FLAG=--narrative-gpu-probe" in diagnostic["run"]
    assert 'EXTRA_ARGUMENTS+=(--source-answer-report "$SOURCE_ANSWER")' in diagnostic["run"]
    assert (
        "narrative_gpu"
        in next(s for s in steps if s.get("with", {}).get("path") == "reviewer-source-answer")["if"]
    )


def test_gpu_diagnostic_preserves_exact_source_bytes_and_never_approves(tmp_path, monkeypatch):
    from clipper import editorial_benchmark
    from scripts import tjr_semantic_editor as editor

    monkeypatch.setattr(editorial_benchmark, "load_frozen_relations", lambda *_: [object()] * 15)
    monkeypatch.setattr(editor, "_thinking_review_profile", lambda: {"candidate": "reasoning-test"})
    monkeypatch.setattr(editor, "qualification_code_hash", lambda *_: "contract-code-hash")
    entries = {
        "fixture": '{"source_sha256": "' + "a" * 64 + '"}\n',
        "proof": '{  "frozen" : true }\n',
        "transcript": '[{"start":0,"end":3,"text":"Old story."}]\n',
        "provenance": '{"identity":{"source_sha256":"' + "a" * 64 + '"}}',
        "source_answer": json.dumps(
            {
                "experiment": "claim_blind_source_answer_v1",
                "experiment_complete": True,
                "production_approved": False,
            }
        ),
    }
    paths = {}
    for key, value in entries.items():
        path = tmp_path / f"{key}.json"
        path.write_text(value)
        paths[key] = path
    report_path = tmp_path / "reviewer-preflight.json"
    observed = []

    def remote(packed, profile, job_key, code_hash):
        unpacked = json.loads(gzip.decompress(packed))
        observed.append(unpacked)
        assert {key: unpacked[key] for key in entries} == entries
        assert profile == {"candidate": "reasoning-test"}
        assert code_hash == "contract-code-hash"
        assert (
            job_key
            == hashlib.sha256(
                packed + json.dumps(profile, sort_keys=True).encode() + code_hash.encode()
            ).hexdigest()
        )
        return {"experiment_complete": True, "semantic_pass": False, "files": []}

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(enable_output=lambda: nullcontext()))
    monkeypatch.setitem(
        sys.modules,
        "scripts.tjr_modal_probe",
        SimpleNamespace(
            app=SimpleNamespace(run=lambda: nullcontext()),
            qualify_podcast_narrative_gpu=SimpleNamespace(remote=remote),
            volume=SimpleNamespace(read_file=lambda *_: []),
        ),
    )
    outcome = editor.podcast_narrative_gpu_qualification(
        paths["fixture"],
        paths["proof"],
        paths["transcript"],
        paths["provenance"],
        paths["source_answer"],
        report_path,
    )
    assert outcome == 1 and len(observed) == 1
    report = json.loads(report_path.read_text())
    assert report["experiment_complete"] is True
    assert report["semantic_pass"] is False
    assert report["production_approved"] is False
    assert report["diagnostic_only"] is True
