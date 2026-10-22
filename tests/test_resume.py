from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper.resume import validate_resume_artifact

EDITOR = {
    "model_id": "editor",
    "revision": "rev",
    "quantization": "q",
    "inference_engine": "engine",
    "prompt_version": "prompt",
    "schema_version": "schema",
}
ASR = {**EDITOR, "model_id": "asr"}
VISION = {**EDITOR, "model_id": "vision"}


def _write_prior(root: Path, *, editorial: dict[str, str] | None = None) -> None:
    run = root / "prior-run"
    run.mkdir(parents=True, exist_ok=True)
    brief = {"campaign_id": "campaign", "targets": {"mode": "explicit"}}
    (run / "brief.normalized.json").write_text(json.dumps(brief), encoding="utf-8")
    manifest = {
        "campaign_id": "campaign",
        "status": "FAILED",
        "status_reason": "explicit_target_grounding_failed",
        "run_metadata": {
            "architecture": "autonomous-multimodal-quality-graph",
            "git_sha": "a" * 40,
            "compute_profile": "balanced",
            "source_hashes": {"video": "s" * 64},
            "editorial_inference": {
                "model": editorial or EDITOR,
                "model_invocations": [],
            },
            "grounding_inference": {
                "models": [
                    {
                        "video_id": "video",
                        "transcription": {"model": ASR},
                    }
                ]
            },
            "visual_inference": {
                "scout": [{"video_id": "video", "model": VISION}],
                "review": [],
            },
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _provenance() -> dict[str, object]:
    return {
        "schema_version": "clipper-resume-provenance-v1",
        "workflow_run_id": "123",
        "artifact_run_path": "/prior-run",
        "artifact_origin_workflow_run_id": "122",
        "artifact_origin_head_sha": "a" * 40,
        "campaign_id": "campaign",
        "source_hashes": {"video": "s" * 64},
        "cache_root": "/artifacts/_cache",
    }


def _validate(root: Path, provenance: dict[str, object] | None = None) -> dict[str, object]:
    return validate_resume_artifact(
        artifact_root=root,
        requested_run_id="123",
        provenance=dict(provenance or _provenance()),
        current_brief={"campaign_id": "campaign", "targets": {"mode": "explicit"}},
        current_source_hashes={"video": "s" * 64},
        current_compute_profile="balanced",
        current_cache_root="/artifacts/_cache",
        current_models={
            "editorial": EDITOR,
            "transcription": ASR,
            "alignment": {**EDITOR, "model_id": "alignment"},
            "diarization": {**EDITOR, "model_id": "diarization"},
            "vision_scout": VISION,
        },
    )


def test_resume_provenance_binds_exact_prior_artifact(tmp_path: Path) -> None:
    _write_prior(tmp_path)

    result = _validate(tmp_path)

    assert result["status"] == "PASS"
    assert result["workflow_run_id"] == "123"
    assert result["artifact_run_path"] == "/prior-run"
    assert result["source_hashes"] == {"video": "s" * 64}
    assert result["grounding_model_evidence_checked"] == 1
    assert result["visual_model_evidence_checked"] == 1
    assert result["cache_reuse_policy"].endswith("keys-only")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("workflow_run_id", "wrong", "workflow run ID"),
        ("artifact_run_path", "/../escape", "direct artifact run directory"),
        ("campaign_id", "other", "registry campaign"),
        ("artifact_origin_head_sha", "b" * 40, "source SHA"),
        ("cache_root", "/other", "cache root"),
    ],
)
def test_resume_provenance_rejects_registry_drift(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    _write_prior(tmp_path)
    provenance = _provenance()
    provenance[field] = value

    with pytest.raises(RuntimeError, match=match):
        _validate(tmp_path, provenance)


def test_resume_provenance_rejects_source_or_brief_or_model_drift(tmp_path: Path) -> None:
    _write_prior(tmp_path)

    with pytest.raises(RuntimeError, match="source hashes"):
        validate_resume_artifact(
            artifact_root=tmp_path,
            requested_run_id="123",
            provenance=_provenance(),
            current_brief={"campaign_id": "campaign", "targets": {"mode": "explicit"}},
            current_source_hashes={"video": "x" * 64},
            current_compute_profile="balanced",
            current_cache_root="/artifacts/_cache",
            current_models={"editorial": EDITOR, "transcription": ASR, "vision_scout": VISION},
        )

    run = tmp_path / "prior-run"
    (run / "brief.normalized.json").write_text(
        json.dumps({"campaign_id": "campaign", "targets": {"mode": "other"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="normalized brief"):
        _validate(tmp_path)

    _write_prior(tmp_path, editorial={**EDITOR, "revision": "other"})
    with pytest.raises(RuntimeError, match="editorial model identity"):
        _validate(tmp_path)
