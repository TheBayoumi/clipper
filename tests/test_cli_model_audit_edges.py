from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from clipper.cli import (
    _assert_runtime_dependencies,
    _audit_model_evidence,
    _model_cache_fingerprint,
    _model_id,
    _resolved_model_plan,
)
from clipper.pipeline import PipelineSettings


class Identity:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id}


def _provider(model_id: str):
    return SimpleNamespace(identity=Identity(model_id))


def _write_manifest(run_dir: Path, run_metadata: object) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "SUCCESS", "run_metadata": run_metadata}), encoding="utf-8"
    )


def _editorial_evidence(model_id: str = "editor") -> dict[str, object]:
    return {
        "editorial_inference": {
            "model_invocations": [{"cache_hit": False, "model": {"model_id": model_id}}]
        }
    }


def test_resolved_model_plan_reads_all_required_provider_identities() -> None:
    settings = PipelineSettings(compute_profile="balanced")
    with (
        patch("clipper.cli.editorial_provider", return_value=_provider("editor")),
        patch(
            "clipper.cli.speech_providers",
            return_value=(_provider("asr"), _provider("align"), _provider("diarize")),
        ),
    ):
        plan = _resolved_model_plan(settings)
    assert plan["architecture"] == "autonomous-multimodal-quality-graph"
    assert plan["compute_profile"] == "balanced"
    assert plan["editorial"] == {"model_id": "editor"}
    assert plan["transcription"] == {"model_id": "asr"}
    assert plan["alignment"] == {"model_id": "align"}
    assert plan["diarization"] == {"model_id": "diarize"}
    assert "embedding" not in plan


def test_resolved_model_plan_has_no_legacy_provider_disable_switches() -> None:
    settings = PipelineSettings(compute_profile="local-lite")
    with (
        patch("clipper.cli.editorial_provider", return_value=_provider("editor")) as editorial,
        patch(
            "clipper.cli.speech_providers",
            return_value=(_provider("asr"), _provider("align"), _provider("diarize")),
        ) as speech,
    ):
        plan = _resolved_model_plan(settings)
    editorial.assert_called_once_with("local-lite")
    speech.assert_called_once_with("local-lite")
    assert plan["architecture"] == "autonomous-multimodal-quality-graph"


def test_runtime_dependency_preflight_checks_resolved_local_runtime_modules() -> None:
    with patch("clipper.cli.importlib.util.find_spec", return_value=object()) as find_spec:
        _assert_runtime_dependencies(
            {"editorial": {"inference_engine": "transformers"}, "compute_profile": "local-lite"}
        )
    find_spec.assert_called_once_with("transformers")


def test_runtime_dependency_preflight_accepts_installed_modal_sdk() -> None:
    plan = {"editorial": {"inference_engine": "modal-transformers"}}
    with patch("clipper.cli.importlib.util.find_spec", return_value=object()) as find_spec:
        _assert_runtime_dependencies(plan)
    find_spec.assert_called_once_with("modal")


def test_runtime_dependency_preflight_rejects_missing_modal_sdk() -> None:
    plan = {
        "editorial": {"inference_engine": "modal-transformers"},
        "transcription": {"inference_engine": "modal-faster-whisper"},
    }
    with (
        patch("clipper.cli.importlib.util.find_spec", return_value=None),
        pytest.raises(RuntimeError, match="missing runtime module") as captured,
    ):
        _assert_runtime_dependencies(plan)
    assert "modal" in str(captured.value)
    assert 'pip install -e ".[open-models]"' in str(captured.value)


def test_runtime_dependency_preflight_handles_invalid_module_spec() -> None:
    plan = {"alignment": {"inference_engine": "modal-whisperx"}}
    with (
        patch("clipper.cli.importlib.util.find_spec", side_effect=ValueError("bad spec")),
        pytest.raises(RuntimeError, match=r"missing runtime module.*modal"),
    ):
        _assert_runtime_dependencies(plan)


def test_model_id_rejects_non_mapping_and_empty_identity() -> None:
    assert _model_id(None) is None
    assert _model_id("model") is None
    assert _model_id({}) is None
    assert _model_id({"model_id": "m"}) == "m"


def test_model_audit_rejects_non_mapping_run_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad-metadata"
    _write_manifest(run_dir, [])
    with pytest.raises(RuntimeError, match="missing run_metadata"):
        _audit_model_evidence(run_dir, {})


def test_model_audit_accepts_fully_cached_editorial_resume_with_bound_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "cached-editorial"
    editorial = {
        "model_id": "editor",
        "revision": "rev",
        "quantization": "int4",
        "inference_engine": "modal-transformers",
        "prompt_version": "p",
        "schema_version": "s",
    }
    fingerprint = _model_cache_fingerprint(editorial)
    _write_manifest(
        run_dir,
        {
            "editorial_inference": {
                "model_invocations": [],
                "cache_summary": {
                    "stage_cache_hits": 7,
                    "stage_executions": 0,
                    "editorial_model_fingerprint": fingerprint,
                    "editorial_model": editorial,
                },
            },
            "grounding_inference": {
                "models": [
                    {
                        "transcription": {
                            "cache_hit": True,
                            "model": {"model_id": "asr"},
                        },
                        "alignment": {
                            "cache_hit": True,
                            "model": {"model_id": "align"},
                        },
                        "diarization": {
                            "cache_hit": True,
                            "model": {"model_id": "diarize"},
                        },
                    }
                ]
            },
        },
    )
    audit_result = _audit_model_evidence(
        run_dir,
        {
            "editorial": editorial,
            "transcription": {"model_id": "asr"},
            "alignment": {"model_id": "align"},
            "diarization": {"model_id": "diarize"},
        },
    )
    editorial_audit = audit_result["editorial"]
    assert isinstance(editorial_audit, dict)
    assert editorial_audit["fully_cached_resume"] is True
    assert editorial_audit["invocations"] == 0
    assert editorial_audit["cache_hits"] == 7


def test_model_audit_rejects_fully_cached_resume_with_wrong_model_fingerprint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bad-cached-editorial"
    editorial = {"model_id": "editor", "revision": "rev"}
    _write_manifest(
        run_dir,
        {
            "editorial_inference": {
                "model_invocations": [],
                "cache_summary": {
                    "stage_cache_hits": 1,
                    "stage_executions": 0,
                    "editorial_model_fingerprint": "wrong",
                    "editorial_model": editorial,
                },
            },
            "grounding_inference": {"models": [{"transcription": {"cache_hit": True}}]},
        },
    )
    with pytest.raises(RuntimeError, match="not bound to the resolved model identity"):
        _audit_model_evidence(run_dir, {"editorial": editorial})


def test_model_audit_rejects_missing_grounding_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "no-grounding"
    _write_manifest(
        run_dir,
        {
            **_editorial_evidence(),
            "grounding_inference": {"models": []},
        },
    )
    plan = {"editorial": {"model_id": "editor"}}
    with pytest.raises(RuntimeError, match="no model evidence"):
        _audit_model_evidence(run_dir, plan)


def test_model_audit_ignores_malformed_grounding_items_but_records_valid_ones(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "mixed-grounding"
    _write_manifest(
        run_dir,
        {
            **_editorial_evidence(),
            "grounding_inference": {
                "models": [
                    "not-a-source",
                    {
                        "transcription": "not-evidence",
                        "alignment": {
                            "cache_hit": True,
                            "model": {"model_id": "align"},
                        },
                        "diarization": {
                            "cache_hit": False,
                            "model": {},
                        },
                    },
                ]
            },
        },
    )
    audit = _audit_model_evidence(
        run_dir,
        {
            "editorial": {"model_id": "editor"},
            "alignment": {},
            "transcription": {},
            "diarization": {},
        },
    )
    grounding = audit["grounding"]
    assert isinstance(grounding, dict)
    assert grounding["observed_models"] == ["align"]
    assert grounding["evidence_records"] == 2
    assert grounding["cache_hits"] == 1
    assert grounding["live_invocations"] == 1


def test_model_audit_rejects_missing_expected_grounding_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing-grounding-model"
    _write_manifest(
        run_dir,
        {
            **_editorial_evidence(),
            "grounding_inference": {
                "models": [
                    {
                        "transcription": {
                            "cache_hit": False,
                            "model": {"model_id": "asr"},
                        }
                    }
                ]
            },
        },
    )
    plan = {
        "editorial": {"model_id": "editor"},
        "transcription": {"model_id": "asr"},
        "alignment": {"model_id": "align"},
        "diarization": {"model_id": "diarize"},
    }
    with pytest.raises(RuntimeError, match="missing resolved models") as captured:
        _audit_model_evidence(run_dir, plan)
    assert "align" in str(captured.value)
    assert "diarize" in str(captured.value)
