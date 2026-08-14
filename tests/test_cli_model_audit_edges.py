from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from clipper.cli import _audit_model_evidence, _model_id, _resolved_model_plan
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


def test_resolved_model_plan_reads_all_open_provider_identities() -> None:
    settings = PipelineSettings(
        editorial_engine="open", grounding_engine="open", compute_profile="balanced"
    )
    with (
        patch(
            "clipper.cli.editorial_and_embedding_providers",
            return_value=(_provider("editor"), _provider("embed")),
        ),
        patch(
            "clipper.cli.speech_providers",
            return_value=(_provider("asr"), _provider("align"), _provider("diarize")),
        ),
    ):
        plan = _resolved_model_plan(settings)
    assert plan["editorial"] == {"model_id": "editor"}
    assert plan["embedding"] == {"model_id": "embed"}
    assert plan["transcription"] == {"model_id": "asr"}
    assert plan["alignment"] == {"model_id": "align"}
    assert plan["diarization"] == {"model_id": "diarize"}


def test_resolved_model_plan_skips_disabled_provider_families() -> None:
    settings = PipelineSettings(editorial_engine="heuristic", grounding_engine="legacy")
    with (
        patch("clipper.cli.editorial_and_embedding_providers") as editorial,
        patch("clipper.cli.speech_providers") as speech,
    ):
        plan = _resolved_model_plan(settings)
    editorial.assert_not_called()
    speech.assert_not_called()
    assert plan == {
        "editorial_engine": "heuristic",
        "grounding_engine": "legacy",
        "compute_profile": "balanced",
    }


def test_model_id_rejects_non_mapping_and_empty_identity() -> None:
    assert _model_id(None) is None
    assert _model_id("model") is None
    assert _model_id({}) is None
    assert _model_id({"model_id": "m"}) == "m"


def test_model_audit_rejects_non_mapping_run_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad-metadata"
    _write_manifest(run_dir, [])
    settings = PipelineSettings(editorial_engine="open", grounding_engine="open")
    with pytest.raises(RuntimeError, match="missing run_metadata"):
        _audit_model_evidence(run_dir, settings, {})


def test_model_audit_rejects_missing_grounding_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "no-grounding"
    _write_manifest(
        run_dir,
        {
            "editorial_inference": {
                "model_invocations": [{"cache_hit": False, "model": {"model_id": "editor"}}]
            },
            "grounding_inference": {"models": []},
        },
    )
    settings = PipelineSettings(editorial_engine="open", grounding_engine="open")
    plan = {"editorial": {"model_id": "editor"}}
    with pytest.raises(RuntimeError, match="no model evidence"):
        _audit_model_evidence(run_dir, settings, plan)


def test_model_audit_ignores_malformed_grounding_items_but_records_valid_ones(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "mixed-grounding"
    _write_manifest(
        run_dir,
        {
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
            }
        },
    )
    settings = PipelineSettings(editorial_engine="heuristic", grounding_engine="open")
    audit = _audit_model_evidence(
        run_dir,
        settings,
        {
            "alignment": {},
            "transcription": {},
            "diarization": {},
        },
    )
    grounding = audit["grounding"]
    assert grounding["observed_models"] == ["align"]
    assert grounding["evidence_records"] == 2
    assert grounding["cache_hits"] == 1
    assert grounding["live_invocations"] == 1


def test_model_audit_rejects_missing_expected_grounding_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing-grounding-model"
    _write_manifest(
        run_dir,
        {
            "grounding_inference": {
                "models": [
                    {
                        "transcription": {
                            "cache_hit": False,
                            "model": {"model_id": "asr"},
                        }
                    }
                ]
            }
        },
    )
    settings = PipelineSettings(editorial_engine="heuristic", grounding_engine="open")
    plan = {
        "transcription": {"model_id": "asr"},
        "alignment": {"model_id": "align"},
        "diarization": {"model_id": "diarize"},
    }
    with pytest.raises(RuntimeError, match="missing resolved models") as captured:
        _audit_model_evidence(run_dir, settings, plan)
    assert "align" in str(captured.value)
    assert "diarize" in str(captured.value)
