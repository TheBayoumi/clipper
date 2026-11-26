from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from clipper.cli import main


def _write_brief(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "campaign_id": "c",
                "title": "Podcast",
                "objective": "Find independently worthwhile clips",
                "targets": {
                    "mode": "explicit",
                    "videos": [
                        {
                            "video_id": "v1",
                            "url": "https://www.youtube.com/watch?v=v1",
                            "channel_id": "UC1",
                        }
                    ],
                },
                "rights": {"confirmed": True, "authorized_channels": ["UC1"]},
            }
        ),
        encoding="utf-8",
    )


def _modal_plan() -> dict[str, object]:
    return {
        "architecture": "autonomous-multimodal-quality-graph",
        "compute_profile": "balanced",
        "editorial": {
            "model_id": "editorial-test",
            "inference_engine": "modal-transformers",
        },
        "transcription": {
            "model_id": "asr-test",
            "inference_engine": "modal-faster-whisper",
        },
        "alignment": {
            "model_id": "alignment-test",
            "inference_engine": "modal-whisperx",
        },
        "diarization": {
            "model_id": "diarization-test",
            "inference_engine": "modal-pyannote",
        },
    }


def _write_manifest(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "run_metadata": {
                    "editorial_inference": {
                        "model_invocations": [
                            {
                                "cache_hit": False,
                                "model": {"model_id": "editorial-test"},
                            }
                        ]
                    },
                    "grounding_inference": {
                        "models": [
                            {
                                "video_id": "v1",
                                "transcription": {
                                    "cache_hit": False,
                                    "model": {"model_id": "asr-test"},
                                },
                                "alignment": {
                                    "cache_hit": False,
                                    "model": {"model_id": "alignment-test"},
                                },
                                "diarization": {
                                    "cache_hit": False,
                                    "model": {"model_id": "diarization-test"},
                                },
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_modal_resume_ignores_previous_local_source_and_dispatches_cloud(tmp_path: Path) -> None:
    brief = tmp_path / "brief.json"
    _write_brief(brief)
    artifact_root = tmp_path / "artifacts"
    previous = artifact_root / "c-20260814T102639Z"
    previous.mkdir(parents=True)
    assert not list(previous.rglob("*.mkv"))

    remote_run = artifact_root / "c-remote-run"
    _write_manifest(remote_run)

    with (
        patch("clipper.cli._resolved_model_plan", return_value=_modal_plan()),
        patch("clipper.cli._assert_runtime_dependencies"),
        patch("clipper.cli._assert_modal_functions_available"),
        patch("clipper.cli._seed_resume_source_cache") as seed,
        patch("clipper.cli.run_pipeline") as local_pipeline,
        patch("clipper.cli.run_modal_pipeline", return_value=remote_run) as modal_pipeline,
    ):
        result = main(
            [
                "run",
                "--brief",
                str(brief),
                "--artifact-root",
                str(artifact_root),
                "--resume",
                previous.name,
            ]
        )

    assert result == 0
    seed.assert_not_called()
    local_pipeline.assert_not_called()
    modal_pipeline.assert_called_once_with(
        brief,
        artifact_root=artifact_root,
        resume_from_run_id=previous.name,
        render=True,
        fresh_inference=False,
    )


def test_fixture_mode_keeps_explicit_local_pipeline_fallback(tmp_path: Path, monkeypatch) -> None:
    brief = tmp_path / "brief.json"
    _write_brief(brief)
    artifact_root = tmp_path / "artifacts"
    local_run = artifact_root / "c-local-run"
    _write_manifest(local_run)
    monkeypatch.setenv("CLIPPER_SOURCE_FIXTURE_DIR", str(tmp_path / "fixture"))

    with (
        patch("clipper.cli._resolved_model_plan", return_value=_modal_plan()),
        patch("clipper.cli._assert_runtime_dependencies"),
        patch("clipper.cli._assert_modal_functions_available"),
        patch("clipper.cli._source_client_for_run", return_value=None),
        patch("clipper.cli.run_pipeline", return_value=local_run) as local_pipeline,
        patch("clipper.cli.run_modal_pipeline") as modal_pipeline,
    ):
        assert (
            main(
                [
                    "run",
                    "--brief",
                    str(brief),
                    "--artifact-root",
                    str(artifact_root),
                ]
            )
            == 0
        )

    local_pipeline.assert_called_once()
    modal_pipeline.assert_not_called()
