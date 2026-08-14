import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clipper.cli import _audit_model_evidence, main
from clipper.models import VideoCandidate
from clipper.pipeline import PipelineSettings


def make_brief(tmp_path: Path) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": "c",
                "title": "AI",
                "objective": "Goal",
                "keywords": ["automation"],
                "allowed_video_ids": ["v1"],
                "rights_confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def open_plan() -> dict[str, object]:
    return {
        "editorial_engine": "open",
        "grounding_engine": "open",
        "compute_profile": "balanced",
        "editorial": {"model_id": "editorial-test"},
        "embedding": {"model_id": "embedding-test"},
        "transcription": {"model_id": "asr-test"},
        "alignment": {"model_id": "alignment-test"},
        "diarization": {"model_id": "diarization-test"},
    }


def write_open_manifest(run_dir: Path, *, status: str = "SUCCESS") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "run_metadata": {
            "editorial_inference": {
                "model_invocations": [
                    {
                        "stage": "episode_editorial_profile",
                        "cache_hit": False,
                        "model": {"model_id": "editorial-test"},
                    },
                    {
                        "stage": "semantic_embeddings",
                        "cache_hit": True,
                        "model": {"model_id": "embedding-test"},
                    },
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
                            "cache_hit": True,
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
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_cli_validate(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    assert main(["--verbose", "validate", "--brief", str(path)]) == 0
    assert '"campaign_id": "c"' in capsys.readouterr().out


def test_cli_discover(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    video = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with patch("clipper.cli.YouTubeClient") as client_cls:
        client_cls.return_value.discover.return_value = [video]
        assert main(["discover", "--brief", str(path)]) == 0
    assert '"video_id": "v1"' in capsys.readouterr().out


def test_cli_discover_filters_video_outside_allow_list(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    video = VideoCandidate("v2", "Other", "UC2", "Other Channel", "https://youtu.be/v2")
    with patch("clipper.cli.YouTubeClient") as client_cls:
        client_cls.return_value.discover.return_value = [video]
        assert main(["discover", "--brief", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_cli_run_defaults_to_audited_open_v10(tmp_path: Path, capsys, monkeypatch) -> None:
    path = make_brief(tmp_path)
    run_dir = tmp_path / "run"
    write_open_manifest(run_dir)
    monkeypatch.setenv("CLIPPER_WHISPER_MODEL", "base.en")
    monkeypatch.delenv("CLIPPER_EDITORIAL_ENGINE", raising=False)
    monkeypatch.delenv("CLIPPER_GROUNDING_ENGINE", raising=False)
    monkeypatch.delenv("CLIPPER_COMPUTE_PROFILE", raising=False)
    with (
        patch("clipper.cli._resolved_model_plan", return_value=open_plan()),
        patch("clipper.cli.run_pipeline", return_value=run_dir) as run,
    ):
        assert (
            main(
                [
                    "run",
                    "--brief",
                    str(path),
                    "--artifact-root",
                    str(tmp_path / "artifacts"),
                    "--no-render",
                    "--fresh-inference",
                ]
            )
            == 0
        )
        settings = run.call_args.kwargs["settings"]
        assert run.call_args.kwargs["render"] is False
        assert settings.whisper_model == "base.en"
        assert settings.editorial_engine == "open"
        assert settings.grounding_engine == "open"
        assert settings.compute_profile == "balanced"
        assert settings.cache_root is not None
        assert "_fresh-cache" in str(settings.cache_root)
    assert str(run_dir) in capsys.readouterr().out
    audit = json.loads((run_dir / "model-execution.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["editorial"]["live_invocations"] == 1
    assert audit["editorial"]["cache_hits"] == 1
    assert audit["grounding"]["live_invocations"] == 2
    assert audit["grounding"]["cache_hits"] == 1


def test_cli_refuses_accidental_legacy_and_local_lite(tmp_path: Path, monkeypatch) -> None:
    path = make_brief(tmp_path)
    monkeypatch.setenv("CLIPPER_EDITORIAL_ENGINE", "heuristic")
    monkeypatch.setenv("CLIPPER_GROUNDING_ENGINE", "legacy")
    monkeypatch.setenv("CLIPPER_COMPUTE_PROFILE", "local-lite")
    with patch("clipper.cli.run_pipeline") as run:
        assert main(["run", "--brief", str(path), "--no-render"]) == 1
        run.assert_not_called()

    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "SUCCESS", "run_metadata": {}}), encoding="utf-8"
    )
    with patch("clipper.cli.run_pipeline", return_value=run_dir) as run:
        assert (
            main(
                [
                    "run",
                    "--brief",
                    str(path),
                    "--no-render",
                    "--allow-legacy",
                    "--allow-local-lite",
                ]
            )
            == 0
        )
    settings = run.call_args.kwargs["settings"]
    assert settings.editorial_engine == "heuristic"
    assert settings.grounding_engine == "legacy"
    assert settings.compute_profile == "local-lite"


def test_cli_refuses_local_lite_open_without_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    path = make_brief(tmp_path)
    monkeypatch.delenv("CLIPPER_EDITORIAL_ENGINE", raising=False)
    monkeypatch.delenv("CLIPPER_GROUNDING_ENGINE", raising=False)
    monkeypatch.setenv("CLIPPER_COMPUTE_PROFILE", "local-lite")
    with patch("clipper.cli.run_pipeline") as run:
        assert main(["run", "--brief", str(path), "--no-render"]) == 1
        run.assert_not_called()


def test_model_evidence_is_fail_closed_and_auditable(tmp_path: Path) -> None:
    settings = PipelineSettings(
        editorial_engine="open", grounding_engine="open", compute_profile="balanced"
    )
    run_dir = tmp_path / "evidence"
    write_open_manifest(run_dir)
    audit = _audit_model_evidence(run_dir, settings, open_plan())
    assert audit["editorial"]["observed_models"] == ["editorial-test", "embedding-test"]
    assert audit["grounding"]["observed_models"] == [
        "alignment-test",
        "asr-test",
        "diarization-test",
    ]

    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "manifest.json").write_text(
        json.dumps({"status": "SUCCESS", "run_metadata": {}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="no model invocation evidence"):
        _audit_model_evidence(missing, settings, open_plan())

    no_manifest = tmp_path / "no-manifest"
    no_manifest.mkdir()
    with pytest.raises(RuntimeError, match=r"manifest\.json"):
        _audit_model_evidence(no_manifest, settings, open_plan())


def test_model_evidence_rejects_wrong_model_identity(tmp_path: Path) -> None:
    settings = PipelineSettings(
        editorial_engine="open", grounding_engine="open", compute_profile="balanced"
    )
    run_dir = tmp_path / "wrong-model"
    write_open_manifest(run_dir)
    plan = open_plan()
    plan["editorial"] = {"model_id": "different-editorial"}
    with pytest.raises(RuntimeError, match="does not contain"):
        _audit_model_evidence(run_dir, settings, plan)


def test_cli_run_returns_failure_for_failed_production_manifest(tmp_path: Path) -> None:
    path = make_brief(tmp_path)
    run_dir = tmp_path / "failed-run"
    write_open_manifest(run_dir, status="FAILED")
    with (
        patch("clipper.cli._resolved_model_plan", return_value=open_plan()),
        patch("clipper.cli.run_pipeline", return_value=run_dir),
    ):
        assert main(["run", "--brief", str(path), "--artifact-root", str(tmp_path / "out")]) == 1


def test_cli_error_path(tmp_path: Path) -> None:
    path = make_brief(tmp_path)
    with patch("clipper.cli.load_brief", side_effect=RuntimeError("boom")):
        assert main(["validate", "--brief", str(path)]) == 1


def test_cli_benchmark_writes_report_and_returns_threshold_status(tmp_path: Path, capsys) -> None:
    domains = [
        "gaming",
        "business",
        "comedy_conversational",
        "science_education",
        "interview_personal",
    ]
    episodes = []
    for index, domain in enumerate(domains):
        stories = tmp_path / f"s{index}.json"
        concepts = tmp_path / f"c{index}.json"
        stories.write_text(json.dumps([{"start": 10, "end": 20}]))
        concepts.write_text(json.dumps([{"source_start": 10, "source_end": 20}]))
        episodes.append(
            {
                "episode_id": str(index),
                "domain": domain,
                "references": [
                    {
                        "reference_id": str(index),
                        "start": 10,
                        "end": 20,
                        "semantic_group": str(index),
                    }
                ],
                "predictions": {"story_moments": stories.name, "concepts": concepts.name},
            }
        )
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps({"schema_version": "clipper-benchmark-corpus-v1", "episodes": episodes})
    )
    output = tmp_path / "report.json"
    assert main(["benchmark", "--manifest", str(manifest), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "PASS"
    assert '"status": "PASS"' in capsys.readouterr().out
