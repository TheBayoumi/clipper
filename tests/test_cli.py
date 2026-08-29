import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clipper.cli import _audit_model_evidence, _seed_resume_source_cache, main
from clipper.models import VideoCandidate
from clipper.pipeline import PipelineSettings


def make_brief(tmp_path: Path) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": "c",
                "title": "AI",
                "objective": "Find every independently worthwhile moment.",
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
    return path


def local_plan(*, profile: str = "balanced") -> dict[str, object]:
    return {
        "architecture": "autonomous-multimodal-quality-graph",
        "compute_profile": profile,
        "editorial": {
            "model_id": "editorial-test",
            "revision": "r1",
            "quantization": "test",
            "inference_engine": "local-test",
        },
        "transcription": {
            "model_id": "asr-test",
            "revision": "r1",
            "quantization": "test",
            "inference_engine": "local-test",
        },
        "alignment": {
            "model_id": "alignment-test",
            "revision": "r1",
            "quantization": "test",
            "inference_engine": "local-test",
        },
        "diarization": {
            "model_id": "diarization-test",
            "revision": "r1",
            "quantization": "test",
            "inference_engine": "local-test",
        },
    }


def write_model_manifest(run_dir: Path, *, status: str = "SUCCESS") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "run_metadata": {
            "editorial_inference": {
                "model_invocations": [
                    {
                        "stage": "semantic_cores:v1",
                        "cache_hit": False,
                        "model": {"model_id": "editorial-test"},
                    },
                    {
                        "stage": "quality_windows:core-1",
                        "cache_hit": True,
                        "model": {"model_id": "editorial-test"},
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


def test_cli_validate_explicit_target_brief(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    assert main(["--verbose", "validate", "--brief", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["campaign_id"] == "c"
    assert output["allowed_video_ids"] == ["v1"]
    assert "keywords" not in output


def test_cli_discover_is_separate_from_production_brief(tmp_path: Path, capsys) -> None:
    del tmp_path
    video = VideoCandidate("v2", "Title", "UC2", "Channel", "https://youtu.be/v2")
    with patch("clipper.cli.YouTubeClient") as client_cls:
        client_cls.return_value.discover.return_value = [video]
        assert (
            main(
                [
                    "discover",
                    "--query",
                    "podcast",
                    "--channel-id",
                    "UC2",
                    "--limit",
                    "4",
                ]
            )
            == 0
        )
    request = client_cls.return_value.discover.call_args.args[0]
    assert request.query == "podcast"
    assert request.channel_ids == ("UC2",)
    assert request.limit == 4
    assert json.loads(capsys.readouterr().out)[0]["video_id"] == "v2"


def test_cli_run_uses_autonomous_quality_graph_and_fresh_cache(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    run_dir = tmp_path / "run"
    write_model_manifest(run_dir)
    with (
        patch("clipper.cli._resolved_model_plan", return_value=local_plan()),
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
    assert settings.compute_profile == "balanced"
    assert settings.cache_root is not None
    assert "_fresh-cache" in str(settings.cache_root)
    assert str(run_dir) in capsys.readouterr().out
    audit = json.loads((run_dir / "model-execution.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["resolved_plan"]["architecture"] == "autonomous-multimodal-quality-graph"
    assert audit["editorial"]["live_invocations"] == 1
    assert audit["editorial"]["cache_hits"] == 1
    assert audit["grounding"]["live_invocations"] == 2
    assert audit["grounding"]["cache_hits"] == 1


def test_resume_reuses_interrupted_source_master(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    previous = artifact_root / "c-20260814T102639Z"
    source_dir = previous / "work" / "v1"
    source_dir.mkdir(parents=True)
    source = source_dir / "v1.mkv"
    source.write_bytes(b"source-master")
    source.with_suffix(".source.json").write_text('{"quality":"source"}', encoding="utf-8")

    settings = PipelineSettings(artifact_root=artifact_root)
    recovered = _seed_resume_source_cache(settings, previous.name, campaign_id="c")

    assert recovered == previous.resolve()
    cached = artifact_root / "_source-media-cache" / "v1" / "v1.mkv"
    assert cached.read_bytes() == b"source-master"
    assert json.loads(cached.with_suffix(".source.json").read_text()) == {"quality": "source"}


def test_cli_resume_accepts_run_id_and_seeds_before_local_pipeline(tmp_path: Path) -> None:
    path = make_brief(tmp_path)
    artifact_root = tmp_path / "artifacts"
    previous = artifact_root / "c-20260814T102639Z"
    source_dir = previous / "work" / "v1"
    source_dir.mkdir(parents=True)
    (source_dir / "v1.mkv").write_bytes(b"source-master")

    run_dir = tmp_path / "continued-run"
    write_model_manifest(run_dir)
    with (
        patch("clipper.cli._resolved_model_plan", return_value=local_plan()),
        patch("clipper.cli.run_pipeline", return_value=run_dir) as run,
    ):
        assert (
            main(
                [
                    "run",
                    "--brief",
                    str(path),
                    "--artifact-root",
                    str(artifact_root),
                    "--resume",
                    previous.name,
                    "--no-render",
                ]
            )
            == 0
        )
    assert (artifact_root / "_source-media-cache" / "v1" / "v1.mkv").is_file()
    run.assert_called_once()


def test_resume_rejects_invalid_or_completed_run(tmp_path: Path) -> None:
    settings = PipelineSettings(artifact_root=tmp_path / "artifacts")
    with pytest.raises(RuntimeError, match="run ID"):
        _seed_resume_source_cache(settings, "../escape", campaign_id="c")

    completed = settings.artifact_root / "c-20260814T102639Z"
    write_model_manifest(completed, status="SUCCESS")
    with pytest.raises(RuntimeError, match="already completed successfully"):
        _seed_resume_source_cache(settings, completed.name, campaign_id="c")


def test_cli_refuses_local_lite_without_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    path = make_brief(tmp_path)
    monkeypatch.setenv("CLIPPER_COMPUTE_PROFILE", "local-lite")
    with patch("clipper.cli.run_pipeline") as run:
        assert main(["run", "--brief", str(path), "--no-render"]) == 1
        run.assert_not_called()


def test_cli_allows_local_lite_only_with_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    path = make_brief(tmp_path)
    monkeypatch.setenv("CLIPPER_COMPUTE_PROFILE", "local-lite")
    run_dir = tmp_path / "local-lite"
    write_model_manifest(run_dir)
    with (
        patch("clipper.cli._resolved_model_plan", return_value=local_plan(profile="local-lite")),
        patch("clipper.cli.run_pipeline", return_value=run_dir) as run,
    ):
        assert (
            main(
                [
                    "run",
                    "--brief",
                    str(path),
                    "--no-render",
                    "--allow-local-lite",
                ]
            )
            == 0
        )
    assert run.call_args.kwargs["settings"].compute_profile == "local-lite"


def test_model_evidence_is_fail_closed_and_auditable(tmp_path: Path) -> None:
    run_dir = tmp_path / "evidence"
    write_model_manifest(run_dir)
    audit = _audit_model_evidence(run_dir, local_plan())
    assert audit["editorial"]["observed_models"] == ["editorial-test"]
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
        _audit_model_evidence(missing, local_plan())

    no_manifest = tmp_path / "no-manifest"
    no_manifest.mkdir()
    with pytest.raises(RuntimeError, match=r"manifest\.json"):
        _audit_model_evidence(no_manifest, local_plan())


def test_model_evidence_rejects_wrong_model_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "wrong-model"
    write_model_manifest(run_dir)
    plan = local_plan()
    plan["editorial"] = {"model_id": "different-editorial"}
    with pytest.raises(RuntimeError, match="does not contain"):
        _audit_model_evidence(run_dir, plan)


def test_cli_run_returns_failure_for_failed_production_manifest(tmp_path: Path) -> None:
    path = make_brief(tmp_path)
    run_dir = tmp_path / "failed-run"
    write_model_manifest(run_dir, status="FAILED")
    with (
        patch("clipper.cli._resolved_model_plan", return_value=local_plan()),
        patch("clipper.cli.run_pipeline", return_value=run_dir),
    ):
        assert main(["run", "--brief", str(path), "--artifact-root", str(tmp_path / "out")]) == 1



def test_cli_run_returns_failure_for_failed_analysis_only_manifest(tmp_path: Path) -> None:
    path = make_brief(tmp_path)
    run_dir = tmp_path / "failed-analysis"
    write_model_manifest(run_dir, status="FAILED")
    with (
        patch("clipper.cli._resolved_model_plan", return_value=local_plan()),
        patch("clipper.cli.run_pipeline", return_value=run_dir),
    ):
        assert (
            main(
                [
                    "run",
                    "--brief",
                    str(path),
                    "--artifact-root",
                    str(tmp_path / "out"),
                    "--no-render",
                ]
            )
            == 1
        )


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
