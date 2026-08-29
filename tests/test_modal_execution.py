from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper.modal_execution import (
    _acquire_remote_source,
    _deploy,
    _explicit_candidates,
    _function,
    _invoke_remote_with_budget,
    _local_git_sha,
    _materialize_remote_run,
    _positive_budget,
    _validate_model_access,
    _verify_deployed_runtime_sha,
    ensure_modal_runtime,
    run_modal_pipeline,
)
from clipper.models import VideoCandidate


class NotFoundError(RuntimeError):
    pass


class ServiceError(RuntimeError):
    pass


def _write_brief(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign",
                "title": "Podcast",
                "objective": "Find worthwhile complete clips",
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
                "content_constraints": {"min_clip_seconds": 20, "max_clip_seconds": 45},
            }
        ),
        encoding="utf-8",
    )

def test_function_hydrates_deployed_handle() -> None:
    handle = Mock()
    from_name = Mock(return_value=handle)
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=from_name))
    with patch("clipper.modal_execution.importlib.import_module", return_value=modal):
        assert _function("app", "worker") is handle
    from_name.assert_called_once_with("app", "worker")
    handle.hydrate.assert_called_once_with()


def test_function_retries_transient_service_errors() -> None:
    handle = Mock()
    handle.hydrate.side_effect = [ServiceError("unavailable"), ServiceError("unavailable"), None]
    from_name = Mock(return_value=handle)
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=from_name))
    with (
        patch("clipper.modal_execution.time.sleep") as sleep,
        patch("clipper.modal_execution.importlib.import_module", return_value=modal),
    ):
        assert _function("app", "worker") is handle
    assert from_name.call_count == 3
    assert [item.args[0] for item in sleep.call_args_list] == [2.0, 5.0]


def test_deploy_requires_modal_cli_and_existing_source(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    with (
        patch("clipper.modal_execution.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="Modal CLI"),
    ):
        _deploy(script)

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        pytest.raises(RuntimeError, match="deployment source"),
    ):
        _deploy(script)

    script.write_text("# worker\n", encoding="utf-8")
    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution._repo_root", return_value=tmp_path),
        patch("clipper.modal_execution._local_git_sha", return_value="a" * 40),
        patch("clipper.modal_execution.subprocess.run") as run,
    ):
        _deploy(script)
    run.assert_called_once_with(
        ["modal", "deploy", str(script)],
        check=True,
        timeout=1800,
        cwd=tmp_path,
        env=run.call_args.kwargs["env"],
    )
    assert run.call_args.kwargs["env"]["CLIPPER_DEPLOYED_GIT_SHA"] == "a" * 40


def test_deploy_retries_cli_failure(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text("# worker\n", encoding="utf-8")
    failure = subprocess.CalledProcessError(1, ["modal", "deploy", str(script)])
    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution._repo_root", return_value=tmp_path),
        patch("clipper.modal_execution._local_git_sha", return_value="a" * 40),
        patch("clipper.modal_execution.subprocess.run", side_effect=[failure, None]) as run,
        patch("clipper.modal_execution.time.sleep") as sleep,
    ):
        _deploy(script)
    assert run.call_count == 2
    sleep.assert_called_once_with(2.0)


def test_ensure_modal_runtime_attaches_without_deploying_when_apps_exist() -> None:
    with (
        patch("clipper.modal_execution._function", return_value=Mock()) as function,
        patch("clipper.modal_execution._deploy") as deploy,
        patch("clipper.modal_execution._repo_script", side_effect=lambda name: Path(name)),
        patch("clipper.modal_execution._validate_model_access") as validate,
    ):
        ensure_modal_runtime()
    assert function.call_count == 10
    deploy.assert_not_called()
    validate.assert_called_once_with("clipper-open-editor")


def test_ensure_modal_runtime_repairs_missing_model_without_redeploying_pipeline() -> None:
    calls: list[tuple[str, str]] = []
    model_failed = False

    def fake_function(app: str, name: str) -> Mock:
        nonlocal model_failed
        calls.append((app, name))
        if name == "transcribe" and not model_failed:
            model_failed = True
            raise NotFoundError("missing model")
        return Mock()

    with (
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch("clipper.modal_execution._deploy") as deploy,
        patch("clipper.modal_execution._repo_script", side_effect=lambda name: Path(name)),
        patch("clipper.modal_execution._validate_model_access") as validate,
    ):
        ensure_modal_runtime()

    assert model_failed is True
    assert [call.args[0].name for call in deploy.call_args_list] == ["modal_open_models.py"]
    assert ("clipper-open-editor", "transcribe") in calls
    assert ("clipper-open-editor", "hf_access_smoke") in calls
    assert ("clipper-production-pipeline", "run_full_cycle") in calls
    validate.assert_called_once_with("clipper-open-editor")


def test_ensure_modal_runtime_repairs_missing_pipeline_only() -> None:
    calls: list[tuple[str, str]] = []
    pipeline_failed = False

    def fake_function(app: str, name: str) -> Mock:
        nonlocal pipeline_failed
        calls.append((app, name))
        if app == "clipper-production-pipeline" and name == "acquire_source" and not pipeline_failed:
            pipeline_failed = True
            raise NotFoundError("missing pipeline")
        return Mock()

    with (
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch("clipper.modal_execution._deploy") as deploy,
        patch("clipper.modal_execution._repo_script", side_effect=lambda name: Path(name)),
        patch("clipper.modal_execution._validate_model_access") as validate,
    ):
        ensure_modal_runtime()

    assert pipeline_failed is True
    deploy.assert_called_once_with(Path("modal_pipeline.py"))
    assert ("clipper-open-editor", "vision") in calls
    assert ("clipper-production-pipeline", "run_full_cycle") in calls
    validate.assert_called_once_with("clipper-open-editor")


def test_ensure_modal_runtime_does_not_redeploy_on_connectivity_failure() -> None:
    with (
        patch("clipper.modal_execution._function", side_effect=ServiceError("unavailable")),
        patch("clipper.modal_execution._deploy") as deploy,
        pytest.raises(RuntimeError, match="control-plane validation failed"),
    ):
        ensure_modal_runtime()
    deploy.assert_not_called()


def test_ensure_modal_runtime_fails_closed_after_unsuccessful_redeploy() -> None:
    with (
        patch(
            "clipper.modal_execution._function",
            side_effect=[NotFoundError("missing"), RuntimeError("still missing")],
        ),
        patch("clipper.modal_execution._deploy"),
        pytest.raises(RuntimeError, match="unavailable after runtime repair"),
    ):
        ensure_modal_runtime()


def test_validate_model_access_requires_successful_remote_smoke() -> None:
    smoke = Mock()
    smoke.remote.return_value = {
        "ok": True,
        "model_id": "pyannote/speaker-diarization-community-1",
        "revision": "revision",
    }
    with patch("clipper.modal_execution._function", return_value=smoke) as function:
        _validate_model_access("clipper-open-editor")
    function.assert_called_once_with("clipper-open-editor", "hf_access_smoke")

    smoke.remote.return_value = {"ok": False}
    with (
        patch("clipper.modal_execution._function", return_value=smoke),
        pytest.raises(RuntimeError, match="invalid result"),
    ):
        _validate_model_access("clipper-open-editor")

    smoke.remote.side_effect = RuntimeError("denied")
    with (
        patch("clipper.modal_execution._function", return_value=smoke),
        pytest.raises(RuntimeError, match="Hugging Face access preflight failed"),
    ):
        _validate_model_access("clipper-open-editor")


def test_explicit_candidates_resolve_only_campaign_targets(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidates = _explicit_candidates(brief_path)
    assert [item.video_id for item in candidates] == ["v1"]
    assert candidates[0].url == "https://www.youtube.com/watch?v=v1"
    assert candidates[0].channel_id == "UC1"

def test_explicit_candidate_keeps_youtube_url_when_supplemental_media_url_exists(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    payload = json.loads(brief_path.read_text(encoding="utf-8"))
    payload["targets"]["videos"][0]["media_url"] = (
        "https://drive.google.com/file/d/supplemental/view"
    )
    brief_path.write_text(json.dumps(payload), encoding="utf-8")

    candidates = _explicit_candidates(brief_path)
    assert candidates[0].url == "https://www.youtube.com/watch?v=v1"


def test_acquire_remote_source_uses_modal_egress_and_validates_quality() -> None:
    function = Mock()
    remote = Mock(
        return_value={
            "video_id": "v1",
            "channel_id": "UC1",
            "quality_policy": "highest_available_no_transcode",
            "bytes": 123,
            "sha256": "abc",
        }
    )
    function.with_options.return_value.remote = remote
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    result = _acquire_remote_source(function, candidate, expected_git_sha="a" * 40)
    assert result["sha256"] == "abc"
    function.with_options.assert_called_once_with(cloud="gcp", timeout=1800)
    remote.assert_called_once_with(
        {
            "video_id": "v1",
            "channel_id": "UC1",
            "video_url": "https://youtu.be/v1",
            "expected_git_sha": "a" * 40,
        }
    )


def test_acquire_remote_source_exhausts_invalid_and_failed_egress() -> None:
    class AlwaysBad:
        def with_options(self, **_kwargs: object) -> AlwaysBad:
            return self

        def remote(self, _payload: dict[str, str]) -> object:
            raise RuntimeError("blocked")

    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(AlwaysBad(), candidate, expected_git_sha="a" * 40)

    function = Mock()
    function.with_options.return_value.remote.return_value = {
        "video_id": "v1",
        "channel_id": "UC1",
        "quality_policy": "downgraded",
    }
    function.remote.return_value = "invalid"
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(function, candidate, expected_git_sha="a" * 40)


def test_acquire_remote_source_rejects_wrong_resolved_identity() -> None:
    function = Mock()
    function.with_options.return_value.remote.return_value = {
        "video_id": "v1",
        "channel_id": "UC-other",
        "quality_policy": "highest_available_no_transcode",
    }
    function.remote.return_value = {
        "video_id": "other",
        "channel_id": "UC1",
        "quality_policy": "highest_available_no_transcode",
    }
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(function, candidate, expected_git_sha="a" * 40)


def test_materialize_remote_run_downloads_only_artifact_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"

    def fake_run(command: list[str], **_kwargs: object) -> None:
        staging = Path(command[-1])
        downloaded = staging / "campaign-run"
        downloaded.mkdir(parents=True)
        (downloaded / "manifest.json").write_text("{}", encoding="utf-8")
        (downloaded / "clip.mp4").write_bytes(b"clip")

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution.subprocess.run", side_effect=fake_run) as run,
    ):
        result = _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="artifacts-volume",
            remote_run_path="/campaign-run",
        )

    assert result == artifact_root / "campaign-run"
    assert (result / "manifest.json").is_file()
    assert (result / "clip.mp4").read_bytes() == b"clip"
    command = run.call_args.args[0]
    assert command[:5] == ["modal", "volume", "get", "--force", "artifacts-volume"]
    child_env = run.call_args.kwargs["env"]
    assert child_env["PYTHONUTF8"] == "1"
    assert child_env["PYTHONIOENCODING"] == "utf-8"


def test_materialize_remote_run_handles_flat_download_and_bad_targets(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    with (
        patch("clipper.modal_execution.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="Modal CLI"),
    ):
        _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="volume",
            remote_run_path="/run",
        )

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        pytest.raises(RuntimeError, match="invalid remote run path"),
    ):
        _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="volume",
            remote_run_path="/",
        )

    (artifact_root / "run").mkdir(parents=True)
    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        pytest.raises(RuntimeError, match="overwrite"),
    ):
        _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="volume",
            remote_run_path="/run",
        )

    flat_root = tmp_path / "flat-artifacts"

    def flat_run(command: list[str], **_kwargs: object) -> None:
        staging = Path(command[-1])
        (staging / "manifest.json").write_text("{}", encoding="utf-8")
        (staging / "evidence.json").write_text("{}", encoding="utf-8")

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution.subprocess.run", side_effect=flat_run),
    ):
        result = _materialize_remote_run(
            artifact_root=flat_root,
            volume_name="volume",
            remote_run_path="/flat-run",
        )
    assert (result / "manifest.json").is_file()
    assert (result / "evidence.json").is_file()


def test_materialize_remote_run_rejects_ambiguous_download(tmp_path: Path) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> None:
        staging = Path(command[-1])
        (staging / "a").mkdir()
        (staging / "b").mkdir()
        (staging / "a" / "manifest.json").write_text("{}", encoding="utf-8")
        (staging / "b" / "manifest.json").write_text("{}", encoding="utf-8")

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution.subprocess.run", side_effect=fake_run),
        pytest.raises(RuntimeError, match="expected one manifest"),
    ):
        _materialize_remote_run(
            artifact_root=tmp_path / "artifacts",
            volume_name="volume",
            remote_run_path="/run",
        )


def test_run_modal_pipeline_acquires_in_modal_runs_remote_and_materializes(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()
    remote_result = {
        "execution_id": "e" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "/campaign-run",
        "run_volume": "clipper-production-artifacts",
    }
    materialized = tmp_path / "artifacts" / "campaign-run"

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    with (
        patch("clipper.modal_execution.ensure_modal_runtime") as ensure,
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ) as acquire_remote,
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=remote_result,
        ) as invoke,
        patch(
            "clipper.modal_execution._materialize_remote_run",
            return_value=materialized,
        ) as download,
    ):
        result = run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id="old-run",
            render=True,
            fresh_inference=True,
        )

    assert result == materialized
    ensure.assert_called_once_with()
    acquire_remote.assert_called_once_with(
        acquire,
        candidate,
        expected_git_sha="a" * 40,
    )
    assert invoke.call_args.args[0] is runner
    payload = invoke.call_args.args[1]
    assert invoke.call_args.kwargs["max_gpu_seconds"] == 21600.0
    assert invoke.call_args.kwargs["max_estimated_usd"] == 10.0
    assert payload["resume_from_run_id"] == "old-run"
    assert payload["render"] is True
    assert payload["fresh_inference"] is True
    assert payload["git_sha"] == "a" * 40
    assert payload["execution_id"] == "e" * 32
    assert payload["max_gpu_seconds"] == 21600.0
    assert payload["max_estimated_usd"] == 10.0
    download.assert_called_once_with(
        artifact_root=tmp_path / "artifacts",
        volume_name="clipper-production-artifacts",
        remote_run_path="/campaign-run",
    )


def test_run_modal_pipeline_fails_closed_for_runtime_empty_targets_and_bad_runner(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")

    with (
        patch("clipper.modal_execution.ensure_modal_runtime", side_effect=RuntimeError("model access denied")),
        patch("clipper.modal_execution._function") as function,
        pytest.raises(RuntimeError, match="model access denied"),
    ):
        run_modal_pipeline(
            brief_path, artifact_root=tmp_path / "artifacts", resume_from_run_id=None,
            render=True, fresh_inference=False,
        )
    function.assert_not_called()

    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._function", return_value=Mock()),
        patch("clipper.modal_execution._explicit_candidates", return_value=[]),
        pytest.raises(RuntimeError, match="no explicit authorized targets"),
    ):
        run_modal_pipeline(
            brief_path, artifact_root=tmp_path / "artifacts-empty", resume_from_run_id=None,
            render=True, fresh_inference=False,
        )

    acquire = Mock()
    runner = Mock()
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=lambda _app, name: acquire if name == "acquire_source" else runner),
        patch("clipper.modal_execution._acquire_remote_source", return_value={"quality_policy": "highest_available_no_transcode"}),
        patch("clipper.modal_execution._invoke_remote_with_budget", return_value="invalid"),
        pytest.raises(RuntimeError, match="invalid response"),
    ):
        run_modal_pipeline(
            brief_path, artifact_root=tmp_path / "artifacts-invalid", resume_from_run_id=None,
            render=True, fresh_inference=False,
        )

    no_path_result = {
        "execution_id": "e" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "",
    }
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=lambda _app, name: acquire if name == "acquire_source" else runner),
        patch("clipper.modal_execution._acquire_remote_source", return_value={"quality_policy": "highest_available_no_transcode"}),
        patch("clipper.modal_execution._invoke_remote_with_budget", return_value=no_path_result),
        pytest.raises(RuntimeError, match="no run path"),
    ):
        run_modal_pipeline(
            brief_path, artifact_root=tmp_path / "artifacts-no-path", resume_from_run_id=None,
            render=True, fresh_inference=False,
        )




def test_verify_deployed_runtime_sha_requires_both_apps_match_local_checkout() -> None:
    expected = "a" * 40
    model = Mock()
    model.remote.return_value = {"deployed_git_sha": expected}
    pipeline = Mock()
    pipeline.remote.return_value = {"deployed_git_sha": expected}

    def function(app: str, name: str) -> Mock:
        assert name == "deployment_identity"
        return model if app == "model-app" else pipeline

    with (
        patch("clipper.modal_execution._local_git_sha", return_value=expected),
        patch("clipper.modal_execution._function", side_effect=function),
    ):
        assert (
            _verify_deployed_runtime_sha(model_app="model-app", pipeline_app="pipeline-app")
            == expected
        )

    pipeline.remote.return_value = {"deployed_git_sha": "b" * 40}
    with (
        patch("clipper.modal_execution._local_git_sha", return_value=expected),
        patch("clipper.modal_execution._function", side_effect=function),
        pytest.raises(RuntimeError, match="pipeline deployed SHA mismatch"),
    ):
        _verify_deployed_runtime_sha(model_app="model-app", pipeline_app="pipeline-app")


def test_source_acquisition_rejects_missing_exact_sha_before_remote() -> None:
    function = Mock()
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with pytest.raises(ValueError, match="full expected_git_sha"):
        _acquire_remote_source(function, candidate, expected_git_sha="")
    function.with_options.assert_not_called()
    function.remote.assert_not_called()


def test_local_git_sha_requires_full_hex_checkout() -> None:
    completed = SimpleNamespace(stdout="A" * 40 + "\n")
    with (
        patch("clipper.modal_execution._repo_root", return_value=Path("/repo")),
        patch("clipper.modal_execution.subprocess.run", return_value=completed) as run,
    ):
        assert _local_git_sha() == "a" * 40
    run.assert_called_once_with(
        ["git", "rev-parse", "HEAD"],
        cwd=Path("/repo"),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    with (
        patch("clipper.modal_execution._repo_root", return_value=Path("/repo")),
        patch(
            "clipper.modal_execution.subprocess.run",
            return_value=SimpleNamespace(stdout="not-a-sha\n"),
        ),
        pytest.raises(RuntimeError, match="full git SHA"),
    ):
        _local_git_sha()


def test_verify_deployed_runtime_sha_rejects_invalid_identity_object() -> None:
    handle = Mock()
    handle.remote.return_value = "invalid"
    with (
        patch("clipper.modal_execution._local_git_sha", return_value="a" * 40),
        patch("clipper.modal_execution._function", return_value=handle),
        pytest.raises(RuntimeError, match="deployment identity is not an object"),
    ):
        _verify_deployed_runtime_sha(model_app="model-app", pipeline_app="pipeline-app")


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_positive_budget_rejects_nonfinite_or_nonpositive_values(value: float) -> None:
    assert _positive_budget(1.5, name="budget") == 1.5
    with pytest.raises(ValueError, match="budget must be finite and positive"):
        _positive_budget(value, name="budget")


def test_run_modal_pipeline_rejects_mismatched_execution_or_sha_before_download(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    wrong_execution_result = {
        "execution_id": "f" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "/run",
    }
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ),
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=wrong_execution_result,
        ),
        patch("clipper.modal_execution._materialize_remote_run") as download,
        pytest.raises(RuntimeError, match="mismatched execution ID"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts-execution",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )
    download.assert_not_called()

    wrong_sha_result = {
        "execution_id": "e" * 32,
        "deployed_git_sha": "b" * 40,
        "run_path": "/run",
    }
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ),
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=wrong_sha_result,
        ),
        patch("clipper.modal_execution._materialize_remote_run") as download,
        pytest.raises(RuntimeError, match="mismatched deployed SHA"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts-sha",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )
    download.assert_not_called()



def test_invoke_remote_with_budget_cancels_exact_call_while_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "0.1")

    class Call:
        def __init__(self) -> None:
            self.cancel_args: list[bool] = []

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            assert timeout == 0.1
            clock["now"] = 1.0
            raise TimeoutError

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_args.append(terminate_containers)

    call = Call()
    function = SimpleNamespace(spawn=Mock(return_value=call))

    with pytest.raises(RuntimeError, match="in-flight compute budget"):
        _invoke_remote_with_budget(
            function,
            {"request": True},
            max_gpu_seconds=1.0,
            max_estimated_usd=100.0,
        )

    function.spawn.assert_called_once_with({"request": True})
    assert call.cancel_args == [False]


def test_invoke_remote_with_budget_caps_poll_to_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5")

    class Call:
        def __init__(self) -> None:
            self.timeouts: list[float] = []
            self.cancel_args: list[bool] = []

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            self.timeouts.append(timeout)
            clock["now"] = timeout
            raise TimeoutError

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_args.append(terminate_containers)

    call = Call()
    function = SimpleNamespace(spawn=Mock(return_value=call))

    with pytest.raises(RuntimeError, match="in-flight compute budget"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=1.0,
            max_estimated_usd=100.0,
        )

    assert call.timeouts == [pytest.approx(0.5)]
    assert call.cancel_args == [False]


def test_invoke_remote_with_budget_cancels_if_hydration_fails() -> None:
    call = Mock()
    call.hydrate.side_effect = ServiceError("hydrate failed")
    function = SimpleNamespace(spawn=Mock(return_value=call))

    with pytest.raises(ServiceError, match="hydrate failed"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )

    call.cancel.assert_called_once_with(terminate_containers=False)


def test_invoke_remote_with_budget_cancels_non_timeout_poll_failure() -> None:
    call = Mock()
    call.get.side_effect = ServiceError("poll failed")
    function = SimpleNamespace(spawn=Mock(return_value=call))

    with pytest.raises(ServiceError, match="poll failed"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )

    call.cancel.assert_called_once_with(terminate_containers=False)


def test_invoke_remote_with_budget_rechecks_successful_final_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "0.1")

    class Call:
        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            assert timeout == 0.1
            clock["now"] = 1.0
            return {"status": "PASS"}

        def cancel(self, *, terminate_containers: bool) -> None:
            raise AssertionError(f"completed call must not be cancelled: {terminate_containers}")

    function = SimpleNamespace(spawn=Mock(return_value=Call()))
    with pytest.raises(RuntimeError, match="final poll"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=1.0,
            max_estimated_usd=100.0,
        )


def test_invoke_remote_with_budget_rejects_nonfinite_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = SimpleNamespace(spawn=Mock())
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "nan")
    with pytest.raises(ValueError, match="finite and positive"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )
    function.spawn.assert_not_called()


def test_failed_runner_response_is_authenticated_and_materialized(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()
    failed_local = tmp_path / "artifacts" / "failed-run"
    failed_result = {
        "status": "FAIL",
        "execution_mode": "content-addressed-resume",
        "execution_id": "e" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "/failed-run",
        "run_volume": "clipper-production-artifacts",
    }

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ),
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=failed_result,
        ),
        patch(
            "clipper.modal_execution._materialize_remote_run",
            return_value=failed_local,
        ) as materialize,
    ):
        result = run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    assert result == failed_local
    materialize.assert_called_once_with(
        artifact_root=tmp_path / "artifacts",
        volume_name="clipper-production-artifacts",
        remote_run_path="/failed-run",
    )
