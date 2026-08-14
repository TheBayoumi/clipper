from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper.modal_execution import (
    _acquire_remote_source,
    _authorized_candidates,
    _deploy,
    _function,
    _materialize_remote_run,
    ensure_modal_runtime,
    run_modal_pipeline,
)
from clipper.models import CampaignBrief, VideoCandidate


def _brief(*, allowed: list[str] | None = None, channels: list[str] | None = None) -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "campaign",
            "title": "Podcast",
            "objective": "Find clips",
            "keywords": ["podcast"],
            "allowed_video_ids": allowed or [],
            "source_channel_ids": channels or [],
            "rights_confirmed": True,
            "source_limit": 2,
        }
    )


def _write_brief(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign",
                "title": "Podcast",
                "objective": "Find clips",
                "keywords": ["podcast"],
                "allowed_video_ids": ["v1"],
                "source_channel_ids": ["UC1"],
                "rights_confirmed": True,
                "source_limit": 1,
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
        patch("clipper.modal_execution.subprocess.run") as run,
    ):
        _deploy(script)
    run.assert_called_once_with(["modal", "deploy", str(script)], check=True, timeout=1800)


def test_ensure_modal_runtime_skips_deploy_when_every_function_hydrates() -> None:
    with (
        patch("clipper.modal_execution._function", return_value=Mock()) as function,
        patch("clipper.modal_execution._deploy") as deploy,
    ):
        ensure_modal_runtime()
    assert function.call_count == 8
    deploy.assert_not_called()


def test_ensure_modal_runtime_repairs_missing_model_and_pipeline_workers() -> None:
    calls: list[tuple[str, str]] = []
    failed = {"model": False, "pipeline": False}

    def fake_function(app: str, name: str) -> Mock:
        calls.append((app, name))
        if name == "transcribe" and not failed["model"]:
            failed["model"] = True
            raise RuntimeError("missing model")
        if name == "acquire_source" and not failed["pipeline"]:
            failed["pipeline"] = True
            raise RuntimeError("missing pipeline")
        return Mock()

    with (
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch("clipper.modal_execution._deploy") as deploy,
        patch("clipper.modal_execution._repo_script", side_effect=lambda name: Path(name)),
    ):
        ensure_modal_runtime()

    assert failed == {"model": True, "pipeline": True}
    assert [call.args[0].name for call in deploy.call_args_list] == [
        "modal_open_models.py",
        "modal_v10_cycle.py",
    ]
    assert ("clipper-open-editor", "transcribe") in calls
    assert ("clipper-v10-cycle", "run_full_cycle") in calls


def test_ensure_modal_runtime_fails_closed_after_unsuccessful_redeploy() -> None:
    with (
        patch("clipper.modal_execution._function", side_effect=RuntimeError("still missing")),
        patch("clipper.modal_execution._deploy"),
        pytest.raises(RuntimeError, match="unavailable after deploy"),
    ):
        ensure_modal_runtime()


def test_authorized_candidates_build_direct_youtube_requests_without_download() -> None:
    brief = _brief(allowed=["v1", "v2"], channels=["UC1"])
    candidates = _authorized_candidates(brief)
    assert [item.video_id for item in candidates] == ["v1", "v2"]
    assert candidates[0].url == "https://www.youtube.com/watch?v=v1"
    assert candidates[0].channel_id == "UC1"


def test_authorized_candidates_uses_discovery_only_when_ids_are_absent() -> None:
    brief = _brief(channels=["UC1"])
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with patch("clipper.modal_execution.YouTubeClient") as client:
        client.return_value.discover.return_value = [candidate]
        assert _authorized_candidates(brief) == [candidate]
    client.return_value.discover.assert_called_once_with(brief)


def test_acquire_remote_source_uses_modal_egress_and_validates_quality() -> None:
    function = Mock()
    remote = Mock(
        return_value={
            "quality_policy": "highest_available_no_transcode",
            "bytes": 123,
            "sha256": "abc",
        }
    )
    function.with_options.return_value.remote = remote
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    result = _acquire_remote_source(function, candidate)
    assert result["sha256"] == "abc"
    function.with_options.assert_called_once_with(cloud="gcp", timeout=1800)
    remote.assert_called_once_with({"video_id": "v1", "video_url": "https://youtu.be/v1"})


def test_acquire_remote_source_exhausts_invalid_and_failed_egress() -> None:
    class AlwaysBad:
        def with_options(self, **_kwargs: object) -> AlwaysBad:
            return self

        def remote(self, _payload: dict[str, str]) -> object:
            raise RuntimeError("blocked")

    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(AlwaysBad(), candidate)

    function = Mock()
    function.with_options.return_value.remote.return_value = {"quality_policy": "downgraded"}
    function.remote.return_value = "invalid"
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(function, candidate)


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


def test_materialize_remote_run_handles_flat_download_and_rejects_bad_targets(tmp_path: Path) -> None:
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

    shutil_root = tmp_path / "flat-artifacts"

    def flat_run(command: list[str], **_kwargs: object) -> None:
        staging = Path(command[-1])
        (staging / "manifest.json").write_text("{}", encoding="utf-8")
        (staging / "evidence.json").write_text("{}", encoding="utf-8")

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution.subprocess.run", side_effect=flat_run),
    ):
        result = _materialize_remote_run(
            artifact_root=shutil_root,
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
    runner.remote.return_value = {
        "run_path": "/campaign-run",
        "run_volume": "clipper-v10-artifacts",
    }
    materialized = tmp_path / "artifacts" / "campaign-run"

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    with (
        patch("clipper.modal_execution.ensure_modal_runtime") as ensure,
        patch("clipper.modal_execution._authorized_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ) as acquire_remote,
        patch("clipper.modal_execution._materialize_remote_run", return_value=materialized) as download,
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
    acquire_remote.assert_called_once_with(acquire, candidate)
    payload = runner.remote.call_args.args[0]
    assert payload["resume_from_run_id"] == "old-run"
    assert payload["render"] is True
    assert payload["fresh_inference"] is True
    download.assert_called_once_with(
        artifact_root=tmp_path / "artifacts",
        volume_name="clipper-v10-artifacts",
        remote_run_path="/campaign-run",
    )


def test_run_modal_pipeline_rejects_missing_or_multi_source_and_bad_runner(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")

    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._function", return_value=Mock()),
        patch("clipper.modal_execution._authorized_candidates", return_value=[]),
        pytest.raises(RuntimeError, match="no authorized source"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._function", return_value=Mock()),
        patch("clipper.modal_execution._authorized_candidates", return_value=[candidate, candidate]),
        pytest.raises(RuntimeError, match="source_limit=1"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    acquire = Mock()
    runner = Mock()
    runner.remote.return_value = "invalid"
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._authorized_candidates", return_value=[candidate]),
        patch(
            "clipper.modal_execution._function",
            side_effect=lambda _app, name: acquire if name == "acquire_source" else runner,
        ),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode"},
        ),
        pytest.raises(RuntimeError, match="invalid response"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    runner.remote.return_value = {"run_path": ""}
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._authorized_candidates", return_value=[candidate]),
        patch(
            "clipper.modal_execution._function",
            side_effect=lambda _app, name: acquire if name == "acquire_source" else runner,
        ),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode"},
        ),
        pytest.raises(RuntimeError, match="no run path"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )
