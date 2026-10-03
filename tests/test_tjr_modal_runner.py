"""Offline verification of Modal production retry and exact-source provenance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from clipper.editorial_run import EditorialRunConfig, load_editorial_run_config
from scripts.tjr_modal_runner import (
    _purge_remote,
    _transfer_verified_original,
    run_modal_production,
)
from scripts.tjr_youtube_preview import NoEditorialMoments

CHANNEL = "UCf1q6dhccWr6eQEcFFnJSbA"
BRIEF = Path(__file__).resolve().parents[1] / "campaigns/reach-double-coverage-dedicated.yaml"


def _staged(video_id: str) -> dict[str, object]:
    return {
        "status": "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED",
        "video_id": video_id,
        "channel_id": CHANNEL,
        "public_video_url": f"https://www.youtube.com/watch?v={video_id}",
        "source_remote_path": f"runs/36340591348-1/{video_id}/original.mp4",
        "source_sha256": "a" * 64,
        "duration": 3400,
        "staged_duration_seconds": 3400,
        "source_scan_complete": True,
        "title": "Original video",
    }


def test_editorial_failure_never_switches_to_older_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.tjr_modal_runner as runner

    monkeypatch.delenv("TJR_SOURCE_VIDEO_ID", raising=False)
    video = _staged("X7msxvyQd_U")

    def transfer(staging: dict[str, object], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = destination.with_suffix(".mp4")
        original.write_bytes(b"validated temporary source")
        destination.with_suffix(".json").write_text("{}", encoding="utf-8")
        return original

    with (
        patch.object(runner, "_acquire_original", return_value=video) as acquire,
        patch.object(runner, "_transfer_verified_original", side_effect=transfer),
        patch.object(runner, "_purge_remote"),
        patch.object(
            runner,
            "render_youtube_previews",
            side_effect=NoEditorialMoments("no qualified moments in this original"),
        ),
        pytest.raises(NoEditorialMoments, match="no qualified moments"),
    ):
        run_modal_production(root=tmp_path / "artifacts")
    assert acquire.call_count == 1
    assert not (tmp_path / "artifacts" / "attempt-2").exists()


def test_pinned_source_never_switches_original_on_editorial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.tjr_modal_runner as runner

    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "X7msxvyQd_U")
    video = _staged("X7msxvyQd_U")
    with (
        patch.object(runner, "_acquire_original", return_value=video) as acquire,
        patch.object(runner, "_transfer_verified_original", return_value=None),
        patch.object(runner, "_purge_remote"),
        patch.object(runner, "render_youtube_previews", side_effect=NoEditorialMoments("weak")),
        pytest.raises(NoEditorialMoments, match="weak"),
    ):
        run_modal_production(root=tmp_path / "artifacts")
    assert acquire.call_count == 1


def test_modal_editor_uses_config_and_preserves_cache_roots(tmp_path: Path, monkeypatch):
    import scripts.tjr_modal_runner as runner

    monkeypatch.setenv("TJR_PERSIST_SOURCE", "0")
    config = EditorialRunConfig(
        brief=BRIEF,
        artifact_root=tmp_path / "artifacts",
        source_video_id="X7msxvyQd_U",
        target_channel_id=CHANNEL,
        editorial_cache_root=tmp_path / "editorial-cache",
        transcript_cache_root=tmp_path / "transcript-cache",
        render_cache_root=tmp_path / "render-cache",
    )
    seen = []

    def render(root, brief, *, run_config):
        seen.append((root, brief, run_config))
        return root / "complete"

    with (
        patch.object(runner, "_acquire_original", return_value=_staged("X7msxvyQd_U")),
        patch.object(runner, "_transfer_verified_original", return_value=None),
        patch.object(runner, "_purge_remote"),
        patch.object(runner, "_save_pipeline_completion"),
        patch.object(runner, "render_youtube_previews", side_effect=render),
    ):
        result = run_modal_production(probe_root=tmp_path / "probe", run_config=config)
    assert result == config.artifact_root / "attempt-1" / "complete"
    root, brief, selected = seen[0]
    assert root == config.artifact_root / "attempt-1" and brief == BRIEF
    assert selected.source_video_id == config.source_video_id
    assert selected.target_channel_id == config.target_channel_id
    assert selected.require_staged_original is True
    assert selected.browser_capture_file == (root / "source.json").resolve()
    assert selected.editorial_cache_root == config.editorial_cache_root
    assert selected.transcript_cache_root == config.transcript_cache_root
    assert selected.render_cache_root == config.render_cache_root
    assert load_editorial_run_config(root / "editorial-run.json") == selected


def test_modal_config_cannot_switch_verified_original(tmp_path: Path, monkeypatch):
    import scripts.tjr_modal_runner as runner

    monkeypatch.setenv("TJR_PERSIST_SOURCE", "0")
    config = EditorialRunConfig(
        brief=BRIEF,
        artifact_root=tmp_path / "artifacts",
        source_video_id="_kDrxucOx9g",
        target_channel_id=CHANNEL,
    )
    with (
        patch.object(runner, "_acquire_original", return_value=_staged("X7msxvyQd_U")),
        patch.object(runner, "_transfer_verified_original", return_value=None),
        patch.object(runner, "_purge_remote"),
        patch.object(runner, "render_youtube_previews") as render,
        pytest.raises(RuntimeError, match="configured source differs"),
    ):
        run_modal_production(probe_root=tmp_path / "probe", run_config=config)
    render.assert_not_called()


def test_transfer_rejects_wrong_sha_and_removes_unverified_original(tmp_path: Path) -> None:
    import scripts.tjr_modal_runner as runner

    stage = _staged("X7msxvyQd_U")
    destination = tmp_path / "source"

    def fake_run(args: list[str], **kwargs: object) -> Mock:
        Path(args[-1]).write_bytes(b"not matching the approved digest")
        return Mock(returncode=0)

    with (
        patch.object(runner.subprocess, "run", side_effect=fake_run),
        pytest.raises(RuntimeError, match="SHA-256 mismatch"),
    ):
        _transfer_verified_original(stage, destination)
    assert not destination.with_suffix(".mp4").exists()
    assert not destination.with_suffix(".json").exists()


def test_remote_source_identity_is_mandatory(tmp_path: Path) -> None:
    stage = _staged("X7msxvyQd_U")
    stage["public_video_url"] = "https://untrusted.invalid/not-the-original"
    with pytest.raises(RuntimeError, match="source or hash validation"):
        _transfer_verified_original(stage, tmp_path / "source")


def test_modal_rejects_partial_source_before_transferring(tmp_path: Path) -> None:
    staging = _staged("X7msxvyQd_U")
    staging["staged_duration_seconds"] = 840
    staging["source_scan_complete"] = False
    with pytest.raises(RuntimeError, match="source or hash validation"):
        _transfer_verified_original(staging, tmp_path / "source")


def test_failed_remote_cleanup_is_reported_and_fails(tmp_path: Path) -> None:
    import json

    import scripts.tjr_modal_runner as runner

    report = tmp_path / "cleanup-error.json"
    with (
        patch.object(runner.subprocess, "run", return_value=Mock(returncode=1, stderr="denied")),
        pytest.raises(RuntimeError, match="MODAL_VOLUME_CLEANUP_FAILED"),
    ):
        _purge_remote(_staged("X7msxvyQd_U"), report)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["exit_code"] == 1
    assert data["source_remote_path"].endswith("/X7msxvyQd_U/original.mp4")
    assert data["stderr"] == "denied"


def test_preacquired_staging_is_reused_without_second_modal_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import scripts.tjr_modal_runner as runner

    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    staging = _staged("X7msxvyQd_U")
    (probe_root / "staged-original.json").write_text(json.dumps(staging), encoding="utf-8")
    monkeypatch.setenv("TJR_MODAL_USE_STAGED", "1")
    with patch.object(runner.subprocess, "run") as command:
        actual = runner._acquire_original(set(), probe_root)
    command.assert_not_called()
    assert actual["video_id"] == "X7msxvyQd_U"
    assert actual["status"] == "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED"


def test_watermark_preflight_reuses_verified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from PIL import Image

    from scripts.tjr_modal_runner import prepare_watermark_cache

    manifest = tmp_path / "asset.json"
    image = manifest.with_suffix(".png")
    Image.new("RGBA", (8, 8)).save(image)
    expected = hashlib.sha256(image.read_bytes()).hexdigest()
    monkeypatch.setenv("CLIPPER_APPROVED_ASSET_SHA256", expected)
    with (
        patch("scripts.tjr_modal_runner.subprocess.run", return_value=Mock(returncode=0)),
        patch("clipper.pipeline._download_asset") as download,
    ):
        prepare_watermark_cache(Path("campaigns/reach-double-coverage-dedicated.yaml"), manifest)
        download.assert_not_called()
    assert json.loads(manifest.read_text())["sha256"] == expected
    image.write_bytes(b"corrupt")
    with (
        patch("scripts.tjr_modal_runner.subprocess.run", return_value=Mock(returncode=0)),
        pytest.raises(RuntimeError, match="SHA-256 mismatch"),
    ):
        prepare_watermark_cache(Path("campaigns/reach-double-coverage-dedicated.yaml"), manifest)


def test_watermark_preflight_rejects_invalid_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.tjr_modal_runner import prepare_watermark_cache

    monkeypatch.setenv("CLIPPER_APPROVED_ASSET_SHA256", "a" * 64)
    monkeypatch.setenv("CLIPPER_APPROVED_ASSET_BOOTSTRAP_URL", "https://example.com/bootstrap")

    def download(url: str, output: Path, **kwargs: object) -> Path:
        output.write_bytes(b"quota HTML")
        return output

    with (
        patch(
            "scripts.tjr_modal_runner.subprocess.run", return_value=Mock(returncode=1)
        ) as command,
        patch("clipper.pipeline._download_asset", side_effect=download),
        pytest.raises(RuntimeError, match="bootstrap failed"),
    ):
        prepare_watermark_cache(
            Path("campaigns/reach-double-coverage-dedicated.yaml"), tmp_path / "asset.json"
        )
    assert command.call_count == 1
    assert not (tmp_path / "asset.png").exists()


def test_exact_source_challenge_retries_same_modal_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from scripts.tjr_modal_runner import _acquire_original

    monkeypatch.delenv("TJR_MODAL_USE_STAGED", raising=False)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "_kDrxucOx9g")
    calls: list[list[str]] = []

    def acquire(command: list[str], **kwargs: object) -> Mock:
        calls.append(command)
        env = kwargs["env"]
        assert env["TJR_SOURCE_VIDEO_ID"] == "_kDrxucOx9g"
        assert env["TJR_MODAL_EXCLUDE_VIDEO_IDS"] == ""
        if len(calls) == 1:
            report = {
                "status": "YOUTUBE_EGRESS_BOT_CHALLENGE",
                "attempts": [
                    {
                        "url": "https://www.youtube.com/watch?v=_kDrxucOx9g",
                        "stage": "metadata",
                        "reason": "YOUTUBE_IP_OR_LOGIN_CHALLENGE",
                    }
                ],
            }
            (tmp_path / "verified-original-egress.json").write_text(json.dumps(report))
            return Mock(returncode=1)
        (tmp_path / "verified-original-egress.json").write_text(
            json.dumps({"status": "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED"})
        )
        (tmp_path / "staged-original.json").write_text(json.dumps(_staged("_kDrxucOx9g")))
        return Mock(returncode=0)

    with patch("scripts.tjr_modal_runner.subprocess.run", side_effect=acquire):
        assert _acquire_original(set(), tmp_path)["video_id"] == "_kDrxucOx9g"
    assert calls[0] == calls[1] == ["modal", "run", "-m", "scripts.tjr_modal_probe"]
    assert (tmp_path / "acquisition-attempt-1.json").is_file()
    assert (tmp_path / "acquisition-attempt-2.json").is_file()


def test_acquisition_does_not_retry_other_failure_or_change_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import subprocess

    from scripts.tjr_modal_runner import _acquire_original

    monkeypatch.delenv("TJR_MODAL_USE_STAGED", raising=False)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "_kDrxucOx9g")

    def fail(command: list[str], **kwargs: object) -> Mock:
        (tmp_path / "verified-original-egress.json").write_text(
            json.dumps({"status": "MODAL_PROBE_FAILED_BEFORE_REMOTE_RESULT"})
        )
        return Mock(returncode=1)

    with (
        patch("scripts.tjr_modal_runner.subprocess.run", side_effect=fail) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        _acquire_original(set(), tmp_path)
    assert run.call_count == 1


def test_same_source_challenge_retry_is_bounded_to_three_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import subprocess

    from scripts.tjr_modal_runner import _acquire_original

    monkeypatch.delenv("TJR_MODAL_USE_STAGED", raising=False)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "_kDrxucOx9g")

    def challenged(command: list[str], **kwargs: object) -> Mock:
        (tmp_path / "verified-original-egress.json").write_text(
            json.dumps(
                {
                    "status": "YOUTUBE_EGRESS_BOT_CHALLENGE",
                    "attempts": [
                        {
                            "url": "https://www.youtube.com/watch?v=_kDrxucOx9g",
                            "stage": "metadata",
                            "reason": "YOUTUBE_IP_OR_LOGIN_CHALLENGE",
                        }
                    ],
                }
            )
        )
        return Mock(returncode=1)

    with (
        patch("scripts.tjr_modal_runner.subprocess.run", side_effect=challenged) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        _acquire_original(set(), tmp_path)
    assert run.call_count == 3
    assert len(list(tmp_path.glob("acquisition-attempt-*.json"))) == 3


def test_source_cache_skips_youtube_acquisition(tmp_path, monkeypatch):
    import json

    import scripts.tjr_modal_runner as runner

    cached = tmp_path / "prior"
    cached.mkdir()
    staged = _staged("_kDrxucOx9g")
    (cached / "staged-original.json").write_text(json.dumps(staged))
    local = tmp_path / "verified.mp4"
    local.write_bytes(b"verified source")
    monkeypatch.setenv("TJR_SOURCE_CACHE_ROOT", str(cached))
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "_kDrxucOx9g")
    monkeypatch.setenv("TJR_MODAL_CHANNEL_ID", staged["channel_id"])
    monkeypatch.delenv("TJR_MODAL_USE_STAGED", raising=False)
    with (
        patch.object(runner, "_transfer_verified_original", return_value=local),
        patch.object(runner.subprocess, "run") as run,
    ):
        result = runner._acquire_original(set(), tmp_path / "retry")
    assert result["source_cache_reused"] is True
    run.assert_not_called()


def test_source_cache_does_not_reuse_a_different_video(tmp_path, monkeypatch):
    import json

    import scripts.tjr_modal_runner as runner

    (tmp_path / "staged-original.json").write_text(json.dumps(_staged("_kDrxucOx9g")))
    monkeypatch.setenv("TJR_SOURCE_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "X7msxvyQd_U")
    assert runner._restore_source_cache(tmp_path / "retry") is None


def test_completed_pipeline_replay_skips_source_and_invalidates_changed_settings(
    tmp_path, monkeypatch
):
    import json

    import scripts.tjr_modal_runner as runner

    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "_kDrxucOx9g")
    monkeypatch.setenv("TJR_MODAL_CHANNEL_ID", "UCf1q6dhccWr6eQEcFFnJSbA")
    prior = tmp_path / "previous-run"
    prior.mkdir()
    (prior / "tjr-youtube-qa-report.json").write_text(json.dumps({"selected_clip_count": 1}))
    clip = prior / "clips" / "01-video.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"unchanged delivery")
    runner._save_pipeline_completion(prior, "a" * 64)
    monkeypatch.setenv("TJR_RENDER_CACHE_ROOT", str(prior))
    with patch.object(runner, "_acquire_original") as acquire:
        assert runner._restore_completed_pipeline(tmp_path / "retry")
        acquire.assert_not_called()
    monkeypatch.setenv("TJR_CAPTION_STYLE", "changed-style")
    assert not runner._restore_completed_pipeline(tmp_path / "changed")
    monkeypatch.delenv("TJR_CAPTION_STYLE", raising=False)
    clip.write_bytes(b"corrupted delivery")
    assert not runner._restore_completed_pipeline(tmp_path / "corrupted")
