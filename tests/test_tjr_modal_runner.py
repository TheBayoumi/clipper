"""Offline verification of Modal production retry and exact-source provenance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scripts.tjr_modal_runner import (
    _purge_remote,
    _transfer_verified_original,
    run_modal_production,
)
from scripts.tjr_youtube_preview import NoEditorialMoments

CHANNEL = "UCZen39LQJPx04GjPj7FOMcw"


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


def test_second_approved_original_after_first_editorial_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.tjr_modal_runner as runner

    monkeypatch.delenv("TJR_SOURCE_VIDEO_ID", raising=False)
    excluded_seen: list[set[str]] = []
    originals = iter(["X7msxvyQd_U", "Xa-4kOvpGok"])
    attempts = 0

    def acquire(excluded: set[str], root: Path) -> dict[str, object]:
        excluded_seen.append(set(excluded))
        return _staged(next(originals))

    def transfer(staging: dict[str, object], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = destination.with_suffix(".mp4")
        original.write_bytes(b"validated temporary source")
        destination.with_suffix(".json").write_text("{}", encoding="utf-8")
        return original

    def render(root: Path, brief: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise NoEditorialMoments("no qualified moments in this original")
        result = root / "successful-render"
        result.mkdir()
        return result

    with (
        patch.object(runner, "_acquire_original", side_effect=acquire),
        patch.object(runner, "_transfer_verified_original", side_effect=transfer),
        patch.object(runner, "_purge_remote"),
        patch.object(runner, "render_youtube_previews", side_effect=render),
    ):
        path = run_modal_production(root=tmp_path / "artifacts", max_sources=2)
    assert path.name == "successful-render"
    assert excluded_seen == [set(), {"X7msxvyQd_U"}]
    assert attempts == 2
    assert (tmp_path / "artifacts" / "attempt-1" / "editorial-failure.json").is_file()
    assert not (tmp_path / "artifacts" / "attempt-1" / "source.mp4").exists()


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
        run_modal_production(root=tmp_path / "artifacts", max_sources=2)
    assert acquire.call_count == 1


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


def test_retry_count_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_sources"):
        run_modal_production(root=tmp_path / "artifacts", max_sources=5)


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
