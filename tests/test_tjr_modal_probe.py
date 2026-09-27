"""Offline regression: one challenged upload cannot hide another approved original."""

from __future__ import annotations

import inspect
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _FakeApp:
    def function(self, **kwargs: object) -> object:
        return lambda fn: fn

    def local_entrypoint(self) -> object:
        return lambda fn: fn


def _load_modal_probe(monkeypatch: pytest.MonkeyPatch) -> object:
    mock_modal = SimpleNamespace(
        App=lambda *_args, **_kwargs: _FakeApp(),
        Image=MagicMock(),
        Volume=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "modal", mock_modal)
    path = Path(__file__).resolve().parents[1] / "scripts" / "tjr_modal_probe.py"
    return runpy.run_path(str(path))["inspect_original_youtube"]


def test_ip_challenge_skips_only_one_video_not_entire_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    first, second = "lxu_J1Ec1XI", "X7msxvyQd_U"
    channel = "UCZen39LQJPx04GjPj7FOMcw"
    urls_seen: list[str] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        url = command[-1]
        if "--dump-single-json" in command:
            urls_seen.append(url)
            if first in url:
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="Sign in to confirm you're not a bot"
                )
            metadata = {
                "id": second,
                "channel_id": channel,
                "duration": 3994,
                "live_status": "not_live",
                "title": "LIVE TRADING",
                "formats": [{"height": 1080, "vcodec": "avc1"}],
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metadata), stderr="")
        if "--test" in command:
            template = Path(command[command.index("-o") + 1])
            output = Path(str(template).replace("%(ext)s", "mp4"))
            output.write_bytes(b"verified HD test source bytes" * 80)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess call: {command}")

    inputs = [
        {"video_id": first, "channel_id": channel},
        {"video_id": second, "channel_id": channel},
    ]
    with patch("subprocess.run", side_effect=run):
        result = inspect(inputs)
    assert result["status"] == "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED"
    assert result["source_video_id"] == second
    assert urls_seen.count(f"https://www.youtube.com/watch?v={first}") == 2
    assert urls_seen[-1].endswith(second)
    assert any(
        attempt["reason"] == "YOUTUBE_IP_OR_LOGIN_CHALLENGE" for attempt in result["attempts"]
    )


def test_all_blocked_videos_still_report_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    channel = "UCZen39LQJPx04GjPj7FOMcw"
    videos = [
        {"video_id": video, "channel_id": channel} for video in ("lxu_J1Ec1XI", "X7msxvyQd_U")
    ]
    calls = 0

    def blocked(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Sign in to confirm you're not a bot"
        )

    with patch("subprocess.run", side_effect=blocked):
        result = inspect(videos)
    assert result["status"] == "YOUTUBE_EGRESS_BOT_CHALLENGE"
    assert calls == 4
    assert len({attempt["url"] for attempt in result["attempts"]}) == 2


def test_region_aborts_after_two_distinct_videos_are_ip_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    channel = "UCZen39LQJPx04GjPj7FOMcw"
    videos = [
        {"video_id": vid, "channel_id": channel}
        for vid in ("lxu_J1Ec1XI", "X7msxvyQd_U", "Xa-4kOvpGok")
    ]
    calls: list[str] = []

    def blocked(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command[-1])
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Sign in to confirm you're not a bot"
        )

    with patch("subprocess.run", side_effect=blocked):
        result = inspect(videos)
    assert result["status"] == "YOUTUBE_EGRESS_BOT_CHALLENGE"
    assert len(calls) == 4
    assert all("Xa-4kOvpGok" not in url for url in calls)


def test_plain_client_succeeds_after_provider_bot_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    video_id = "X7msxvyQd_U"
    channel = "UCZen39LQJPx04GjPj7FOMcw"
    client_attempts: list[str] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if "--dump-single-json" in command:
            extractor = command[command.index("--extractor-args") + 1]
            client_attempts.append(extractor)
            if extractor == "youtube:player_client=default,mweb":
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="Sign in to confirm you're not a bot"
                )
            assert extractor == "youtube:player_client=tv,web_safari"
            metadata = {
                "id": video_id,
                "channel_id": channel,
                "duration": 3600,
                "live_status": "was_live",
                "title": "LIVE TRADING",
                "formats": [{"height": 1080, "vcodec": "avc1"}],
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metadata), stderr="")
        if "--test" in command:
            template = Path(command[command.index("-o") + 1])
            output = Path(str(template).replace("%(ext)s", "mp4"))
            output.write_bytes(b"verified independent source data" * 90)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected transport: {command}")

    with patch("subprocess.run", side_effect=run):
        result = inspect([{"video_id": video_id, "channel_id": channel}])
    assert result["status"] == "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED"
    assert result["transport_strategy"] == "plain_tv_web_safari"
    assert client_attempts == [
        "youtube:player_client=default,mweb",
        "youtube:player_client=tv,web_safari",
    ]
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["reason"] == "YOUTUBE_IP_OR_LOGIN_CHALLENGE"


def test_modal_source_windowing_matches_runner_without_remote_editor_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.tjr_youtube_preview import youtube_scan_section_args

    fake = SimpleNamespace(
        App=lambda *_args, **_kwargs: _FakeApp(),
        Image=MagicMock(),
        Volume=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "tjr_modal_probe.py")
    )
    source_sections = script["source_download_sections"]
    for seconds in (120, 2642, 3600, 3601, 7200):
        assert source_sections(seconds) == youtube_scan_section_args(seconds)
    assert "tjr_youtube_preview" not in inspect.getsource(script["stage_official_original"])


def test_staging_probe_rejects_corrupt_partial_and_missing_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metadata hit must never turn a damaged download into a staged original."""
    fake = SimpleNamespace(
        App=lambda *_args, **_kwargs: _FakeApp(),
        Image=MagicMock(),
        Volume=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "tjr_modal_probe.py")
    )
    verify = script["verify_staged_media_probe"]
    good_streams = [
        {"codec_type": "video", "height": 1080},
        {"codec_type": "audio"},
    ]
    with pytest.raises(RuntimeError, match="CORRUPT_OR_INCOMPLETE_HD_TRANSFER"):
        verify(
            returncode=1,
            stdout="",
            stderr="moov atom not found",
            expected_seconds=2642,
        )
    with pytest.raises(RuntimeError, match="SOURCE_DURATION_INCOMPLETE"):
        verify(
            returncode=0,
            stdout=json.dumps({"streams": good_streams, "format": {"duration": "840"}}),
            stderr="",
            expected_seconds=2642,
        )
    with pytest.raises(RuntimeError, match="MISSING_AUDIO_STREAM"):
        verify(
            returncode=0,
            stdout=json.dumps({"streams": good_streams[:1], "format": {"duration": "2642"}}),
            stderr="",
            expected_seconds=2642,
        )
    assert (
        verify(
            returncode=0,
            stdout=json.dumps({"streams": good_streams, "format": {"duration": "2642"}}),
            stderr="",
            expected_seconds=2642,
        )
        == 2642
    )
    # An approved two-hour original may stage only its declared first hour.
    assert (
        verify(
            returncode=0,
            stdout=json.dumps({"streams": good_streams, "format": {"duration": "3600"}}),
            stderr="",
            expected_seconds=7200,
        )
        == 3600
    )
