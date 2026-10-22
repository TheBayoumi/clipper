from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from clipper.models import VideoCandidate
from clipper.youtube import (
    DiscoveryRequest,
    YouTubeClient,
    YouTubeError,
    _is_http_403,
    _json_object,
    _object,
    _object_list,
    _retry_player_clients,
    _run,
)


def _format_payload(*, include_4k: bool = True) -> dict[str, object]:
    formats: list[dict[str, object]] = []
    if include_4k:
        formats.extend(
            [
                {
                    "format_id": "313",
                    "height": 2160,
                    "width": 3840,
                    "fps": 60,
                    "vcodec": "vp9",
                    "acodec": "none",
                    "ext": "webm",
                    "tbr": 10000,
                },
                {
                    "format_id": "401",
                    "height": 2160,
                    "width": 3840,
                    "fps": 30,
                    "vcodec": "av01",
                    "acodec": "none",
                    "ext": "mp4",
                    "tbr": 4500,
                },
            ]
        )
    formats.append(
        {
            "format_id": "137",
            "height": 1080,
            "width": 1920,
            "fps": 30,
            "vcodec": "avc1",
            "acodec": "none",
            "ext": "mp4",
            "tbr": 1900,
        }
    )
    return {
        "id": "v1",
        "channel_id": "UC1",
        "webpage_url": "https://www.youtube.com/watch?v=v1",
        "formats": formats,
    }


def test_youtube_helper_edge_cases(monkeypatch) -> None:
    with pytest.raises(YouTubeError, match="expected a JSON object"):
        _json_object("[]")
    assert _object_list(None) == []
    assert _object(None) == {}
    assert _is_http_403(YouTubeError("HTTP Error 403: Forbidden"))
    assert _is_http_403(YouTubeError("403 Forbidden"))
    assert not _is_http_403(YouTubeError("HTTP Error 500"))

    monkeypatch.setenv("CLIPPER_YTDLP_RETRY_CLIENTS", "android_vr, web_embedded")
    assert _retry_player_clients() == ("android_vr", "web_embedded")
    monkeypatch.setenv("CLIPPER_YTDLP_RETRY_CLIENTS", "")
    assert _retry_player_clients() == ("android_vr", "web_embedded")


def test_visible_run_timeout_kills_process() -> None:
    class Process:
        stdout = io.BytesIO(b"")
        killed = False

        def wait(self, timeout):
            raise subprocess.TimeoutExpired(["yt-dlp"], timeout)

        def kill(self):
            self.killed = True

    process = Process()
    with (
        patch("clipper.youtube.subprocess.Popen", return_value=process),
        pytest.raises(YouTubeError, match="timed out after 1s"),
    ):
        _run(["yt-dlp"], visible=True, timeout=1)
    assert process.killed is True


def test_visible_run_flushes_incomplete_utf8_tail(capsys) -> None:
    class Process:
        stdout = io.BytesIO(b"\xe2\x82")

        def wait(self, timeout):
            assert timeout == 900
            return 0

        def kill(self):
            return None

    with patch("clipper.youtube.subprocess.Popen", return_value=Process()):
        result = _run(["yt-dlp"], visible=True)
    assert result.returncode == 0
    assert "\ufffd" in capsys.readouterr().err


def test_direct_video_discovery_uses_metadata_path() -> None:
    request = DiscoveryRequest(video_ids=("v1",), limit=1)
    payload = {
        "id": "v1",
        "title": "Allowed",
        "channel_id": "UC1",
        "channel": "Channel",
        "webpage_url": "https://www.youtube.com/watch?v=v1",
        "duration": 42,
    }
    client = YouTubeClient(None)
    with (
        patch("clipper.youtube.shutil.which", return_value="yt-dlp"),
        patch("clipper.youtube._run", return_value=Mock(stdout=json.dumps(payload))) as run,
    ):
        videos = client.discover(request)
    assert [video.video_id for video in videos] == ["v1"]
    assert "https://www.youtube.com/watch?v=v1" in run.call_args.args[0]


def test_api_discovery_forwards_published_after() -> None:
    request = DiscoveryRequest(
        query="neutral",
        channel_ids=("UC1",),
        published_after="2026-01-01T00:00:00Z",
        limit=1,
    )
    client = YouTubeClient("key")
    with patch.object(
        client,
        "_api_get",
        side_effect=[{"items": []}],
    ) as api_get:
        assert client.discover(request) == []
    search_params = api_get.call_args_list[0].args[1]
    assert search_params["publishedAfter"] == "2026-01-01T00:00:00Z"


def test_private_api_requires_key_and_format_selector_handles_muxed() -> None:
    client = YouTubeClient(None)
    with pytest.raises(YouTubeError, match="API key is required"):
        client._api_get("search", {})
    assert client._format_selector({"format_id": "22", "audio_codec": "aac"}) == "22"
    assert client._format_selector({"format_id": "313", "audio_codec": "none"}).startswith(
        "313+bestaudio"
    )
    assert client._extractor_args(None) == []
    assert client._extractor_args("android_vr") == [
        "--extractor-args",
        "youtube:player_client=android_vr",
    ]


def test_format_selection_ignores_non_video_formats() -> None:
    payload = _format_payload()
    formats = payload["formats"]
    assert isinstance(formats, list)
    formats.insert(0, {"format_id": "audio", "height": 0, "vcodec": "none"})
    selected, _ = YouTubeClient._select_video_format(payload, None)
    assert selected["height"] == 2160


def test_non_403_download_failure_does_not_enter_recovery(tmp_path: Path) -> None:
    client = YouTubeClient(None)
    video = VideoCandidate("v1", "T", "UC1", "C", "https://www.youtube.com/watch?v=v1")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "--dump-single-json" in command:
            return Mock(stdout=json.dumps(_format_payload()))
        raise YouTubeError("HTTP Error 500: upstream failure")

    with (
        patch("clipper.youtube._run", side_effect=fake_run),
        pytest.raises(YouTubeError, match="HTTP Error 500"),
    ):
        client.download_media(video, tmp_path)
    assert len([call for call in calls if "--dump-single-json" in call]) == 1
    assert all("--extractor-args" not in call for call in calls)


def test_403_exhaustion_records_refresh_and_quality_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLIPPER_YTDLP_RETRY_CLIENTS", "android_vr,web_embedded")
    client = YouTubeClient(None)
    video = VideoCandidate("v1", "T", "UC1", "C", "https://www.youtube.com/watch?v=v1")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        is_metadata = "--dump-single-json" in command
        client_arg = next(
            (item for item in command if item.startswith("youtube:player_client=")), None
        )
        if is_metadata and client_arg is None:
            return Mock(stdout=json.dumps(_format_payload()))
        if is_metadata and client_arg == "youtube:player_client=android_vr":
            raise YouTubeError("android metadata refresh failed")
        if is_metadata and client_arg == "youtube:player_client=web_embedded":
            return Mock(stdout=json.dumps(_format_payload(include_4k=False)))
        raise YouTubeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")

    with (
        patch("clipper.youtube._run", side_effect=fake_run),
        pytest.raises(YouTubeError, match="quality remained locked to 3840x2160") as captured,
    ):
        client.download_media(video, tmp_path)

    assert "android metadata refresh failed" in str(captured.value.__cause__)
    assert any("youtube:player_client=android_vr" in call for call in calls)
    assert any("youtube:player_client=web_embedded" in call for call in calls)


def test_403_retry_candidates_remain_exact_resolution() -> None:
    client = YouTubeClient(None)
    selected = {
        "format_id": "313",
        "height": 2160,
        "width": 3840,
        "fps": 60.0,
        "bitrate_kbps": 10000.0,
    }
    available = [
        selected,
        {
            "format_id": "401",
            "height": 2160,
            "width": 3840,
            "fps": 30.0,
            "bitrate_kbps": 4500.0,
        },
        {
            "format_id": "137",
            "height": 1080,
            "width": 1920,
            "fps": 30.0,
            "bitrate_kbps": 1900.0,
        },
    ]
    assert [item["format_id"] for item in client._same_quality_formats(available, selected)] == [
        "313",
        "401",
    ]
