import io
import json
import subprocess
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from clipper.models import CampaignBrief, VideoCandidate
from clipper.youtube import YouTubeClient, YouTubeError, _run


def brief() -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "c",
            "title": "AI",
            "objective": "automation",
            "keywords": ["agents"],
            "source_channel_ids": ["UC1"],
            "rights_confirmed": True,
        }
    )


def test_api_discovery_maps_results() -> None:
    client = YouTubeClient("key")
    search = {"items": [{"id": {"videoId": "v1"}}]}
    details = {
        "items": [
            {
                "id": "v1",
                "snippet": {"title": "T", "channelId": "UC1", "channelTitle": "C"},
                "statistics": {"viewCount": "12"},
            }
        ]
    }
    client._api_get = Mock(side_effect=[search, details])  # type: ignore[method-assign]
    result = client.discover(brief())
    assert result[0].video_id == "v1"
    assert result[0].view_count == 12


def test_ytdlp_discovery_and_downloads_preserve_highest_source_quality(tmp_path: Path) -> None:
    payload = {
        "entries": [
            {
                "id": "v1",
                "title": "T",
                "channel_id": "UC1",
                "channel": "C",
                "webpage_url": "https://youtube.test/v1",
                "duration": 30,
                "view_count": 99,
            }
        ]
    }
    client = YouTubeClient(None)
    with (
        patch("clipper.youtube.shutil.which", return_value="/bin/yt-dlp"),
        patch("clipper.youtube._run", return_value=Mock(stdout=json.dumps(payload))),
    ):
        assert client._discover_ytdlp(brief())[0].duration_seconds == 30

    video = VideoCandidate("v1", "T", "UC1", "C", "https://youtube.test/v1")
    subtitle = tmp_path / "v1.en.vtt"

    def fake_subtitles(*_args, **_kwargs):
        subtitle.write_text("WEBVTT", encoding="utf-8")
        return Mock(stdout="")

    with patch("clipper.youtube._run", side_effect=fake_subtitles):
        assert client.download_subtitles(video, tmp_path, "en") == subtitle

    media = tmp_path / "v1.mkv"
    format_payload = {
        "formats": [
            {
                "format_id": "401",
                "height": 2160,
                "width": 3840,
                "fps": 24,
                "vcodec": "av01",
                "acodec": "none",
                "ext": "mp4",
                "tbr": 4500,
            },
            {
                "format_id": "137",
                "height": 1080,
                "width": 1920,
                "fps": 24,
                "vcodec": "avc1",
                "acodec": "none",
                "ext": "mp4",
                "tbr": 1900,
            },
        ]
    }
    calls: list[list[str]] = []

    def fake_media(command, **_kwargs):
        calls.append(command)
        if "--dump-single-json" in command:
            return Mock(stdout=json.dumps(format_payload))
        media.write_bytes(b"video")
        return Mock(stdout="")

    with patch("clipper.youtube._run", side_effect=fake_media):
        assert client.download_media(video, tmp_path) == media
        assert client.download_media(video, tmp_path) == media
    assert any("401+bestaudio/401" in item for item in calls[-1])
    assert "--remux-video" in calls[-1]
    assert "--newline" not in calls[-1]
    assert calls[-1][calls[-1].index("--retries") + 1] == "3"
    evidence = json.loads(media.with_suffix(".source.json").read_text())
    assert evidence["quality_policy"] == "highest_available_no_transcode"
    assert evidence["selected"]["height"] == 2160
    assert evidence["selected"]["format_id"] == "401"
    assert evidence["recovered_after_http_403"] is False


def test_ytdlp_403_refreshes_and_uses_same_quality_alternate(tmp_path: Path) -> None:
    client = YouTubeClient(None)
    video = VideoCandidate("v1", "T", "UC1", "C", "https://youtube.test/v1")
    media = tmp_path / "v1.mkv"
    payload = {
        "formats": [
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
            {
                "format_id": "137",
                "height": 1080,
                "width": 1920,
                "fps": 30,
                "vcodec": "avc1",
                "acodec": "none",
                "ext": "mp4",
                "tbr": 1900,
            },
        ]
    }
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "--dump-single-json" in command:
            return Mock(stdout=json.dumps(payload))
        selector = command[command.index("-f") + 1]
        if selector.startswith("313+"):
            raise YouTubeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        media.write_bytes(b"video")
        return Mock(stdout="")

    with patch("clipper.youtube._run", side_effect=fake_run):
        assert client.download_media(video, tmp_path) == media

    download_calls = [item for item in calls if "-f" in item]
    assert len(download_calls) == 3
    assert download_calls[0][download_calls[0].index("-f") + 1].startswith("313+")
    assert "youtube:player_client=android_vr" in download_calls[1]
    assert "youtube:player_client=android_vr" in download_calls[2]
    assert download_calls[2][download_calls[2].index("-f") + 1].startswith("401+")
    assert all("137" not in item[item.index("-f") + 1] for item in download_calls)
    evidence = json.loads(media.with_suffix(".source.json").read_text())
    assert evidence["recovered_after_http_403"] is True
    assert evidence["selected"]["height"] == 2160
    assert evidence["selected"]["format_id"] == "401"
    assert evidence["selected_player_client"] == "android_vr"
    assert "HTTP Error 403" in evidence["download_attempts"][0]["error"]


def test_ytdlp_missing_and_run_errors() -> None:
    client = YouTubeClient(None)
    with patch("clipper.youtube.shutil.which", return_value=None), pytest.raises(YouTubeError):
        client._discover_ytdlp(brief())
    with (
        patch("clipper.youtube.subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(YouTubeError, match="not found"),
    ):
        _run(["missing"])


def test_run_visible_streams_one_line_progress_and_preserves_error(capsys) -> None:
    class Process:
        stdout = io.BytesIO(
            b"[download] 1.8% of 3.62GiB\rERROR: unable to download video data: "
            b"HTTP Error 403: Forbidden\n"
        )

        def wait(self, timeout):
            assert timeout == 900
            return 1

        def kill(self):
            return None

    with (
        patch("clipper.youtube.subprocess.Popen", return_value=Process()),
        pytest.raises(YouTubeError, match="HTTP Error 403"),
    ):
        _run(["yt-dlp"], visible=True)
    visible = capsys.readouterr().err
    assert "[download] 1.8%" in visible
    assert "HTTP Error 403: Forbidden" in visible


def test_run_visible_success_does_not_capture_away_progress(capsys) -> None:
    class Process:
        stdout = io.BytesIO(b"[download] 50%\r[download] 100%\n")

        def wait(self, timeout):
            assert timeout == 900
            return 0

        def kill(self):
            return None

    with patch("clipper.youtube.subprocess.Popen", return_value=Process()) as popen:
        result = _run(["yt-dlp"], visible=True)
    assert result.returncode == 0
    assert "[download] 100%" in capsys.readouterr().err
    kwargs = popen.call_args.kwargs
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.STDOUT


def test_api_get_success_and_error() -> None:
    client = YouTubeClient("key")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"items": []}'

    with patch("clipper.youtube.urllib.request.urlopen", return_value=Response()):
        assert client._api_get("search", {"q": "x"}) == {"items": []}
    with (
        patch(
            "clipper.youtube.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ),
        pytest.raises(YouTubeError, match="request failed"),
    ):
        client._api_get("search", {"q": "x"})


def test_api_discovery_empty_and_optional_fields() -> None:
    client = YouTubeClient("key")
    client._api_get = Mock(return_value={"items": []})  # type: ignore[method-assign]
    assert client._discover_api(brief()) == []

    candidate = client._candidate_from_api(
        {"id": "v", "snippet": {"title": "T", "channelId": "C", "channelTitle": "N"}}
    )
    assert candidate.view_count is None


def test_run_command_failures() -> None:
    with (
        patch(
            "clipper.youtube.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["x"], stderr="failure"),
        ),
        pytest.raises(YouTubeError, match="failure"),
    ):
        _run(["x"])
    with (
        patch("clipper.youtube.subprocess.run", side_effect=subprocess.TimeoutExpired(["x"], 1)),
        pytest.raises(YouTubeError, match="timed out"),
    ):
        _run(["x"], timeout=1)


def test_download_subtitle_and_media_failure_paths(tmp_path: Path) -> None:
    client = YouTubeClient(None)
    video = VideoCandidate("v1", "T", "UC1", "C", "https://youtube.test/v1")
    with patch("clipper.youtube._run", side_effect=YouTubeError("no captions")):
        assert client.download_subtitles(video, tmp_path, "en") is None
    with patch("clipper.youtube._run", return_value=Mock(stdout="")):
        assert client.download_subtitles(video, tmp_path, "en") is None
    format_payload = {
        "formats": [
            {
                "format_id": "137",
                "height": 1080,
                "width": 1920,
                "fps": 24,
                "vcodec": "avc1",
                "acodec": "none",
                "ext": "mp4",
                "tbr": 1900,
            }
        ]
    }
    with (
        patch(
            "clipper.youtube._run",
            side_effect=[Mock(stdout=json.dumps(format_payload)), Mock(stdout="")],
        ),
        pytest.raises(YouTubeError, match="without creating"),
    ):
        client.download_media(video, tmp_path)


def test_format_selection_prefers_quality_without_container_bias() -> None:
    payload = {
        "formats": [
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
            {
                "format_id": "400",
                "height": 1440,
                "width": 2560,
                "fps": 60,
                "vcodec": "av01",
                "acodec": "none",
                "ext": "mp4",
                "tbr": 2100,
            },
        ]
    }
    selected, available = YouTubeClient._select_video_format(payload, None)
    assert selected["format_id"] == "313"
    selected_1440, _ = YouTubeClient._select_video_format(payload, 1440)
    assert selected_1440["format_id"] == "400"
    assert available[0]["format_id"] == "313"
    with pytest.raises(YouTubeError, match="no video format"):
        YouTubeClient._select_video_format(payload, 360)
    with pytest.raises(YouTubeError, match="at least 360"):
        YouTubeClient(None, max_height=200)
