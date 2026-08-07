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


def test_ytdlp_discovery_and_downloads(tmp_path: Path) -> None:
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
    with patch("clipper.youtube.shutil.which", return_value="/bin/yt-dlp"), patch(
        "clipper.youtube._run", return_value=Mock(stdout=json.dumps(payload))
    ):
        assert client._discover_ytdlp(brief())[0].duration_seconds == 30

    video = VideoCandidate("v1", "T", "UC1", "C", "https://youtube.test/v1")
    subtitle = tmp_path / "v1.en.vtt"

    def fake_subtitles(*_args, **_kwargs):
        subtitle.write_text("WEBVTT", encoding="utf-8")
        return Mock(stdout="")

    with patch("clipper.youtube._run", side_effect=fake_subtitles):
        assert client.download_subtitles(video, tmp_path, "en") == subtitle

    media = tmp_path / "v1.mp4"

    def fake_media(*_args, **_kwargs):
        media.write_bytes(b"video")
        return Mock(stdout="")

    with patch("clipper.youtube._run", side_effect=fake_media):
        assert client.download_media(video, tmp_path) == media
        assert client.download_media(video, tmp_path) == media


def test_ytdlp_missing_and_run_errors() -> None:
    client = YouTubeClient(None)
    with patch("clipper.youtube.shutil.which", return_value=None), pytest.raises(YouTubeError):
        client._discover_ytdlp(brief())
    with patch("clipper.youtube.subprocess.run", side_effect=FileNotFoundError), pytest.raises(
        YouTubeError, match="not found"
    ):
        _run(["missing"])


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
    with patch(
        "clipper.youtube.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ), pytest.raises(YouTubeError, match="request failed"):
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
    with patch(
        "clipper.youtube.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["x"], stderr="failure"),
    ), pytest.raises(YouTubeError, match="failure"):
        _run(["x"])
    with patch(
        "clipper.youtube.subprocess.run", side_effect=subprocess.TimeoutExpired(["x"], 1)
    ), pytest.raises(YouTubeError, match="timed out"):
        _run(["x"], timeout=1)


def test_download_subtitle_and_media_failure_paths(tmp_path: Path) -> None:
    client = YouTubeClient(None)
    video = VideoCandidate("v1", "T", "UC1", "C", "https://youtube.test/v1")
    with patch("clipper.youtube._run", side_effect=YouTubeError("no captions")):
        assert client.download_subtitles(video, tmp_path, "en") is None
    with patch("clipper.youtube._run", return_value=Mock(stdout="")):
        assert client.download_subtitles(video, tmp_path, "en") is None
    with patch("clipper.youtube._run", return_value=Mock(stdout="")), pytest.raises(
        YouTubeError, match="without creating"
    ):
        client.download_media(video, tmp_path)
