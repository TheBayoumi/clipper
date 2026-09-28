"""Regression checks for Chrome-based playback of exact official YouTube URLs."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROBE = runpy.run_path(str(ROOT / "scripts" / "tjr_browser_probe.py"))
parse_player_response = PROBE["parse_player_response"]
select_browser_formats = PROBE["select_browser_formats"]
trusted_media_url = PROBE["trusted_media_url"]


def test_chrome_watch_page_json_is_parsed_without_exposing_media_urls() -> None:
    player = {"videoDetails": {"videoId": "p2LU37eat70"}, "playabilityStatus": {"status": "OK"}}
    html = "<script>var ytInitialPlayerResponse = " + json.dumps(player) + ";</script>"
    parsed = parse_player_response(html)
    assert parsed is not None
    assert parsed["videoDetails"]["videoId"] == "p2LU37eat70"


@pytest.mark.parametrize(
    "url",
    [
        "https://googlevideo.com.evil.test/videoplayback?id=1",
        "https://youtube.com/videoplayback?id=1",
        "http://r1.googlevideo.com/videoplayback?id=1",
        "https://r1.googlevideo.com/private-data?id=1",
        "file:///etc/passwd",
    ],
)
def test_browser_transport_rejects_untrusted_media_hosts(url: str) -> None:
    assert not trusted_media_url(url)


def test_browser_transport_chooses_real_hd_video_plus_audio() -> None:
    page = {
        "streamingData": {
            "adaptiveFormats": [
                {
                    "mimeType": 'video/mp4; codecs="avc1"',
                    "height": 1080,
                    "bitrate": 5000000,
                    "url": "https://r1.googlevideo.com/videoplayback?itag=137",
                },
                {
                    "mimeType": 'video/webm; codecs="vp9"',
                    "height": 720,
                    "bitrate": 2000000,
                    "url": "https://r1.googlevideo.com/videoplayback?itag=247",
                },
                {
                    "mimeType": 'audio/mp4; codecs="mp4a"',
                    "bitrate": 120000,
                    "url": "https://r1.googlevideo.com/videoplayback?itag=140",
                },
            ]
        }
    }
    urls = select_browser_formats(page)
    assert urls is not None
    assert "itag=137" in urls[0]
    assert "itag=140" in urls[1]


def test_browser_transport_rejects_non_hd_and_missing_player() -> None:
    assert parse_player_response("<html>challenge</html>") is None
    assert (
        select_browser_formats(
            {
                "streamingData": {
                    "adaptiveFormats": [
                        {
                            "mimeType": "video/mp4",
                            "height": 480,
                            "url": "https://r1.googlevideo.com/videoplayback?itag=135",
                        }
                    ]
                }
            }
        )
        is None
    )


def test_browser_capture_is_full_length_and_rejects_partial_source(tmp_path: Path) -> None:
    fetch = PROBE["fetch_browser_original"]
    output = tmp_path / "source.mkv"
    commands: list[list[str]] = []
    duration = 2642.0

    def fake_run(command: list[str], **_kwargs: object) -> Mock:
        commands.append(command)
        if command[0] == "ffmpeg":
            output.write_bytes(b"verified original source")
            return Mock(returncode=0)
        if command[0] == "ffprobe":
            return Mock(returncode=0, stdout=str(duration))
        raise AssertionError(command)

    with patch.dict(
        fetch.__globals__,
        {"subprocess": Mock(run=fake_run), "probe_original": lambda _: {"width": 1920}},
    ):
        assert (
            fetch(
                "https://r1.googlevideo.com/videoplayback",
                "https://r1.googlevideo.com/videoplayback",
                output,
                expected_seconds=2642,
            )
            == "CAPTURED_HD_ORIGINAL"
        )
        assert "-t" not in commands[0]
        assert (
            fetch(
                "https://r1.googlevideo.com/videoplayback",
                "https://r1.googlevideo.com/videoplayback",
                output,
                expected_seconds=7200,
            )
            == "BROWSER_SOURCE_OUTSIDE_SUPPORTED_DURATION"
        )
        duration = 840.0
        assert (
            fetch(
                "https://r1.googlevideo.com/videoplayback",
                "https://r1.googlevideo.com/videoplayback",
                output,
                expected_seconds=2642,
            )
            == "CAPTURE_INCOMPLETE_SOURCE"
        )


def test_pinned_source_and_parallel_routing_are_explicit_in_workflow() -> None:
    workflow = (ROOT / ".github" / "workflows" / "tjr-weekly-hd.yml").read_text()
    assert (
        "ALT_EGRESS_PLATFORM: ${{ matrix.os }}\n"
        "          TJR_SOURCE_VIDEO_ID: ${{ inputs.source_video_id }}"
        in workflow
    )
    assert "fromJSON(inputs.source_video_id != ''" in workflow
    assert "max-parallel: 2" in workflow
