"""Regression checks for Chrome-based playback of exact official YouTube URLs."""

import json
import runpy
import sys
from pathlib import Path

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
