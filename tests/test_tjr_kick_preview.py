"""Offline regression checks for direct official Kick-source video rendering."""

import runpy
import sys
from pathlib import Path

import pytest

from clipper.models import ClipCandidate, TranscriptSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
script = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "tjr_kick_preview.py")
)
assert_official_vod = script["assert_official_vod"]
PINNED_VODS = script["PINNED_VODS"]
skip_high_risk_context = script["skip_high_risk_context"]
_trusted_kick_playlist = script["_trusted_kick_playlist"]


@pytest.mark.parametrize("url", PINNED_VODS)
def test_only_known_official_tjr_vods(url: str) -> None:
    video_id = assert_official_vod(url)
    assert len(video_id) == 36


@pytest.mark.parametrize(
    "url",
    [
        "https://kick.com/anothercreator/videos/01a0aa4e-7680-7082-ae4f-62301f7a038b",
        "http://kick.com/tjr/videos/01a0aa4e-7680-7082-ae4f-62301f7a038b",
        "https://evil.invalid/tjr/videos/01a0aa4e-7680-7082-ae4f-62301f7a038b",
        "https://kick.com/tjr/videos/unauthorized",
    ],
)
def test_non_allowlisted_sources_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="pinned official"):
        assert_official_vod(url)


def test_official_feed_can_select_fresh_vods() -> None:
    fresh = "https://kick.com/tjr/videos/01a0faaa-1234-7678-abcd-123456789abc"
    assert assert_official_vod(fresh, allow_unpinned=True) == fresh.rsplit("/", 1)[-1]
    with pytest.raises(ValueError):
        assert_official_vod(fresh)


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://stream.kick.com/media/hls/master.m3u8", True),
        ("https://video.kickcdn.com/vod/master.m3u8", True),
        ("http://stream.kick.com/master.m3u8", False),
        ("https://untrusted.example/master.m3u8", False),
        ("https://stream.kick.com/not-a-video.mp4", False),
    ],
)
def test_only_expected_kick_playback_hosts(url: str, allowed: bool) -> None:
    assert _trusted_kick_playlist(url) is allowed


def test_high_risk_spoken_context_is_not_selected() -> None:
    transcript = [
        TranscriptSegment(0, 3, "Don't copy a trade without a plan"),
        TranscriptSegment(14, 16, "That person's retarded"),
        TranscriptSegment(44, 47, "Set a stop-loss before entry"),
    ]
    candidates = [
        ClipCandidate("official-tjr", 0, 30, "opening moment", 8.0),
        ClipCandidate("official-tjr", 31, 60, "independent advice", 7.0),
    ]
    assert skip_high_risk_context(candidates, transcript) == [candidates[1]]
