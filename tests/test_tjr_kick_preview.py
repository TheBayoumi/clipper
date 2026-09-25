"""Offline regression checks for direct official Kick-source video rendering."""
import runpy
from pathlib import Path

import pytest

script = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "tjr_kick_preview.py")
)
assert_official_vod = script["assert_official_vod"]
PINNED_VODS = script["PINNED_VODS"]


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
