"""Offline enforcement of the Reach TJR YouTube-only source allowlist."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = runpy.run_path(str(ROOT / "scripts" / "tjr_youtube_preview.py"))
parse_official_feed = SCRIPT["parse_official_feed"]
verified_youtube_metadata = SCRIPT["verified_youtube_metadata"]
select_separate_clips = SCRIPT["select_separate_clips"]
OfficialVideo = SCRIPT["OfficialVideo"]
CHANNELS = SCRIPT["CHANNELS"]
_auth_args = SCRIPT["_auth_args"]
prioritize_campaign_moments = SCRIPT["prioritize_campaign_moments"]
dynamic_browser_variants = SCRIPT["dynamic_browser_variants"]
constrain_official_sources = SCRIPT["constrain_official_sources"]


def atom_feed(channel_id: str, entry_channel: str | None = None) -> bytes:
    owner = entry_channel or channel_id
    return f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <yt:channelId>{channel_id}</yt:channelId>
  <entry>
    <yt:videoId>8PYgFVB0GHE</yt:videoId>
    <yt:channelId>{owner}</yt:channelId>
    <title>Official TJR content</title>
    <published>2026-09-25T12:00:00+00:00</published>
  </entry>
</feed>""".encode()


@pytest.mark.parametrize("channel_id", list(CHANNELS))
def test_only_official_reach_channel_feeds_are_accepted(channel_id: str) -> None:
    videos = parse_official_feed(atom_feed(channel_id), channel_id)
    assert len(videos) == 1
    assert videos[0].channel_id == channel_id
    assert videos[0].url == "https://www.youtube.com/watch?v=8PYgFVB0GHE"


def test_live_official_feed_with_uc_prefix_elided_is_normalized() -> None:
    channel = "UCGHBUXjDCeiIXNdKR0HUZnA"
    raw = atom_feed(channel).replace(channel.encode(), channel.removeprefix("UC").encode())
    videos = parse_official_feed(raw, channel)
    assert len(videos) == 1
    assert videos[0].channel_id == channel


def test_wrong_feed_owner_fails_closed() -> None:
    with pytest.raises(ValueError, match="owner mismatch"):
        parse_official_feed(atom_feed("UC-not-approved"), "UCGHBUXjDCeiIXNdKR0HUZnA")


def test_root_atom_channel_id_is_accepted_if_yt_channel_id_is_absent() -> None:
    channel = "UCGHBUXjDCeiIXNdKR0HUZnA"
    source = atom_feed(channel)
    source = source.replace(
        f"<yt:channelId>{channel}</yt:channelId>".encode(),
        f"<id>yt:channel:{channel}</id>".encode(),
        1,
    )
    videos = parse_official_feed(source, channel)
    assert len(videos) == 1
    assert videos[0].channel_id == channel


def test_foreign_video_entries_are_ignored() -> None:
    result = parse_official_feed(
        atom_feed("UCGHBUXjDCeiIXNdKR0HUZnA", entry_channel="UC-not-approved"),
        "UCGHBUXjDCeiIXNdKR0HUZnA",
    )
    assert result == []


def test_unlisted_channel_is_never_a_source() -> None:
    with pytest.raises(ValueError, match="non-campaign"):
        parse_official_feed(atom_feed("UC-not-approved"), "UC-not-approved")


def test_independent_moments_do_not_overlap() -> None:
    from clipper.models import ClipCandidate

    candidates = [
        ClipCandidate("8PYgFVB0GHE", 0, 31, "first", 10.0),
        ClipCandidate("8PYgFVB0GHE", 8, 35, "overlap", 9.0),
        ClipCandidate("8PYgFVB0GHE", 42, 72, "separate", 8.0),
    ]
    chosen = select_separate_clips(candidates)
    assert [(clip.start, clip.end) for clip in chosen] == [(0, 31), (42, 72)]


def test_youtube_metadata_must_match_original_video_and_exact_channel() -> None:
    official = OfficialVideo(
        "8PYgFVB0GHE",
        "UCGHBUXjDCeiIXNdKR0HUZnA",
        "Original title",
        "2026-09-25T12:00:00+00:00",
    )
    wrong = Mock(
        stdout=json.dumps(
            {
                "id": official.video_id,
                "channel_id": "UC-third-party-repost",
                "duration": 600,
            }
        )
    )
    with (
        patch.dict(verified_youtube_metadata.__globals__, {"invoke": Mock(return_value=wrong)}),
        pytest.raises(RuntimeError, match="metadata unavailable"),
    ):
        verified_youtube_metadata(official)


def test_metadata_accepts_matching_official_channel_and_nonlive_video() -> None:
    official = OfficialVideo(
        "8PYgFVB0GHE",
        "UCGHBUXjDCeiIXNdKR0HUZnA",
        "Original title",
        "2026-09-25T12:00:00+00:00",
    )
    valid = Mock(
        stdout=json.dumps(
            {
                "id": official.video_id,
                "channel_id": official.channel_id,
                "duration": 900,
                "live_status": "was_live",
            }
        )
    )
    with patch.dict(verified_youtube_metadata.__globals__, {"invoke": Mock(return_value=valid)}):
        assert verified_youtube_metadata(official)["channel_id"] == official.channel_id


def test_campaign_prefers_a_recent_eligible_full_video_to_newer_hashtag_shorts() -> None:
    full = OfficialVideo(
        "p2LU37eat70",
        "UCZen39LQJPx04GjPj7FOMcw",
        "Live Day Trading Making $18,350",
        "2026-09-25T16:01:53+00:00",
    )
    short = OfficialVideo(
        "Uwlp9JBpdLc",
        "UCGHBUXjDCeiIXNdKR0HUZnA",
        "Method #tjrtrades #tjr",
        "2026-09-26T00:58:31+00:00",
    )
    unknown = OfficialVideo(
        "5JR-dmmcpfw",
        "UCGHBUXjDCeiIXNdKR0HUZnA",
        "unknown",
        "2026-09-25T22:04:46+00:00",
    )
    ordered = prioritize_campaign_moments([short, unknown, full])
    assert ordered == [full, short, unknown]


def test_optional_auth_uses_only_explicit_existing_cookie_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    assert _auth_args() == []
    jar = tmp_path / "dedicated-viewer-cookies.txt"
    jar.write_text("# Netscape HTTP Cookie File\\n", encoding="utf-8")
    monkeypatch.setenv("YOUTUBE_COOKIES_FILE", str(jar))
    assert _auth_args() == ["--cookies", str(jar)]
    jar.unlink()
    with pytest.raises(RuntimeError, match="empty or unavailable"):
        _auth_args()


def test_explicit_youtube_source_cannot_fall_back_to_other_videos() -> None:
    official = OfficialVideo(
        "p2LU37eat70",
        "UCZen39LQJPx04GjPj7FOMcw",
        "Live Day Trading Making $18,350",
        "2026-09-25T16:01:53+00:00",
    )
    other = OfficialVideo(
        "D9J3-dqV6JI",
        "UCZen39LQJPx04GjPj7FOMcw",
        "TJR Reacts to the TJR and Aiden videos",
        "2026-09-25T13:54:41+00:00",
    )
    assert constrain_official_sources([other, official], official.video_id) == [official]
    with pytest.raises(RuntimeError, match="not in either Reach-listed"):
        constrain_official_sources([other, official], "aaaaaaaaaaa")
    with pytest.raises(RuntimeError, match="exactly 11"):
        constrain_official_sources([other, official], "not-a-video-id")


def test_dynamic_guest_tokens_do_not_require_exported_account_cookies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    browser = tmp_path / "chrome"
    browser.write_text("test executable", encoding="utf-8")
    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.setenv("YT_DLP_WPC_BROWSER_PATH", str(browser))
    variants = dynamic_browser_variants()
    assert len(variants) == 2
    assert all(
        "youtubepot-wpc:browser_path=" + str(browser) in variant
        for variant in variants
    )
    assert "youtube:player_client=mweb" in variants[0]
    assert _auth_args() == []


def test_dynamic_browser_is_optional_but_bad_config_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("YT_DLP_WPC_BROWSER_PATH", raising=False)
    assert dynamic_browser_variants() == ()
    monkeypatch.setenv("YT_DLP_WPC_BROWSER_PATH", str(tmp_path / "missing-chrome"))
    with pytest.raises(RuntimeError, match="browser executable is missing"):
        dynamic_browser_variants()
