"""Offline enforcement of the Reach Double Coverage YouTube source allowlist."""

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
dynamic_bgutil_variants = SCRIPT["dynamic_bgutil_variants"]
dynamic_browser_variants = SCRIPT["dynamic_browser_variants"]
dynamic_token_variants = SCRIPT["dynamic_token_variants"]
constrain_official_sources = SCRIPT["constrain_official_sources"]
_flat_channel_playlist = SCRIPT["_flat_channel_playlist"]
discover_official_uploads = SCRIPT["discover_official_uploads"]


def atom_feed(channel_id: str, entry_channel: str | None = None) -> bytes:
    owner = entry_channel or channel_id
    return f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <yt:channelId>{channel_id}</yt:channelId>
  <entry>
    <yt:videoId>8PYgFVB0GHE</yt:videoId>
    <yt:channelId>{owner}</yt:channelId>
    <title>Official Double Coverage content</title>
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
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    raw = atom_feed(channel).replace(channel.encode(), channel.removeprefix("UC").encode())
    videos = parse_official_feed(raw, channel)
    assert len(videos) == 1
    assert videos[0].channel_id == channel


def test_wrong_feed_owner_fails_closed() -> None:
    with pytest.raises(ValueError, match="owner mismatch"):
        parse_official_feed(atom_feed("UC-not-approved"), "UCf1q6dhccWr6eQEcFFnJSbA")


def test_root_atom_channel_id_is_accepted_if_yt_channel_id_is_absent() -> None:
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
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
        atom_feed("UCf1q6dhccWr6eQEcFFnJSbA", entry_channel="UC-not-approved"),
        "UCf1q6dhccWr6eQEcFFnJSbA",
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
        "UCf1q6dhccWr6eQEcFFnJSbA",
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
        "UCf1q6dhccWr6eQEcFFnJSbA",
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


def test_metadata_rejects_live_and_upcoming_streams() -> None:
    official = OfficialVideo(
        "8PYgFVB0GHE",
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "Scheduled stream",
        "2026-09-30T12:00:00+00:00",
    )
    for live_status in ("is_live", "is_upcoming"):
        response = Mock(
            stdout=json.dumps(
                {
                    "id": official.video_id,
                    "channel_id": official.channel_id,
                    "duration": 900,
                    "live_status": live_status,
                }
            )
        )
        with (
            patch.dict(
                verified_youtube_metadata.__globals__,
                {"invoke": Mock(return_value=response)},
            ),
            pytest.raises(RuntimeError, match="metadata unavailable"),
        ):
            verified_youtube_metadata(official)


def test_campaign_prefers_a_recent_eligible_full_video_to_newer_hashtag_shorts() -> None:
    full = OfficialVideo(
        "p2LU37eat70",
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "Live Day Trading Making $18,350",
        "2026-09-25T16:01:53+00:00",
    )
    short = OfficialVideo(
        "Uwlp9JBpdLc",
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "Method #tjrtrades #tjr",
        "2026-09-26T00:58:31+00:00",
    )
    unknown = OfficialVideo(
        "5JR-dmmcpfw",
        "UCf1q6dhccWr6eQEcFFnJSbA",
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
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "Live Day Trading Making $18,350",
        "2026-09-25T16:01:53+00:00",
    )
    other = OfficialVideo(
        "D9J3-dqV6JI",
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "TJR Reacts to the TJR and Aiden videos",
        "2026-09-25T13:54:41+00:00",
    )
    assert constrain_official_sources([other, official], official.video_id) == [official]
    with pytest.raises(RuntimeError, match="not in"):
        constrain_official_sources([other, official], "aaaaaaaaaaa")
    with pytest.raises(RuntimeError, match="exactly 11"):
        constrain_official_sources([other, official], "not-a-video-id")


def test_channel_bound_discovery_never_spills_and_is_newest_first() -> None:
    target = "UCf1q6dhccWr6eQEcFFnJSbA"
    other_channel = "UC-not-approved-secondary"
    older_target = OfficialVideo(
        "AAAABBBB111",
        target,
        "Older target-channel upload",
        "2026-09-29T18:00:00Z",
    )
    newest_target = OfficialVideo(
        "EEEEFFFF333",
        target,
        "Newest target-channel upload",
        "2026-09-30T00:30:00Z",
    )
    newer_other_video = OfficialVideo(
        "CCCCDDDD222",
        other_channel,
        "Newer upload on the other channel",
        "2026-09-30T01:00:00Z",
    )
    selected = constrain_official_sources(
        [newer_other_video, older_target, newest_target],
        None,
        target_channel_id=target,
    )
    assert selected == [newest_target, older_target]


def test_explicit_video_must_belong_to_target_channel() -> None:
    target = "UCf1q6dhccWr6eQEcFFnJSbA"
    other_channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    other_video = OfficialVideo(
        "CCCCDDDD222",
        other_channel,
        "Other-channel upload",
        "2026-09-30T01:00:00Z",
    )
    with pytest.raises(RuntimeError, match="target Reach-listed channel"):
        constrain_official_sources(
            [other_video],
            other_video.video_id,
            target_channel_id=target,
        )


def test_bgutil_tokens_are_primary_and_loopback_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    browser = tmp_path / "chrome"
    browser.write_text("test executable", encoding="utf-8")
    monkeypatch.setenv("TJR_BGUTIL_POT_PROVIDER_URL", "http://127.0.0.1:4416")
    monkeypatch.setenv("YT_DLP_WPC_BROWSER_PATH", str(browser))
    variants = dynamic_token_variants()
    assert len(variants) == 4
    assert "youtube:player_client=mweb" in variants[0]
    assert "youtubepot-bgutilhttp:base_url=http://127.0.0.1:4416" in variants[0]
    assert "youtubepot-wpc:browser_path=" + str(browser) in variants[2]

    monkeypatch.setenv("TJR_BGUTIL_POT_PROVIDER_URL", "http://0.0.0.0:4416")
    with pytest.raises(RuntimeError, match="runner loopback"):
        dynamic_bgutil_variants()


def test_dynamic_guest_tokens_do_not_require_exported_account_cookies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    browser = tmp_path / "chrome"
    browser.write_text("test executable", encoding="utf-8")
    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.setenv("YT_DLP_WPC_BROWSER_PATH", str(browser))
    variants = dynamic_browser_variants()
    assert len(variants) == 2
    assert all("youtubepot-wpc:browser_path=" + str(browser) in variant for variant in variants)
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


def test_required_staged_original_never_falls_back_to_another_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.tjr_youtube_preview import render_youtube_previews

    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    official = OfficialVideo("p2LU37eat70", channel, "Official", "2026-09-25T16:00:00Z")
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", official.video_id)
    monkeypatch.setenv("TJR_BROWSER_CAPTURE_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setenv("TJR_REQUIRE_STAGED_ORIGINAL", "1")
    with (
        patch.dict(
            render_youtube_previews.__globals__,
            {
                "discover_official_uploads": lambda: ([official], []),
                "verified_youtube_metadata": Mock(
                    side_effect=AssertionError("Unverified fallback must not run")
                ),
            },
        ),
        pytest.raises(RuntimeError, match="required approved staged original"),
    ):
        render_youtube_previews(tmp_path / "artifacts", Path("campaigns/reach-double-coverage-dedicated.yaml"))


def test_failed_editorial_writes_transcript_and_screening_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clipper.models import TranscriptSegment
    from scripts.tjr_youtube_preview import render_youtube_previews

    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    official = OfficialVideo("X7msxvyQd_U", channel, "LIVE TRADING", "2026-09-27T15:45:16Z")
    media = tmp_path / "source.mp4"
    media.write_bytes(b"verified source fixture")
    manifest = tmp_path / "stage.json"
    manifest.write_text("verified fixture", encoding="utf-8")
    monkeypatch.setenv("TJR_BROWSER_CAPTURE_FILE", str(manifest))
    monkeypatch.setenv("TJR_REQUIRE_STAGED_ORIGINAL", "1")
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", official.video_id)
    segments = [
        TranscriptSegment(0, 8, "Welcome back everybody for this stream today"),
        TranscriptSegment(8, 17, "We are waiting for the chart to load"),
        TranscriptSegment(17, 27, "The chart is still loading please stand by."),
    ]
    with (
        patch.dict(
            render_youtube_previews.__globals__,
            {
                "discover_official_uploads": lambda: ([official], []),
                "load_verified_browser_original": lambda *_: (
                    official,
                    media,
                    {"title": official.title, "duration": 100},
                ),
                "probe_source_profile": lambda *_: Mock(as_dict=lambda: {"fps": "60/1"}),
                "probe_original": lambda *_: {"width": 1920, "height": 1080},
                "transcribe_source_chunks": lambda *_args, **_kwargs: ([segments], 100.0),
                "build_semantic_editorial_candidates": lambda *_args, **_kwargs: (
                    [],
                    {
                        "architecture": "source_level_semantic_event_segmentation_v1",
                        "candidate_count": 0,
                        "fixed_candidate_or_output_quota": False,
                    },
                ),
            },
        ),
    ):
        render_youtube_previews(tmp_path / "renders", Path("campaigns/reach-double-coverage-dedicated.yaml"))
    runs = list((tmp_path / "renders").glob("reach-tjr-youtube-*"))
    assert len(runs) == 1
    run = runs[0]
    assert len(json.loads((run / "transcript.json").read_text())) == 3
    audit = json.loads((run / "editorial-candidate-audit.json").read_text())
    assert audit["selected"] == []
    assert audit["semantic_candidate_count"] == 0
    assert audit["semantic_architecture"]["fixed_candidate_or_output_quota"] is False
    assert audit["screening_mode"] == audit["semantic_architecture"]["architecture"]
    report = json.loads((run / "tjr-youtube-qa-report.json").read_text())
    assert report["status"] == "NO_CREATOR_GRADE_MOMENTS"
    assert report["clips"] == []
    assert report["selection_policy"] == "quality_driven_zero_to_n"
    assert "ranked-candidates.json" in {p.name for p in run.iterdir()}


def test_official_playlist_finds_older_full_video_without_third_party_reposts() -> None:
    import json

    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    official_long = {
        "id": "ABCD1234xyz",
        "channel_id": channel,
        "title": "Double Coverage Podcast Episode",
        "duration": 2400,
        "timestamp": 1790451200,
    }
    repost = {
        "id": "badR3post_X",
        "channel_id": "UC-foreign-channel",
        "title": "Reposted Double Coverage compilation",
        "duration": 1200,
    }
    result = Mock(stdout="\n".join(json.dumps(x) for x in (official_long, repost)))
    with patch.dict(_flat_channel_playlist.__globals__, {"invoke": Mock(return_value=result)}):
        found = _flat_channel_playlist(channel, section="streams", limit=36)
    assert [video.video_id for video in found] == ["ABCD1234xyz"]
    assert found[0].duration_seconds == 2400
    with pytest.raises(ValueError, match="approved playlist"):
        _flat_channel_playlist("UC-foreign-channel", section="videos")


def test_short_filled_rss_is_enriched_with_older_official_long_form() -> None:
    import io

    short_feed = {
        channel: atom_feed(channel).replace(b"Official Double Coverage content", b"#DoubleCoverage")
        for channel in CHANNELS
    }
    long_channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    older = OfficialVideo(
        "ABCD1234xyz",
        long_channel,
        "Double Coverage Podcast Episode",
        "",
        duration_seconds=3600,
    )
    calls: list[tuple[str, str]] = []

    def response(request: object, timeout: int) -> io.BytesIO:
        del timeout
        address = request.full_url
        channel = address.split("channel_id=", 1)[1]
        return io.BytesIO(short_feed[channel])

    def playlist(channel: str, *, section: str = "videos", limit: int = 36) -> list[OfficialVideo]:
        del limit
        calls.append((channel, section))
        return [older] if channel == long_channel and section == "streams" else []

    with (
        patch("urllib.request.urlopen", side_effect=response),
        patch.dict(
            discover_official_uploads.__globals__,
            {"_flat_channel_playlist": playlist},
        ),
    ):
        videos, _ = discover_official_uploads()
    assert (long_channel, "streams") in calls
    assert {video.video_id for video in videos} >= {"8PYgFVB0GHE", older.video_id}
    assert prioritize_campaign_moments(videos)[0].video_id == older.video_id


def test_playlist_duration_prevents_known_short_from_blocking_long_source() -> None:
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    short = OfficialVideo("8PYgFVB0GHE", channel, "Great trading", "2026-09-27", 31)
    older = OfficialVideo("ABCD1234xyz", channel, "Market update", "2026-09-25", 1800)
    assert prioritize_campaign_moments([short, older])[0] == older


def test_full_livestream_scan_is_not_cut_to_fourteen_minutes() -> None:
    from scripts.tjr_youtube_preview import youtube_scan_section_args

    assert youtube_scan_section_args(2642) == []
    assert youtube_scan_section_args(3600) == []
    with pytest.raises(ValueError, match="SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT"):
        youtube_scan_section_args(7200)


def test_chunked_asr_preserves_absolute_video_and_word_offsets(
    tmp_path: Path,
) -> None:
    from clipper.models import TranscriptSegment, WordTiming
    from scripts.tjr_youtube_preview import transcribe_source_chunks

    source = tmp_path / "source.mp4"
    source.touch()
    received: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> Mock:
        received.append(command)
        if command[0] == "ffprobe":
            return Mock(stdout="2642.0\n")
        if command[0] == "ffmpeg":
            Path(command[-1]).touch()
            return Mock(returncode=0)
        raise AssertionError(command)

    initialized = 0
    transcribed = 0

    class FakeTranscriber:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal initialized
            initialized += 1

        def transcribe(self, *_args: object, **_kwargs: object) -> list[TranscriptSegment]:
            nonlocal transcribed
            transcribed += 1
            return [
                TranscriptSegment(
                    1.0,
                    2.0,
                    "Trade now.",
                    words=(WordTiming(1.0, 1.4, "Trade"), WordTiming(1.4, 2.0, "now.")),
                )
            ]

    with patch.dict(
        transcribe_source_chunks.__globals__,
        {
            "subprocess": Mock(run=fake_run),
            "FasterWhisperTranscriber": FakeTranscriber,
        },
    ):
        chunks, duration = transcribe_source_chunks(source, tmp_path)
    assert duration == 2642
    assert len(chunks) == 4
    assert [chunk[0].start for chunk in chunks] == [1, 841, 1681, 2521]
    assert [chunk[0].words[0].start for chunk in chunks] == [1, 841, 1681, 2521]
    assert len([call for call in received if call[0] == "ffmpeg"]) == 4
    assert initialized == 1
    assert transcribed == 4
    assert not list((tmp_path / "work").glob("audio-chunk-*.wav"))


def test_editorial_scoring_sees_neighbors_across_audio_chunk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clipper.models import TranscriptSegment
    from scripts.tjr_youtube_preview import render_youtube_previews

    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    original = OfficialVideo("X7msxvyQd_U", channel, "Trade setup", "2026-09-27T15:00:00Z")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"test source")
    manifest = tmp_path / "stage.json"
    manifest.write_text("fixture", encoding="utf-8")
    first = TranscriptSegment(835, 839.5, "we are not quite ready to")
    second = TranscriptSegment(840, 844, "enter until the price returns.")
    scored: list[list[TranscriptSegment]] = []

    def inspect_candidates(
        _brief: object, _video_id: str, words: list[TranscriptSegment], **_kwargs: object
    ) -> tuple[list[object], dict[str, object]]:
        scored.append(list(words))
        return [], {
            "architecture": "source_level_semantic_event_segmentation_v1",
            "candidate_count": 0,
        }

    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", original.video_id)
    monkeypatch.setenv("TJR_REQUIRE_STAGED_ORIGINAL", "1")
    monkeypatch.setenv("TJR_BROWSER_CAPTURE_FILE", str(manifest))
    with (
        patch.dict(
            render_youtube_previews.__globals__,
            {
                "discover_official_uploads": lambda: ([original], []),
                "load_verified_browser_original": lambda *_: (
                    original,
                    source,
                    {"title": original.title, "duration": 900},
                ),
                "probe_source_profile": lambda *_: Mock(as_dict=lambda: {"fps": "60/1"}),
                "probe_original": lambda *_: {"width": 1920, "height": 1080},
                "transcribe_source_chunks": lambda *_a, **_kw: ([[first], [second]], 900.0),
                "build_semantic_editorial_candidates": inspect_candidates,
            },
        ),
    ):
        render_youtube_previews(tmp_path / "renders", Path("campaigns/reach-double-coverage-dedicated.yaml"))
    assert scored and all(items == [first, second] for items in scored)


def test_campaign_source_cutoff_rejects_old_and_unknown_uploads() -> None:
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    old = OfficialVideo("ABCD1234xyz", channel, "Old show", "2026-08-29T18:00:00Z", 2200)
    unknown = OfficialVideo("ABCD1234xyy", channel, "Unverified date", "", 1800)
    fresh = OfficialVideo("X7msxvyQd_U", channel, "Current trading", "2026-09-27", 2200)
    cutoff = "2026-09-01T00:00:00Z"
    assert constrain_official_sources([old, unknown, fresh], None, published_after=cutoff) == [
        fresh
    ]
    with pytest.raises(RuntimeError, match="not in channel"):
        constrain_official_sources([old, fresh], old.video_id, published_after=cutoff)


def test_direct_download_requires_full_verified_duration(tmp_path: Path) -> None:
    from scripts.tjr_youtube_preview import (
        download_original_excerpt,
        verify_complete_download,
    )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"valid-looking partial media")
    verify = verify_complete_download
    with (
        patch.dict(
            verify.__globals__,
            {"subprocess": Mock(run=lambda *_a, **_kw: Mock(returncode=0, stdout="840\n"))},
        ),
        pytest.raises(RuntimeError, match="SOURCE_DURATION_INCOMPLETE"),
    ):
        verify(source, 2642)
    attempts: list[list[str]] = []

    def download(command: list[str], *, timeout: int) -> Mock:
        del timeout
        attempts.append(command)
        source.write_bytes(b"incomplete source fragment")
        return Mock(returncode=0)

    video = OfficialVideo(
        "X7msxvyQd_U",
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "Verified video",
        "2026-09-27T15:00:00Z",
        2642,
    )
    with (
        patch.dict(
            download_original_excerpt.__globals__,
            {
                "_auth_args": lambda: [],
                "dynamic_browser_variants": lambda: (),
                "invoke": download,
                "probe_original": lambda _: {"width": 1920, "height": 1080},
                "verify_complete_download": Mock(
                    side_effect=RuntimeError("SOURCE_DURATION_INCOMPLETE")
                ),
            },
        ),
        pytest.raises(RuntimeError, match="SOURCE_DURATION_INCOMPLETE"),
    ):
        download_original_excerpt(video, tmp_path, metadata={"duration": 2642})
    assert attempts
    assert all("--abort-on-unavailable-fragments" in args for args in attempts)
    assert not list(tmp_path.glob("source.*"))


def test_direct_mode_artifacts_include_all_referenced_qa_evidence() -> None:
    workflow = (ROOT / ".github" / "workflows" / "tjr-weekly-hd.yml").read_text(encoding="utf-8")
    for root in ("tjr-youtube-artifacts",):
        for suffix in (
            "**/clips/*.ass",
            "**/clips/*.quality.json",
            "**/clips/*.ssim.txt",
            "**/clips/*-contact.png",
            "**/source-analysis-coverage.json",
        ):
            assert f"{root}/{suffix}" in workflow


def test_double_coverage_uses_campaign_default_layout_and_required_watermark() -> None:
    source = (ROOT / "scripts" / "tjr_youtube_preview.py").read_text(encoding="utf-8")
    assert 'layout = "default"' in source
    assert "double-coverage-watermark.png" in source
    assert 'chosen_video.video_id == "LvnemCfJpQU"' not in source
    assert 'chosen_video.video_id == "p2LU37eat70"' not in source
