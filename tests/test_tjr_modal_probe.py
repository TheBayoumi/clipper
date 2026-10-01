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


def test_modal_discovery_binds_channel_before_source_ordering() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "tjr_modal_probe.py").read_text(
        encoding="utf-8"
    )
    assert "target_channel_id=channel_id or None" in source
    assert "official = [item for item in official if item.channel_id == channel_id]" not in source


def test_ip_challenge_skips_only_one_video_not_entire_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    first, second = "lxu_J1Ec1XI", "X7msxvyQd_U"
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
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
                "duration": 3400,
                "live_status": "not_live",
                "title": "DOUBLE COVERAGE PODCAST",
                "formats": [{"width": 1920, "height": 1080, "vcodec": "avc1"}],
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
    assert urls_seen.count(f"https://www.youtube.com/watch?v={first}") == 5
    assert urls_seen[-1].endswith(second)
    assert any(
        attempt["reason"] == "YOUTUBE_IP_OR_LOGIN_CHALLENGE" for attempt in result["attempts"]
    )


def test_all_blocked_videos_still_report_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
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
    assert calls == 10
    assert len({attempt["url"] for attempt in result["attempts"]}) == 2


def test_region_aborts_after_two_distinct_videos_are_ip_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
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
    assert len(calls) == 10
    assert all("Xa-4kOvpGok" not in url for url in calls)


def test_plain_client_succeeds_after_provider_bot_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = _load_modal_probe(monkeypatch)
    video_id = "X7msxvyQd_U"
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
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
                "title": "DOUBLE COVERAGE PODCAST",
                "formats": [{"width": 1920, "height": 1080, "vcodec": "avc1"}],
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
    for seconds in (120, 2642, 3600):
        assert source_sections(seconds) == youtube_scan_section_args(seconds)
    for seconds in (3601, 7200):
        with pytest.raises(ValueError, match="SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT"):
            source_sections(seconds)
        with pytest.raises(ValueError, match="SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT"):
            youtube_scan_section_args(seconds)
    stage_source = inspect.getsource(script["stage_official_original"])
    assert "tjr_youtube_preview" not in stage_source
    assert "--abort-on-unavailable-fragments" in stage_source


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
        {"codec_type": "video", "width": 1920, "height": 1080},
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
    with pytest.raises(RuntimeError, match="MISSING_PRODUCTION_HD_VIDEO_STREAM"):
        verify(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "width": 960, "height": 720},
                        {"codec_type": "audio"},
                    ],
                    "format": {"duration": "2642"},
                }
            ),
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
    # A two-hour original is unsupported, never mislabeled as fully staged.
    with pytest.raises(RuntimeError, match="SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT"):
        verify(
            returncode=0,
            stdout=json.dumps({"streams": good_streams, "format": {"duration": "3600"}}),
            stderr="",
            expected_seconds=7200,
        )


def test_modal_route_count_is_bounded() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "tjr_modal_probe.py"
    script = path.read_text(encoding="utf-8")
    assert "MAX_MODAL_EGRESS_ATTEMPTS = 1" in script
    assert "routes[:MAX_MODAL_EGRESS_ATTEMPTS]" in script


@pytest.mark.parametrize("fail_inside", [False, True])
def test_viewer_cookie_secret_is_ephemeral_and_used_by_extractor(
    monkeypatch: pytest.MonkeyPatch, fail_inside: bool
) -> None:
    import base64

    mock_modal = SimpleNamespace(
        App=lambda *_args, **_kwargs: _FakeApp(), Image=MagicMock(), Volume=MagicMock()
    )
    monkeypatch.setitem(sys.modules, "modal", mock_modal)
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "tjr_modal_probe.py")
    )
    cookie = b"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSESSION\tfake\n"
    monkeypatch.setenv("TJR_YOUTUBE_COOKIES_B64", base64.b64encode(cookie).decode())
    monkeypatch.setenv("YOUTUBE_COOKIES_FILE", "previous-session")
    cookie_path = None
    try:
        with script["viewer_cookie_session"]():
            args = script["viewer_cookie_args"]()
            cookie_path = Path(args[1])
            assert args[0] == "--cookies"
            assert cookie_path.read_bytes() == cookie
            assert cookie_path.stat().st_mode & 0o777 == 0o600
            assert (
                args
                == script["_yt_command"]((), "https://www.youtube.com/watch?v=976-d0RlyfQ")[-3:-1]
            )
            if fail_inside:
                raise RuntimeError("simulated transport failure")
    except RuntimeError:
        assert fail_inside
    assert cookie_path is not None and not cookie_path.exists()
    import os

    assert os.environ["YOUTUBE_COOKIES_FILE"] == "previous-session"


@pytest.mark.parametrize("encoded", ["not base64", "c2VjcmV0"])
def test_invalid_viewer_secret_fails_without_exposing_value(
    monkeypatch: pytest.MonkeyPatch, encoded: str
) -> None:
    mock_modal = SimpleNamespace(
        App=lambda *_args, **_kwargs: _FakeApp(), Image=MagicMock(), Volume=MagicMock()
    )
    monkeypatch.setitem(sys.modules, "modal", mock_modal)
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "tjr_modal_probe.py")
    )
    monkeypatch.setenv("TJR_YOUTUBE_COOKIES_B64", encoded)
    with (
        pytest.raises(RuntimeError, match="INVALID_YOUTUBE_VIEWER_SESSION") as exc,
        script["viewer_cookie_session"](),
    ):
        pytest.fail("Invalid credentials must not reach acquisition")
    assert encoded not in str(exc.value)


def test_late_cookie_free_strategy_is_not_suppressed_by_two_challenges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_modal_probe(monkeypatch)
    video_id = "X7msxvyQd_U"
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    calls = 0

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        if "--dump-single-json" in command:
            calls += 1
            assert "--cookies" not in command
            if calls < 5:
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="Sign in to confirm you're not a bot"
                )
            assert "youtube:player_client=web_embedded,android_vr" in command
            metadata = {
                "id": video_id,
                "channel_id": channel,
                "duration": 600,
                "formats": [{"width": 1920, "height": 1080, "vcodec": "avc1"}],
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metadata), stderr="")
        assert "--test" in command
        template = Path(command[command.index("-o") + 1])
        Path(str(template).replace("%(ext)s", "mp4")).write_bytes(b"real test bytes" * 100)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.delenv("TJR_YOUTUBE_COOKIES_B64", raising=False)
    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    with patch("subprocess.run", side_effect=run):
        result = probe([{"video_id": video_id, "channel_id": channel}])
    assert calls == 5
    assert result["status"] == "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED"
    assert result["transport_strategy"] == "bgutil_embedded_android_vr"


def test_browser_guest_uses_isolated_provider_and_player_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_modal_probe(monkeypatch)
    functions = probe.__globals__
    monkeypatch.setenv("TJR_YOUTUBE_SESSION_MODE", "browser_guest")
    monkeypatch.delenv("TJR_YOUTUBE_COOKIES_B64", raising=False)
    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    strategies = functions["acquisition_strategies"]()
    assert len(strategies) == 1
    command = functions["_yt_command"](
        strategies[0][1], "https://www.youtube.com/watch?v=976-d0RlyfQ"
    )
    assert command[:3] == ["xvfb-run", "--auto-servernum", "/opt/youtube-wpc/bin/yt-dlp"]
    assert "youtube:player_client=mweb;fetch_pot=always" in command
    assert "--cookies" not in command
    assert not any("bgutil" in arg for arg in command)
    monkeypatch.setenv("TJR_YOUTUBE_COOKIES_B64", "secret-must-not-be-used")
    with pytest.raises(RuntimeError, match="GUEST_ACQUISITION_REJECTS_ACCOUNT_COOKIES"):
        probe([])


def test_browser_timeout_kills_entire_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_modal_probe(monkeypatch)
    monkeypatch.setenv("TJR_YOUTUBE_SESSION_MODE", "browser_guest")
    process = MagicMock()
    process.pid = 123456
    process.communicate.side_effect = [subprocess.TimeoutExpired("browser", 10), (b"", b"")]
    popen = MagicMock()
    popen.__enter__.return_value = process
    with patch("subprocess.Popen", return_value=popen), patch("os.killpg") as kill:
        with pytest.raises(subprocess.TimeoutExpired):
            probe.__globals__["acquisition_run"](["browser"], timeout=10)
        kill.assert_called_once()
        assert kill.call_args.args[0] == process.pid
        assert process.communicate.call_count == 2


def test_guest_watch_page_status_is_bound_to_exact_video(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_modal_probe(monkeypatch)
    parse = probe.__globals__["guest_watch_playability"]
    details = {"videoId": "X7msxvyQd_U", "channelId": "UCf1q6dhccWr6eQEcFFnJSbA"}
    html = "var ytInitialPlayerResponse = " + json.dumps(
        {"videoDetails": details, "playabilityStatus": {"status": "OK"}}
    )
    assert parse(html, details["videoId"], details["channelId"]) == "OK"
    assert parse(html, "different", details["channelId"]) == "SOURCE_IDENTITY_MISMATCH"
    assert parse("no player", details["videoId"], details["channelId"]) == "NO_PLAYER_RESPONSE"


def test_fresh_guest_profile_is_extracted_on_the_fly_and_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_modal_probe(monkeypatch)
    functions = probe.__globals__
    monkeypatch.setenv("TJR_YOUTUBE_SESSION_MODE", "browser_guest")
    monkeypatch.delenv("TJR_GUEST_BROWSER_PROFILE", raising=False)
    monkeypatch.delenv("TJR_YOUTUBE_COOKIES_B64", raising=False)
    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    candidate = {"video_id": "X7msxvyQd_U", "channel_id": "UCf1q6dhccWr6eQEcFFnJSbA"}
    profile_path = None

    def bootstrap(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--password-store=basic" in command
        assert command[-1].endswith(candidate["video_id"])
        profile = Path(
            next(arg.split("=", 1)[1] for arg in command if arg.startswith("--user-data-dir="))
        )
        profile.mkdir()
        (profile / "Cookies").write_bytes(b"fake guest database")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with (
        patch.dict(functions, {"acquisition_run": bootstrap}),
        functions["anonymous_watch_session"]([candidate]) as evidence,
    ):
        args = functions["viewer_cookie_args"]()
        assert args[0] == "--cookies-from-browser"
        profile_path = Path(args[1].split(":", 1)[1])
        assert profile_path.is_dir()
        assert evidence["account_login_used"] is False
        assert evidence["cookie_database_created"] is True
    assert profile_path is not None and not profile_path.exists()
    import os

    assert "TJR_GUEST_BROWSER_PROFILE" not in os.environ


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "socks5://host:1080",
        "http://host",
        "http://host:bad",
        "http://host:1234/path",
        "http://host:1234?rotate=1",
    ],
)
def test_static_egress_rejects_missing_or_invalid_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    functions = _load_modal_probe(monkeypatch).__globals__
    monkeypatch.setenv("TJR_YOUTUBE_SESSION_MODE", "browser_guest")
    monkeypatch.setenv("TJR_YOUTUBE_EGRESS_MODE", "static_proxy")
    monkeypatch.setenv("TJR_YOUTUBE_EGRESS_PROXY_URL", endpoint)
    with pytest.raises(RuntimeError, match="STATIC_PROXY_REQUIRES_VALID_PRIVATE_ENDPOINT"):
        functions["egress_configuration"]()


def test_static_proxy_stops_without_forwarder_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functions = _load_modal_probe(monkeypatch).__globals__
    monkeypatch.setenv("TJR_YOUTUBE_EGRESS_MODE", "static_proxy")
    monkeypatch.delenv("TJR_LOCAL_YOUTUBE_PROXY", raising=False)
    with pytest.raises(RuntimeError, match="STATIC_PROXY_FORWARDER_REQUIRED"):
        functions["_yt_command"]((), "https://www.youtube.com/watch?v=X7msxvyQd_U")
    monkeypatch.setenv("TJR_YOUTUBE_SESSION_MODE", "bgutil_guest")
    with pytest.raises(RuntimeError, match="STATIC_PROXY_REQUIRES_BROWSER_GUEST_MODE"):
        functions["egress_configuration"]()


def test_fixed_proxy_credentials_stay_out_of_commands_and_cleanup_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    functions = _load_modal_probe(monkeypatch).__globals__
    endpoint = "http://private-user:private-password@fixed.example:3128"
    monkeypatch.setenv("TJR_YOUTUBE_SESSION_MODE", "browser_guest")
    monkeypatch.setenv("TJR_YOUTUBE_EGRESS_MODE", "static_proxy")
    monkeypatch.setenv("TJR_YOUTUBE_EGRESS_PROXY_URL", endpoint)
    monkeypatch.delenv("TJR_LOCAL_YOUTUBE_PROXY", raising=False)
    process = MagicMock()
    process.stdout.readline.return_value = "http://127.0.0.1:43210\n"
    popen = MagicMock()
    popen.__enter__.return_value = process
    with (
        patch("subprocess.Popen", return_value=popen) as launch,
        patch("select.select", return_value=([process.stdout], [], [])),
        pytest.raises(ValueError, match="render failure"),
        functions["fixed_egress_session"](),
    ):
        command = functions["_yt_command"]((), "https://www.youtube.com/watch?v=X7msxvyQd_U")
        assert command[command.index("--proxy") + 1] == "http://127.0.0.1:43210"
        assert "private-password" not in str(command)
        raise ValueError("render failure")
    assert "TJR_LOCAL_YOUTUBE_PROXY" not in os.environ
    process.terminate.assert_called_once()
    assert "private-password" not in str(launch.call_args.args)
    assert launch.call_args.kwargs["env"]["TJR_YOUTUBE_EGRESS_PROXY_URL"] == endpoint


def test_direct_mode_ignores_unselected_proxy_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    functions = _load_modal_probe(monkeypatch).__globals__
    monkeypatch.setenv("TJR_YOUTUBE_EGRESS_MODE", "direct")
    monkeypatch.setenv("TJR_YOUTUBE_EGRESS_PROXY_URL", "unused-secret")
    assert functions["egress_configuration"]() == ("direct", "")
