from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "niche_video.py"
_SPEC = importlib.util.spec_from_file_location("niche_video", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
nv = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = nv
_SPEC.loader.exec_module(nv)


@pytest.fixture
def config(tmp_path: Path) -> nv.NicheConfig:
    config_file = tmp_path / "niche.yaml"
    config_file.write_text(
        "niche: AI tips\n"
        "audience: Builders\n"
        "brand: WORKFLOW LAB\n"
        "topic: Agent reliability\n"
        "voice: marin\n"
        "facts: []\n",
        encoding="utf-8",
    )
    return nv.NicheConfig.load(config_file)


@pytest.fixture
def episode() -> nv.Episode:
    return nv.Episode.from_json(
        {
            "title": "Reliable agents",
            "scenes": [
                {
                    "headline": "The problem",
                    "narration": "An agent without boundaries can make your workflow much less predictable over time.",
                },
                {
                    "headline": "The solution",
                    "narration": "Define the allowed actions clearly and validate each tool result before deciding the next step.",
                },
                {
                    "headline": "The takeaway",
                    "narration": "Keep an execution trace and make failures visible so a human can review uncertain decisions.",
                },
                {
                    "headline": "Start small",
                    "narration": "Test your workflow against real edge cases before allowing it to trigger irreversible changes.",
                },
            ],
        }
    )


def test_niche_config_defaults_and_override(config: nv.NicheConfig, tmp_path: Path) -> None:
    path = tmp_path / "override.yml"
    path.write_text(
        "niche: Test\naudience: Devs\nbrand: LAB\ntopic: Default\nfacts:\n"
        "  - Explicitly approved test statement\n",
        encoding="utf-8",
    )
    changed = nv.NicheConfig.load(path, "Overridden topic")
    assert config.voice == "marin"
    assert changed.topic == "Overridden topic"
    assert changed.facts == ("Explicitly approved test statement",)


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "niche: x\naudience: y\nbrand: z\ntopic: t\nextra: value\n",
        "niche: x\naudience: y\nbrand: z\ntopic: t\nvoice: fake\n",
        "niche: x\naudience: y\nbrand: z\ntopic: t\nfacts: nope\n",
        "niche: x\naudience: y\nbrand: z\ntopic: t\nfacts: [5]\n",
        "niche: x\naudience: y\nbrand: z\n",
    ],
)
def test_invalid_config_fails_closed(contents: str, tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(nv.NicheVideoError):
        nv.NicheConfig.load(path)


def test_episode_contract(episode: nv.Episode) -> None:
    assert nv.Episode.from_json(episode.to_dict()) == episode
    assert sum(len(s.narration.split()) for s in episode.scenes) >= 45


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"title": "X", "scenes": []},
        {"title": "X", "scenes": [{"headline": "a", "narration": "b"}] * 3},
        {"title": "X", "scenes": [{"headline": "a"}] * 3},
        {"title": "X", "scenes": "wrong"},
    ],
)
def test_invalid_episode(raw: object) -> None:
    with pytest.raises(nv.NicheVideoError):
        nv.Episode.from_json(raw)


def test_script_request_is_structured_and_constrained(config: nv.NicheConfig) -> None:
    request = nv._script_request(config)
    assert request["model"] == nv.SCRIPT_MODEL
    schema: Any = request["response_format"]
    assert schema["json_schema"]["strict"] is True
    assert "approved_facts" in str(request)


def test_generate_episode_reuses_identical_script(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        return_value=json.dumps(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(episode.to_dict())}}
                ]
            }
        ).encode()
    )
    monkeypatch.setattr(nv, "_openai_post", post)
    first, fingerprint = nv.generate_episode(config, tmp_path)
    second, again = nv.generate_episode(config, tmp_path)
    assert first == second == episode
    assert fingerprint == again
    post.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": [{"finish_reason": "length", "message": {}}]},
        {"choices": [{"finish_reason": "stop", "message": {"refusal": "no"}}]},
    ],
)
def test_generation_rejects_bad_completions(
    config: nv.NicheConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    monkeypatch.setattr(nv, "_openai_post", lambda *_args, **_kw: json.dumps(payload).encode())
    with pytest.raises(nv.NicheVideoError):
        nv.generate_episode(config, tmp_path)


def test_tts_content_cache(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(return_value=b"x" * 2048)
    monkeypatch.setattr(nv, "_openai_post", post)
    a = nv.synthesize_scene(episode.scenes[0], config, tmp_path)
    b = nv.synthesize_scene(episode.scenes[0], config, tmp_path)
    assert a == b
    assert a.stat().st_size == 2048
    post.assert_called_once()


def test_short_tts_fails(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nv, "_openai_post", lambda *_args, **_kw: b"bad")
    with pytest.raises(nv.NicheVideoError):
        nv.synthesize_scene(episode.scenes[0], config, tmp_path)


def test_api_missing_key(config: nv.NicheConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(nv.NicheVideoError, match="OPENAI_API_KEY"):
        nv._openai_post("audio/speech", {"input": "hello"}, audio=True)


def test_api_error_redacts_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-that-must-not-appear")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("endpoint", 429, "limit", {}, None)

    monkeypatch.setattr(nv.urllib.request, "urlopen", fail)
    with pytest.raises(nv.NicheVideoError) as caught:
        nv._openai_post("audio/speech", {"input": "test"}, audio=True)
    assert "secret-that-must-not-appear" not in str(caught.value)


def test_atomic_write(tmp_path: Path) -> None:
    dest = tmp_path / "a" / "data.json"
    nv._write_atomic(dest, b"first")
    nv._write_atomic(dest, b"second")
    assert dest.read_bytes() == b"second"


def test_media_command_checks_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nv.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffmpeg"], 1, "", "failed"),
    )
    with pytest.raises(nv.NicheVideoError, match="failed"):
        nv._process(["ffmpeg", "-version"])


def test_duration_boundaries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "voice.mp3"
    monkeypatch.setattr(nv, "_probe", lambda _path: {"format": {"duration": "3.6"}})
    assert nv._duration(audio) == 3.6
    monkeypatch.setattr(nv, "_probe", lambda _path: {"format": {"duration": "nan"}})
    with pytest.raises(nv.NicheVideoError):
        nv._duration(audio)


def test_scene_render_uses_portrait_geometry_and_no_shell(
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_process(command: list[str], *, timeout: int = 240) -> str:
        commands.append(command)
        return ""

    monkeypatch.setattr(nv, "_duration", lambda _path: 3.6)
    monkeypatch.setattr(nv, "_process", fake_process)
    target = tmp_path / "scene-00.mp4"
    assert nv._render_scene(episode.scenes[0], tmp_path / "a.mp3", target, "BRAND") >= 3.8
    assert any("1080x1920" in part for part in commands[0])
    assert "shell" not in str(commands[0])
    assert (tmp_path / "scene-00.headline.txt").read_text() == "The problem"


def test_verify_media_and_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media: dict[str, Any] = {
        "format": {"duration": "15.0"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    monkeypatch.setattr(nv, "_probe", lambda _path: media)
    monkeypatch.setattr(nv, "_process", lambda _command, **_kw: "")
    assert nv._verify_video(tmp_path / "clip.mp4", 15)["status"] == "PASS"
    with pytest.raises(nv.NicheVideoError, match="duration"):
        nv._verify_video(tmp_path / "clip.mp4", 30)
    media["streams"][0]["width"] = 720
    with pytest.raises(nv.NicheVideoError, match="geometry"):
        nv._verify_video(tmp_path / "clip.mp4", 15)


def test_produce_is_draft_only_with_hash_evidence(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nv.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(nv, "generate_episode", lambda *_args: (episode, "a" * 64))
    monkeypatch.setattr(nv, "synthesize_scene", lambda *_args: tmp_path / "audio.mp3")

    def fake_render(_scene: nv.Scene, _audio: Path, target: Path, _brand: str) -> float:
        target.write_bytes(b"encoded")
        return 4.0

    def fake_concat(_clips: list[Path], target: Path) -> None:
        target.write_bytes(b"complete-video")

    monkeypatch.setattr(nv, "_render_scene", fake_render)
    monkeypatch.setattr(nv, "_concat", fake_concat)
    monkeypatch.setattr(nv, "_verify_video", lambda *_args: {"status": "PASS"})
    draft = nv.produce(config, tmp_path)
    manifest = json.loads(draft.with_name("manifest.json").read_text())
    assert manifest["status"] == "DRAFT_REQUIRES_HUMAN_REVIEW"
    assert manifest["published"] is False
    assert manifest["video_sha256"] == nv.hashlib.sha256(b"complete-video").hexdigest()
    assert manifest["disclosure"] == "Includes AI-generated narration"


def test_validate_only_requires_no_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    config.write_text("niche: AI\naudience: Devs\nbrand: LAB\ntopic: Example\n", encoding="utf-8")
    monkeypatch.setattr(nv, "_openai_post", Mock(side_effect=AssertionError("API called")))
    assert nv.main(["--config", str(config), "--validate-only"]) == 0
