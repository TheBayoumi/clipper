from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import wave
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
        "voice: af_heart\n"
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
    assert config.voice == "af_heart"
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


def _valid_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x01\x00" * 24000)
    return buffer.getvalue()


def test_script_request_is_local_and_constrained(config: nv.NicheConfig) -> None:
    request = nv._script_request(config)
    assert request["model"] == nv.SCRIPT_MODEL
    assert request["revision"] == nv.SCRIPT_REVISION
    assert request["seed"] == 42
    assert "approved_facts" in str(request)


def test_generate_episode_reuses_identical_script(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock(return_value=json.dumps(episode.to_dict()))
    monkeypatch.setattr(nv, "_local_script_completion", model)
    first, fingerprint = nv.generate_episode(config, tmp_path)
    second, again = nv.generate_episode(config, tmp_path)
    assert first == second == episode
    assert fingerprint == again
    model.assert_called_once()


@pytest.mark.parametrize("completion", ["", "not json", "[]", "{}"])
def test_generation_rejects_bad_completions(
    config: nv.NicheConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completion: str,
) -> None:
    monkeypatch.setattr(nv, "_local_script_completion", lambda _request: completion)
    with pytest.raises((nv.NicheVideoError, json.JSONDecodeError)):
        nv.generate_episode(config, tmp_path)


def test_failed_word_count_receives_targeted_repair(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short = episode.to_dict()
    for scene in short["scenes"]:
        scene["narration"] = "One two three four five six seven"
    model = Mock(side_effect=[json.dumps(short), json.dumps(episode.to_dict())])
    monkeypatch.setattr(nv, "_local_script_completion", model)

    result, fingerprint = nv.generate_episode(config, tmp_path)
    assert result == episode
    assert model.call_count == 2
    revision = model.call_args_list[1].args[0]
    assert "total_narration_words=28" in str(revision)
    assert config.topic in str(revision)
    assert (tmp_path / "scripts" / f"{fingerprint}.json").is_file()
    audit = json.loads((tmp_path / "script_diagnostics" / f"{fingerprint}.json").read_text())
    assert [item["accepted"] for item in audit] == [False, True]
    nv.generate_episode(config, tmp_path)
    assert model.call_count == 2


def test_repair_exhaustion_preserves_original_context(
    config: nv.NicheConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_fact = nv.NicheConfig(
        niche=config.niche,
        audience=config.audience,
        brand=config.brand,
        voice=config.voice,
        facts=("The only approved statement",),
        topic="Responsible AI agents",
    )
    model = Mock(return_value="not json")
    monkeypatch.setattr(nv, "_local_script_completion", model)
    with pytest.raises(nv.NicheVideoError, match="after 3 attempts"):
        nv.generate_episode(with_fact, tmp_path)
    assert model.call_count == nv.MAX_SCRIPT_ATTEMPTS
    last = model.call_args_list[-1].args[0]
    assert "The only approved statement" in str(last)
    assert "Responsible AI agents" in str(last)
    assert "Revise your previous JSON" in str(last)
    assert not list((tmp_path / "scripts").glob("*.json"))
    reports = list((tmp_path / "script_diagnostics").glob("*.json"))
    assert len(reports) == 1
    assert len(json.loads(reports[0].read_text())) == 3


def test_complete_sentence_book_edit_accepts_oversized_script_without_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = nv.NicheConfig.load(
        Path(__file__).resolve().parents[1] / "niches" / "book-lessons.yaml"
    )
    narration = (
        "Start with one small action that feels easy to repeat. "
        "Set up the space so the next step is right in front of you. "
        "A tiny example can make the lesson feel practical and memorable. "
        "You can try the idea once today without making big promises."
    )
    original = {
        "title": "Action over intention",
        "scenes": [{"headline": f"STEP {index}", "narration": narration} for index in range(1, 5)],
    }
    assert sum(len(s["narration"].split()) for s in original["scenes"]) > 125
    model = Mock(return_value=json.dumps(original))
    monkeypatch.setattr(nv, "_local_script_completion", model)
    episode, digest = nv.generate_episode(config, tmp_path)
    assert model.call_count == 1
    assert len(episode.scenes) == 4
    assert 45 <= sum(len(scene.narration.split()) for scene in episode.scenes) <= 125
    assert episode.scenes[0].narration.startswith("Start with one small action")
    assert episode.scenes[-1].narration.endswith("without making big promises.")
    for scene in episode.scenes:
        assert 6 <= len(scene.narration.split()) <= 45
        assert all(sentence in narration for sentence in scene.narration.split(". ") if sentence)
    audit = json.loads((tmp_path / "script_diagnostics" / f"{digest}.json").read_text())
    assert audit[0]["accepted"] is True
    assert audit[0]["complete_sentence_edit"] is True
    assert (tmp_path / "scripts" / f"{digest}.json").is_file()


def test_book_edit_does_not_chop_unbreakable_long_sentence() -> None:
    very_long = " ".join(["unbroken"] * 48) + "."
    raw = {
        "title": "Too long",
        "scenes": [{"headline": f"SCENE {i}", "narration": very_long} for i in range(3)],
    }
    with pytest.raises(nv.NicheVideoError, match="whole-sentence boundaries"):
        nv._compact_book_sentences(raw)


def test_recovered_full_sentence_script_preserves_payoff_and_no_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).resolve().parents[1] / "niches" / "book-lessons.yaml"
    config = nv.NicheConfig.load(source)
    candidate = {
        "title": "A habit that sticks",
        "scenes": [
            {
                "headline": "The start",
                "narration": (
                    "A habit is not about how hard it is to do, but how easy it is to start. "
                    "The key is to begin with a small, manageable action that feels rewarding. "
                    "For example, instead of trying to read a whole book in one sitting, "
                    "start with just one page a day. "
                    "This approach makes the task feel achievable and gives you a sense of accomplishment."
                ),
            },
            {
                "headline": "The cue",
                "narration": (
                    "A cue is a trigger that signals your brain to take action. "
                    "You can create a cue by placing your phone on a visible desk, "
                    "or by setting a specific time each day to do something. "
                    "The more obvious your cue is, the more likely you are to respond automatically. "
                    "For instance, if you want to exercise, set your phone to ring at "
                    "7 AM, and you will be motivated to wake up and move."
                ),
            },
            {
                "headline": "The reward",
                "narration": (
                    "Every habit should feel good to do. "
                    "The more satisfying the action is, the more likely you are to repeat it. "
                    "For example, instead of trying to eat healthy every day, start with "
                    "a small snack that you enjoy. "
                    "Over time, this becomes a habit that you look forward to, "
                    "and it becomes easier to maintain."
                ),
            },
            {
                "headline": "The finale",
                "narration": (
                    "Habits are not about perfection, they are about consistency. "
                    "The more you practice, the more your brain will start to automate the behavior. "
                    "If you want to make a habit easier to maintain, try to build it gradually. "
                    "For example, if you want to write every day, start with just 10 minutes "
                    "a day and increase the time as you get comfortable with the routine."
                ),
            },
        ],
    }
    raw_words = sum(len(s["narration"].split()) for s in candidate["scenes"])
    assert raw_words > 125
    audit = tmp_path / "rejected.json"
    audit.write_text(
        json.dumps([{"attempt": 1, "accepted": False, "completion": json.dumps(candidate)}]),
        encoding="utf-8",
    )
    model = Mock(side_effect=AssertionError("model must not rerun on recovery"))
    monkeypatch.setattr(nv, "_local_script_completion", model)
    digest = nv._recover_book_script(config, tmp_path / "_cache", audit, "40045892788")
    episode, loaded_digest = nv.generate_episode(config, tmp_path / "_cache")
    assert digest == loaded_digest
    assert model.call_count == 0
    assert episode.scenes[-1].narration.endswith("comfortable with the routine.")
    count = sum(len(scene.narration.split()) for scene in episode.scenes)
    assert 45 <= count <= 125
    recovery = json.loads(
        (tmp_path / "_cache" / "script_recoveries" / f"{digest}.json").read_text()
    )
    assert recovery["source_run_id"] == "40045892788"
    assert recovery["edited"] is True


def test_recovery_rejects_unrecorded_script(tmp_path: Path) -> None:
    config = nv.NicheConfig.load(
        Path(__file__).resolve().parents[1] / "niches" / "book-lessons.yaml"
    )
    audit = tmp_path / "bad.json"
    audit.write_text('[{"accepted": true, "completion": "{}"}]', encoding="utf-8")
    with pytest.raises(nv.NicheVideoError, match="original rejected"):
        nv._recover_book_script(config, tmp_path / "_cache", audit, "42")


def test_source_attributed_book_lesson_config(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "niches" / "book-lessons.yaml"
    book = nv.NicheConfig.load(source)
    assert book.format == "book_lesson"
    assert book.pillar == "self_improvement"
    assert book.book_title == "Atomic Habits"
    assert book.book_author == "James Clear"
    assert book.source_url.startswith("https://jamesclear.com/")
    assert book.facts
    request = nv._script_request(book)
    assert "Atomic Habits" in str(request)
    assert "James Clear" in str(request)
    assert "approved_facts" in str(request)


@pytest.mark.parametrize(
    "source_url", ["http://example.com/fact", "https://user:pass@example.com/x"]
)
def test_book_lesson_rejects_untrusted_source(tmp_path: Path, source_url: str) -> None:
    path = tmp_path / "book.yaml"
    path.write_text(
        "format: book_lesson\nniche: Books\naudience: Readers\nbrand: PAGE TO LIFE\n"
        "topic: Get started\npillar: psychology\nbook_title: Example Book\n"
        "book_author: Example Author\nfacts:\n  - Reviewed fact.\n"
        f"source_url: {source_url}\n",
        encoding="utf-8",
    )
    with pytest.raises(nv.NicheVideoError, match="HTTPS"):
        nv.NicheConfig.load(path)


def test_book_lesson_refuses_unsourced_claims(tmp_path: Path) -> None:
    path = tmp_path / "book.yaml"
    path.write_text(
        "format: book_lesson\nniche: Books\naudience: Readers\nbrand: PAGE TO LIFE\n"
        "topic: Get started\npillar: psychology\nbook_title: Example Book\n"
        "book_author: Example Author\nsource_url: https://example.com/author\n"
        "facts: []\n",
        encoding="utf-8",
    )
    with pytest.raises(nv.NicheVideoError, match="approved, sourced facts"):
        nv.NicheConfig.load(path)


def test_cartoon_original_artwork_and_animated_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    monkeypatch.syspath_prepend(str(_SCRIPT.parent))
    from book_cartoon import CANVAS, HOST_SIZE, create_assets

    bg, speaking, idle = create_assets(
        tmp_path, "PAGE TO LIFE", "Atomic Habits", "James Clear", "START WITH AN OBVIOUS CUE", 0, 4
    )
    assert Image.open(bg).size == CANVAS == (1080, 1920)
    assert Image.open(speaking).size == HOST_SIZE
    assert Image.open(speaking).mode == "RGBA"
    assert speaking.read_bytes() != idle.read_bytes()

    config = nv.NicheConfig.load(
        Path(__file__).resolve().parents[1] / "niches" / "book-lessons.yaml"
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(nv, "_duration", lambda _audio: 2.6)
    monkeypatch.setattr(nv, "_process", lambda command, **_kwargs: commands.append(command) or "")
    scene = nv.Scene("START WITH AN OBVIOUS CUE", "An original and actionable lesson.")
    seconds = nv._render_scene(
        scene, tmp_path / "narration.wav", tmp_path / "scene.mp4", config.brand, config, 0, 4
    )
    assert seconds >= 2.8
    assert "-filter_complex" in commands[0]
    assert "overlay=" in str(commands[0])
    assert "-loop" in commands[0]


def test_candidate_backlog_never_auto_eligible() -> None:
    import yaml

    source = Path(__file__).resolve().parents[1] / "niches" / "book-lesson-backlog.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert data["production_eligible"] is False
    assert len(data["episodes"]) == 20
    assert {e["pillar"] for e in data["episodes"]} == {"self_improvement", "psychology", "business"}
    assert all(e["source_status"] == "research_required" for e in data["episodes"])


def test_tts_content_cache(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice = Mock(return_value=_valid_wav())
    monkeypatch.setattr(nv, "_kokoro_wav", voice)
    a = nv.synthesize_scene(episode.scenes[0], config, tmp_path)
    b = nv.synthesize_scene(episode.scenes[0], config, tmp_path)
    assert a == b
    assert a.suffix == ".wav"
    assert nv._wav_metadata(a.read_bytes()) == (1, 24000, 24000)
    voice.assert_called_once()


def test_empty_tts_fails(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nv, "_kokoro_wav", lambda *_args: b"invalid")
    with pytest.raises(nv.NicheVideoError):
        nv.synthesize_scene(episode.scenes[0], config, tmp_path)


def test_no_openai_key_required(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(nv, "_local_script_completion", lambda _req: json.dumps(episode.to_dict()))
    monkeypatch.setattr(nv, "_kokoro_wav", lambda *_args: _valid_wav())
    generated, _ = nv.generate_episode(config, tmp_path)
    assert generated == episode
    assert nv.synthesize_scene(episode.scenes[0], config, tmp_path).is_file()


def test_cached_audio_corruption_fails(
    config: nv.NicheConfig,
    episode: nv.Episode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nv, "_kokoro_wav", lambda *_args: _valid_wav())
    location = nv.synthesize_scene(episode.scenes[0], config, tmp_path)
    location.write_bytes(b"not a wave")
    with pytest.raises(nv.NicheVideoError):
        nv.synthesize_scene(episode.scenes[0], config, tmp_path)


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

    def fake_render(
        _scene: nv.Scene, _audio: Path, target: Path, _brand: str, *_args: object
    ) -> float:
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
    monkeypatch.setattr(
        nv, "_local_script_completion", Mock(side_effect=AssertionError("model called"))
    )
    assert nv.main(["--config", str(config), "--validate-only"]) == 0
