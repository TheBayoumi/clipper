"""Book pipeline DAG records deterministic stages, cached resumes and closed gates."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import narration, selection
from clipper.books.contracts import BOOK_STAGE_CONTRACTS
from clipper.books.orchestrator import BookOrchestrator, BookRunError, NotPortedStage

ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict, dict]:
    return (
        json.loads((ROOT / "niches/book-editorial-catalog.json").read_text()),
        json.loads((ROOT / "niches/atomic-habits-approved-storyboard.json").read_text()),
    )


def test_plan_matches_legacy_output_and_reuses_exact_dag_stage(tmp_path: Path) -> None:
    catalog, board = _inputs()
    orchestrator = BookOrchestrator(tmp_path)
    first = orchestrator.plan(catalog, board, book_id="atomic-habits", lesson_id="two-minute-rule")
    second = orchestrator.plan(catalog, board, book_id="atomic-habits", lesson_id="two-minute-rule")
    assert first == second
    expected = selection.choose_lesson(
        catalog, book_id="atomic-habits", lesson_id="two-minute-rule"
    )
    assert (
        first.selection_json.encode()
        == (json.dumps(expected, indent=2, ensure_ascii=False) + "\n").encode()
    )
    assert first.selected.legacy_fingerprint == expected["fingerprint"]
    assert json.loads(first.outline_json)["selection_fingerprint"] == expected["fingerprint"]
    assert json.loads(first.niche_config_json)["selection_fingerprint"] == expected["fingerprint"]
    records = list((tmp_path / "book-dag/books-selection").glob("*/stage.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["attempt_count"] == 1


def _recorded_wavs(root: Path, count: int = 4) -> Path:
    directory = root / "recorded-wavs"
    directory.mkdir()
    for index in range(count):
        with wave.open(str(directory / f"scene-{index:02d}.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            output.writeframes(bytes((index, 0)) * 24000)
    return directory


def test_recorded_episode_resumes_narration_then_fails_closed_at_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, board = _inputs()
    runner = BookOrchestrator(tmp_path)
    recordings = _recorded_wavs(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a producer must not be invoked from recorded WAVs")

    monkeypatch.setattr(narration, "synthesize_scene", forbidden)
    args = {
        "source": ROOT / "niches/atomic-habits-episode-02.json",
        "narration_dir": recordings,
        "book_id": "atomic-habits",
        "lesson_id": "two-minute-rule",
    }
    with pytest.raises(NotPortedStage, match=r"NOT_PORTED: books.direction"):
        runner.run(catalog, board, **args)
    monkeypatch.setattr(narration, "_wav_metadata", forbidden)
    with pytest.raises(NotPortedStage, match=r"NOT_PORTED: books.direction"):
        runner.run(catalog, board, **args)

    for name in ("selection", "scripting", "narration", "direction"):
        paths = list((tmp_path / f"book-dag/books-{name}").glob("*/stage.json"))
        assert len(paths) == 1
        record = json.loads(paths[0].read_text())
        assert record["status"] == ("FAILED" if name == "direction" else "PASS")
        assert record["attempt_count"] == (2 if name == "direction" else 1)
        if name == "direction":
            assert record["error_type"] == "NotPortedStage"
        if name == "narration":
            assert record["identity"]["model_revision"] == "v1.0"
            assert record["identity"]["decoding_parameters"]["sample_rate"] == 24000
            assert record["identity"]["decoding_parameters"]["voice"] == "am_adam"


def test_script_stage_uses_existing_completion_cache_without_loading_weights(
    tmp_path: Path,
) -> None:
    catalog, board = _inputs()
    runner = BookOrchestrator(tmp_path)
    plan = runner.plan(catalog, board, book_id="atomic-habits", lesson_id="two-minute-rule")
    completed = json.loads((ROOT / "niches/atomic-habits-episode-02.json").read_text())["script"]
    fake = Mock(return_value=json.dumps(completed))
    first = runner.script(plan, completion=fake)
    second = runner.script(plan, completion=fake)
    assert first == second
    fake.assert_called_once()
    assert first.selection_fingerprint == plan.selected.legacy_fingerprint
    assert first.script_cache_key
    assert len(first.scenes) == 4


@pytest.mark.parametrize("stage", ["direction", "visuals", "assembly", "review"])
def test_every_unported_stage_records_an_explicit_failure(tmp_path: Path, stage: str) -> None:
    assert stage in BOOK_STAGE_CONTRACTS
    runner = BookOrchestrator(tmp_path)
    with pytest.raises(NotPortedStage, match=f"NOT_PORTED: books.{stage}"):
        runner.require_unported(stage, source_hash="a" * 64, dependency_hashes=("b" * 64,))
    records = list((tmp_path / f"book-dag/books-{stage}").glob("*/stage.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["status"] == "FAILED"


def test_invalid_stage_and_mismatched_recorded_episode_never_start_model(
    tmp_path: Path,
) -> None:
    catalog, board = _inputs()
    runner = BookOrchestrator(tmp_path)
    with pytest.raises(BookRunError, match="unknown or implemented"):
        runner.require_unported("selection", source_hash="a", dependency_hashes=())
    plan = runner.plan(catalog, board, book_id="deep-work")
    with pytest.raises(BookRunError, match="does not match selection"):
        runner.script(plan, source=ROOT / "niches/atomic-habits-episode-02.json")


def test_narration_uses_legacy_audio_cache_and_dag_resume_with_fake_producer(
    tmp_path: Path,
) -> None:
    catalog, board = _inputs()
    runner = BookOrchestrator(tmp_path)
    plan = runner.plan(catalog, board, book_id="atomic-habits", lesson_id="two-minute-rule")
    script = runner.script(plan, source=ROOT / "niches/atomic-habits-episode-02.json")
    payload = tmp_path / "recorded-wavs"
    payload.mkdir()
    with wave.open(str(payload / "single.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(bytes((1, 0)) * 24000)
    wav = (payload / "single.wav").read_bytes()
    producer = Mock(return_value=wav)
    tracks = runner.narrate(plan, script, producer=producer)
    again = runner.narrate(plan, script, producer=producer)
    assert tracks == again
    assert len(tracks) == len(script.scenes)
    assert all(item.sample_rate == 24000 and item.voice_id == "am_adam" for item in tracks)
    assert producer.call_count == len(script.scenes)
    saved = list((tmp_path / "book-script-cache/tts").glob("*.wav"))
    assert len(saved) == len(script.scenes)
    assert all(path.read_bytes() == wav for path in saved)


def test_recorded_narration_rejects_bad_geometry_with_dag_failure_evidence(
    tmp_path: Path,
) -> None:
    catalog, board = _inputs()
    runner = BookOrchestrator(tmp_path)
    plan = runner.plan(catalog, board, book_id="atomic-habits", lesson_id="two-minute-rule")
    script = runner.script(plan, source=ROOT / "niches/atomic-habits-episode-02.json")
    recordings = _recorded_wavs(tmp_path, count=len(script.scenes))
    (recordings / "scene-00.wav").write_bytes(b"not-a-wave")
    with pytest.raises(Exception, match="invalid WAV audio"):
        runner.narrate(plan, script, recorded_dir=recordings)
    records = list((tmp_path / "book-dag/books-narration").glob("*/stage.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["status"] == "FAILED"
    assert not list((tmp_path / "book-script-cache/tts").glob("*.wav"))
