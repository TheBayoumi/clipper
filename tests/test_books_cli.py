"""Book CLI parity, validation and fail-closed DAG execution; no model calls."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from clipper import cli
from clipper.books import scripting, selection

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "niches/book-editorial-catalog.json"
BOARD = ROOT / "niches/atomic-habits-approved-storyboard.json"
BRIEF = ROOT / "niches/atomic-habits-episode-02.json"
CONFIG = ROOT / "niches/book-lessons.yaml"


def test_books_plan_matches_legacy_atomic_selection_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    legacy = tmp_path / "legacy"
    target = tmp_path / "books"
    common = [
        "--catalog",
        str(CATALOG),
        "--book-id",
        "atomic-habits",
        "--lesson-id",
        "two-minute-rule",
    ]
    monkeypatch.setattr(
        "sys.argv",
        [
            "book_episode_selection.py",
            *common,
            "--output",
            str(legacy / "selection.json"),
            "--niche-config-output",
            str(legacy / "niche.json"),
            "--storyboard-template",
            str(BOARD),
            "--storyboard-output",
            str(legacy / "outline.json"),
        ],
    )
    selection.main()
    original_stdout = capsys.readouterr().out
    result = cli.main(
        [
            "books",
            "plan",
            *common,
            "--storyboard-template",
            str(BOARD),
            "--artifact-root",
            str(tmp_path / "dag"),
            "--output",
            str(target / "selection.json"),
            "--niche-config-output",
            str(target / "niche.json"),
            "--storyboard-output",
            str(target / "outline.json"),
        ]
    )
    assert result == 0
    assert capsys.readouterr().out == original_stdout
    for filename in ("selection.json", "niche.json", "outline.json"):
        assert (target / filename).read_bytes() == (legacy / filename).read_bytes()
    selected = json.loads((target / "selection.json").read_text())
    assert selected["episode_id"] == json.loads(BRIEF.read_text())["episode_id"]


def test_books_validate_catalog_and_book_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["books", "validate", "--catalog", str(CATALOG), "--config", str(CONFIG)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema":"unknown", "books":[]}', encoding="utf-8")
    assert cli.main(["books", "validate", "--catalog", str(invalid), "--config", str(CONFIG)]) == 1


def test_books_run_uses_recorded_script_and_fails_at_not_ported_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("live model inference forbidden in test")

    monkeypatch.setattr(scripting, "generate_episode", forbidden)
    recordings = tmp_path / "recorded"
    recordings.mkdir()
    for index in range(4):
        with wave.open(str(recordings / f"scene-{index:02d}.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            output.writeframes(bytes((index, 0)) * 24000)
    code = cli.main(
        [
            "books",
            "run",
            "--catalog",
            str(CATALOG),
            "--storyboard-template",
            str(BOARD),
            "--script-brief",
            str(BRIEF),
            "--narration-dir",
            str(recordings),
            "--book-id",
            "atomic-habits",
            "--lesson-id",
            "two-minute-rule",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "NOT_PORTED: books.direction" in caplog.text
    passed = list((tmp_path / "book-dag/books-narration").glob("*/stage.json"))
    records = list((tmp_path / "book-dag/books-direction").glob("*/stage.json"))
    assert len(passed) == len(records) == 1
    assert json.loads(passed[0].read_text())["status"] == "PASS"
    assert json.loads(records[0].read_text())["status"] == "FAILED"


def test_books_plan_rejects_bad_history_without_model_inference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    history = tmp_path / "history.json"
    history.write_text('{"invalid":"dict"}', encoding="utf-8")
    assert (
        cli.main(
            [
                "books",
                "plan",
                "--catalog",
                str(CATALOG),
                "--history",
                str(history),
                "--output",
                str(tmp_path / "selection.json"),
            ]
        )
        == 1
    )
    assert "history requires a list" in caplog.text


def test_books_validate_requires_actual_book_lesson_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = tmp_path / "general.yaml"
    config.write_text(
        "niche: AI\naudience: Builders\nbrand: LAB\ntopic: Example\n", encoding="utf-8"
    )
    assert cli.main(["books", "validate", "--catalog", str(CATALOG), "--config", str(config)]) == 1
    assert "format: book_lesson" in caplog.text



def test_books_run_requires_recorded_audio_before_any_inference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert (
        cli.main(
            [
                "books",
                "run",
                "--catalog",
                str(CATALOG),
                "--script-brief",
                str(BRIEF),
                "--artifact-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "requires --narration-dir" in caplog.text
    assert not (tmp_path / "book-dag").exists()
