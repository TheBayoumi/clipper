"""Book pipeline DAG records deterministic stages, cached resumes and closed gates."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import selection
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
    first = orchestrator.plan(
        catalog, board, book_id="atomic-habits", lesson_id="two-minute-rule"
    )
    second = orchestrator.plan(
        catalog, board, book_id="atomic-habits", lesson_id="two-minute-rule"
    )
    assert first == second
    expected = selection.choose_lesson(
        catalog, book_id="atomic-habits", lesson_id="two-minute-rule"
    )
    assert first.selection_json.encode() == (
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    assert first.selected.legacy_fingerprint == expected["fingerprint"]
    assert json.loads(first.outline_json)["selection_fingerprint"] == expected["fingerprint"]
    assert json.loads(first.niche_config_json)["selection_fingerprint"] == expected["fingerprint"]
    records = list((tmp_path / "book-dag/books-selection").glob("*/stage.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["attempt_count"] == 1


def test_recorded_episode_runs_selection_and_script_then_fails_at_narration(
    tmp_path: Path,
) -> None:
    catalog, board = _inputs()
    runner = BookOrchestrator(tmp_path)
    with pytest.raises(NotPortedStage, match=r"NOT_PORTED: books.narration"):
        runner.run(
            catalog,
            board,
            source=ROOT / "niches/atomic-habits-episode-02.json",
            book_id="atomic-habits",
            lesson_id="two-minute-rule",
        )
    for name in ("selection", "scripting", "narration"):
        paths = list((tmp_path / f"book-dag/books-{name}").glob("*/stage.json"))
        assert len(paths) == 1
        record = json.loads(paths[0].read_text())
        assert record["status"] == ("FAILED" if name == "narration" else "PASS")
        if name == "narration":
            assert record["error_type"] == "NotPortedStage"


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


@pytest.mark.parametrize("stage", ["narration", "direction", "visuals", "assembly", "review"])
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
