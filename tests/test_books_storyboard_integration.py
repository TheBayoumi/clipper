"""Book-specific storyboard CLI parity and DagStore template-provenance regression."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from clipper import cli
from clipper.books import selection
from clipper.books.orchestrator import BookOrchestrator

NICHES = Path(__file__).resolve().parents[1] / "niches"
CATALOG = NICHES / "book-editorial-catalog.json"


def _catalog() -> dict:
    return json.loads(CATALOG.read_text())


@pytest.mark.parametrize("book_id", ["atomic-habits", "deep-work", "psychology-of-money"])
def test_every_catalog_book_plans_and_records_template_identity(
    book_id: str, tmp_path: Path
) -> None:
    catalog = _catalog()
    runner = BookOrchestrator(tmp_path)
    plan = runner.plan(catalog, catalog_dir=NICHES, book_id=book_id)
    picked = selection.choose_lesson(catalog, book_id=book_id)
    expected_file = NICHES / f"{book_id}-approved-storyboard.json"
    assert plan.selection_json == json.dumps(picked, indent=2, ensure_ascii=False) + "\n"
    assert plan.selected.legacy_fingerprint == picked["fingerprint"]
    assert (
        plan.storyboard_provenance.template_sha256
        == hashlib.sha256(expected_file.read_bytes()).hexdigest()
    )
    assert plan.storyboard_provenance.template_path == str(expected_file)
    record_paths = list((tmp_path / "book-dag/books-selection").glob("*/stage.json"))
    assert len(record_paths) == 1
    record = json.loads(record_paths[0].read_text())
    output = json.loads(record_paths[0].with_name("output.json").read_text())
    assert record["status"] == "PASS"
    assert output["storyboard"] == asdict(plan.storyboard_provenance)


def test_template_bytes_rekey_selection_stage_without_changing_legacy_selection(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    board_file = tmp_path / "atomic-habits-approved-storyboard.json"
    original_bytes = (NICHES / board_file.name).read_bytes()
    board_file.write_bytes(original_bytes)
    runner = BookOrchestrator(tmp_path / "results")
    first = runner.plan(catalog, catalog_dir=tmp_path, book_id="atomic-habits")
    changed = json.loads(original_bytes)
    changed["production_storyboard"]["editorial_sequence"][0]["objective"] += (
        " Open with an original action."
    )
    board_file.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    second = runner.plan(catalog, catalog_dir=tmp_path, book_id="atomic-habits")
    assert first.selection_json == second.selection_json
    assert first.niche_config_json == second.niche_config_json
    assert first.outline_json != second.outline_json
    assert first.storyboard_provenance.template_sha256 != (
        second.storyboard_provenance.template_sha256
    )
    saved = list((tmp_path / "results/book-dag/books-selection").glob("*/stage.json"))
    assert len(saved) == 2
    assert len({json.loads(path.read_text())["identity"]["cache_key"] for path in saved}) == 2


def test_deep_work_cli_default_matches_legacy_selection_output_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = tmp_path / "legacy"
    actual = tmp_path / "books"
    common = [
        "--catalog",
        str(CATALOG),
        "--book-id",
        "deep-work",
        "--lesson-id",
        "focus-without-distraction",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "book_episode_selection.py",
            *common,
            "--output",
            str(expected / "selection.json"),
            "--niche-config-output",
            str(expected / "config.json"),
            "--storyboard-template",
            str(NICHES / "deep-work-approved-storyboard.json"),
            "--storyboard-output",
            str(expected / "outline.json"),
        ],
    )
    selection.main()
    legacy_stdout = capsys.readouterr().out
    result = cli.main(
        [
            "books",
            "plan",
            *common,
            "--artifact-root",
            str(tmp_path / "dag"),
            "--output",
            str(actual / "selection.json"),
            "--niche-config-output",
            str(actual / "config.json"),
            "--storyboard-output",
            str(actual / "outline.json"),
        ]
    )
    assert result == 0
    assert capsys.readouterr().out == legacy_stdout
    for filename in ("selection.json", "config.json", "outline.json"):
        assert (actual / filename).read_bytes() == (expected / filename).read_bytes()


def test_atomic_default_matches_explicit_legacy_template_override(
    tmp_path: Path,
) -> None:
    results: list[tuple[bytes, bytes, bytes]] = []
    for index, extra in enumerate(
        [[], ["--storyboard-template", str(NICHES / "atomic-habits-approved-storyboard.json")]]
    ):
        root = tmp_path / str(index)
        argv = [
            "books",
            "plan",
            "--catalog",
            str(CATALOG),
            "--book-id",
            "atomic-habits",
            "--lesson-id",
            "two-minute-rule",
            "--artifact-root",
            str(root / "dag"),
            "--output",
            str(root / "selection.json"),
            "--niche-config-output",
            str(root / "config.json"),
            "--storyboard-output",
            str(root / "outline.json"),
            *extra,
        ]
        assert cli.main(argv) == 0
        results.append(
            (
                (root / "selection.json").read_bytes(),
                (root / "config.json").read_bytes(),
                (root / "outline.json").read_bytes(),
            )
        )
    assert results[0] == results[1]


def test_psychology_of_money_plans_without_atomic_fallback(tmp_path: Path) -> None:
    target = tmp_path / "selection.json"
    assert (
        cli.main(
            [
                "books",
                "plan",
                "--catalog",
                str(CATALOG),
                "--book-id",
                "psychology-of-money",
                "--output",
                str(target),
                "--artifact-root",
                str(tmp_path / "dag"),
            ]
        )
        == 0
    )
    selected = json.loads(target.read_text())
    assert selected["book"]["title"] == "The Psychology of Money"
    assert selected["book_id"] == "psychology-of-money"


def test_cross_book_and_unapproved_overrides_fail_before_selection_pass(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = tmp_path / "failed.json"
    base = [
        "books",
        "plan",
        "--catalog",
        str(CATALOG),
        "--book-id",
        "deep-work",
        "--output",
        str(first),
        "--artifact-root",
        str(tmp_path / "dag"),
    ]
    assert (
        cli.main(
            [*base, "--storyboard-template", str(NICHES / "atomic-habits-approved-storyboard.json")]
        )
        == 1
    )
    assert "book mismatch" in caplog.text
    corrupted = copy.deepcopy(
        json.loads((NICHES / "deep-work-approved-storyboard.json").read_text())
    )
    corrupted["production_storyboard"]["publish_eligible"] = True
    invalid = tmp_path / "unapproved.json"
    invalid.write_text(json.dumps(corrupted), encoding="utf-8")
    assert cli.main([*base, "--storyboard-template", str(invalid)]) == 1
    assert "unapproved" in caplog.text
    assert not first.exists()
    assert not (tmp_path / "dag/book-dag/books-selection").exists()
