"""Golden compatibility checks for the source-backed book-selection package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper.books import selection

ROOT = Path(__file__).resolve().parents[1]


def test_selection_and_storyboard_retain_exact_serialized_contracts() -> None:
    catalog = json.loads((ROOT / "niches/book-editorial-catalog.json").read_text())
    board = json.loads((ROOT / "niches/atomic-habits-approved-storyboard.json").read_text())
    picked = selection.choose_lesson(catalog, book_id="atomic-habits", lesson_id="two-minute-rule")
    assert picked["episode_id"] == "atomic-habits--two-minute-rule--primary"
    assert picked["book"]["author"] == "James Clear"
    assert picked["fingerprint"] == selection._digest(
        {key: value for key, value in picked.items() if key != "fingerprint"}
    )
    outline = selection.to_storyboard_outline(picked, board)
    assert outline["fingerprint"] == selection._digest(
        {key: value for key, value in outline.items() if key != "fingerprint"}
    )
    config = selection.to_niche_config(picked)
    assert config["selection_fingerprint"] == picked["fingerprint"]
    assert json.loads(json.dumps(config, ensure_ascii=False)) == config


def test_selection_cli_writes_the_existing_json_encodings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "book_episode_selection.py",
            "--catalog",
            str(ROOT / "niches/book-editorial-catalog.json"),
            "--book-id",
            "atomic-habits",
            "--lesson-id",
            "two-minute-rule",
            "--output",
            str(tmp_path / "selection.json"),
            "--niche-config-output",
            str(tmp_path / "niche.json"),
            "--storyboard-template",
            str(ROOT / "niches/atomic-habits-approved-storyboard.json"),
            "--storyboard-output",
            str(tmp_path / "outline.json"),
        ],
    )
    selection.main()
    data = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert (tmp_path / "selection.json").read_bytes() == (
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert json.loads((tmp_path / "niche.json").read_text()) == selection.to_niche_config(data)
    assert json.loads((tmp_path / "outline.json").read_text()) == selection.to_storyboard_outline(
        data, json.loads((ROOT / "niches/atomic-habits-approved-storyboard.json").read_text())
    )
    report = json.loads(capsys.readouterr().out)
    assert report["fingerprint"] == data["fingerprint"]


def test_selection_rejects_empty_catalog_and_invalid_storyboard() -> None:
    with pytest.raises(selection.BookSelectionError, match="empty book library"):
        selection.choose_lesson({"schema": "clipper-book-editorial-catalog-v1", "books": []})
    with pytest.raises(selection.BookSelectionError, match="unsupported"):
        selection.to_storyboard_outline(
            {"fingerprint": "a", "episode_id": "b"}, {"production_storyboard": {}}
        )



@pytest.mark.parametrize(
    "defect,message",
    [
        ("schema", "unknown editorial"),
        ("book_type", "invalid book entry"),
        ("book_title", "unattributed book"),
        ("book_url", "book source URL"),
        ("book_lessons", "no eligible lesson"),
        ("lesson_id", "invalid lesson ID"),
        ("lesson_url", "claim source must be HTTPS"),
        ("angles_empty", "nonempty list"),
        ("angles_shape", "invalid editorial angle"),
    ],
)
def test_catalog_invalid_contract_variants_fail_closed(defect: str, message: str) -> None:
    catalog = json.loads((ROOT / "niches/book-editorial-catalog.json").read_text())
    book = catalog["books"][0]
    lesson = book["lessons"][0]
    if defect == "schema":
        catalog["schema"] = "unrecognized"
    elif defect == "book_type":
        catalog["books"][0] = None
    elif defect == "book_title":
        book["title"] = " "
    elif defect == "book_url":
        book["source_url"] = "http://example.net/book"
    elif defect == "book_lessons":
        book["lessons"] = []
    elif defect == "lesson_id":
        lesson["id"] = "Bad Lesson"
    elif defect == "lesson_url":
        lesson["claim_source_url"] = "http://example.net/claim"
    elif defect == "angles_empty":
        lesson["angles"] = []
    else:
        lesson["angles"] = [{"unknown": "shape"}]
    with pytest.raises(selection.BookSelectionError, match=message):
        selection.choose_lesson(catalog)


def test_selection_history_and_storyboard_boundaries() -> None:
    catalog = json.loads((ROOT / "niches/book-editorial-catalog.json").read_text())
    with pytest.raises(selection.BookSelectionError, match="history requires"):
        selection.choose_lesson(catalog, [{"book_id": "atomic-habits"}])
    with pytest.raises(selection.BookSelectionError, match="no verified book"):
        selection.choose_lesson(catalog, pillar="unknown")
    picked = selection.choose_lesson(catalog)
    with pytest.raises(selection.BookSelectionError, match="unqualified"):
        selection.to_niche_config({"schema": "wrong"})
    template = {
        "production_storyboard": {
            "schema": "clipper-production-storyboard-v2",
            "editorial_sequence": [],
        }
    }
    with pytest.raises(selection.BookSelectionError, match="missing narrative roles"):
        selection.to_storyboard_outline(picked, template)
