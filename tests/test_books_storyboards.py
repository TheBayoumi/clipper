"""Book-specific approved planning storyboards enforce binding and human-review blocks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from clipper.books import selection, storyboards

NICHES = Path(__file__).resolve().parents[1] / "niches"


def _catalog() -> dict:
    return json.loads((NICHES / "book-editorial-catalog.json").read_text())


@pytest.mark.parametrize("book_id", ["atomic-habits", "deep-work", "psychology-of-money"])
def test_all_catalog_books_have_matching_planning_storyboards(book_id: str) -> None:
    catalog = _catalog()
    picked = selection.choose_lesson(catalog, book_id=book_id)
    board, source = storyboards.resolve_template(catalog, picked, catalog_dir=NICHES)
    assert source.template_path.endswith(f"{book_id}-approved-storyboard.json")
    assert len(source.template_sha256) == 64
    assert board["production_storyboard"]["source_context"]["book_id"] == book_id
    assert board["production_storyboard"]["publish_eligible"] is False


def test_missing_or_undeclared_storyboard_never_falls_back(tmp_path: Path) -> None:
    catalog = _catalog()
    picked = selection.choose_lesson(catalog, book_id="deep-work")
    with pytest.raises(selection.BookSelectionError, match="missing or unreadable"):
        storyboards.resolve_template(catalog, picked, catalog_dir=tmp_path)
    catalog["books"][1].pop("storyboard_template")
    with pytest.raises(selection.BookSelectionError, match="requires declared"):
        storyboards.resolve_template(catalog, picked, catalog_dir=NICHES)
    catalog["books"][1]["storyboard_template"] = "../atomic-habits-approved-storyboard.json"
    with pytest.raises(selection.BookSelectionError, match="book-specific"):
        storyboards.resolve_template(catalog, picked, catalog_dir=NICHES)


def test_cross_book_override_is_not_accepted() -> None:
    catalog = _catalog()
    picked = selection.choose_lesson(catalog, book_id="deep-work")
    with pytest.raises(selection.BookSelectionError, match="book mismatch"):
        storyboards.resolve_template(
            catalog,
            picked,
            catalog_dir=NICHES,
            override=NICHES / "atomic-habits-approved-storyboard.json",
        )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("book", "book mismatch"),
        ("context", "source_context book mismatch"),
        ("schema", "unsupported"),
        ("delivery", "delivery contract"),
        ("roles", "narrative roles"),
        ("direction", "shot-direction"),
        ("status", "unapproved"),
        ("publish", "unapproved"),
        ("review", "unapproved"),
    ],
)
def test_bad_storyboard_is_rejected(change: str, error: str) -> None:
    catalog = _catalog()
    picked = selection.choose_lesson(catalog, book_id="deep-work")
    original, _proof = storyboards.resolve_template(catalog, picked, catalog_dir=NICHES)
    board = copy.deepcopy(original)
    spec = board["production_storyboard"]
    if change == "book":
        board["book"]["author"] = "Invented author"
    elif change == "context":
        spec["source_context"]["book_id"] = "atomic-habits"
    elif change == "schema":
        spec["schema"] = "other"
    elif change == "delivery":
        spec["delivery"]["duration_seconds"]["min"] = 20
    elif change == "roles":
        spec["editorial_sequence"].pop()
    elif change == "direction":
        spec["shot_direction_requirements"] = {}
    elif change == "status":
        spec["status"] = "APPROVED_FOR_PUBLICATION"
    elif change == "publish":
        spec["publish_eligible"] = True
    else:
        board["acceptance"].pop("requires_exact_mp4_sha256_human_approval")
    with pytest.raises(selection.BookSelectionError, match=error):
        storyboards.validate_template(board, picked)
