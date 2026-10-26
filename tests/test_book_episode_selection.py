"""Source-backed automatic book rotation and non-frozen production storyboard."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "book_episode_selection_test", ROOT / "scripts/book_episode_selection.py"
)
assert SPEC and SPEC.loader
CHOOSER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHOOSER
SPEC.loader.exec_module(CHOOSER)

CATALOG = json.loads((ROOT / "niches/book-editorial-catalog.json").read_text())
BOARD = json.loads((ROOT / "niches/atomic-habits-approved-storyboard.json").read_text())


def test_default_auto_book_selection_uses_source_evidence_and_no_publish() -> None:
    result = CHOOSER.choose_lesson(CATALOG)
    assert result["mode"] == "automatic_catalog_rotation"
    assert result["book"]["title"] == "Atomic Habits"
    assert result["lesson"]["id"] == "make-it-easy"
    assert result["sources"] and all(x.startswith("https://") for x in result["sources"])
    assert result["approved_facts"] == [result["lesson"]["claim"]]
    assert result["min_duration_seconds"] == 30
    assert result["max_duration_seconds"] == 60
    assert result["publish_eligible"] is False
    assert result["fingerprint"] == CHOOSER.choose_lesson(CATALOG)["fingerprint"]


def test_auto_rotation_favors_new_book_and_no_repeated_lesson() -> None:
    first = CHOOSER.choose_lesson(CATALOG)
    history = [
        {
            "book_id": first["book"]["id"],
            "lesson_id": first["lesson"]["id"],
        }
    ]
    second = CHOOSER.choose_lesson(CATALOG, history)
    assert second["book"]["id"] != first["book"]["id"]
    history.append(
        {
            "book_id": second["book"]["id"],
            "lesson_id": second["lesson"]["id"],
        }
    )
    third = CHOOSER.choose_lesson(CATALOG, history)
    assert third["book"]["id"] not in {first["book"]["id"], second["book"]["id"]}
    assert third["lesson"]["id"] != first["lesson"]["id"]


def test_manual_user_override_and_topic_filter() -> None:
    manual = CHOOSER.choose_lesson(CATALOG, book_id="atomic-habits", lesson_id="two-minute-rule")
    assert manual["mode"] == "manual_override"
    assert manual["lesson"]["id"] == "two-minute-rule"
    assert CHOOSER.choose_lesson(CATALOG, pillar="psychology")["book"]["id"] == (
        "psychology-of-money"
    )
    with pytest.raises(CHOOSER.BookSelectionError, match="requires book_id"):
        CHOOSER.choose_lesson(CATALOG, lesson_id="two-minute-rule")
    with pytest.raises(CHOOSER.BookSelectionError, match="not in source-backed"):
        CHOOSER.choose_lesson(CATALOG, book_id="invented-book")


def test_unsourced_claims_are_rejected() -> None:
    altered = json.loads(json.dumps(CATALOG))
    altered["books"][0]["lessons"][0]["claim_source_url"] = ""
    with pytest.raises(CHOOSER.BookSelectionError, match="missing evidence"):
        CHOOSER.choose_lesson(altered)
    altered = json.loads(json.dumps(CATALOG))
    altered["books"][0]["status"] = "unverified"
    with pytest.raises(CHOOSER.BookSelectionError, match="unverified book"):
        CHOOSER.choose_lesson(altered)


def test_new_storyboard_is_semantic_not_frozen_60s() -> None:
    spec = BOARD["production_storyboard"]
    assert spec["delivery"]["duration_seconds"] == {"min": 30, "max": 60, "preferred": 45}
    assert spec["delivery"]["shot_count"].startswith("adaptive")
    assert len(spec["editorial_sequence"]) >= 6
    assert all("start" not in beat and "end" not in beat for beat in spec["editorial_sequence"])
    assert spec["selection"]["default"].startswith("select one book")
    assert spec["status"].endswith("AWAITING_DYNAMIC_AI_DIRECTOR")
    assert BOARD["pilot"]["duration_seconds"] == 15
    assert "full_episode" not in BOARD
    assert spec["publish_eligible"] is False


def test_source_backed_selection_is_usable_by_current_qwen_script() -> None:
    selection = CHOOSER.choose_lesson(
        CATALOG, book_id="deep-work", lesson_id="focus-without-distraction"
    )
    config = CHOOSER.to_niche_config(selection)
    assert config["format"] == "book_lesson"
    assert config["book_title"] == "Deep Work"
    assert config["book_author"] == "Cal Newport"
    assert config["facts"] == [selection["lesson"]["claim"]]
    assert config["source_url"] == selection["lesson"]["claim_source_url"]
    assert config["topic"] == selection["lesson"]["topic"]


def test_selected_lesson_builds_dynamic_source_led_storyboard_outline() -> None:
    selection = CHOOSER.choose_lesson(CATALOG, book_id="psychology-of-money")
    outline = CHOOSER.to_storyboard_outline(selection, BOARD)
    assert outline["schema"] == "clipper-book-story-outline-v1"
    assert outline["book"]["title"] == "The Psychology of Money"
    assert outline["selected_hook_angle"] == selection["lesson"]["hook_angle"]
    assert outline["selected_concrete_visual_action"] == selection["lesson"]["visual_action"]
    assert outline["duration_range_seconds"] == [30, 60]
    assert all("start" not in beat and "end" not in beat for beat in outline["narrative_roles"])
    assert outline["timing"].startswith("WAIT_FOR_KOKORO")
    assert outline["publish_eligible"] is False


def test_stable_episode_identity_is_distinct_across_lessons() -> None:
    first = CHOOSER.choose_lesson(CATALOG, book_id="atomic-habits", lesson_id="make-it-easy")
    second = CHOOSER.choose_lesson(CATALOG, book_id="atomic-habits", lesson_id="two-minute-rule")
    assert first["episode_id"] != second["episode_id"]
    assert first["angle_id"] == "primary"
    assert first["script_version"] == CHOOSER.SCRIPT_VERSION
    assert (
        CHOOSER.choose_lesson(CATALOG, book_id="atomic-habits", lesson_id="make-it-easy")[
            "episode_id"
        ]
        == first["episode_id"]
    )
    config = CHOOSER.to_niche_config(first)
    assert config["voice"] == "am_adam"
    assert config["selection_fingerprint"] == first["fingerprint"]


def test_angle_rotation_avoids_repeat_when_alternative_available() -> None:
    catalog = json.loads(json.dumps(CATALOG))
    lesson = catalog["books"][0]["lessons"][0]
    lesson["angles"] = [
        {"id": "friction", "hook_angle": "Remove resistance", "visual_action": "Move a pen"},
        {"id": "setup", "hook_angle": "Prepare ahead", "visual_action": "Set a notebook"},
    ]
    previous = CHOOSER.choose_lesson(
        catalog, book_id="atomic-habits", lesson_id="make-it-easy", angle_id="friction"
    )
    selected = CHOOSER.choose_lesson(
        catalog,
        [
            {
                "book_id": previous["book_id"],
                "lesson_id": previous["lesson_id"],
                "angle_id": previous["angle_id"],
            }
        ],
        book_id="atomic-habits",
        lesson_id="make-it-easy",
    )
    assert selected["angle_id"] == "setup"
    assert selected["episode_id"] != previous["episode_id"]
    assert selected["approved_facts"] == previous["approved_facts"]


def test_invalid_angles_and_unsafe_source_urls_fail_closed() -> None:
    catalog = json.loads(json.dumps(CATALOG))
    catalog["books"][0]["lessons"][0]["angles"] = [
        {"id": "same", "hook_angle": "One", "visual_action": "A"},
        {"id": "same", "hook_angle": "Two", "visual_action": "B"},
    ]
    with pytest.raises(CHOOSER.BookSelectionError, match="duplicate"):
        CHOOSER.choose_lesson(catalog)
    catalog = json.loads(json.dumps(CATALOG))
    catalog["books"][0]["lessons"][0]["claim_source_url"] = (
        "https://attacker:secret@example.org/claim"
    )
    with pytest.raises(CHOOSER.BookSelectionError, match="HTTPS"):
        CHOOSER.choose_lesson(catalog)
    with pytest.raises(CHOOSER.BookSelectionError, match="angle override requires"):
        CHOOSER.choose_lesson(CATALOG, angle_id="primary")
