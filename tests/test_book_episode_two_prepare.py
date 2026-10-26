"""Episode 02 must differ from the established Atomic Habits pilot."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import book_episode_two_prepare as pilot  # noqa: E402
import niche_video  # noqa: E402

BRIEF = json.loads((ROOT / "niches/atomic-habits-episode-02.json").read_text())
CATALOG = json.loads((ROOT / "niches/book-editorial-catalog.json").read_text())


def test_second_lesson_is_verified_and_has_distinct_cinematic_shots() -> None:
    selected, request, plan = pilot.build_contract(BRIEF, CATALOG)
    assert selected["lesson_id"] == "two-minute-rule"
    assert selected["episode_id"] == "atomic-habits--two-minute-rule--primary"
    assert plan["shot_count"] == 5
    assert len(BRIEF["script"]["scenes"]) == 4
    assert 85 <= request["total_words"] <= 125
    assert selected["lesson"]["claim_source_url"] == BRIEF["source_url"]
    assert plan["shots"][0]["role"] == "hook"
    assert plan["shots"][-1]["role"] == "payoff"
    assert [item["scene_index"] for item in plan["shots"]] == [0, 1, 2, 2, 3]
    assert all(shot["renderer"] == "s2v" for shot in plan["shots"])
    assert all(shot["publish_eligible"] is False for shot in [plan])
    assert sum(len(x.narration.split()) for x in niche_video.Episode.from_json(BRIEF["script"]).scenes) == 105


@pytest.mark.parametrize("change", ["lesson", "source", "duplicate", "skip", "bad_voice"])
def test_refuse_identity_or_timeline_mutation(change: str) -> None:
    edited = copy.deepcopy(BRIEF)
    if change == "lesson":
        edited["lesson_id"] = "make-it-easy"
    elif change == "source":
        edited["source_url"] = "https://example.org/unsupported"
    elif change == "duplicate":
        edited["shot_briefs"].insert(2, copy.deepcopy(edited["shot_briefs"][1]))
    elif change == "skip":
        edited["shot_briefs"].pop(3)
        edited["shot_briefs"][2]["cut_after_words"] = 12
    else:
        edited["shot_briefs"][-1]["scene_index"] = 2
    with pytest.raises((pilot.EpisodeTwoError, ValueError)):
        pilot.build_contract(edited, CATALOG)


def test_contract_creates_immutable_identified_artifacts(tmp_path: Path) -> None:
    pilot.contract(
        ROOT / "niches/atomic-habits-episode-02.json",
        ROOT / "niches/book-editorial-catalog.json",
        tmp_path,
    )
    assert (tmp_path / "selection.json").is_file()
    assert (tmp_path / "director-request.json").is_file()
    assert (tmp_path / "director-plan.json").is_file()
    selection = pilot.load(tmp_path / "selection.json")
    assert selection["publish_eligible"] is False
    assert selection["lesson"]["id"] == "two-minute-rule"
    assert pilot.load(tmp_path / "director-plan.json")["shot_count"] == 5


def test_word_normalization_only_ignores_punctuation_not_content() -> None:
    assert pilot.normalize("two-minute") == "twominute"
    assert pilot.normalize("Two minute") == "twominute"
    assert pilot.normalize("thirty") != pilot.normalize("twenty")
