"""Prevent accidental storyboard drift when binding Wan video backends."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wan_storyboard_plan_test", ROOT / "scripts/wan_storyboard_plan.py")
assert SPEC and SPEC.loader
BOARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOARD
SPEC.loader.exec_module(BOARD)

SOURCE_SHA = "c3bbc6395e47c04646ad3f0ad44f13873511e4ee3189046eafbc682225e746d2"


def read_board() -> dict[str, object]:
    return json.loads((ROOT / "niches/atomic-habits-approved-storyboard.json").read_text())


def test_plan_preserves_three_actions_and_renderer_routing() -> None:
    plan = BOARD.build_plan(read_board(), SOURCE_SHA)
    assert plan["schema"] == "clipper-wan-storyboard-v1"
    assert [(x["id"], x["renderer"]) for x in plan["shots"]] == [
        ("hook", "s2v"),
        ("book_intro", "i2v"),
        ("lesson_example", "i2v"),
    ]
    assert [x["end_seconds"] for x in plan["shots"]] == [3, 8, 15]
    assert all(x["required_prop_interaction"] and x["prompt"] for x in plan["shots"])
    assert plan["accepted_video_sha256"] is None
    assert plan["publish_eligible"] is False
    assert plan["plan_sha256"] == BOARD.build_plan(read_board(), SOURCE_SHA)["plan_sha256"]


def test_character_or_shot_tampering_rejected() -> None:
    modified = copy.deepcopy(read_board())
    modified["character"]["outfit_lock"] = "business_suit"
    with pytest.raises(BOARD.StoryboardPlanError, match="outfit"):
        BOARD.build_plan(modified, SOURCE_SHA)
    modified = copy.deepcopy(read_board())
    modified["pilot"]["shots"][1]["start"] = 4
    with pytest.raises(BOARD.StoryboardPlanError, match="gaps or overlaps"):
        BOARD.build_plan(modified, SOURCE_SHA)
