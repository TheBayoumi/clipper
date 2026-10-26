"""CPU-only, no-model regression tests for the book AI Director contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


selector = load("book_selection_director_test", "scripts/book_episode_selection.py")
director = load("book_ai_director_test", "scripts/book_ai_director.py")
niche = load("niche_video_director_test", "scripts/niche_video.py")
CATALOG = json.loads((ROOT / "niches/book-editorial-catalog.json").read_text())
EPISODE = {
    "title": "Make your first step easier",
    "scenes": [
        {
            "headline": "Hook",
            "narration": "You keep waiting for motivation but what if your first step is simply too hard today",
        },
        {
            "headline": "Action",
            "narration": (
                "James Clear explains making a desired habit easy. Imagine leaving an open "
                "notebook beside your morning coffee instead of planning a perfect schedule"
            ),
        },
        {
            "headline": "Payoff",
            "narration": (
                "Tomorrow reduce one tiny action until starting feels realistic then repeat "
                "it consistently. Your job is to begin not to feel inspired"
            ),
        },
    ],
}


def selection() -> dict:
    return selector.choose_lesson(CATALOG, book_id="atomic-habits", lesson_id="make-it-easy")


def make_plan(request: dict, shots_per_section: int = 1) -> dict:
    words = [
        word for section in request["narrative_sections"] for word in section["narration"].split()
    ]
    shots = []
    for section in request["narrative_sections"]:
        low, high = section["start_word"], section["end_word"]
        borders = [
            low + (high - low) * i // shots_per_section for i in range(shots_per_section + 1)
        ]
        for part in range(shots_per_section):
            start, end = borders[part], borders[part + 1]
            role = "demonstration"
            if section["index"] == 0 and part == 0:
                role = "hook"
            elif (
                section["index"] == len(request["narrative_sections"]) - 1
                and part == shots_per_section - 1
            ):
                role = "payoff"
            shots.append(
                {
                    "id": f"shot-{section['index']}-{part}",
                    "role": role,
                    "scene_index": section["index"],
                    "start_word": start,
                    "end_word": end,
                    "narration_text": " ".join(words[start:end]),
                    "renderer": "s2v" if part == 0 else "i2v",
                    "character_action": "Host opens a notebook to begin writing",
                    "camera": "Medium close-up dolly",
                    "setting": "Warm study desk",
                    "emotion": "Thoughtful",
                    "required_objects": ["notebook"],
                    "source_evidence": [request["approved_claims"][0]]
                    if section["index"] == 1 and part == 0
                    else [],
                    "qa_criteria": ["Coherent hand movement", "Character identity matches"],
                }
            )
    return {
        "schema": director.PLAN_SCHEMA,
        "request_fingerprint": request["fingerprint"],
        "shots": shots,
    }


def test_three_or_six_variable_shots_preserve_one_script_without_gpu() -> None:
    request = director.build_request(selection(), EPISODE)
    first = director.validate_plan(request, make_plan(request))
    second = director.validate_plan(request, make_plan(request, shots_per_section=2))
    assert request["episode_id"] == "atomic-habits--make-it-easy--primary"
    assert first["shot_count"] == 3
    assert second["shot_count"] == 6
    assert first["fingerprint"] != second["fingerprint"]
    assert first["publish_eligible"] is False
    assert all("start_seconds" not in shot for shot in first["shots"])


@pytest.mark.parametrize("defect", ["gap", "duplicate", "tamper", "cross"])
def test_rejects_inconsistent_shot_spans(defect: str) -> None:
    request = director.build_request(selection(), EPISODE)
    candidate = make_plan(request)
    if defect == "gap":
        candidate["shots"][1]["start_word"] += 1
    elif defect == "duplicate":
        candidate["shots"][1]["id"] = candidate["shots"][0]["id"]
    elif defect == "tamper":
        candidate["shots"][0]["narration_text"] = "Invented line"
    else:
        candidate["shots"][0]["end_word"] += 1
    with pytest.raises(director.DirectorContractError):
        director.validate_plan(request, candidate)


def test_rejects_unsupported_evidence_and_wrong_request() -> None:
    request = director.build_request(selection(), EPISODE)
    candidate = make_plan(request)
    candidate["shots"][1]["source_evidence"] = [
        {"claim": "Invented statistic", "source_url": "https://example.org"}
    ]
    with pytest.raises(director.DirectorContractError, match="invented"):
        director.validate_plan(request, candidate)
    candidate = make_plan(request)
    candidate["request_fingerprint"] = "0" * 64
    with pytest.raises(director.DirectorContractError, match="different"):
        director.validate_plan(request, candidate)
    corrupted = copy.deepcopy(request)
    corrupted["narrative_sections"][0]["narration"] = "altered"
    with pytest.raises(director.DirectorContractError, match="fingerprint"):
        director.validate_plan(corrupted, make_plan(request))


def test_rejects_tampered_selection_and_unsourced_script() -> None:
    selected = selection()
    selected["approved_facts"] = ["Everyone succeeds every time"]
    with pytest.raises(director.DirectorContractError, match="fingerprint"):
        director.build_request(selected, EPISODE)
    with pytest.raises(director.DirectorContractError, match="word budget"):
        director.build_request(
            selection(),
            {
                "title": "short",
                "scenes": [
                    {"headline": "a", "narration": "one"},
                    {"headline": "b", "narration": "two"},
                    {"headline": "c", "narration": "three"},
                ],
            },
        )


def test_qwen_selection_prompt_and_cache_are_episode_specific(tmp_path: Path) -> None:
    selected = selection()
    config_data = selector.to_niche_config(selected)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(config_data), encoding="utf-8")
    config = niche.NicheConfig.load(path)
    prompt = json.loads(niche._script_request(config)["messages"][1]["content"])
    assert config.voice == "am_adam"
    assert prompt["selection_fingerprint"] == selected["fingerprint"]
    assert prompt["episode_id"] == selected["episode_id"]
    assert prompt["hook_angle"] == selected["lesson"]["hook_angle"]
    current_digest = niche._script_fingerprint(config)
    config_data["angle_id"] = "alternative"
    config_data["episode_id"] = f"{selected['book_id']}--{selected['lesson_id']}--alternative"
    path.write_text(json.dumps(config_data), encoding="utf-8")
    assert current_digest != niche._script_fingerprint(niche.NicheConfig.load(path))
    config_data["episode_id"] = "mismatch"
    path.write_text(json.dumps(config_data), encoding="utf-8")
    with pytest.raises(niche.NicheVideoError, match="identity mismatch"):
        niche.NicheConfig.load(path)
