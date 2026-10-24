"""The 15-second demo must not accidentally become a production episode."""

from __future__ import annotations

import hashlib
import itertools
import importlib.util
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "clipper_production_episode_plan", ROOT / "scripts/wan_episode_plan.py"
)
assert SPEC and SPEC.loader
PLANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLANNER
SPEC.loader.exec_module(PLANNER)

SOURCE_SHA = "c3bbc6395e47c04646ad3f0ad44f13873511e4ee3189046eafbc682225e746d2"


def build_audio(tmp_path: Path, lengths: list[float]) -> tuple[dict, list[Path]]:
    scenes = []
    files = []
    for index, seconds in enumerate(lengths):
        scenes.append({
            "headline": f"Unique section {index}",
            "narration": "An original story with a meaningful idea and useful example.",
        })
        path = tmp_path / f"scene-{index:02d}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(24000)
            out.writeframes(b"\x01\x00" * int(seconds * 24000))
        files.append(path)
    return {"title": "Book lesson", "scenes": scenes}, files


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([10, 10, 10], 30.0),
        ([13.5, 12.1, 10.4, 11], 47.0),
        ([15, 14, 17, 14], 60.0),
    ],
)
def test_valid_duration_preserves_audio_and_renders_variable_shot_count(
    tmp_path: Path, lengths: list[float], expected: float
) -> None:
    script, audios = build_audio(tmp_path, lengths)
    plan = PLANNER.build_episode_plan(script, audios, SOURCE_SHA)
    assert plan["output"]["duration_seconds"] == expected
    assert plan["output"]["fps"] == 24
    assert (plan["output"]["width"], plan["output"]["height"]) == (1080, 1920)
    assert plan["shot_count"] > len(lengths)
    assert plan["shot_count"] == len(plan["shots"])
    assert plan["shots"][0]["start_frame"] == 0
    assert plan["shots"][-1]["end_frame"] == plan["output"]["total_frames"]
    assert plan["publish_eligible"] is False
    assert all(x["director_brief"] is None and x["renderer"] is None for x in plan["shots"])
    assert all(
        2.5 <= x["end_seconds"] - x["start_seconds"] <= 6.001
        for x in plan["shots"]
    )
    for before, after in itertools.pairwise(plan["shots"]):
        assert before["end_frame"] == after["start_frame"]


@pytest.mark.parametrize("lengths", [[5, 5, 5], [10, 10, 9.9], [20, 20, 20.1]])
def test_rejects_15s_demo_and_out_of_range_video(
    tmp_path: Path, lengths: list[float]
) -> None:
    script, audios = build_audio(tmp_path, lengths)
    with pytest.raises(PLANNER.ProductionTimelineError, match="MUST be 30"):
        PLANNER.build_episode_plan(script, audios, SOURCE_SHA)


def test_voice_cache_fingerprint_and_no_false_creative_approval(tmp_path: Path) -> None:
    script, audios = build_audio(tmp_path, [13, 12, 11, 10])
    first = PLANNER.build_episode_plan(script, audios, SOURCE_SHA)
    assert first["fingerprint"] == PLANNER.build_episode_plan(
        script, audios, SOURCE_SHA
    )["fingerprint"]
    assert first["audio"][0]["sha256"] == hashlib.sha256(audios[0].read_bytes()).hexdigest()
    assert first["stage"] == "AUDIO_LOCKED_AWAITING_AI_STORY_DIRECTION"
    with wave.open(str(audios[0]), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"\x03\x00" * (13 * 24000))
    assert PLANNER.build_episode_plan(script, audios, SOURCE_SHA)["fingerprint"] != first[
        "fingerprint"
    ]


def test_incomplete_audio_and_bad_sample_rate_are_rejected(tmp_path: Path) -> None:
    script, audios = build_audio(tmp_path, [11, 12, 13])
    with pytest.raises(PLANNER.ProductionTimelineError, match="exact cached"):
        PLANNER.build_episode_plan(script, audios[:2], SOURCE_SHA)
    with wave.open(str(audios[1]), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * (16000 * 5))
    with pytest.raises(PLANNER.ProductionTimelineError, match="24 kHz"):
        PLANNER.build_episode_plan(script, audios, SOURCE_SHA)
