"""Tests enforce the approved visuals, rather than only video-codec success."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/storyboard_acceptance.py"
SPEC = importlib.util.spec_from_file_location("storyboard_acceptance", SOURCE)
assert SPEC is not None and SPEC.loader is not None
sa = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sa
SPEC.loader.exec_module(sa)
CONTRACT = ROOT / "niches/atomic-habits-approved-storyboard.json"


def test_approved_storyboard_production_contract_is_locked() -> None:
    config = sa.read_contract(CONTRACT)
    assert [s["id"] for s in config["pilot"]["shots"]] == [
        "hook",
        "book_intro",
        "lesson_example",
    ]
    assert config["character"]["clothing"].find("hoodie") >= 0
    assert config["character"]["status"] == "master_rig_missing"
    assert config["pilot"]["duration_seconds"] == 15
    assert config["acceptance"]["tech_result_is_creative_approval"] is False


def test_generic_stock_rig_cannot_count_as_approved_master(tmp_path: Path) -> None:
    config = sa.read_contract(CONTRACT)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "storyboard_schema": "storyboard-production-v1",
                "character_id": config["character"]["character_id"],
                "master_character_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(sa.CreativeGateError, match="unapproved"):
        sa.audit_render(config, manifest, tmp_path / "video.mp4")


def test_storyboard_changes_cannot_be_hidden(tmp_path: Path) -> None:
    config = sa.read_contract(CONTRACT)
    config["pilot"]["shots"][1]["start"] = 4
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(sa.CreativeGateError, match="gap or overlap"):
        sa.read_contract(f)


def test_static_stage_does_not_satisfy_acting(tmp_path: Path) -> None:
    config = sa.read_contract(CONTRACT)
    config["pilot"]["shots"][0]["meaningful_motion"] = False
    f = tmp_path / "static.json"
    f.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(sa.CreativeGateError, match="physical acting"):
        sa.read_contract(f)


def test_current_generic_character_is_not_storyboard_character() -> None:
    legacy = json.loads((ROOT / "niches/character-host-v1.json").read_text())
    config = sa.read_contract(CONTRACT)
    assert legacy["outfit"] == "male_casualsuit01"
    assert legacy["name"] != config["character"]["character_id"]
    assert legacy["outfit"] in config["character"]["forbidden_substitutes"]


def _physical_sample(prop_id: str = "hand_lifts_book") -> dict:
    return {
        "evaluation_source": "evaluated_3d_scene",
        "character_sha256": "a" * 64,
        "prop_id": prop_id,
        "hand_start": [0.0, 0.0, 0.0],
        "hand_at_contact": [0.97, 0.0, 0.0],
        "prop_start": [1.0, 0.0, 0.0],
        "prop_at_contact": [1.0, 0.0, 0.0],
        "prop_end": [1.2, 0.0, 0.0],
    }


def test_measured_book_pickup_is_admissible_but_not_creative_approval() -> None:
    expected = {"id": "book_intro", "required_prop_interaction": "hand_lifts_book"}
    actual = {"physical_contact_evidence": _physical_sample()}
    assert sa.verify_physical_contact(expected, actual, "a" * 64) is None


def test_hand_wave_without_contact_does_not_count_as_pickup() -> None:
    expected = {"id": "book_intro", "required_prop_interaction": "hand_lifts_book"}
    evidence = _physical_sample()
    evidence["hand_at_contact"] = [0.0, 0.0, 0.0]
    with pytest.raises(sa.CreativeGateError, match="measurable reach"):
        sa.verify_physical_contact(expected, {"physical_contact_evidence": evidence}, "a" * 64)


def test_stationary_book_does_not_count_as_interaction() -> None:
    expected = {"id": "book_intro", "required_prop_interaction": "hand_lifts_book"}
    evidence = _physical_sample()
    evidence["prop_end"] = evidence["prop_start"]
    with pytest.raises(sa.CreativeGateError, match="scene transformation"):
        sa.verify_physical_contact(expected, {"physical_contact_evidence": evidence}, "a" * 64)


def test_prop_proof_cannot_be_transferred_to_another_character() -> None:
    expected = {"id": "book_intro", "required_prop_interaction": "hand_lifts_book"}
    with pytest.raises(sa.CreativeGateError, match="another character"):
        sa.verify_physical_contact(
            expected, {"physical_contact_evidence": _physical_sample()}, "b" * 64
        )


def test_boolean_prop_flags_do_not_substitute_for_scene_measurements() -> None:
    expected = {"id": "hook", "required_prop_interaction": "hand_reaches_for_phone"}
    with pytest.raises(sa.CreativeGateError, match="physical evidence"):
        sa.verify_physical_contact(expected, {"prop_interaction_verified": True}, "a" * 64)


def test_nonfinite_or_boolean_coords_are_never_valid_measurement() -> None:
    expected = {"id": "book_intro", "required_prop_interaction": "hand_lifts_book"}
    for invalid in (float("nan"), True):
        evidence = _physical_sample()
        evidence["hand_at_contact"][0] = invalid
        with pytest.raises(sa.CreativeGateError, match=r"numeric|non-finite"):
            sa.verify_physical_contact(expected, {"physical_contact_evidence": evidence}, "a" * 64)
