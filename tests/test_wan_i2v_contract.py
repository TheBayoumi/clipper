"""I2V action-shot contracts remain draft-only and preserve keyframe provenance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "wan_shot_contract.py"
SPEC = importlib.util.spec_from_file_location("wan_i2v_contract_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
WAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WAN
SPEC.loader.exec_module(WAN)


def test_i2v_action_qualified_and_identity_locked(tmp_path: Path) -> None:
    image = tmp_path / "book-keyframe.png"
    Image.new("RGB", (720, 1280), (40, 35, 31)).save(image)
    prompt = "A cinematic stylized cartoon presenter physically lifts a book from the table."
    first = WAN.prepare_i2v(image, prompt, "book_intro")
    again = WAN.prepare_i2v(image, prompt, "book_intro")
    assert first["fingerprint"] == again["fingerprint"]
    assert first["model_id"] == "Wan-AI/Wan2.2-I2V-A14B"
    assert first["model_revision"] == "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
    assert first["keyframe_creative_review"] == "PENDING"
    assert first["audio"] is None
    assert first["publish_eligible"] is False
    assert WAN.prepare_i2v(image, prompt + " Slowly.", "book_intro")["fingerprint"] != first[
        "fingerprint"
    ]


def test_i2v_rejects_unapproved_shot_or_missing_reference(tmp_path: Path) -> None:
    image = tmp_path / "missing.png"
    with pytest.raises(WAN.ShotContractError, match="action must belong"):
        WAN.prepare_i2v(image, "A cinematic book demonstration with careful hand contact.", "hook")
    with pytest.raises(WAN.ShotContractError, match="reference image missing"):
        WAN.prepare_i2v(image, "A cinematic book demonstration with careful hand contact.", "book_intro")
