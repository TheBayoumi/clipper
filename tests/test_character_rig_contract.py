"""Adversarial GLB rig contract tests, requiring no GPU or hosted models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/character_rig_contract.py"
SPEC = importlib.util.spec_from_file_location("approved_rig_contract", PATH)
assert SPEC is not None and SPEC.loader is not None
rig = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rig
SPEC.loader.exec_module(rig)


def fixture_glb(tmp_path: Path, *, skinned: bool = True, visemes: bool = True) -> Path:
    nodes = [{"name": f"Joint_{i}"} for i in range(16)]
    nodes.append({"name": "Host", "mesh": 0, **({"skin": 0} if skinned else {})})
    attributes = {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}
    data = {
        "asset": {"version": "2.0"},
        "nodes": nodes,
        "skins": [{"joints": list(range(16))}] if skinned else [],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "targets": [
                            {"POSITION": 3},
                            {"POSITION": 4},
                            {"POSITION": 5},
                            {"POSITION": 6},
                        ],
                    }
                ],
                "extras": {
                    "targetNames": ["viseme_A", "viseme_B", "viseme_C", "viseme_D"]
                    if visemes
                    else ["A", "B", "C", "D"]
                },
            }
        ],
        "buffers": [{"byteLength": 240}],
        "accessors": [
            {"count": 3, "type": "VEC3"},
            {"count": 3, "type": "VEC4"},
            {"count": 3, "type": "VEC4"},
            *[{"count": 3, "type": "VEC3"} for _ in range(4)],
        ],
    }
    payload = json.dumps(data, separators=(",", ":")).encode()
    payload += b" " * ((-len(payload)) % 4)
    binary = bytes(240)
    raw = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + 8 + len(binary))
        + struct.pack("<I4s", len(payload), b"JSON")
        + payload
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
    path = tmp_path / "candidate.glb"
    path.write_bytes(raw)
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_glb_structure_is_distinct_from_creative_approval(tmp_path: Path) -> None:
    model = fixture_glb(tmp_path)
    result = rig.audit_rig(model, digest(model))
    assert result["unique_joint_count"] == 16
    assert len(result["viseme_morph_targets"]) == 4
    assert result["structure_gate"] == "PASS"
    assert result["publish_eligible"] is False
    assert result["human_identity_review"] == "PENDING"


def test_unrigged_mesh_does_not_pass(tmp_path: Path) -> None:
    model = fixture_glb(tmp_path, skinned=False)
    with pytest.raises(rig.RigContractError, match="no skinned actor hierarchy"):
        rig.audit_rig(model, digest(model))


def test_no_narration_visemes_does_not_pass(tmp_path: Path) -> None:
    model = fixture_glb(tmp_path, visemes=False)
    with pytest.raises(rig.RigContractError, match="viseme"):
        rig.audit_rig(model, digest(model))


def test_unauthorized_asset_substitution_rejected(tmp_path: Path) -> None:
    model = fixture_glb(tmp_path)
    with pytest.raises(rig.RigContractError, match="differ"):
        rig.audit_rig(model, "0" * 64)


def test_truncated_binary_rejected(tmp_path: Path) -> None:
    model = fixture_glb(tmp_path)
    model.write_bytes(model.read_bytes()[:-4])
    with pytest.raises(rig.RigContractError, match="envelope"):
        rig.audit_rig(model, digest(model))


def test_skin_weights_cannot_be_missing_for_skinned_mesh(tmp_path: Path) -> None:
    model = fixture_glb(tmp_path)
    data = rig.glb_json(model)
    data["meshes"][0]["primitives"][0]["attributes"].pop("JOINTS_0")
    payload = json.dumps(data, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    binary = bytes(240)
    model.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + 8 + 240)
        + struct.pack("<I4s", len(payload), b"JSON")
        + payload
        + struct.pack("<I4s", 240, b"BIN\x00")
        + binary
    )
    with pytest.raises(rig.RigContractError, match="vertex skin weights"):
        rig.audit_rig(model, digest(model))
