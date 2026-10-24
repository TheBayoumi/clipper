"""Assertions for the first genuine skinned 3D character performance gate."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_versioned_identity_and_animated_skeleton() -> None:
    host = json.loads((ROOT / "niches/character-host-v1.json").read_text())
    assert host["rig"] == "game"
    assert host["animations"] is True
    assert host["faceBones"] is True
    assert host["faceShapes"] is True
    assert host["weapon"] == "none"
    assert host["seed"] == 724
    assert host["name"] == "PageToLife_Host_V1"


def test_real_motion_is_required_not_camera_pan() -> None:
    script = (ROOT / "scripts/blender_character_probe.py").read_text()
    for keyword in (
        "bpy.ops.import_scene.gltf",
        "genuine_wave_motion",
        "max_arm_bone_matrix_delta",
        "0.015",
        "ARMATURE",
        "bpy.ops.render.render(write_still=True)",
        '"publish_eligible": False',
    ):
        assert keyword in script


def test_external_source_is_pinned_and_review_is_nonpublishable() -> None:
    workflow = (ROOT / ".github/workflows/rigged-character-proof.yml").read_text()
    assert "8aa993d7141b6f37e5d16f611264e6f1ab520f30" in workflow
    assert "npm ci" in workflow
    assert "40047215101" in workflow
    assert "full" not in workflow.split("name: Rigged human motion")[0].lower()
    assert "if: always()" in workflow
    assert "workflow_dispatch:" in workflow
    assert "\\${{" not in workflow
    assert "inputs.frames || '24'" in workflow
    assert "real-rigged-human-${{ github.sha }}" in workflow
