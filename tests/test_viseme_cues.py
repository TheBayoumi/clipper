"""Voice-derived mouth animation is a rigged jaw effect, never a fake timer."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "viseme_cues.py"
SPEC = importlib.util.spec_from_file_location("viseme_cues", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cues = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cues
SPEC.loader.exec_module(cues)


def test_blender_sidecar_loader_works_without_scripts_on_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the real Blender probe with stubbed bpy to reproduce its imports."""
    blender_source = SCRIPT.with_name("blender_character_probe.py")
    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = object
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)

    spec = importlib.util.spec_from_file_location("clipper_rig_import_probe", blender_source)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    reader, sampled_angles = probe.load_cue_tools()
    assert reader.__module__ == "clipper_viseme_cues"
    assert sampled_angles.__module__ == "clipper_viseme_cues"


def test_rhubarb_speech_shapes_are_timed_and_nontrivial(tmp_path: Path) -> None:
    source = tmp_path / "real-mouths.json"
    source.write_text(
        json.dumps(
            {
                "metadata": {"duration": 1.0},
                "mouthCues": [
                    {"start": 0.0, "end": 0.15, "value": "X"},
                    {"start": 0.15, "end": 0.4, "value": "C"},
                    {"start": 0.4, "end": 0.7, "value": "D"},
                    {"start": 0.7, "end": 1.0, "value": "B"},
                ],
            }
        )
    )
    parsed = cues.read_cues(source)
    values = cues.frame_angles(parsed, fps=12, frames=12)
    assert values[0] == 0
    assert max(values) == pytest.approx(0.28)
    assert len(set(values)) >= 3
    assert cues.angle_at(parsed, 1.5) == 0


def test_invalid_speech_timestamps_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "gapped.json"
    source.write_text(
        json.dumps(
            {
                "mouthCues": [
                    {"start": 0, "end": 0.1, "value": "X"},
                    {"start": 0.3, "end": 0.5, "value": "C"},
                    {"start": 0.5, "end": 0.7, "value": "D"},
                ],
            }
        )
    )
    with pytest.raises(cues.MouthCueError, match="gap or overlap"):
        cues.read_cues(source)


def test_static_or_unrecognized_mouths_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "flat.json"
    source.write_text(
        json.dumps(
            {
                "mouthCues": [
                    {"start": 0, "end": 0.1, "value": "X"},
                    {"start": 0.1, "end": 0.2, "value": "A"},
                    {"start": 0.2, "end": 0.3, "value": "B"},
                ],
            }
        )
    )
    with pytest.raises(cues.MouthCueError, match="visible vowel"):
        cues.read_cues(source)
