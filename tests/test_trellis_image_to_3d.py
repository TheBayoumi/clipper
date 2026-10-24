"""Reject missing/degenerate 3D meshes and keep image-to-3D unapproved."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/trellis_image_to_3d.py"
SPEC = importlib.util.spec_from_file_location("clipper_trellis2", PATH)
assert SPEC is not None and SPEC.loader is not None
trellis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trellis
SPEC.loader.exec_module(trellis)


def test_missing_mesh_does_not_pass(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        trellis.validate_glb(tmp_path / "missing.glb")
    empty = tmp_path / "empty.glb"
    empty.write_bytes(b"glTF")
    with pytest.raises(ValueError, match="small"):
        trellis.validate_glb(empty)


def test_generated_asset_never_receives_false_rigging_claim() -> None:
    script = PATH.read_text(encoding="utf-8")
    assert trellis.SPACE == "microsoft/TRELLIS.2"
    assert '"rigging_status": "NOT_IMPLEMENTED"' in script
    assert '"publish_eligible": False' in script
    assert '"creative_status": "NOT_REVIEWED"' in script
    assert '"CANDIDATE_GLB_GENERATED_NOT_RIGGED"' in script
    assert 'client.predict(100000, 2048, api_name="/extract_glb")' in script
    assert "client.predict(generated[0], 100000" not in script


def test_workflow_uses_approved_reference_and_unprivileged_gpu_service() -> None:
    root = PATH.parents[1]
    wf = (root / ".github/workflows/approved-host-image-to-3d.yml").read_text()
    assert "reference_base64" in wf
    assert "reference_url" in wf
    assert "probe_only" in wf
    assert "secrets.HF_TOKEN" in wf
    assert "github.sha" in wf
    assert "if: always()" in wf
