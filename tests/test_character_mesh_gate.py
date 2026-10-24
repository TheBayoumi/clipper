"""Fail-closed tests for trusted character cleanup and immutable source evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/character_mesh_gate.py"
SPEC = importlib.util.spec_from_file_location("clipper_mesh_gate", SOURCE)
assert SPEC and SPEC.loader
mesh = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mesh
SPEC.loader.exec_module(mesh)


@pytest.mark.parametrize(
    ("extents", "height", "expected"),
    [
        ((1.0, .026, 1.0), -.413, True),
        ((1.0, .07, 1.0), -.45, True),
        ((.2, .02, .8), -.45, False),
        ((1.0, .3, 1.0), -.45, False),
        ((1.0, .03, 1.0), .1, False),
        ((.69, .03, .69), -.45, False),
    ],
)
def test_plane_signature_does_not_delete_character_parts(
    extents: tuple[float, float, float], height: float, expected: bool
) -> None:
    assert mesh.is_background_sheet(extents, height) is expected


def test_probe_cannot_be_cleaned(tmp_path: Path) -> None:
    source, glb, evidence = (tmp_path / name for name in ("source", "model", "evidence"))
    source.write_bytes(b"source")
    glb.write_bytes(b"glTF")
    evidence.write_text(json.dumps({"schema": mesh.SCHEMA, "status": "API_PROBE_ONLY"}))
    with pytest.raises(mesh.MeshGateError, match="did not generate"):
        mesh.read_provenance(source, glb, evidence)


def test_tampered_model_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, glb, evidence = (tmp_path / name for name in ("source", "model", "evidence"))
    source.write_bytes(b"approved")
    glb.write_bytes(b"tampered")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(mesh, "SOURCE_SHA", source_sha)
    evidence.write_text(json.dumps({
        "schema": mesh.SCHEMA, "status": "CANDIDATE_GLB_GENERATED_NOT_RIGGED",
        "model_family": "TRELLIS_v1_community", "publish_eligible": False,
        "source_sha256": source_sha, "glb_sha256": "a" * 64,
    }))
    with pytest.raises(mesh.MeshGateError, match="do not match"):
        mesh.read_provenance(source, glb, evidence)


def test_production_claim_from_provider_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, glb, evidence = (tmp_path / name for name in ("source", "model", "evidence"))
    source.write_bytes(b"approved")
    glb.write_bytes(b"scan")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(mesh, "SOURCE_SHA", source_sha)
    evidence.write_text(json.dumps({
        "schema": mesh.SCHEMA, "status": "CANDIDATE_GLB_GENERATED_NOT_RIGGED",
        "model_family": "TRELLIS_v1_community", "publish_eligible": True,
        "source_sha256": source_sha,
        "glb_sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
    }))
    with pytest.raises(mesh.MeshGateError, match="publication"):
        mesh.read_provenance(source, glb, evidence)


def test_clipper_asset_workflow_never_reruns_model() -> None:
    root = SOURCE.parents[1]
    workflow = (root / ".github/workflows/approved-host-geometry-audit.yml").read_text()
    assert "gh run download" in workflow
    assert "sha256sum --check" in workflow
    assert "scripts/character_mesh_gate.py" in workflow
    assert "scripts/character_turntable.py" in workflow
    assert "trellis_image_to_3d.py --" not in workflow
    assert "actions/upload-artifact@v4" in workflow
