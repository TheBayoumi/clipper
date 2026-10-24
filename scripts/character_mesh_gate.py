"""Audit and remove TRELLIS background geometry without claiming rig acceptance.

Reads existing GitHub Action artifacts; it never re-runs model inference.
Geometry-only cleanup is NOT proof of creative identity or safe auto-skinning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "clipper-trellis2-v1"
SOURCE_SHA = "c3bbc6395e47c04646ad3f0ad44f13873511e4ee3189046eafbc682225e746d2"


class MeshGateError(ValueError):
    """Untrusted model artifact or destructive geometry operation."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_provenance(source: Path, glb: Path, evidence_file: Path) -> dict[str, Any]:
    if not all(p.is_file() for p in (source, glb, evidence_file)):
        raise MeshGateError("source, model or inference evidence missing")
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or evidence.get("schema") != SCHEMA:
        raise MeshGateError("unexpected inference evidence contract")
    if evidence.get("status") != "CANDIDATE_GLB_GENERATED_NOT_RIGGED":
        raise MeshGateError("inference did not generate a GLB")
    if evidence.get("model_family") not in ("TRELLIS_v1_community", "TRELLIS.2"):
        raise MeshGateError("unrecognized generator model")
    if evidence.get("publish_eligible") is not False:
        raise MeshGateError("generator claimed publication without approval")
    if evidence.get("source_sha256") != SOURCE_SHA or sha256(source) != SOURCE_SHA:
        raise MeshGateError("source is not the approved pinned image")
    if evidence.get("glb_sha256") != sha256(glb):
        raise MeshGateError("GLB bytes do not match inference evidence")
    return evidence


def is_background_sheet(extents: tuple[float, float, float], y: float) -> bool:
    """Full-width, shallow horizontal plane near feet; not a thin body part."""
    return extents[0] >= 0.70 and extents[2] >= 0.70 and extents[1] <= 0.07 and y <= -0.25


def prepare(glb: Path, target: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    if glb.stat().st_size < 25000:
        raise MeshGateError("GLB too small for approved full-body scan")
    scene = trimesh.load(str(glb), file_type="glb", force="scene")
    if not scene.geometry or any(g.visual.kind != "texture" for g in scene.geometry.values()):
        raise MeshGateError("missing textured 3D geometry")
    body = scene.to_geometry()
    # split() duplicates triangles on nonmanifold TRELLIS scans; use exact face partition
    face_groups = trimesh.graph.connected_components(
        body.face_adjacency, nodes=np.arange(len(body.faces)), min_len=1
    )
    components = body.submesh(face_groups, append=False, repair=False)
    if sum(len(component.faces) for component in components) != len(body.faces):
        raise MeshGateError("connected components did not conserve faces")
    keep, background = [], []
    for component in components:
        spans = tuple(float(v) for v in component.extents)
        if is_background_sheet(spans, float(component.centroid[1])):
            background.append(component)
        else:
            keep.append(component)
    if not keep or not background:
        raise MeshGateError("background not confidently separable; human review required")
    removed = sum(len(c.faces) for c in background)
    if removed >= len(body.faces) * 0.4:
        raise MeshGateError("cleanup would erase more than 40% of geometry")
    clean = trimesh.Scene()
    for i, part in enumerate(keep):
        clean.add_geometry(part, geom_name=f"scan-part-{i:04d}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(clean.export(file_type="glb"))
    check = trimesh.load(str(target), file_type="glb", force="scene")
    expected = len(body.faces) - removed
    if sum(len(part.faces) for part in check.geometry.values()) != expected:
        target.unlink(missing_ok=True)
        raise MeshGateError("round-trip changed triangle count")
    return {
        "schema": "clipper-host-mesh-audit-v1",
        "source_glb_sha256": sha256(glb),
        "clean_glb_sha256": sha256(target),
        "raw_triangles": len(body.faces),
        "removed_background_triangles": removed,
        "clean_triangles": expected,
        "raw_components": len(components),
        "removed_background_components": len(background),
        "remaining_components": len(keep),
        "open_shells_detected": any(not c.is_watertight for c in keep),
        "retopology_required": len(keep) > 75 or any(not c.is_watertight for c in keep),
        "geometry_cleanup": "PASS",
        "identity_review": "PENDING_HUMAN",
        "rigged": False,
        "publish_eligible": False,
        "production_rig_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.glb.resolve():
        parser.error("output must not overwrite the original GLB")
    report: dict[str, Any] = {"status": "REJECTED", "publish_eligible": False}
    try:
        e = read_provenance(args.source, args.glb, args.evidence)
        report.update(prepare(args.glb, args.output))
        report.update(
            provider=e["provider"],
            model_family=e["model_family"],
            source_image_sha256=e["source_sha256"],
            status="CLEANED_CANDIDATE_NOT_RIGGED",
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path = args.output.with_suffix(".audit.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
