"""Validate a real GLB body's skin and morph-target structure before pilot use.

This is structural validation only: it cannot establish attractive deformation,
facial identity, art quality or a human-approved video performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


class RigContractError(ValueError):
    """Binary asset does not meet the approved animation-ready rig contract."""


def glb_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 128:
        raise RigContractError("missing or undersized GLB")
    if path.stat().st_size > 200_000_000:
        raise RigContractError("GLB exceeds 200MB inspection budget")
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise RigContractError("truncated GLB header")
        magic, version, total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or total_length != path.stat().st_size:
            raise RigContractError("invalid GLB v2 binary envelope")
        chunk = handle.read(8)
        if len(chunk) != 8:
            raise RigContractError("missing glTF JSON chunk")
        json_length, chunk_type = struct.unpack("<I4s", chunk)
        if chunk_type != b"JSON" or not 2 <= json_length <= 20_000_000:
            raise RigContractError("invalid GLB JSON chunk")
        try:
            data = json.loads(handle.read(json_length))
        except (ValueError, UnicodeError) as exc:
            raise RigContractError("unreadable glTF JSON document") from exc
        binary_header = handle.read(8)
        if len(binary_header) != 8:
            raise RigContractError("GLB has no embedded binary geometry")
        binary_size, binary_tag = struct.unpack("<I4s", binary_header)
        if (
            binary_tag != b"BIN\x00"
            or binary_size < 32
            or len(handle.read(binary_size)) != binary_size
        ):
            raise RigContractError("truncated or absent GLB binary geometry")
    if not isinstance(data, dict) or not isinstance(data.get("asset"), dict):
        raise RigContractError("not a glTF v2 asset")
    if data["asset"].get("version") != "2.0":
        raise RigContractError("not a glTF v2 asset")
    return data


def audit_rig(glb: Path, expected_sha256: str) -> dict[str, Any]:
    """Inspect actual GLB skinned nodes, weight attributes and facial shapes.

    Joint counts and attributes are a prerequisite, never a certification of
    weighted deformation quality or fidelity to the reference character.
    """
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise RigContractError("invalid pinned master asset SHA256")
    h = hashlib.sha256()
    with glb.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != expected_sha256:
        raise RigContractError("master GLB bytes differ from approved asset SHA256")
    data = glb_json(glb)
    skins = data.get("skins", [])
    nodes = data.get("nodes", [])
    meshes = data.get("meshes", [])
    accessors = data.get("accessors", [])
    if not skins or not nodes or not meshes or not accessors or not data.get("buffers"):
        raise RigContractError("no skinned actor hierarchy in GLB")
    skinned_nodes = [n for n in nodes if "skin" in n and "mesh" in n]
    if not skinned_nodes:
        raise RigContractError("no mesh instance bound to a skeleton")
    joint_ids: set[int] = set()
    for node in skinned_nodes:
        try:
            joints = skins[node["skin"]]["joints"]
            mesh = meshes[node["mesh"]]
        except (IndexError, KeyError, TypeError) as exc:
            raise RigContractError("invalid glTF skin-node references") from exc
        if len(joints) < 16 or any(type(j) is not int or j < 0 or j >= len(nodes) for j in joints):
            raise RigContractError("skeletal hierarchy lacks essential body joints")
        joint_ids.update(joints)
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if not all(name in attributes for name in ("POSITION", "JOINTS_0", "WEIGHTS_0")):
                raise RigContractError("skinned actor geometry lacks vertex skin weights")
            try:
                position, bone_ids, weights = (
                    accessors[attributes[name]] for name in ("POSITION", "JOINTS_0", "WEIGHTS_0")
                )
            except (IndexError, KeyError, TypeError) as exc:
                raise RigContractError("invalid vertex skin weight accessor references") from exc
            if (
                position.get("count", 0) < 3
                or bone_ids.get("count") != position["count"]
                or weights.get("count") != position["count"]
                or bone_ids.get("type") != "VEC4"
                or weights.get("type") != "VEC4"
            ):
                raise RigContractError("skin weights must map to all geometry vertices")
    visemes: set[str] = set()
    for mesh in meshes:
        names = mesh.get("extras", {}).get("targetNames", [])
        for primitive in mesh.get("primitives", []):
            targets = primitive.get("targets", [])
            if targets and len(targets) == len(names):
                visemes.update(
                    str(name).lower()
                    for name in names
                    if str(name).lower().startswith(("viseme_", "viseme."))
                )
    if len(visemes) < 4:
        raise RigContractError("facial rig lacks four or more named viseme morph targets")
    return {
        "schema": "clipper-approved-master-rig-v1",
        "glb_sha256": expected_sha256,
        "skinned_nodes": len(skinned_nodes),
        "unique_joint_count": len(joint_ids),
        "viseme_morph_targets": sorted(visemes),
        "structure_gate": "PASS",
        "evaluated_animation": "NOT_VERIFIED",
        "human_identity_review": "PENDING",
        "publish_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_rig(args.glb, args.sha256), indent=2))


if __name__ == "__main__":
    main()
