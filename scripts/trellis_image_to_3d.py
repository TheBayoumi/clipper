"""Generate an unrigged candidate GLB from the approved Clipper character image.

TRELLIS.2 runs on the public Hugging Face ZeroGPU Space. Outputs remain
candidate assets: they are NOT approved character rigs or videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

SPACE = "microsoft/TRELLIS.2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_glb(path: Path) -> dict:
    if path.stat().st_size < 25000:
        raise ValueError("GLB missing or implausibly small")

    import trimesh

    scene = trimesh.load(str(path), file_type="glb", force="scene")
    meshes = list(scene.geometry.values())
    if not meshes or sum(len(m.faces) for m in meshes) < 1000:
        raise ValueError("GLB lacks 3D mesh geometry")
    if scene.bounds is None:
        raise ValueError("GLB has no 3D bounds")
    spans = scene.bounds[1] - scene.bounds[0]
    if any(float(x) <= 0.0001 for x in spans):
        raise ValueError("GLB is flat or has degenerate bounds")
    materials = sum(getattr(mesh.visual, "material", None) is not None for mesh in meshes)
    if not materials:
        raise ValueError("GLB lacks material evidence")
    return {
        "vertices": sum(len(m.vertices) for m in meshes),
        "triangles": sum(len(m.faces) for m in meshes),
        "material_meshes": int(materials),
        "bounds": scene.bounds.tolist(),
        "generated_asset_is_rigged": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--space", default=SPACE)
    parser.add_argument("--output", type=Path, default=Path("artifacts/host-3d"))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--resolution", choices=("512", "1024"), default="512")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema": "clipper-trellis2-v1",
        "provider": args.space,
        "seed": args.seed,
        "resolution": args.resolution,
        "publish_eligible": False,
        "creative_status": "NOT_REVIEWED",
        "rigging_status": "NOT_IMPLEMENTED",
        "status": "NOT_STARTED",
    }
    try:
        from gradio_client import Client, handle_file

        client = Client(args.space, token=os.getenv("HF_TOKEN") or None)
        schema = str(client.view_api(return_format="dict"))
        for endpoint in ("/preprocess_image", "/image_to_3d", "/extract_glb"):
            if endpoint not in schema:
                raise ValueError(f"Public Space lacks endpoint {endpoint}")
        evidence["api_health"] = "PASS"
        evidence["authenticated_hf_token_supplied"] = bool(os.getenv("HF_TOKEN"))
        if args.probe:
            evidence["status"] = "API_PROBE_ONLY"
            return
        if not args.reference or not args.reference.is_file():
            raise ValueError("Missing character reference image, no stock fallback")
        from PIL import Image

        with Image.open(args.reference) as image:
            if image.width < 512 or image.height < 700:
                raise ValueError("Low resolution input violates character gate")
        evidence["source_sha256"] = sha256(args.reference)
        prepared = client.predict(handle_file(str(args.reference)), api_name="/preprocess_image")
        generated = client.predict(
            prepared,
            args.seed,
            args.resolution,
            7.5,
            0.7,
            12,
            5.0,
            7.5,
            0.5,
            12,
            3.0,
            1.0,
            0.0,
            12,
            3.0,
            api_name="/image_to_3d",
        )
        if not isinstance(generated, (tuple, list)) or not generated:
            raise ValueError("TRELLIS.2 returned no visible 3D preview")
        # TRELLIS stores its latent state in the same Gradio client session.
        # /extract_glb has exactly two PUBLIC inputs: decimation, texture size.
        # Passing generated[0] as a third input is invalid; it is a preview,
        # NOT a serializable model state.
        exported = client.predict(100000, 2048, api_name="/extract_glb")
        source = exported[0] if isinstance(exported, (tuple, list)) else exported
        candidate = Path(source if isinstance(source, str) else source["path"])
        if not candidate.is_file():
            raise ValueError("No GLB downloaded from public Space")
        target = args.output / "page-to-life-host-candidate.glb"
        shutil.copyfile(candidate, target)
        evidence["mesh"] = validate_glb(target)
        evidence["glb_sha256"] = sha256(target)
        evidence["status"] = "CANDIDATE_GLB_GENERATED_NOT_RIGGED"
    except Exception as error:
        evidence.update(
            status="REMOTE_INFERENCE_BLOCKED", error_type=type(error).__name__, error_reason=str(error)[:1400]
        )
        raise
    finally:
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
