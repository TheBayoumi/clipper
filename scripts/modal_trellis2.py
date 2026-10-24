"""Official Microsoft TRELLIS.2 on Modal A100, hash-bound to Clipper source artwork.

No public Gradio, TRELLIS v1 fallback, or stock character. An unrigged candidate.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "clipper-trellis2-approved-host"
MODEL_ID = "microsoft/TRELLIS.2-4B"
UPSTREAM_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SOURCE_SHA256 = "c3bbc6395e47c04646ad3f0ad44f13873511e4ee3189046eafbc682225e746d2"
MAX_OUTPUT_BYTES = 90 * 1024 * 1024
CUDA_IMAGE = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"

weights = modal.Volume.from_name("clipper-trellis2-weights", create_if_missing=True)


def image_spec() -> modal.Image:
    """Install upstream and its real CUDA extensions in a reproducible image."""
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.10")
        .entrypoint([])
        .apt_install(
            "git",
            "git-lfs",
            "build-essential",
            "clang",
            "cmake",
            "ninja-build",
            "libjpeg-dev",
            "libgl1",
            "libglib2.0-0",
            "libopenexr-dev",
            "ffmpeg",
            "pkg-config",
            "curl",
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
                "MAX_JOBS": "8",
                "HF_HOME": "/model-cache",
                "HF_HUB_CACHE": "/model-cache/hub",
                "OPENCV_IO_ENABLE_OPENEXR": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "ATTN_BACKEND": "flash-attn",
            }
        )
        .run_commands(
            f"git clone --recursive https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2"
            f" && git -C /opt/TRELLIS.2 checkout --detach {UPSTREAM_COMMIT}"
            " && git -C /opt/TRELLIS.2 submodule update --init --recursive",
            "python -m pip install torch==2.6.0 torchvision==0.21.0"
            " --index-url https://download.pytorch.org/whl/cu124",
            "python -m pip install 'numpy<2.5' imageio imageio-ffmpeg tqdm"
            " easydict opencv-python-headless ninja trimesh"
            " transformers==4.57.3 tensorboard pandas lpips zstandard"
            " kornia timm pillow-simd huggingface-hub",
            "python -m pip install"
            " 'git+https://github.com/EasternJournalist/utils3d.git"
            "@9a4eb15e4021b67b12c460c7057d642626897ec8'",
            "python -m pip install flash-attn==2.7.3 --no-build-isolation",
            "git clone --depth 1 --branch v0.4.0"
            " https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast"
            " && python -m pip install /tmp/nvdiffrast --no-build-isolation",
            "git clone --depth 1 --branch renderutils"
            " https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec"
            " && python -m pip install /tmp/nvdiffrec --no-build-isolation",
            "git clone --recursive"
            " https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh"
            " && python -m pip install /tmp/CuMesh --no-build-isolation",
            "git clone --recursive"
            " https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM"
            " && python -m pip install /tmp/FlexGEMM --no-build-isolation",
            "python -m pip install /opt/TRELLIS.2/o-voxel --no-build-isolation",
            "cd /opt/TRELLIS.2 && python -c"
            " 'import torch, o_voxel, flash_attn, nvdiffrast, trellis2; print(torch.__version__)'",
        )
    )


app = modal.App(APP_NAME)


def verified_request(source: bytes, source_sha: str, resolution: str, seed: int) -> None:
    """Reject input tampering and accidental expensive 1536-resolution runs."""
    if not isinstance(source, bytes) or not 25_000 <= len(source) <= 12_000_000:
        raise ValueError("Source is missing or outside the bounded reference size")
    if source_sha != SOURCE_SHA256 or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Modal input does not match the approved character SHA256")
    if resolution not in ("512", "1024_cascade"):
        raise ValueError("Unsupported resolution; require reviewed first candidate")
    if not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("Seed outside accepted deterministic range")


@app.function(
    image=image_spec(),
    gpu="A100-80GB",
    timeout=1800,
    memory=32768,
    cpu=8,
    volumes={"/model-cache": weights},
    max_containers=1,
    scaledown_window=60,
)
def generate_approved_host(
    source: bytes, source_sha: str, resolution: str, seed: int
) -> dict[str, Any]:
    verified_request(source, source_sha, resolution, seed)
    import o_voxel
    import torch
    from PIL import Image

    sys.path.insert(0, "/opt/TRELLIS.2")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 worker has no assigned CUDA GPU")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 24 * 1024**3:
        raise RuntimeError("TRELLIS.2 requires at least 24 GB GPU VRAM")
    os.chdir("/opt/TRELLIS.2")
    model = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
    model.cuda()
    with Image.open(io.BytesIO(source)) as loaded:
        image = loaded.convert("RGBA")
    with torch.no_grad():
        mesh = model.run(image, seed=seed, pipeline_type=resolution)[0]
        mesh.simplify(16777216)
        glb_scene = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=300000,
            texture_size=2048,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as stream:
        output = Path(stream.name)
    glb_scene.export(str(output), extension_webp=True)
    if not output.is_file() or not 25_000 < output.stat().st_size < MAX_OUTPUT_BYTES:
        raise RuntimeError("TRELLIS.2 did not export a bounded valid GLB")
    payload = output.read_bytes()
    return {
        "glb": payload,
        "evidence": {
            "schema": "clipper-trellis2-modal-v1",
            "provider": "Modal self-hosted official TRELLIS.2",
            "model_id": MODEL_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "gpu": device.name,
            "gpu_total_memory_gib": round(device.total_memory / 1024**3, 2),
            "source_sha256": source_sha,
            "resolution": resolution,
            "seed": seed,
            "glb_sha256": hashlib.sha256(payload).hexdigest(),
            "glb_bytes": len(payload),
            "creative_status": "NOT_REVIEWED",
            "rigged": False,
            "publish_eligible": False,
            "status": "CANDIDATE_GLB_GENERATED_NOT_RIGGED",
        },
    }


@app.local_entrypoint()
def main(
    source: str,
    output: str = "artifacts/host-3d-modal",
    resolution: str = "1024_cascade",
    seed: int = 724,
) -> None:
    """Invoked from GitHub as modal run scripts/modal_trellis2.py --source ..."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    evidence_file = target / "evidence.json"
    evidence: dict[str, Any] = {
        "schema": "clipper-trellis2-modal-v1",
        "provider": "Modal self-hosted official TRELLIS.2",
        "source_sha256_expected": SOURCE_SHA256,
        "status": "INCOMPLETE",
        "publish_eligible": False,
    }
    try:
        payload = Path(source).read_bytes()
        verified_request(payload, hashlib.sha256(payload).hexdigest(), resolution, seed)
        outcome = generate_approved_host.remote(payload, SOURCE_SHA256, resolution, seed)
        glb_bytes = outcome.pop("glb")
        if not isinstance(glb_bytes, bytes):
            raise RuntimeError("Modal failed to return real GLB bytes")
        candidate = target / "page-to-life-host-candidate.glb"
        candidate.write_bytes(glb_bytes)
        evidence.update(outcome["evidence"])
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != evidence["glb_sha256"]:
            raise RuntimeError("GPU output hash does not match saved artifact")
    except Exception as exc:
        evidence.update(
            status="REMOTE_INFERENCE_FAILED",
            error_type=type(exc).__name__,
            error_reason=str(exc)[:1300],
        )
        raise
    finally:
        evidence_file.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2), file=sys.stderr)
