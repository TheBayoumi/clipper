"""Reference-conditioned cinematic keyframes generated inside Clipper on Modal.

Official Apache-2.0 Qwen-Image-Edit-2511; no proprietary image generation API.
Only explicitly invoked inference allocates an A100. Every frame derives from
the SAME hash-locked source identity and approved storyboard action.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
MODEL_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
SCHEMA = "clipper-cinematic-keyframe-v1"
SHOTS = ("hook", "book_intro", "lesson_example")
CUDA_IMAGE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
WEIGHTS_DIR = "/model-cache/qwen-image-edit-2511"
CACHE_DIR = Path("/model-cache/clipper-keyframes-v1")
MAX_SOURCE_BYTES = 20_000_000
MAX_OUTPUT_BYTES = 20_000_000

volume = modal.Volume.from_name("clipper-qwen-keyframes", create_if_missing=True)
hf_secret = modal.Secret.from_name("clipper-wan-hf")


def image_spec() -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
        .apt_install("libgl1", "libglib2.0-0", "git")
        .run_commands(
            "python -m pip install torch==2.6.0 torchvision==0.21.0 "
            "--index-url https://download.pytorch.org/whl/cu124",
            "python -m pip install 'diffusers==0.36.0' 'transformers==4.57.3' "
            "'accelerate==1.12.0' 'huggingface-hub==0.36.2' "
            "'Pillow>=11,<13' 'sentencepiece>=0.2' 'protobuf>=5' safetensors",
            # Driverless image-build smoke only; actual model inference on GPU.
            "python -c 'import importlib.util; "
            'assert importlib.util.find_spec("diffusers"); '
            'assert importlib.util.find_spec("transformers"); '
            'print("QWEN_IMAGE_EDIT_DEPENDENCIES_READY")\'',
        )
    )


app = modal.App("clipper-qwen-cinematic-keyframes")


def validate_request(reference: bytes, req: dict[str, Any]) -> None:
    if not isinstance(req, dict) or req.get("schema") != SCHEMA:
        raise ValueError("invalid versioned Qwen keyframe request")
    if req.get("model_id") != MODEL_ID or req.get("model_revision") != MODEL_REVISION:
        raise ValueError("Qwen model id or revision changed")
    if req.get("shot_id") not in SHOTS:
        raise ValueError("unapproved shot")
    if req.get("stage") != "KEYFRAME_REQUEST_NOT_RENDERED":
        raise ValueError("unexpected shot lifecycle state")
    if req.get("publish_eligible") is not False:
        raise ValueError("candidate must not be preapproved")
    if not 0 < len(reference) <= MAX_SOURCE_BYTES:
        raise ValueError("source reference absent or too large")
    sha = hashlib.sha256(reference).hexdigest()
    if sha != req.get("source_image_sha256"):
        raise ValueError("source bytes changed")
    if not isinstance(req.get("fingerprint"), str) or len(req["fingerprint"]) != 64:
        raise ValueError("missing stable request cache identity")
    if len(req.get("prompt", "")) < 350:
        raise ValueError("cinematic keyframe prompt unexpectedly missing")
    if (req.get("render_width"), req.get("render_height"), req.get("steps")) != (832, 1472, 40):
        raise ValueError("unexpected canonical cinematic frame generation contract")


@app.function(
    image=image_spec(),
    gpu="A100-80GB",
    cpu=12,
    memory=131072,
    timeout=7200,
    volumes={"/model-cache": volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=60,
)
def generate_keyframe(reference: bytes, request: dict[str, Any]) -> dict[str, Any]:
    validate_request(reference, request)
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from huggingface_hub import snapshot_download
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen frame generation requires an active Modal A100 GPU")
    volume.reload()
    cache_id = request["fingerprint"]
    cached_png = CACHE_DIR / f"{cache_id}.png"
    if cached_png.is_file():
        data = cached_png.read_bytes()
        if not 100_000 <= len(data) <= MAX_OUTPUT_BYTES:
            raise RuntimeError("cached keyframe size invalid")
        return {
            "png": data,
            "image_sha256": hashlib.sha256(data).hexdigest(),
            "cache": "REUSED",
            "gpu": torch.cuda.get_device_name(0),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    checkpoint = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=WEIGHTS_DIR,
        token=os.environ.get("HF_TOKEN"),
        max_workers=4,
    )
    volume.commit()
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
    with Image.open(io.BytesIO(reference)) as source:
        original = source.convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(request["seed"])
        with torch.inference_mode():
            created = pipeline(
                image=[original],
                prompt=request["prompt"],
                generator=generator,
                true_cfg_scale=4.0,
                negative_prompt="low quality, gray background, identity drift, wrong clothes",
                num_inference_steps=request["steps"],
                guidance_scale=1.0,
                num_images_per_prompt=1,
                width=request["render_width"],
                height=request["render_height"],
            ).images[0]
    if created.width < 720 or created.height < 1250:
        raise RuntimeError("generated shot frame too low-resolution")
    if abs(created.width / created.height - 9 / 16) > 0.035:
        raise RuntimeError("generated frame does not respect vertical 9:16 target")
    out = io.BytesIO()
    created.convert("RGB").save(out, format="PNG", optimize=True)
    data = out.getvalue()
    if not 100_000 <= len(data) <= MAX_OUTPUT_BYTES:
        raise RuntimeError("generated PNG missing meaningful pixel detail")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_png.write_bytes(data)
    volume.commit()
    return {
        "png": data,
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "cache": "GENERATED",
        "gpu": torch.cuda.get_device_name(0),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@app.local_entrypoint()
def main(
    reference: str,
    storyboard: str = "niches/atomic-habits-approved-storyboard.json",
    shot_ids: str = "hook",
    output: str = "artifacts/wan-s2v/output/keyframes",
) -> None:
    # Imported on local GitHub CPU, not at Modal image build time.
    from wan_keyframe_contract import prepare, verified_local_keyframe

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference)
    source = reference_path.read_bytes()
    requested = [item.strip() for item in shot_ids.split(",")]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("shot list is empty or repeated")
    if any(item not in SHOTS for item in requested):
        raise ValueError("unknown shot requested")
    for shot_id in requested:
        req = prepare(reference_path, Path(storyboard), shot_id)
        frame = output_path / f"{shot_id}.png"
        frame_evidence = output_path / f"{shot_id}.json"
        if verified_local_keyframe(frame, frame_evidence, req):
            print(json.dumps({
                "shot": shot_id,
                "status": "FRAME_GENERATED_TECHNICAL_CANDIDATE",
                "cache": "REUSED_LOCAL_VERIFIED",
            }))
            continue
        evidence: dict[str, Any] = {
            **req,
            "status": "INCOMPLETE",
        }
        try:
            response = generate_keyframe.remote(source, req)
            png = response.pop("png")
            frame = output_path / f"{shot_id}.png"
            frame.write_bytes(png)
            if hashlib.sha256(png).hexdigest() != response["image_sha256"]:
                raise RuntimeError("Modal returned image SHA mismatch")
            evidence.update(
                response,
                status="FRAME_GENERATED_TECHNICAL_CANDIDATE",
                stage="KEYFRAME_AWAITING_VISUAL_REVIEW",
                creative_review="PENDING",
                publish_eligible=False,
                output_path=str(frame),
            )
        except Exception as exc:
            evidence.update(
                status="FRAME_GENERATION_FAILED",
                error_type=type(exc).__name__,
                error_reason=str(exc)[:1000],
            )
            raise
        finally:
            (output_path / f"{shot_id}.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "shot": shot_id,
                        "status": evidence["status"],
                        "model": MODEL_ID,
                        "cache": evidence.get("cache"),
                    }
                )
            )
