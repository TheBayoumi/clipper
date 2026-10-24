"""Official Wan2.2-S2V-14B on Modal A100; only explicitly dispatched GPU calls.

Wan weights stay on an independent persistent volume. A passing model render
is a candidate, not proof of the recurring character's creative consistency.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = "clipper-wan-s2v"
UPSTREAM_COMMIT = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
MODEL_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
MODEL_SIZE = "1024*704"  # Frame area; portrait aspect comes from the input image.
CUDA_IMAGE = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
MAX_MP4_BYTES = 180_000_000

weights = modal.Volume.from_name("clipper-wan2-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("clipper-wan-hf")


def image_spec() -> modal.Image:
    """Use the official pinned Wan code and CUDA extensions, not a web demo."""
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.10")
        .entrypoint([])
        .apt_install(
            "git",
            "git-lfs",
            "build-essential",
            "ninja-build",
            "ffmpeg",
            "libgl1",
            "libglib2.0-0",
            "libsndfile1",
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "MAX_JOBS": "8",
                "HF_HOME": "/model-cache/huggingface",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .run_commands(
            "git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2"
            f" && git -C /opt/Wan2.2 checkout --detach {UPSTREAM_COMMIT}",
            "python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"
            " --index-url https://download.pytorch.org/whl/cu124",
            "python -m pip install 'numpy>=1.26,<2' 'transformers==4.51.3'"
            " 'diffusers>=0.31,<0.35' 'accelerate>=1.1,<2'"
            " 'huggingface-hub>=0.30,<1' 'tokenizers>=0.20.3'"
            " 'opencv-python-headless>=4.9,<5' easydict ftfy tqdm"
            " 'imageio[ffmpeg]' imageio-ffmpeg soundfile librosa"
            " einops Pillow sentencepiece dashscope ninja wheel setuptools decord safetensors"
            " 'peft==0.15.2' 'tomli>=2,<3'",
            "python -m pip install flash-attn==2.7.3 --no-build-isolation",
            # Check CUDA package presence, not driver-dependent import in image build.
            "python -c 'import importlib.util;"
            ' assert importlib.util.find_spec("flash_attn") is not None;'
            ' print("WAN_S2V_DEPENDENCIES_READY")\'',
        )
    )


app = modal.App(os.environ.get("CLIPPER_MODAL_APP_NAME", APP_NAME))


def check_request(reference: bytes, wav: bytes, source_sha: str, audio_sha: str) -> None:
    if not 0 < len(reference) <= 20_000_000:
        raise ValueError("Reference image must be an approved, detailed original")
    if not 40_000 <= len(wav) <= 25_000_000:
        raise ValueError("WAV source missing or invalid")
    if hashlib.sha256(reference).hexdigest() != source_sha:
        raise ValueError("Reference source hash mismatch")
    if hashlib.sha256(wav).hexdigest() != audio_sha:
        raise ValueError("Narration audio hash mismatch")


@app.function(
    image=image_spec(),
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=7200,
    volumes={"/model-cache": weights},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=60,
)
def generate_s2v(
    reference: bytes,
    wav: bytes,
    source_sha: str,
    audio_sha: str,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    check_request(reference, wav, source_sha, audio_sha)
    import io

    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("Modal A100 worker has no CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory < 79 * 1024**3:
        raise RuntimeError("Wan S2V requires an 80 GB-class GPU")
    with Image.open(io.BytesIO(reference)) as source:
        source.verify()
    with Image.open(io.BytesIO(reference)) as source:
        if source.width < 384 or source.height < 512:
            raise ValueError("Reference resolution below official S2V technical input floor")
    # Upstream Wan calls torch.cuda.current_device() during package import.
    # Driverless Modal image builders must never import the Wan package.
    sys.path.insert(0, "/opt/Wan2.2")
    import wan

    if not hasattr(wan, "WanS2V"):
        raise RuntimeError("Official Wan S2V runtime import is unavailable")
    import runpy

    # Resolve the entire upstream CLI import graph before model inference.
    runpy.run_path("/opt/Wan2.2/generate.py", run_name="clipper_wan_gpu_import_probe")
    checkpoint = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir="/model-cache/wan-s2v-14b",
        token=os.environ.get("HF_TOKEN"),
        max_workers=4,
    )
    weights.commit()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        reference_file = root / "approved-reference.png"
        audio_file = root / "narration.wav"
        video_file = root / "wan-s2v-candidate.mp4"
        Image.open(io.BytesIO(reference)).convert("RGB").save(reference_file)
        audio_file.write_bytes(wav)
        command = [
            sys.executable,
            "/opt/Wan2.2/generate.py",
            "--task",
            "s2v-14B",
            "--size",
            MODEL_SIZE,
            "--ckpt_dir",
            checkpoint,
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--prompt",
            prompt,
            "--image",
            str(reference_file),
            "--audio",
            str(audio_file),
            "--base_seed",
            str(seed),
            "--save_file",
            str(video_file),
        ]
        subprocess.run(command, cwd="/opt/Wan2.2", check=True, timeout=4800)
        if not video_file.is_file() or not 25_000 < video_file.stat().st_size <= MAX_MP4_BYTES:
            raise RuntimeError("Wan S2V produced no bounded video")
        result = video_file.read_bytes()
    return {
        "video": result,
        "gpu": gpu.name,
        "upstream_commit": UPSTREAM_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "image_sha256": source_sha,
        "audio_sha256": audio_sha,
        "video_sha256": hashlib.sha256(result).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seed": seed,
        "creative_review": "PENDING",
        "publish_eligible": False,
    }


@app.local_entrypoint()
def main(
    reference: str,
    audio: str,
    output: str = "artifacts/wan-s2v",
    prompt: str = (
        "Cinematic stylized dark-haired animated presenter wearing black rectangular "
        "glasses and a charcoal hoodie speaking naturally with expressive features."
    ),
    seed: int = 724,
) -> None:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "schema": "clipper-wan-s2v-modal-v1",
        "model_id": MODEL_ID,
        "creative_review": "PENDING",
        "publish_eligible": False,
        "status": "INCOMPLETE",
    }
    try:
        reference_path, audio_path = Path(reference), Path(audio)
        source, wav = reference_path.read_bytes(), audio_path.read_bytes()
        source_sha, audio_sha = hashlib.sha256(source).hexdigest(), hashlib.sha256(wav).hexdigest()
        check_request(source, wav, source_sha, audio_sha)
        response = generate_s2v.remote(source, wav, source_sha, audio_sha, prompt, seed)
        video = response.pop("video")
        candidate = target / "wan-s2v-candidate.mp4"
        candidate.write_bytes(video)
        if hashlib.sha256(video).hexdigest() != response["video_sha256"]:
            raise RuntimeError("video hash from Modal does not match artifact")
        evidence.update(response, status="CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED")
    except Exception as exc:
        evidence.update(
            status="GENERATION_FAILED",
            error_type=type(exc).__name__,
            error_reason=str(exc)[:1200],
        )
        raise
    finally:
        (target / "model-evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        print(json.dumps(evidence, indent=2), file=sys.stderr)
