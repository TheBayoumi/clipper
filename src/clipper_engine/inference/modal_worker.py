"""Isolated Clipper-owned Modal visual compute; never schedules or renders clips.

Deploy only from an explicit immutable SHA to a non-production Modal app::

    CLIPPER_DEPLOYED_GIT_SHA=<sha> modal deploy src/clipper_engine/inference/modal_worker.py

The app name and GPU configuration are intentionally isolated from the existing
ContentRewards/TJR deployments. Training weights are pinned to the qualified donor.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

import modal

from clipper_engine.inference.vision import (
    VISION_MODEL_ID,
    VISION_PROMPT_VERSION,
    VISION_REVISION,
    VISION_SCHEMA_VERSION,
)

DEPLOYED_GIT_SHA = os.getenv("CLIPPER_DEPLOYED_GIT_SHA", "").strip().lower()
APP_NAME = os.getenv("CLIPPER_NEURAL_MODAL_APP", "clipper-neural-qualification")
if not APP_NAME.startswith("clipper-") or APP_NAME in {
    "clipper-open-editor",
    "clipper-production-pipeline",
}:
    raise ValueError("refusing shared or unscoped Modal neural app name")
APP_GPU = "L4:2"
# Donor ContentRewards Qwen3-VL deployment also uses 2 x L4 for this model.
# These USD figures are cost estimates for GPU only, not a Modal invoice.
GPU_ESTIMATED_USD_PER_SECOND = 2 * 0.000222
app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("clipper-neural-model-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.57.3",
        "accelerate>=1.14,<2",
        "pillow>=11,<13",
        "sentencepiece>=0.2,<1",
    )
    .env({"HF_HOME": "/models", "HF_HUB_CACHE": "/models"})
    .add_local_python_source("clipper_engine")
)


def _validate_request(payload: dict[str, Any]) -> None:
    from clipper_engine.inference.vision import SHA_PATTERN

    if not SHA_PATTERN.fullmatch(DEPLOYED_GIT_SHA):
        raise RuntimeError("neural Modal worker missing immutable deployment SHA")
    if payload.get("expected_git_sha") != DEPLOYED_GIT_SHA:
        raise RuntimeError("neural worker deployment SHA mismatch")
    if payload.get("task") != "source_gameplay_visual_scout":
        raise ValueError("unsupported neural visual task")
    if payload.get("model") != {
        "model_id": VISION_MODEL_ID,
        "revision": VISION_REVISION,
        "prompt_version": VISION_PROMPT_VERSION,
        "schema_version": VISION_SCHEMA_VERSION,
    }:
        raise ValueError("neural vision model contract mismatch")
    identity = payload.get("source_identity")
    if not isinstance(identity, dict) or any(
        not identity.get(k)
        for k in ("source_key", "analysis_sha256", "provider_asset_id", "provider_review_url")
    ):
        raise ValueError("neural worker missing source identity")
    frames = payload.get("frames_base64")
    stamps = payload.get("frame_timestamps")
    if (
        not isinstance(frames, list)
        or not isinstance(stamps, list)
        or not 1 <= len(frames) == len(stamps) <= 16
    ):
        raise ValueError("neural worker requires 1-16 source-stamped images")


def _strict_observations(raw: dict[str, Any], timestamps: list[float]) -> list[dict[str, Any]]:
    observations = raw.get("observations")
    if not isinstance(observations, list) or len(observations) != len(timestamps):
        raise ValueError("neural VLM omitted required source-stamped observations")
    for timestamp, item in zip(timestamps, observations, strict=True):
        if not isinstance(item, dict) or item.get("timestamp") != timestamp:
            raise ValueError("neural VLM produced invalid timestamp or frame ordering")
    return observations


@app.cls(
    image=image,
    gpu=APP_GPU,
    memory=32768,
    volumes={"/models": model_cache},
    timeout=240,
    max_containers=1,
)
class VisionModel:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, revision=VISION_REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(
            VISION_MODEL_ID,
            revision=VISION_REVISION,
            dtype=torch.bfloat16,
            device_map="balanced",
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    @modal.method()
    def ready(self) -> dict[str, Any]:
        import torch

        return {
            "deployed_git_sha": DEPLOYED_GIT_SHA,
            "model_id": VISION_MODEL_ID,
            "revision": VISION_REVISION,
            "gpu_count": torch.cuda.device_count(),
            "cuda_available": torch.cuda.is_available(),
        }

    @modal.method()
    def inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        import torch
        from PIL import Image

        _validate_request(payload)
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("neural inference CUDA GPU configuration unavailable")
        started = time.perf_counter()
        frames = [
            Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB")
            for item in payload["frames_base64"]
        ]
        timestamps = payload["frame_timestamps"]
        instruction = (
            "You are observing source-indexed gameplay or cinematic footage. "
            "For each supplied image, report directly visible source-grounded "
            "actions, confrontations, reveals, traversal, and possible outcomes. "
            "Do not claim confirmed deaths, kills, motives, dialogue or actor "
            "identities without directly visible evidence. Do not infer events "
            "between sampled frames. Return exactly one JSON object with "
            "observations, an array of exactly one item per frame in the given "
            "order; each item has timestamp, summary, event_labels (string array), "
            "confidence (number 0 to 1). Empty labels are valid. Timestamps must "
            "exactly match supplied values. No markdown or extra text. "
            "Frame timestamps: " + json.dumps(timestamps)
        )
        content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_tokens = int(inputs["input_ids"].numel())
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=min(2048, 192 * len(frames) + 256),
                do_sample=False,
            )
        result_tokens = output[0][inputs["input_ids"].shape[-1] :]
        decoded = self.processor.decode(result_tokens, skip_special_tokens=True)
        observations = _strict_observations(json.loads(decoded), timestamps)
        duration = time.perf_counter() - started
        return {
            "deployed_git_sha": DEPLOYED_GIT_SHA,
            "source_identity": payload["source_identity"],
            "model": payload["model"],
            "observations": observations,
            "usage": {
                "started_at": datetime.now(UTC).isoformat(),
                "gpu_type": APP_GPU,
                "gpu_seconds": duration * 2,
                "estimated_cost_usd": duration * GPU_ESTIMATED_USD_PER_SECOND,
                "input_units": input_tokens,
                "output_units": int(result_tokens.numel()),
                "peak_vram_mb": max(
                    torch.cuda.max_memory_allocated(i) / (1024 * 1024)
                    for i in range(torch.cuda.device_count())
                ),
            },
        }
