"""Isolated Clipper-owned Modal visual compute; never schedules or renders clips.

Deploy only from an explicit immutable SHA to a non-production Modal app::

    CLIPPER_DEPLOYED_GIT_SHA=<sha> modal deploy src/clipper_engine/inference/modal_worker.py

The app name and GPU configuration are intentionally isolated from the existing
ContentRewards/TJR deployments. Training weights are pinned to the qualified donor.
"""

from __future__ import annotations

import base64
import hashlib
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
    source_visual_prompt,
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
    .env(
        {
            "HF_HOME": "/models",
            "HF_HUB_CACHE": "/models",
            "CLIPPER_DEPLOYED_GIT_SHA": DEPLOYED_GIT_SHA,
            "CLIPPER_NEURAL_MODAL_APP": APP_NAME,
        }
    )
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
    timeout=600,
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
        total_input_units = 0
        total_output_units = 0
        observations: list[dict[str, Any]] | None = None
        for attempt in range(2):
            instruction = source_visual_prompt(timestamps, recovery_attempt=attempt)
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
            total_input_units += int(inputs["input_ids"].numel())
            inputs = inputs.to(self.model.device)
            generation_budget = min(2048, max(384, 170 * len(frames) + 96) * (attempt + 1))
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=generation_budget, do_sample=False
                )
            result_tokens = output[0][inputs["input_ids"].shape[-1] :]
            total_output_units += int(result_tokens.numel())
            decoded = self.processor.decode(result_tokens, skip_special_tokens=True)
            keys: list[str] = []
            try:
                value = json.loads(decoded)
                if isinstance(value, dict):
                    keys = list(value)[:15]
                else:
                    raise ValueError("neural visual output must be a JSON object")
                observations = _strict_observations(value, timestamps)
                print(
                    json.dumps(
                        {
                            "event": "neural_vision_schema_validation",
                            "valid": True,
                            "attempt": attempt + 1,
                            "frames": len(frames),
                            "output_keys": keys,
                            "output_tokens": int(result_tokens.numel()),
                        }
                    ),
                    flush=True,
                )
                break
            except (json.JSONDecodeError, ValueError) as exc:
                print(
                    json.dumps(
                        {
                            "event": "neural_vision_schema_validation",
                            "valid": False,
                            "attempt": attempt + 1,
                            "frames": len(frames),
                            "output_keys": keys,
                            "error_type": type(exc).__name__,
                            "output_sha256": hashlib.sha256(decoded.encode()).hexdigest()[:16],
                            "output_tokens": int(result_tokens.numel()),
                            "generation_budget": generation_budget,
                        }
                    ),
                    flush=True,
                )
                if attempt == 1:
                    raise ValueError(
                        "neural model did not return a complete source-stamped "
                        "observations array after bounded format recovery"
                    ) from exc
            finally:
                del inputs, output, result_tokens
                torch.cuda.empty_cache()
        if observations is None:
            raise RuntimeError("neural vision schema validation produced no observations")
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
                "input_units": total_input_units,
                "output_units": total_output_units,
                "peak_vram_mb": max(
                    torch.cuda.max_memory_allocated(i) / (1024 * 1024)
                    for i in range(torch.cuda.device_count())
                ),
            },
        }


@app.local_entrypoint()
def smoke(
    frame_path: str,
    source_identity_json: str,
    timestamp: float,
    output_path: str,
    max_estimated_usd: float = 1.0,
) -> None:
    """Validation-only live GPU smoke on one pre-certified source-clock frame."""
    from pathlib import Path

    from clipper_engine.inference.vision import _parse_batch

    if not 0 < max_estimated_usd <= 1.0:
        raise ValueError("smoke hard estimated cost ceiling is one USD")
    frame = Path(frame_path)
    identity = json.loads(Path(source_identity_json).read_text(encoding="utf-8"))
    at = round(float(timestamp), 3)
    source_frame_sha256 = __import__("hashlib").sha256(frame.read_bytes()).hexdigest()
    payload = {
        "task": "source_gameplay_visual_scout",
        "frame_timestamps": [at],
        "source_identity": identity,
        "expected_git_sha": DEPLOYED_GIT_SHA,
        "model": {
            "model_id": VISION_MODEL_ID,
            "revision": VISION_REVISION,
            "prompt_version": VISION_PROMPT_VERSION,
            "schema_version": VISION_SCHEMA_VERSION,
        },
        "frames_base64": [base64.b64encode(frame.read_bytes()).decode("ascii")],
    }
    _validate_request(payload)
    started = time.perf_counter()
    worker = VisionModel()
    ready = worker.ready.remote()
    if (
        ready.get("deployed_git_sha") != DEPLOYED_GIT_SHA
        or ready.get("model_id") != VISION_MODEL_ID
        or ready.get("revision") != VISION_REVISION
        or ready.get("cuda_available") is not True
        or ready.get("gpu_count", 0) < 2
    ):
        raise RuntimeError("neural GPU model readiness/identity qualification failed")
    result = worker.inspect.remote(payload)
    observations, gpu_seconds, gpu_only_cost = _parse_batch(
        result, identity=identity, requested=(at,), deployed_sha=DEPLOYED_GIT_SHA
    )
    wall_elapsed = time.perf_counter() - started
    # Conservatively include startup/ready wall time as if both GPUs were allocated.
    estimated_total_cost = wall_elapsed * GPU_ESTIMATED_USD_PER_SECOND
    if max(gpu_only_cost, estimated_total_cost) > max_estimated_usd:
        raise RuntimeError("neural live qualification exceeded estimated cost budget")
    evidence = {
        "status": "GPU_SMOKE_PASS_PROPOSALS_ONLY",
        "deployed_git_sha": DEPLOYED_GIT_SHA,
        "model": payload["model"],
        "source_identity": identity,
        "frame_timestamp": at,
        "source_frame_sha256": source_frame_sha256,
        "readiness": ready,
        "observations": observations,
        "usage": {
            "gpu_seconds": gpu_seconds,
            "worker_estimated_gpu_cost_usd": gpu_only_cost,
            "conservative_wall_estimated_gpu_cost_usd": estimated_total_cost,
            "wall_seconds": wall_elapsed,
        },
    }
    Path(output_path).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"neural_gpu_smoke": evidence["status"], "usage": evidence["usage"]}))


@app.local_entrypoint()
def source_scout(
    source_path: str,
    manifest_path: str,
    campaign_config: str,
    output_dir: str,
    max_gpu_seconds: float = 1800.0,
    max_estimated_usd: float = 3.0,
    sample_interval_seconds: float = 12.0,
    batch_frames: int = 5,
    max_refinement_anchors: int = 30,
    max_sample_frames: int = 300,
) -> None:
    """Validation-only source-WIDE GPU scout via the canonical Clipper analysis.

    No alternate analyzer, planner, allocator or renderer. GPU model outputs are
    source-stamped PROPOSALS only and are separately queued for human review.
    """
    from pathlib import Path

    from clipper_engine.gameplay import analysis, neural_evidence
    from clipper_engine.inference.modal import ModalVisionProvider
    from clipper_engine.inference.vision import VisionSettings, analyze_source
    from clipper_engine.profiles import load_profile
    from clipper_engine.sources import qa as source_qa

    video = Path(source_path)
    source_certificate = Path(manifest_path)
    output = Path(output_dir)
    if not 0 < max_estimated_usd <= 3.0 or not 0 < max_gpu_seconds <= 3600:
        raise ValueError("source-wide scout budget must be finite and bounded")
    if not 4 <= sample_interval_seconds <= 15 or not 1 <= batch_frames <= 8:
        raise ValueError("source-wide scout sampling must be bounded")
    if not 1 <= max_refinement_anchors <= 100 or not 20 <= max_sample_frames <= 500:
        raise ValueError("source-wide scout anchor/frame budget must be bounded")
    if not DEPLOYED_GIT_SHA or len(DEPLOYED_GIT_SHA) != 40:
        raise RuntimeError("source-wide qualification requires immutable git SHA")
    if os.environ.get("GITHUB_SHA") and os.environ["GITHUB_SHA"] != DEPLOYED_GIT_SHA:
        raise RuntimeError("source-wide GPU worker SHA differs from checked-out source")
    identity = source_qa.verify(video, source_certificate, require_original=False)
    source_key = str(identity.get("source_key") or "")
    if identity.get("derivative_type") != "proxy" or not source_key:
        raise ValueError("source-wide neural qualification requires certified analysis proxy")
    config = load_profile("mw4", Path(campaign_config)).config
    capability = dict((config.get("capabilities") or {}).get("neural_visual_events") or {})
    capability.update(
        {
            "enabled": True,
            "backend": "modal",
            "deployed_sha": DEPLOYED_GIT_SHA,
            "model_id": VISION_MODEL_ID,
            "model_revision": VISION_REVISION,
            "sample_interval_seconds": sample_interval_seconds,
            "batch_frames": batch_frames,
            "max_gpu_seconds": max_gpu_seconds,
            "max_estimated_usd": max_estimated_usd,
            "max_sample_frames": max_sample_frames,
            "max_refinement_anchors": max_refinement_anchors,
            # The source-wide qualification MUST exercise the exact same
            # per-batch inference deadline as the production gameplay profile.
            # Previously this silently forced 180s over MW4's 480s setting.
        }
    )
    config["capabilities"]["neural_visual_events"] = capability
    settings = VisionSettings.from_profile(config)
    worker = VisionModel()
    ready = worker.ready.remote()
    if (
        ready.get("deployed_git_sha") != DEPLOYED_GIT_SHA
        or ready.get("model_id") != VISION_MODEL_ID
        or ready.get("revision") != VISION_REVISION
        or ready.get("cuda_available") is not True
        or ready.get("gpu_count", 0) < 2
    ):
        raise RuntimeError("source-wide neural worker CUDA/model/SHA readiness mismatch")
    timeline = analysis.analyze_source(video, config)
    output.mkdir(parents=True, exist_ok=True)
    report = analyze_source(
        video,
        source_certificate,
        config,
        output,
        provider=ModalVisionProvider(settings, worker=worker),
        source_key=source_key,
        timeline=timeline,
    )
    if report is None or report.get("status") != "PROPOSALS_ONLY":
        raise RuntimeError("source-wide visual scout did not yield complete proposals")
    review = neural_evidence.reconcile(report, timeline, source_key, config)
    (output / "neural_review_queue.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    summary = {
        "status": "GPU_SOURCE_WIDE_PROPOSALS_ONLY",
        "execution_sha": DEPLOYED_GIT_SHA,
        "source_key": source_key,
        "source_sha256": identity["sha256"],
        "model": report["model"],
        "source_clock": report.get("source_clock"),
        "coverage": report["coverage"],
        "usage": report.get("usage"),
        "review_queue_coverage": review["coverage"],
        "readiness": ready,
        "requires_independent_source_review": True,
        "may_create_verified_outcomes": False,
    }
    (output / "source_wide_gpu_qualification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
