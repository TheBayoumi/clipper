from __future__ import annotations

import base64
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
# This is a Modal resource name, not credential material.
HF_SECRET_NAME = "custom-secret"  # noqa: S105
HF_CACHE = "/model-cache"
MEDIA_ROOT = "/media"
L4_USD_PER_SECOND = 0.000222
L40S_USD_PER_SECOND = 0.000542
EDITORIAL_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EDITORIAL_MODEL_REVISION = "110954009be4a882781a90356c7d2b8a9e3428dc"
DEPLOYED_GIT_SHA = os.getenv("CLIPPER_DEPLOYED_GIT_SHA", "").strip().lower()
EDITORIAL_EXECUTION_TIMEOUT_SECONDS = int(
    os.getenv("CLIPPER_EDITORIAL_EXECUTION_TIMEOUT_SECONDS", "900")
)
EDITORIAL_STARTUP_TIMEOUT_SECONDS = int(
    os.getenv("CLIPPER_EDITORIAL_STARTUP_TIMEOUT_SECONDS", "1800")
)
EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS = int(
    os.getenv("CLIPPER_EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS", "65536")
)
if EDITORIAL_EXECUTION_TIMEOUT_SECONDS <= 0:
    raise ValueError("CLIPPER_EDITORIAL_EXECUTION_TIMEOUT_SECONDS must be positive")
if EDITORIAL_STARTUP_TIMEOUT_SECONDS <= 0:
    raise ValueError("CLIPPER_EDITORIAL_STARTUP_TIMEOUT_SECONDS must be positive")
if EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS <= 0:
    raise ValueError("CLIPPER_EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS must be positive")

model_cache = modal.Volume.from_name("clipper-hf-cache", create_if_missing=True)
media_cache = modal.Volume.from_name("clipper-media-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_HUB_CACHE": HF_CACHE,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CLIPPER_DEPLOYED_GIT_SHA": DEPLOYED_GIT_SHA,
        }
    )
)
media_image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.12")
    .entrypoint([])
    .apt_install("ffmpeg", "git")
    .uv_pip_install(
        "yt-dlp>=2026.7.4,<2027",
        "bgutil-ytdlp-pot-provider==1.3.1",
    )
    .run_commands(
        "git clone --depth 1 --branch 1.3.1 "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "
        "/root/bgutil-ytdlp-pot-provider",
        "cd /root/bgutil-ytdlp-pot-provider/server && npm ci && npx tsc",
    )
)
text_image = base_image.uv_pip_install(
    "torch==2.8.0",
    "torchvision==0.23.0",
    "transformers>=4.57,<5",
    "accelerate>=1.14,<2",
    "sentencepiece>=0.2,<1",
    "tiktoken>=0.11,<1",
    "bitsandbytes>=0.47,<1",
    "pillow>=11,<13",
    "outlines>=1.3,<2",
).add_local_python_source("clipper")
speech_image = base_image.uv_pip_install(
    "torch>=2.8,<3",
    "faster-whisper>=1.2.1,<2",
    "whisperx>=3.8.6,<4",
    "pyannote.audio>=4.0.7,<5",
)

app = modal.App(APP_NAME)
_whisper_model: Any | None = None
_diarization_pipeline: Any | None = None
_model_revisions: dict[str, str] = {}


def _model_revision(model_id: str) -> str:
    cached = _model_revisions.get(model_id)
    if cached is not None:
        return cached
    from huggingface_hub import HfApi

    revision = str(
        HfApi(token=os.environ.get("HF_TOKEN") or None).model_info(model_id).sha or "unknown"
    )
    _model_revisions[model_id] = revision
    return revision


def _model_evidence(model_id: str, *, revision: str | None = None) -> dict[str, str]:
    return {"model_id": model_id, "revision": revision or _model_revision(model_id)}


def _gpu_rate(gpu: str) -> float:
    if gpu == "L40S":
        return L40S_USD_PER_SECOND
    if gpu.startswith("L4"):
        count = int(gpu.split(":", 1)[1]) if ":" in gpu else 1
        return L4_USD_PER_SECOND * count
    raise ValueError(f"unsupported GPU rate: {gpu}")


def _gpu_count(gpu: str) -> int:
    return int(gpu.split(":", 1)[1]) if ":" in gpu else 1


def _usage(
    started: float, gpu: str, *, input_units: int = 0, output_units: int = 0
) -> dict[str, Any]:
    import torch

    duration = max(0.0, time.perf_counter() - started)
    rate = _gpu_rate(gpu)
    peak_by_device = (
        {
            str(index): float(torch.cuda.max_memory_allocated(index) / (1024 * 1024))
            for index in range(torch.cuda.device_count())
        }
        if torch.cuda.is_available()
        else {}
    )
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "gpu_type": gpu,
        "gpu_seconds": duration * _gpu_count(gpu),
        "peak_vram_mb": max(peak_by_device.values(), default=None),
        "peak_vram_mb_by_device": peak_by_device,
        "input_units": input_units,
        "output_units": output_units,
        "estimated_cost_usd": duration * rate,
    }


class EditorialOutputTruncated(ValueError):
    """Editorial generation exhausted runtime-derived output capacity."""

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        self.details = dict(details)
        super().__init__(message)


def _json_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0].strip().casefold() in {"```", "```json"}:
            candidate = "\n".join(lines[1:-1]).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def _vision_contract(task: str) -> str:
    if task == "source_policy_visual_scout":
        return (
            'Schema: {"observations":[{"timestamp":0.0,"scene_id":"scene-1",'
            '"summary":"visible source evidence","visible_speakers":["speaker labels"],'
            '"event_labels":["branding:name","ocr:text","hazard:description"],'
            '"confidence":0.0}]}. '
            "Return exactly one observation for every supplied frame_timestamps value, in the "
            "same order. timestamp must equal a supplied value. Use empty event_labels when no "
            "policy-relevant label is visible. Never invent observations for unsupplied times."
        )
    if task == "visual_timeline_scout":
        return (
            'Schema: {"events":[{"start":0.0,"end":1.0,"scene_id":"scene-1",'
            '"summary":"visible evidence","visible_speakers":["speaker labels"],'
            '"event_labels":["short labels"],"confidence":0.0}]}. '
            "Use only supplied frame_timestamps for start and end. Return events=[] when no "
            "reliable visual event is visible."
        )
    if task in {"rendered_clip_review", "rendered_clip_review_escalation"}:
        return (
            'Schema: {"decision":"PASS","summary":"short review",'
            '"overall_confidence":0.0,"issues":[{"issue_type":"short_label",'
            '"start":0.0,"end":1.0,"severity":"LOW","confidence":0.0,'
            '"repair_target":"TRACKING","description":"visible problem"}]}. '
            "decision must be PASS, REPAIR, REJECT, or ESCALATE; severity must be LOW, "
            "MEDIUM, or HIGH."
        )
    return "Return one JSON object whose fields satisfy the supplied task and context."


def _vision_prompt(payload: dict[str, Any], attempt: int) -> str:
    task = str(payload.get("task") or "")
    recovery = ""
    if attempt > 1:
        recovery = (
            " Recovery attempt: the previous response was empty or malformed. Return one "
            "complete JSON object, with no markdown fence or explanatory text."
        )
    return (
        "Return only one complete valid JSON object. Do not retranscribe audio. "
        + _vision_contract(task)
        + recovery
        + " Payload: "
        + json.dumps({"task": task, "context": payload.get("context")}, ensure_ascii=False)
    )


class VisionOutputCapacityError(ValueError):
    """Vision generation exhausted its runtime-derived output capacity."""


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _vision_context_limit(model: Any, processor: Any) -> int:
    text_config = getattr(model.config, "text_config", model.config)
    for value in (
        getattr(text_config, "max_position_embeddings", None),
        getattr(model.config, "max_position_embeddings", None),
        getattr(getattr(processor, "tokenizer", None), "model_max_length", None),
    ):
        parsed = _positive_int(value)
        if parsed is not None:
            return parsed
    raise ValueError("vision model does not expose a usable context limit")


def _vision_output_cardinality(payload: dict[str, Any], frame_count: int) -> int:
    context = payload.get("context")
    if isinstance(context, dict):
        timestamps = context.get("frame_timestamps")
        if isinstance(timestamps, list) and timestamps:
            return len(timestamps)
    return max(1, frame_count)


def _vision_output_template(task: str, cardinality: int) -> dict[str, Any]:
    if task == "source_policy_visual_scout":
        observation = {
            "timestamp": 0.0,
            "scene_id": "scene-visible-source-evidence",
            "summary": (
                "visible source evidence including people, scene, branding, overlays, "
                "on-screen text, and policy-relevant hazards"
            ),
            "visible_speakers": ["visible-speaker-a", "visible-speaker-b"],
            "event_labels": ["branding:visible", "ocr:visible text", "hazard:visible condition"],
            "confidence": 0.0,
        }
        return {"observations": [dict(observation) for _ in range(cardinality)]}
    if task == "visual_timeline_scout":
        event = {
            "start": 0.0,
            "end": 0.0,
            "scene_id": "scene-visible-evidence",
            "summary": "visible scene evidence relevant to editorial continuity",
            "visible_speakers": ["visible-speaker"],
            "event_labels": ["visible-event"],
            "confidence": 0.0,
        }
        return {"events": [dict(event) for _ in range(cardinality)]}
    if task in {"rendered_clip_review", "rendered_clip_review_escalation"}:
        issue = {
            "issue_type": "visible_issue",
            "start": 0.0,
            "end": 0.0,
            "severity": "MEDIUM",
            "confidence": 0.0,
            "repair_target": "TRACKING",
            "description": "visible problem requiring deterministic repair or rejection",
        }
        return {
            "decision": "REPAIR",
            "summary": "visual review of the rendered clip",
            "overall_confidence": 0.0,
            "issues": [dict(issue) for _ in range(cardinality)],
        }
    return {
        "items": [
            {"summary": "structured task output derived from supplied visual evidence"}
            for _ in range(cardinality)
        ]
    }


def _vision_structural_output_tokens(processor: Any, task: str, cardinality: int) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise ValueError(
            "vision processor does not expose a tokenizer for output-capacity planning"
        )
    template = json.dumps(
        _vision_output_template(task, cardinality),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, len(tokenizer.encode(template, add_special_tokens=False)))


def _vision_history_output_tokens_per_item(payload: dict[str, Any]) -> float | None:
    context = payload.get("context")
    if not isinstance(context, dict):
        return None
    capacity = context.get("generation_capacity")
    if not isinstance(capacity, dict):
        return None
    raw = capacity.get("observed_output_tokens_per_item")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0.0 and math.isfinite(value) else None


def _vision_generation_capacity(
    payload: dict[str, Any],
    *,
    frame_count: int,
    input_units: int,
    processor: Any,
    model: Any,
    minimum_budget: int = 0,
) -> dict[str, Any]:
    task = str(payload.get("task") or "")
    context_limit = _vision_context_limit(model, processor)
    available_output = context_limit - input_units
    if available_output <= 0:
        raise ValueError(
            "vision request exceeds model context: "
            f"input_tokens={input_units} context_limit={context_limit} frames={frame_count}"
        )
    cardinality = _vision_output_cardinality(payload, frame_count)
    structural_floor = _vision_structural_output_tokens(processor, task, cardinality)
    history_per_item = _vision_history_output_tokens_per_item(payload)
    history_budget = (
        math.ceil(history_per_item * cardinality) if history_per_item is not None else 0
    )
    predicted_budget = max(structural_floor, history_budget, minimum_budget, 1)
    generation_budget = min(available_output, predicted_budget)
    return {
        "task": task,
        "cardinality": cardinality,
        "context_limit_tokens": context_limit,
        "available_output_tokens": available_output,
        "structural_floor_tokens": structural_floor,
        "history_output_tokens_per_item": history_per_item,
        "generation_budget_tokens": generation_budget,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _editorial_context_limit(model: Any, tokenizer: Any) -> int:
    text_config = getattr(model.config, "text_config", model.config)
    for value in (
        getattr(text_config, "max_position_embeddings", None),
        getattr(model.config, "max_position_embeddings", None),
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed > 0:
            return parsed
    raise ValueError("editorial model does not expose a usable context limit")


def _editorial_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("payload")
    return raw if isinstance(raw, dict) else {}


def _editorial_discourse_units(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    raw_words = _editorial_payload(payload).get("words")
    if not isinstance(raw_words, list):
        raw_words = _editorial_payload(payload).get("source_context_words")
    words = [item for item in raw_words or [] if isinstance(item, dict)]
    if not words:
        return []
    terminal = (".", "!", "?", "…", "。")
    units: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_speaker: object = None
    for word in words:
        speaker = word.get("speaker_id")
        if (
            current
            and previous_speaker is not None
            and speaker is not None
            and speaker != previous_speaker
        ):
            units.append(current)
            current = []
        current.append(word)
        if str(word.get("text") or "").rstrip().endswith(terminal):
            units.append(current)
            current = []
        previous_speaker = speaker
    if current:
        units.append(current)
    return units


def _editorial_output_template(payload: dict[str, Any]) -> dict[str, Any]:
    from clipper.providers.editorial_prompt import editorial_task_family

    task = str(payload.get("task") or "")
    family = editorial_task_family(task)
    actual = _editorial_payload(payload)
    units = _editorial_discourse_units(payload)

    def unit_text(unit: list[dict[str, Any]]) -> str:
        return " ".join(str(item.get("text") or "") for item in unit).strip()

    def first_ref(unit: list[dict[str, Any]]) -> str:
        return str(unit[0].get("word_ref") or "word-start") if unit else "word-start"

    def last_ref(unit: list[dict[str, Any]]) -> str:
        return str(unit[-1].get("word_ref") or "word-end") if unit else "word-end"

    if family == "semantic_cores":
        return {
            "cores": [
                {
                    "core_id": f"core-{first_ref(unit)}-{last_ref(unit)}",
                    "start_word_id": first_ref(unit),
                    "end_word_id": last_ref(unit),
                    "semantic_summary": unit_text(unit),
                    "editorial_reason": unit_text(unit),
                    "confidence": 0.0,
                }
                for unit in units
            ]
        }
    if family == "source_hazards":
        return {
            "segments": [
                {
                    "start_word_id": first_ref(unit),
                    "end_word_id": last_ref(unit),
                    "classification": "editorial_content",
                    "confidence": 0.0,
                    "evidence": [unit_text(unit)],
                }
                for unit in units
            ]
        }
    context_text = " ".join(unit_text(unit) for unit in units).strip()
    core = actual.get("core") if isinstance(actual.get("core"), dict) else {}
    if family == "narrative_envelope":
        context_words = actual.get("source_context_words")
        refs = [item for item in context_words or [] if isinstance(item, dict)]
        return {
            "envelope_id": "envelope",
            "core_id": str(core.get("core_id") or "core"),
            "start_word_id": str(refs[0].get("word_ref") or "word-start") if refs else "word-start",
            "end_word_id": str(refs[-1].get("word_ref") or "word-end") if refs else "word-end",
            "required_prior_context": context_text,
            "required_followup_context": context_text,
            "setup_resolved": True,
            "payoff_resolved": True,
            "reference_resolution": [],
            "confidence": 0.0,
        }
    windows = actual.get("feasible_windows")
    first_window = (
        windows[0] if isinstance(windows, list) and windows and isinstance(windows[0], dict) else {}
    )
    return {
        "core_id": str(core.get("core_id") or "core"),
        "selected_window_id": first_window.get("window_id"),
        "decision": "PASS",
        "quality_score": 0.0,
        "opening_strategy": context_text,
        "rationale": context_text,
        "confidence": 0.0,
    }


def _editorial_structural_output_tokens(payload: dict[str, Any], tokenizer: Any) -> int:
    template = json.dumps(
        _editorial_output_template(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, len(tokenizer.encode(template, add_special_tokens=False)))


def _editorial_device_map_policy() -> str:
    import torch

    return "balanced" if torch.cuda.device_count() > 1 else "auto"


def _editorial_cuda_device_indices(device_map: dict[str, Any]) -> tuple[int, ...]:
    indices: set[int] = set()
    for value in device_map.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if value >= 0:
                indices.add(value)
            continue
        rendered = str(value)
        if rendered.startswith("cuda:"):
            suffix = rendered.split(":", 1)[1]
            if suffix.isdigit():
                indices.add(int(suffix))
    return tuple(sorted(indices))


def _editorial_model_bytes_by_device(model: Any) -> dict[str, int]:
    totals: dict[str, int] = {}
    seen: set[int] = set()
    for tensor in (*tuple(model.parameters()), *tuple(model.buffers())):
        marker = id(tensor)
        if marker in seen:
            continue
        seen.add(marker)
        device = str(tensor.device)
        totals[device] = totals.get(device, 0) + int(tensor.numel()) * int(tensor.element_size())
    return totals


def _editorial_device_distribution(model: Any) -> dict[str, Any]:
    import torch

    raw_map = dict(getattr(model, "hf_device_map", {}) or {})
    model_gpu_indices = _editorial_cuda_device_indices(raw_map)
    expected_gpu_count = torch.cuda.device_count()
    if expected_gpu_count > 1 and len(model_gpu_indices) != expected_gpu_count:
        raise RuntimeError(
            "editorial model did not distribute across all allocated GPUs: "
            f"policy={_editorial_device_map_policy()} expected={expected_gpu_count} "
            f"observed={model_gpu_indices} hf_device_map={raw_map}"
        )
    return {
        "placement_policy": _editorial_device_map_policy(),
        "expected_gpu_count": expected_gpu_count,
        "model_gpu_indices": list(model_gpu_indices),
        "hf_device_map": raw_map,
        "model_bytes_by_device": _editorial_model_bytes_by_device(model),
    }


def _editorial_topology_key() -> str:
    import torch

    material = {
        "gpu_names": [
            str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())
        ],
        "placement_policy": _editorial_device_map_policy(),
    }
    encoded = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _editorial_capacity_state_path() -> Path:
    return (
        Path(HF_CACHE)
        / "clipper-editorial-capacity"
        / f"{EDITORIAL_MODEL_REVISION}-{_editorial_topology_key()}.json"
    )


def _load_editorial_capacity_state() -> dict[str, Any]:
    path = _editorial_capacity_state_path()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _persist_editorial_capacity_state(state: dict[str, Any]) -> None:
    path = _editorial_capacity_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    model_cache.commit()


def _editorial_history_entry(state: dict[str, Any], task: str) -> dict[str, Any]:
    from clipper.providers.editorial_prompt import editorial_task_family

    family = editorial_task_family(task)
    raw = state.get(family)
    if isinstance(raw, dict):
        return raw
    entry: dict[str, Any] = {}
    state[family] = entry
    return entry


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if parsed > 0 and math.isfinite(parsed) else None


def _execution_id(payload: dict[str, Any]) -> str:
    return str(payload.get("execution_id") or "").strip()


def _assert_expected_git_sha(payload: dict[str, Any]) -> None:
    expected = str(payload.get("expected_git_sha") or "").strip().lower()
    if not expected:
        return
    if not DEPLOYED_GIT_SHA:
        raise RuntimeError("open-model worker has no embedded deployment SHA")
    if expected != DEPLOYED_GIT_SHA:
        raise RuntimeError(
            "open-model worker SHA mismatch: "
            f"expected={expected} deployed={DEPLOYED_GIT_SHA}"
        )


def _editorial_generation_plan(
    payload: dict[str, Any],
    *,
    tokenizer: Any,
    model: Any,
    input_units: int,
    capacity_state: dict[str, Any],
) -> dict[str, Any]:
    from clipper.providers.base import EditorialCapacityError

    task = str(payload.get("task") or "")
    actual_payload = _editorial_payload(payload)
    repartitionable = actual_payload.get("capacity_repartitionable") is True
    context_limit = _editorial_context_limit(model, tokenizer)
    available_output = context_limit - input_units
    if available_output <= 0:
        raise EditorialCapacityError(
            "editorial request exceeds model context",
            details={
                "reason": "context_exhausted",
                "task": task,
                "input_tokens": input_units,
                "context_limit_tokens": context_limit,
            },
        )
    structural = _editorial_structural_output_tokens(payload, tokenizer)
    raw_minimum = payload.get("generation_minimum_output_tokens")
    minimum = (
        int(raw_minimum)
        if isinstance(raw_minimum, int) and not isinstance(raw_minimum, bool) and raw_minimum > 0
        else 0
    )
    history = _editorial_history_entry(capacity_state, task)
    ratio = _positive_number(history.get("output_tokens_per_input_token"))
    history_budget = math.ceil(ratio * input_units) if ratio is not None else 0
    requested_output = max(structural, minimum, history_budget, 1)
    if requested_output > available_output:
        raise EditorialCapacityError(
            "editorial input and task-derived output demand exceed model context",
            details={
                "reason": "input_output_context_exhausted",
                "task": task,
                "input_tokens": input_units,
                "context_limit_tokens": context_limit,
                "available_output_tokens": available_output,
                "structural_output_tokens": structural,
                "history_output_tokens": history_budget,
                "requested_output_tokens": requested_output,
            },
        )

    if repartitionable and input_units > EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS:
        raise EditorialCapacityError(
            "editorial request exceeds the measured runtime-safe input bootstrap",
            details={
                "reason": "runtime_input_guard",
                "task": task,
                "input_tokens": input_units,
                "context_limit_tokens": context_limit,
                "available_output_tokens": available_output,
                "requested_output_tokens": requested_output,
                "runtime_safe_input_tokens": EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS,
            },
        )

    smallest_bad = history.get("smallest_bad_input_tokens")
    largest_good = history.get("largest_good_input_tokens")
    smallest_dynamic_bad = history.get("smallest_dynamic_oom_input_tokens")
    largest_dynamic_good = history.get("largest_dynamic_good_input_tokens")
    largest_offloaded_good = history.get("largest_offloaded_good_input_tokens")

    if (
        repartitionable
        and isinstance(smallest_dynamic_bad, int)
        and not isinstance(smallest_dynamic_bad, bool)
        and smallest_dynamic_bad > 0
        and input_units >= smallest_dynamic_bad
        and (
            not isinstance(largest_dynamic_good, int)
            or isinstance(largest_dynamic_good, bool)
            or input_units > largest_dynamic_good
        )
    ):
        raise EditorialCapacityError(
            "editorial request is at or above a learned dynamic-KV OOM boundary",
            details={
                "reason": "history_dynamic_oom_boundary",
                "task": task,
                "input_tokens": input_units,
                "context_limit_tokens": context_limit,
                "largest_good_input_tokens": largest_dynamic_good,
                "smallest_bad_input_tokens": smallest_dynamic_bad,
                "generation_budget_tokens": requested_output,
            },
        )

    if (
        isinstance(smallest_bad, int)
        and not isinstance(smallest_bad, bool)
        and smallest_bad > 0
        and input_units >= smallest_bad
        and (
            not isinstance(largest_good, int)
            or isinstance(largest_good, bool)
            or input_units > largest_good
        )
    ):
        raise EditorialCapacityError(
            "editorial request is at or above a previously observed OOM boundary",
            details={
                "reason": "history_capacity_boundary",
                "task": task,
                "input_tokens": input_units,
                "context_limit_tokens": context_limit,
                "largest_good_input_tokens": largest_good,
                "smallest_bad_input_tokens": smallest_bad,
                "generation_budget_tokens": requested_output,
            },
        )
    return {
        "task": task,
        "input_tokens": input_units,
        "context_limit_tokens": context_limit,
        "available_output_tokens": available_output,
        "structural_output_tokens": structural,
        "history_output_tokens": history_budget,
        "generation_budget_tokens": requested_output,
        "largest_good_input_tokens": largest_good,
        "smallest_bad_input_tokens": smallest_bad,
        "largest_dynamic_good_input_tokens": largest_dynamic_good,
        "smallest_dynamic_oom_input_tokens": smallest_dynamic_bad,
        "largest_offloaded_good_input_tokens": largest_offloaded_good,
        "runtime_safe_input_tokens": EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS,
        "capacity_repartitionable": repartitionable,
    }


def _update_editorial_success_history(
    state: dict[str, Any],
    *,
    task: str,
    input_units: int,
    output_units: int,
    cache_implementation: str,
) -> None:
    history = _editorial_history_entry(state, task)
    previous_good = history.get("largest_good_input_tokens")
    if (
        not isinstance(previous_good, int)
        or isinstance(previous_good, bool)
        or input_units > previous_good
    ):
        history["largest_good_input_tokens"] = input_units

    cache_good_key = (
        "largest_dynamic_good_input_tokens"
        if cache_implementation == "dynamic"
        else "largest_offloaded_good_input_tokens"
    )
    previous_cache_good = history.get(cache_good_key)
    if (
        not isinstance(previous_cache_good, int)
        or isinstance(previous_cache_good, bool)
        or input_units > previous_cache_good
    ):
        history[cache_good_key] = input_units

    bad = history.get("smallest_bad_input_tokens")
    if isinstance(bad, int) and not isinstance(bad, bool) and input_units >= bad:
        history.pop("smallest_bad_input_tokens", None)
    if input_units > 0 and output_units > 0:
        ratio = output_units / input_units
        previous_ratio = _positive_number(history.get("output_tokens_per_input_token"))
        history["output_tokens_per_input_token"] = (
            ratio if previous_ratio is None else max(previous_ratio, ratio)
        )
    history["successful_cache_implementation"] = cache_implementation
    history["cuda_memory_by_device"] = _cuda_memory_snapshot()
    _persist_editorial_capacity_state(state)


def _update_editorial_oom_history(
    state: dict[str, Any],
    *,
    task: str,
    input_units: int,
    cache_implementation: str,
) -> None:
    history = _editorial_history_entry(state, task)
    cache_bad_key = (
        "smallest_dynamic_oom_input_tokens"
        if cache_implementation == "dynamic"
        else "smallest_offloaded_oom_input_tokens"
    )
    previous_cache_bad = history.get(cache_bad_key)
    if (
        not isinstance(previous_cache_bad, int)
        or isinstance(previous_cache_bad, bool)
        or input_units < previous_cache_bad
    ):
        history[cache_bad_key] = input_units

    if cache_implementation == "offloaded":
        previous_bad = history.get("smallest_bad_input_tokens")
        if (
            not isinstance(previous_bad, int)
            or isinstance(previous_bad, bool)
            or input_units < previous_bad
        ):
            history["smallest_bad_input_tokens"] = input_units

    history[f"cuda_memory_by_device_at_{cache_implementation}_oom"] = _cuda_memory_snapshot()
    _persist_editorial_capacity_state(state)


def _transport_error(
    exc: Exception,
    *,
    context: str | None = None,
    execution_id: str = "",
) -> dict[str, Any]:
    error_type = type(exc).__name__
    capacity_rejected = error_type == "OutOfMemoryError" or "CapacityError" in error_type
    if not capacity_rejected:
        traceback.print_exception(exc)
    message = str(exc)
    if context:
        message = f"{context}: {message}"
    application_status = "CAPACITY_REJECTED" if capacity_rejected else "FAILED"
    recovery_action = "REPARTITION" if capacity_rejected else "NONE"
    raw_details = getattr(exc, "details", None)
    details = dict(raw_details) if isinstance(raw_details, dict) else {}
    details.setdefault("application_status", application_status)
    details.setdefault("recovery_action", recovery_action)
    print(
        json.dumps(
            {
                "event": "application_result",
                "application_status": application_status,
                "error_type": error_type,
                "recovery_action": recovery_action,
                "execution_id": execution_id,
            },
            sort_keys=True,
        )
    )
    return {
        "application_status": application_status,
        "recovery_action": recovery_action,
        "error": {
            "type": error_type,
            "message": message,
            "details": details,
        },
    }


def _diarization_audio_source(source_path: str) -> tuple[Path, bool]:
    """Return an audio file whose stream header has a concrete duration.

    YouTube MKV masters can carry duration only at the container level. Pyannote 4's
    TorchCodec-backed crop path reads the audio-stream duration and eventually evaluates
    ``None * sample_rate`` when that field is absent. A compact PCM WAV gives TorchCodec
    complete stream metadata while preserving the canonical source master for rendering.
    """

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"diarization source does not exist: {source}")
    if source.suffix.lower() in {".wav", ".wave"}:
        return source, False

    key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:24]
    target = Path(tempfile.gettempdir()) / f"clipper-pyannote-{key}.wav"
    temporary = target.with_suffix(".tmp.wav")
    target.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise RuntimeError("ffmpeg produced no usable diarization WAV")
        temporary.replace(target)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffmpeg returned no diagnostic output").strip()
        raise RuntimeError(f"diarization WAV extraction failed: {detail[-4000:]}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target, True


@app.function(
    image=media_image,
    volumes={MEDIA_ROOT: media_cache},
    timeout=7200,
    memory=4096,
    scaledown_window=2,
)
def acquire_source(payload: dict[str, Any]) -> dict[str, Any]:
    video_url = str(payload.get("video_url") or "").strip()
    video_id = str(payload.get("video_id") or "").strip()
    if not video_url.startswith("https://"):
        raise ValueError("source acquisition requires an https video_url")
    safe_video_id_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not video_id or any(char not in safe_video_id_chars for char in video_id):
        raise ValueError("source acquisition requires a safe video_id")

    staging = Path(MEDIA_ROOT) / "staging" / video_id
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    output_template = staging / "source.%(ext)s"
    command = [
        "yt-dlp",
        "--no-playlist",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--concurrent-fragments",
        "4",
        "--extractor-args",
        "youtube:player_client=mweb",
        "--format",
        "bestvideo+bestaudio/best",
        "--merge-output-format",
        "mkv",
        "--output",
        str(output_template),
        "--print",
        "after_move:filepath",
        video_url,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=7000,
        )
    except subprocess.CalledProcessError as exc:
        detail = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip()
        )
        detail = detail[-6000:] if detail else "yt-dlp returned no diagnostic output"
        raise RuntimeError(f"yt-dlp source acquisition failed:\n{detail}") from exc
    printed_paths = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    source = printed_paths[-1] if printed_paths else Path()
    if not source.is_file() or source.stat().st_size <= 0:
        candidates = [path for path in staging.glob("source.*") if path.is_file()]
        if not candidates:
            raise RuntimeError("yt-dlp completed without creating a source master")
        source = max(candidates, key=lambda path: path.stat().st_size)

    digest = _sha256_file(source)
    suffix = source.suffix.lower() or ".mkv"
    volume_path = f"/inputs/{digest}{suffix}"
    target = Path(MEDIA_ROOT) / volume_path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        if _sha256_file(target) != digest:
            raise RuntimeError("existing Modal source master hash mismatch")
        source.unlink(missing_ok=True)
    else:
        source.replace(target)
    media_cache.commit()
    shutil.rmtree(staging, ignore_errors=True)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    probe_payload = json.loads(probe.stdout)
    streams = probe_payload.get("streams") if isinstance(probe_payload, dict) else []
    stream_list = streams if isinstance(streams, list) else []
    video_stream = next(
        (
            item
            for item in stream_list
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ),
        {},
    )
    audio_stream = next(
        (
            item
            for item in stream_list
            if isinstance(item, dict) and item.get("codec_type") == "audio"
        ),
        {},
    )
    return {
        "video_id": video_id,
        "source_url": video_url,
        "sha256": digest,
        "bytes": target.stat().st_size,
        "volume_path": volume_path,
        "mount_path": str(target),
        "container": suffix.lstrip("."),
        "quality_policy": "highest_available_no_transcode",
        "video": {
            "width": int(video_stream.get("width") or 0),
            "height": int(video_stream.get("height") or 0),
            "codec": str(video_stream.get("codec_name") or "unknown"),
            "pixel_format": str(video_stream.get("pix_fmt") or "unknown"),
            "frame_rate": str(video_stream.get("avg_frame_rate") or "unknown"),
        },
        "audio": {
            "codec": str(audio_stream.get("codec_name") or "unknown"),
            "sample_rate": str(audio_stream.get("sample_rate") or "unknown"),
            "channels": int(audio_stream.get("channels") or 0),
        },
    }


def _cuda_memory_snapshot() -> dict[str, dict[str, float]]:
    import torch

    if not torch.cuda.is_available():
        return {}
    snapshot: dict[str, dict[str, float]] = {}
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        snapshot[str(index)] = {
            "free_mb": float(free_bytes / (1024 * 1024)),
            "total_mb": float(total_bytes / (1024 * 1024)),
            "allocated_mb": float(torch.cuda.memory_allocated(index) / (1024 * 1024)),
            "reserved_mb": float(torch.cuda.memory_reserved(index) / (1024 * 1024)),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated(index) / (1024 * 1024)),
        }
    return snapshot


def _worker_runtime(
    lifecycle_id: str,
    *,
    model_load_count: int,
    batch_frame_count: int | None = None,
    generation_capacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "worker_lifecycle_id": lifecycle_id,
        "model_load_count": model_load_count,
        "cuda_memory_by_device": _cuda_memory_snapshot(),
    }
    if batch_frame_count is not None:
        runtime["batch_frame_count"] = batch_frame_count
    if generation_capacity is not None:
        runtime["generation_capacity"] = dict(generation_capacity)
    return runtime


def _load_editorial_model() -> tuple[Any, Any, Any]:
    import torch
    from outlines import from_transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        EDITORIAL_MODEL_ID,
        revision=EDITORIAL_MODEL_REVISION,
        device_map=_editorial_device_map_policy(),
        dtype=torch.bfloat16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    _editorial_device_distribution(model)
    return tokenizer, model, from_transformers(model, tokenizer)


def _editorial_infer(
    payload: dict[str, Any],
    tokenizer: Any,
    model: Any,
    structured_model: Any,
    capacity_state: dict[str, Any],
    *,
    lifecycle_id: str,
) -> dict[str, Any]:
    import torch
    from outlines.types import JsonSchema

    from clipper.providers.base import EditorialCapacityError
    from clipper.providers.editorial_prompt import editorial_contract, editorial_json_schema

    started = time.perf_counter()
    task = str(payload.get("task") or "")
    execution_id = _execution_id(payload)
    system_content = (
        "You are a source-grounded multimodal short-form editor. "
        "Never invent source evidence, spoken words, timestamps, or IDs. "
        + editorial_contract(task)
    )
    recovery_instruction = payload.get("generation_recovery_instruction")
    if isinstance(recovery_instruction, str) and recovery_instruction.strip():
        system_content += " " + recovery_instruction.strip()
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    schema = JsonSchema(editorial_json_schema(task))
    input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    input_units = len(input_ids)
    serialized_request_bytes = len(rendered.encode("utf-8"))
    try:
        plan = _editorial_generation_plan(
            payload,
            tokenizer=tokenizer,
            model=model,
            input_units=input_units,
            capacity_state=capacity_state,
        )
    except EditorialCapacityError as exc:
        exc.details.setdefault("serialized_request_bytes", serialized_request_bytes)
        raise
    plan["serialized_request_bytes"] = serialized_request_bytes
    output_budget = int(plan["generation_budget_tokens"])
    print(
        json.dumps(
            {
                "event": "editorial_request_plan",
                "worker_lifecycle_id": lifecycle_id,
                "task": task,
                "execution_id": execution_id,
                **plan,
                "serialized_request_bytes": serialized_request_bytes,
                "cuda_memory_by_device": _cuda_memory_snapshot(),
            },
            sort_keys=True,
        )
    )

    generated_text: str | None = None
    cache_implementation = "dynamic"
    generation_started = time.perf_counter()
    for cache_policy in ("dynamic", "offloaded"):
        kwargs: dict[str, Any] = {
            "max_new_tokens": output_budget,
            "do_sample": False,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if cache_policy == "offloaded":
            kwargs["cache_implementation"] = "offloaded"
        try:
            print(
                json.dumps(
                    {
                        "event": "editorial_generation_start",
                        "worker_lifecycle_id": lifecycle_id,
                        "task": task,
                        "execution_id": execution_id,
                        "cache_implementation": cache_policy,
                        "input_tokens": input_units,
                        "generation_budget_tokens": output_budget,
                        "cuda_memory_by_device": _cuda_memory_snapshot(),
                    },
                    sort_keys=True,
                )
            )
            with torch.inference_mode():
                candidate = structured_model(rendered, schema, **kwargs)
            if not isinstance(candidate, str):
                raise TypeError(
                    "Outlines transformers generation returned a non-string response: "
                    f"{type(candidate).__name__}"
                )
            generated_text = candidate
            cache_implementation = cache_policy
            break
        except torch.OutOfMemoryError as exc:
            print(
                json.dumps(
                    {
                        "event": "editorial_oom",
                        "worker_lifecycle_id": lifecycle_id,
                        "task": task,
                        "execution_id": execution_id,
                        "cache_implementation": cache_policy,
                        "input_tokens": input_units,
                        "generation_budget_tokens": output_budget,
                        "cuda_memory_by_device": _cuda_memory_snapshot(),
                        "error": str(exc),
                    },
                    sort_keys=True,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
            _update_editorial_oom_history(
                capacity_state,
                task=task,
                input_units=input_units,
                cache_implementation=cache_policy,
            )
            if cache_policy == "dynamic":
                if plan.get("capacity_repartitionable") is True:
                    raise EditorialCapacityError(
                        "editorial generation exceeded dynamic-KV working-set capacity",
                        details={
                            "reason": "cuda_oom_dynamic_cache",
                            **plan,
                            "cache_implementation": cache_policy,
                            "cuda_memory_by_device": _cuda_memory_snapshot(),
                        },
                    ) from exc
                print(
                    json.dumps(
                        {
                            "event": "editorial_capacity_fallback",
                            "worker_lifecycle_id": lifecycle_id,
                            "task": task,
                            "from_cache_implementation": "dynamic",
                            "to_cache_implementation": "offloaded",
                        },
                        sort_keys=True,
                    )
                )
                continue
            raise EditorialCapacityError(
                "editorial generation exhausted GPU working-set capacity",
                details={
                    "reason": "cuda_oom_after_offloaded_cache",
                    **plan,
                    "cache_implementation": cache_policy,
                    "cuda_memory_by_device": _cuda_memory_snapshot(),
                },
            ) from exc

    if generated_text is None:
        raise AssertionError("editorial generation cache-policy ladder produced no result")

    output_ids = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
    output_units = len(output_ids)
    generated_sha = hashlib.sha256(generated_text.encode("utf-8")).hexdigest()[:16]
    print(
        json.dumps(
            {
                "event": "editorial_generation_complete",
                "worker_lifecycle_id": lifecycle_id,
                "task": task,
                "execution_id": execution_id,
                "cache_implementation": cache_implementation,
                "input_tokens": input_units,
                "output_tokens": output_units,
                "generation_budget_tokens": output_budget,
                "duration_seconds": max(0.0, time.perf_counter() - generation_started),
                "generated_sha256": generated_sha,
                "cuda_memory_by_device": _cuda_memory_snapshot(),
            },
            sort_keys=True,
        )
    )
    try:
        value = _json_text(generated_text)
    except json.JSONDecodeError as exc:
        if output_units >= output_budget:
            available = int(plan["available_output_tokens"])
            next_budget = min(
                available,
                max(output_budget + 1, output_budget * 2),
            )
            raise EditorialOutputTruncated(
                f"task={task} exhausted runtime-derived generation budget={output_budget}: {exc}",
                details={
                    **plan,
                    "generated_sha256": generated_sha,
                    "output_tokens": output_units,
                    "next_output_budget_tokens": next_budget,
                },
            ) from exc
        raise RuntimeError(
            "constrained editorial generation returned invalid JSON despite Outlines "
            f"schema enforcement for task={task}: {exc}"
        ) from exc

    _update_editorial_success_history(
        capacity_state,
        task=task,
        input_units=input_units,
        output_units=output_units,
        cache_implementation=cache_implementation,
    )
    runtime = _worker_runtime(lifecycle_id, model_load_count=1)
    runtime["editorial_capacity"] = {
        **plan,
        "output_tokens": output_units,
        "cache_implementation": cache_implementation,
        "logits_to_keep": 1,
        "placement": _editorial_device_distribution(model),
    }
    return {
        "value": value,
        "model": _model_evidence(EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION),
        "structured_generation": {
            "engine": "outlines-transformers",
            "schema_version": "editorial-json-v2",
            "constrained": True,
            "cache_implementation": cache_implementation,
            "logits_to_keep": 1,
        },
        "usage": _usage(
            started,
            "L4:2",
            input_units=input_units,
            output_units=output_units,
        ),
        "runtime": runtime,
    }


def _editorial_capacity_probe(
    payload: dict[str, Any],
    tokenizer: Any,
    model: Any,
    capacity_state: dict[str, Any],
    *,
    lifecycle_id: str,
) -> dict[str, Any]:
    """Measure a real editorial request against live model capacity without generation."""

    from clipper.providers.base import EditorialCapacityError
    from clipper.providers.editorial_prompt import editorial_contract

    started = time.perf_counter()
    task = str(payload.get("task") or "")
    execution_id = _execution_id(payload)
    system_content = (
        "You are a source-grounded multimodal short-form editor. "
        "Never invent source evidence, spoken words, timestamps, or IDs. "
        + editorial_contract(task)
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_units = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    serialized_request_bytes = len(rendered.encode("utf-8"))

    try:
        plan = _editorial_generation_plan(
            payload,
            tokenizer=tokenizer,
            model=model,
            input_units=input_units,
            capacity_state=capacity_state,
        )
        value: dict[str, Any] = {
            "event": "editorial_capacity_probe",
            "status": "FIT",
            "task": task,
            "execution_id": execution_id,
            **plan,
            "serialized_request_bytes": serialized_request_bytes,
        }
    except EditorialCapacityError as exc:
        value = {
            "event": "editorial_capacity_probe",
            "status": "CAPACITY_REJECTED",
            "task": task,
            "execution_id": execution_id,
            **exc.details,
            "serialized_request_bytes": serialized_request_bytes,
        }

    value["worker_lifecycle_id"] = lifecycle_id
    value["duration_seconds"] = max(0.0, time.perf_counter() - started)
    print(json.dumps(value, sort_keys=True))
    runtime = _worker_runtime(lifecycle_id, model_load_count=1)
    runtime["editorial_placement"] = dict(_editorial_device_distribution(model))
    return {
        "value": value,
        "model": _model_evidence(EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION),
        "usage": _usage(started, "L4:2", input_units=input_units, output_units=0),
        "runtime": runtime,
    }


@app.cls(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=EDITORIAL_EXECUTION_TIMEOUT_SECONDS,
    startup_timeout=EDITORIAL_STARTUP_TIMEOUT_SECONDS,
    memory=32768,
)
class EditorialModel:
    @modal.enter()
    def load_model(self) -> None:
        self.lifecycle_id = uuid.uuid4().hex
        self.tokenizer, self.model, self.structured_model = _load_editorial_model()
        self.placement = _editorial_device_distribution(self.model)
        self.capacity_state = _load_editorial_capacity_state()
        print(
            json.dumps(
                {
                    "event": "editorial_model_ready",
                    "worker_lifecycle_id": self.lifecycle_id,
                    "model": _model_evidence(EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION),
                    "cuda_memory_by_device": _cuda_memory_snapshot(),
                    **self.placement,
                    "capacity_state_path": str(_editorial_capacity_state_path()),
                },
                sort_keys=True,
            )
        )

    @modal.method()
    def ready(self) -> dict[str, Any]:
        runtime = _worker_runtime(self.lifecycle_id, model_load_count=1)
        runtime["editorial_placement"] = dict(self.placement)
        return {
            "value": {"ready": True},
            "model": _model_evidence(EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION),
            "runtime": runtime,
        }

    @modal.method()
    def capacity_probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task") or "")
        try:
            _assert_expected_git_sha(payload)
            return _editorial_capacity_probe(
                payload,
                self.tokenizer,
                self.model,
                self.capacity_state,
                lifecycle_id=self.lifecycle_id,
            )
        except Exception as exc:
            return _transport_error(
                exc,
                context=f"capacity_probe task={task or '<missing>'}",
                execution_id=_execution_id(payload),
            )

    @modal.method()
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task") or "")
        try:
            _assert_expected_git_sha(payload)
            return _editorial_infer(
                payload,
                self.tokenizer,
                self.model,
                self.structured_model,
                self.capacity_state,
                lifecycle_id=self.lifecycle_id,
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return _transport_error(
                exc,
                context=f"task={task or '<missing>'}",
                execution_id=_execution_id(payload),
            )
        finally:
            gc.collect()
            with contextlib.suppress(Exception):
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def _load_vision_model(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    kwargs: dict[str, Any] = {
        "device_map": "balanced" if torch.cuda.device_count() > 1 else "auto",
        "dtype": torch.bfloat16,
    }
    if "30B" in model_id:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return processor, model


def _vision_infer(
    payload: dict[str, Any],
    model_id: str,
    gpu: str,
    processor: Any,
    model: Any,
    *,
    lifecycle_id: str,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    started = time.perf_counter()
    raw_frames = payload.get("frames_base64")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("vision payload requires frames_base64")
    frames = [Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB") for item in raw_frames]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)

    total_input_units = 0
    total_output_units = 0
    minimum_budget = 0
    capacity_expansion_used = False
    format_recovery_used = False
    attempt = 0
    try:
        while True:
            attempt += 1
            inputs: Any | None = None
            output: Any | None = None
            generated: Any | None = None
            try:
                content: list[dict[str, Any]] = [
                    {"type": "image", "image": frame} for frame in frames
                ]
                content.append({"type": "text", "text": _vision_prompt(payload, attempt)})
                messages = [{"role": "user", "content": content}]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                input_units = int(inputs["input_ids"].numel())
                capacity = _vision_generation_capacity(
                    payload,
                    frame_count=len(frames),
                    input_units=input_units,
                    processor=processor,
                    model=model,
                    minimum_budget=minimum_budget,
                )
                generation_budget = int(capacity["generation_budget_tokens"])
                total_input_units += input_units
                print(
                    json.dumps(
                        {
                            "event": "vision_generation_start",
                            "worker_lifecycle_id": lifecycle_id,
                            "attempt": attempt,
                            "frames": len(frames),
                            **capacity,
                        },
                        sort_keys=True,
                    )
                )
                inputs = inputs.to(model.device)
                attempt_started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=generation_budget,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )
                generated = output[0][inputs["input_ids"].shape[-1] :]
                output_units = int(generated.numel())
                total_output_units += output_units
                saturated = output_units >= generation_budget
                print(
                    json.dumps(
                        {
                            "event": "vision_generation_complete",
                            "worker_lifecycle_id": lifecycle_id,
                            "attempt": attempt,
                            "frames": len(frames),
                            "generated_tokens": output_units,
                            "generation_budget_tokens": generation_budget,
                            "saturated": saturated,
                            "duration_seconds": max(0.0, time.perf_counter() - attempt_started),
                        },
                        sort_keys=True,
                    )
                )
                generated_text = processor.decode(generated, skip_special_tokens=True)
                try:
                    value = _json_text(generated_text)
                except (json.JSONDecodeError, ValueError) as exc:
                    digest = hashlib.sha256(generated_text.encode("utf-8")).hexdigest()[:16]
                    print(
                        json.dumps(
                            {
                                "event": "vision_json_validation",
                                "worker_lifecycle_id": lifecycle_id,
                                "attempt": attempt,
                                "frames": len(frames),
                                "valid": False,
                                "saturated": saturated,
                                "generated_tokens": output_units,
                                "generation_budget_tokens": generation_budget,
                                "sha256": digest,
                                "error_type": type(exc).__name__,
                            },
                            sort_keys=True,
                        )
                    )
                    if saturated:
                        available_output = int(capacity["available_output_tokens"])
                        if not capacity_expansion_used and generation_budget < available_output:
                            capacity_expansion_used = True
                            minimum_budget = min(
                                available_output,
                                max(generation_budget + 1, generation_budget * 2),
                            )
                            print(
                                json.dumps(
                                    {
                                        "event": "vision_generation_capacity_expand",
                                        "worker_lifecycle_id": lifecycle_id,
                                        "frames": len(frames),
                                        "previous_budget_tokens": generation_budget,
                                        "next_minimum_budget_tokens": minimum_budget,
                                        "available_output_tokens": available_output,
                                    },
                                    sort_keys=True,
                                )
                            )
                            continue
                        raise VisionOutputCapacityError(
                            "vision output capacity exhausted: "
                            f"task={payload.get('task')!s} frames={len(frames)} "
                            f"generated_tokens={output_units} "
                            f"generation_budget_tokens={generation_budget} "
                            f"available_output_tokens={available_output} "
                            f"context_limit_tokens={capacity['context_limit_tokens']}"
                        ) from exc
                    if not format_recovery_used:
                        format_recovery_used = True
                        continue
                    raise ValueError(
                        "vision model did not return valid JSON after format recovery: "
                        f"task={payload.get('task')!s} attempts={attempt} "
                        f"tokens={output_units} chars={len(generated_text)} "
                        f"sha256={digest}"
                    ) from exc

                cardinality = int(capacity["cardinality"])
                generation_runtime = {
                    **capacity,
                    "output_tokens": output_units,
                    "output_tokens_per_item": output_units / cardinality,
                    "budget_utilization": output_units / generation_budget,
                    "capacity_expansion_used": capacity_expansion_used,
                    "format_recovery_used": format_recovery_used,
                }
                print(
                    json.dumps(
                        {
                            "event": "vision_json_validation",
                            "worker_lifecycle_id": lifecycle_id,
                            "attempt": attempt,
                            "frames": len(frames),
                            "valid": True,
                            "generated_tokens": output_units,
                            "generation_budget_tokens": generation_budget,
                        },
                        sort_keys=True,
                    )
                )
                return {
                    "value": value,
                    "model": _model_evidence(model_id),
                    "usage": _usage(
                        started,
                        gpu,
                        input_units=total_input_units,
                        output_units=total_output_units,
                    ),
                    "runtime": _worker_runtime(
                        lifecycle_id,
                        model_load_count=1,
                        batch_frame_count=len(frames),
                        generation_capacity=generation_runtime,
                    ),
                }
            finally:
                inputs = None
                output = None
                generated = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        frames.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_vision_worker(worker: Any, model_id: str) -> None:
    worker.lifecycle_id = uuid.uuid4().hex
    worker.processor, worker.model = _load_vision_model(model_id)
    print(
        json.dumps(
            {
                "event": "vision_model_ready",
                "worker_lifecycle_id": worker.lifecycle_id,
                "model": _model_evidence(model_id),
                "cuda_memory_by_device": _cuda_memory_snapshot(),
            },
            sort_keys=True,
        )
    )


def _vision_worker_ready(worker: Any, model_id: str) -> dict[str, Any]:
    return {
        "value": {"ready": True},
        "model": _model_evidence(model_id),
        "runtime": _worker_runtime(worker.lifecycle_id, model_load_count=1),
    }


def _vision_worker_inspect(
    worker: Any,
    payload: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    try:
        _assert_expected_git_sha(payload)
        return _vision_infer(
            payload,
            model_id,
            "L4:2",
            worker.processor,
            worker.model,
            lifecycle_id=worker.lifecycle_id,
        )
    except Exception as exc:
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                print(
                    json.dumps(
                        {
                            "event": "vision_inference_error",
                            "worker_lifecycle_id": worker.lifecycle_id,
                            "model_id": model_id,
                            "error_type": type(exc).__name__,
                            "cuda_memory_by_device": _cuda_memory_snapshot(),
                        },
                        sort_keys=True,
                    )
                )
                torch.cuda.empty_cache()
        return _transport_error(
            exc,
            context=(
                f"task={payload.get('task') or '<missing>'} "
                f"frames={len(payload.get('frames_base64') or [])}"
            ),
        )


@app.cls(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
)
class VisionModel:
    @modal.enter()
    def load_model(self) -> None:
        _load_vision_worker(self, "Qwen/Qwen3-VL-8B-Instruct")

    @modal.method()
    def ready(self) -> dict[str, Any]:
        return _vision_worker_ready(self, "Qwen/Qwen3-VL-8B-Instruct")

    @modal.method()
    def inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _vision_worker_inspect(self, payload, "Qwen/Qwen3-VL-8B-Instruct")


@app.cls(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
)
class VisionModelLarge:
    @modal.enter()
    def load_model(self) -> None:
        _load_vision_worker(self, "Qwen/Qwen3-VL-30B-A3B-Instruct")

    @modal.method()
    def ready(self) -> dict[str, Any]:
        return _vision_worker_ready(self, "Qwen/Qwen3-VL-30B-A3B-Instruct")

    @modal.method()
    def inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _vision_worker_inspect(
            self,
            payload,
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        )


@app.function(
    image=speech_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache, MEDIA_ROOT: media_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=24576,
)
def transcribe(payload: dict[str, Any]) -> dict[str, Any]:
    global _whisper_model
    _assert_expected_git_sha(payload)
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    source_path = str(payload["source_path"])
    if not source_path.startswith(f"{MEDIA_ROOT}/"):
        raise ValueError("source_path must be mounted media")
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )
    segments, _info = _whisper_model.transcribe(
        source_path,
        word_timestamps=True,
        vad_filter=True,
    )
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.words or ():
            if word.start is None or word.end is None or not word.word.strip():
                continue
            words.append(
                {
                    "text": word.word.strip(),
                    "start": float(word.start),
                    "end": float(word.end),
                    "confidence": (
                        float(word.probability) if word.probability is not None else None
                    ),
                }
            )
    if not words:
        raise ValueError("faster-whisper produced no timestamped words")
    return {
        "words": words,
        "model": _model_evidence("mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
        "usage": _usage(started, "L4", output_units=len(words)),
    }


@app.function(
    image=speech_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache, MEDIA_ROOT: media_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=24576,
    scaledown_window=2,
)
def align(payload: dict[str, Any]) -> dict[str, Any]:
    _assert_expected_git_sha(payload)
    import whisperx

    started = time.perf_counter()
    source_path = str(payload["source_path"])
    timeline = payload.get("timeline")
    if not isinstance(timeline, dict) or not isinstance(timeline.get("words"), list):
        raise ValueError("alignment requires canonical timeline")
    audio = whisperx.load_audio(source_path)
    model, metadata = whisperx.load_align_model(language_code="en", device="cuda")
    raw_words = timeline["words"]
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in raw_words:
        if not isinstance(word, dict):
            continue
        current.append(word)
        if len(current) >= 28:
            segments.append(
                {
                    "start": float(current[0]["source_start"]),
                    "end": float(current[-1]["source_end"]),
                    "text": " ".join(str(item["text"]) for item in current),
                }
            )
            current = []
    if current:
        segments.append(
            {
                "start": float(current[0]["source_start"]),
                "end": float(current[-1]["source_end"]),
                "text": " ".join(str(item["text"]) for item in current),
            }
        )
    aligned = whisperx.align(segments, model, metadata, audio, "cuda", return_char_alignments=False)
    raw_segments = aligned.get("segments") if isinstance(aligned, dict) else None
    if not isinstance(raw_segments, list):
        raise ValueError("WhisperX returned no aligned segments")
    return {
        "segments": raw_segments,
        "usage": _usage(started, "L4", output_units=len(raw_segments)),
    }


@app.function(
    image=speech_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache, MEDIA_ROOT: media_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=24576,
    scaledown_window=2,
)
def diarize(payload: dict[str, Any]) -> dict[str, Any]:
    global _diarization_pipeline
    _assert_expected_git_sha(payload)
    cleanup_source: Path | None = None
    try:
        import torch
        from pyannote.audio import Pipeline

        started = time.perf_counter()
        source_path = str(payload["source_path"])
        if not source_path.startswith(f"{MEDIA_ROOT}/"):
            raise ValueError("source_path must be mounted media")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for pyannote community-1")
        if _diarization_pipeline is None:
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=token
            )
            _diarization_pipeline.to(torch.device("cuda"))
        diarization_source, should_cleanup = _diarization_audio_source(source_path)
        cleanup_source = diarization_source if should_cleanup else None
        result = _diarization_pipeline(str(diarization_source))
        annotation = getattr(result, "speaker_diarization", result)
        turns = [
            [float(segment.start), float(segment.end), str(speaker)]
            for segment, _track, speaker in annotation.itertracks(yield_label=True)
            if segment.end > segment.start
        ]
        if not turns:
            raise ValueError("pyannote produced no speaker turns")
        return {
            "turns": turns,
            "model": _model_evidence("pyannote/speaker-diarization-community-1"),
            "usage": _usage(started, "L4", output_units=len(turns)),
        }
    except Exception as exc:
        return _transport_error(exc)
    finally:
        if cleanup_source is not None:
            cleanup_source.unlink(missing_ok=True)


@app.function(
    image=modal.Image.debian_slim().env({"CLIPPER_DEPLOYED_GIT_SHA": DEPLOYED_GIT_SHA}),
    timeout=60,
    scaledown_window=2,
)
def deployment_identity() -> dict[str, Any]:
    return {"app": APP_NAME, "deployed_git_sha": DEPLOYED_GIT_SHA}


@app.function(image=modal.Image.debian_slim(), timeout=120, scaledown_window=2)
def credential_smoke() -> dict[str, Any]:
    return {"app": APP_NAME, "ok": True}


@app.function(image=text_image, timeout=180, memory=4096, scaledown_window=2)
def editorial_schema_smoke() -> dict[str, Any]:
    from clipper.providers.editorial_prompt import editorial_json_schema
    from outlines.types import JsonSchema

    tasks = [
        "source_hazards:smoke",
        "semantic_cores:smoke",
        "narrative_envelope:smoke",
        "quality_windows:smoke",
    ]
    for task in tasks:
        JsonSchema(editorial_json_schema(task))
    return {
        "app": APP_NAME,
        "ok": True,
        "engine": "outlines-transformers",
        "task_families": len(tasks),
    }


@app.function(
    image=base_image.uv_pip_install("huggingface_hub>=0.35,<2"),
    secrets=[hf_secret],
    timeout=180,
    scaledown_window=2,
)
def hf_access_smoke() -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not available inside Modal")
    model_id = "pyannote/speaker-diarization-community-1"
    info = HfApi(token=token).model_info(model_id)
    hf_hub_download(repo_id=model_id, filename="config.yaml", token=token)
    return {"ok": True, "model_id": info.id, "revision": info.sha, "gated_file": "config.yaml"}
