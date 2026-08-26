from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
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
VISION_MAX_PIXELS_PER_FRAME = 512 * 28 * 28
VISION_MAX_NEW_TOKENS = 2048
VISION_FALLBACK_CONTEXT_LIMIT = 262_144
VISION_MAX_ATTEMPTS = 2

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
_editorial_tokenizer: Any | None = None
_editorial_model: Any | None = None
_editorial_structured_model: Any | None = None
_vision_models: dict[str, tuple[Any, Any]] = {}
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
    """Editorial generation exhausted its output budget before valid JSON completed."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _editorial_recovery_attempt(payload: dict[str, Any]) -> int:
    raw = payload.get("generation_recovery_attempt")
    try:
        attempt = int(raw) if raw is not None else 1
    except (TypeError, ValueError):
        attempt = 1
    return max(1, min(attempt, 3))


def _editorial_output_budget(payload: dict[str, Any]) -> int:
    from clipper.providers.editorial_prompt import editorial_output_budget

    base_budget = editorial_output_budget(payload)
    return min(4096, base_budget * _editorial_recovery_attempt(payload))


def _transport_error(exc: Exception, *, context: str | None = None) -> dict[str, Any]:
    traceback.print_exception(exc)
    message = str(exc)
    if context:
        message = f"{context}: {message}"
    return {"error": {"type": type(exc).__name__, "message": message}}


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


@app.function(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
    scaledown_window=2,
)
def editorial(payload: dict[str, Any]) -> dict[str, Any]:
    from clipper.providers.editorial_prompt import editorial_contract, editorial_json_schema

    global _editorial_model, _editorial_structured_model, _editorial_tokenizer
    task = str(payload.get("task") or "")
    recovery_attempt = _editorial_recovery_attempt(payload)
    try:
        import torch
        from outlines import from_transformers
        from outlines.types import JsonSchema
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        started = time.perf_counter()
        if _editorial_model is None or _editorial_tokenizer is None:
            _editorial_tokenizer = AutoTokenizer.from_pretrained(
                EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION
            )
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            _editorial_model = AutoModelForCausalLM.from_pretrained(
                EDITORIAL_MODEL_ID,
                revision=EDITORIAL_MODEL_REVISION,
                device_map="auto",
                dtype=torch.bfloat16,
                quantization_config=quantization,
                max_memory={0: "22GiB", 1: "22GiB"},
                low_cpu_mem_usage=True,
            )
            _editorial_structured_model = None

        if _editorial_structured_model is None:
            _editorial_structured_model = from_transformers(
                _editorial_model,
                _editorial_tokenizer,
            )

        system_content = (
            "You are a source-grounded multimodal short-form editor. "
            "Never invent source evidence, spoken words, timestamps, or IDs. "
            + editorial_contract(task)
        )
        if recovery_attempt > 1:
            system_content += (
                " This is a constrained recovery generation after a previous invalid, truncated, "
                "or semantically rejected response. Return a complete object satisfying the JSON "
                "Schema. If needed, return fewer valid items with shorter prose rather than "
                "exhausting the output budget."
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        rendered = _editorial_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        schema = JsonSchema(editorial_json_schema(task))
        output_budget = _editorial_output_budget(payload)

        input_ids = _editorial_tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]
        input_units = len(input_ids)

        text = _editorial_structured_model(
            rendered,
            schema,
            max_new_tokens=output_budget,
            do_sample=False,
            use_cache=True,
        )
        if not isinstance(text, str):
            raise TypeError(
                "Outlines transformers generation returned a non-string response: "
                f"{type(text).__name__}"
            )

        output_ids = _editorial_tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        output_units = len(output_ids)

        try:
            value = _json_text(text)
        except json.JSONDecodeError as exc:
            # Outlines constrains syntax during decoding, so reaching this branch generally
            # means generation was cut off by the output budget or an upstream library error.
            if output_units >= output_budget:
                raise EditorialOutputTruncated(
                    f"task={task} attempt={recovery_attempt} exhausted "
                    f"max_new_tokens={output_budget}: {exc}"
                ) from exc
            raise RuntimeError(
                "constrained editorial generation returned invalid JSON despite Outlines "
                f"schema enforcement for task={task}: {exc}"
            ) from exc

        return {
            "value": value,
            "model": _model_evidence(EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION),
            "structured_generation": {
                "engine": "outlines-transformers",
                "schema_version": "editorial-json-v2",
                "constrained": True,
                "recovery_attempt": recovery_attempt,
            },
            "usage": _usage(
                started,
                "L4:2",
                input_units=input_units,
                output_units=output_units,
            ),
        }
    except Exception as exc:
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return _transport_error(
            exc, context=f"task={task or '<missing>'} attempt={recovery_attempt}"
        )


def _vision_infer(payload: dict[str, Any], model_id: str, gpu: str) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    started = time.perf_counter()
    raw_frames = payload.get("frames_base64")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("vision payload requires frames_base64")
    frames = [Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB") for item in raw_frames]
    cached = _vision_models.get(model_id)
    if cached is None:
        processor = AutoProcessor.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"device_map": "auto", "dtype": torch.bfloat16}
        if "30B" in model_id:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["max_memory"] = {0: "22GiB", 1: "22GiB"}
        else:
            # Split the dense 8B weights so neither L4 is crowded before visual input arrives.
            kwargs["device_map"] = "balanced"
            kwargs["max_memory"] = {0: "20GiB", 1: "20GiB"}
        processor.image_processor.size["longest_edge"] = VISION_MAX_PIXELS_PER_FRAME
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        _vision_models[model_id] = (processor, model)
    else:
        processor, model = cached
    total_input_units = 0
    total_output_units = 0
    for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
        content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
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
        text_config = getattr(model.config, "text_config", model.config)
        context_limit = int(
            getattr(text_config, "max_position_embeddings", VISION_FALLBACK_CONTEXT_LIMIT)
        )
        if input_units + VISION_MAX_NEW_TOKENS > context_limit:
            raise ValueError(
                "vision request exceeds model context after pixel bounding: "
                f"input_tokens={input_units} output_reserve={VISION_MAX_NEW_TOKENS} "
                f"context_limit={context_limit} frames={len(frames)}"
            )
        total_input_units += input_units
        inputs = inputs.to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=VISION_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        output_units = int(generated.numel())
        total_output_units += output_units
        text = processor.decode(generated, skip_special_tokens=True)
        try:
            value = _json_text(text)
        except (json.JSONDecodeError, ValueError) as exc:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            print(
                "vision JSON validation failed: "
                f"task={payload.get('task')!s} attempt={attempt} tokens={output_units} "
                f"chars={len(text)} sha256={digest} error={type(exc).__name__}: {exc}"
            )
            if attempt < VISION_MAX_ATTEMPTS:
                del output, generated, inputs
                torch.cuda.empty_cache()
                continue
            raise ValueError(
                "vision model did not return valid JSON after recovery: "
                f"task={payload.get('task')!s} attempts={attempt} tokens={output_units} "
                f"chars={len(text)} sha256={digest}"
            ) from exc
        return {
            "value": value,
            "model": _model_evidence(model_id),
            "usage": _usage(
                started,
                gpu,
                input_units=total_input_units,
                output_units=total_output_units,
            ),
        }
    raise AssertionError("vision recovery loop exhausted without returning")


@app.function(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1200,
    memory=24576,
    scaledown_window=2,
)
def vision(payload: dict[str, Any]) -> dict[str, Any]:
    return _vision_infer(payload, "Qwen/Qwen3-VL-8B-Instruct", "L4:2")


@app.function(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
    scaledown_window=2,
)
def vision_large(payload: dict[str, Any]) -> dict[str, Any]:
    return _vision_infer(payload, "Qwen/Qwen3-VL-30B-A3B-Instruct", "L4:2")


@app.function(
    image=speech_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache, MEDIA_ROOT: media_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=24576,
    scaledown_window=2,
)
def transcribe(payload: dict[str, Any]) -> dict[str, Any]:
    global _whisper_model
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    source_path = str(payload["source_path"])
    if not source_path.startswith(f"{MEDIA_ROOT}/"):
        raise ValueError("source_path must be mounted media")
    if _whisper_model is None:
        _whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    segments, _info = _whisper_model.transcribe(source_path, word_timestamps=True, vad_filter=True)
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
                    "confidence": float(word.probability) if word.probability is not None else None,
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
