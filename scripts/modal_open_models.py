from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
HF_CACHE = "/model-cache"
MEDIA_ROOT = "/media"
L4_USD_PER_SECOND = 0.000222
L40S_USD_PER_SECOND = 0.000542
EDITORIAL_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EDITORIAL_MODEL_REVISION = "110954009be4a882781a90356c7d2b8a9e3428dc"

model_cache = modal.Volume.from_name("clipper-hf-cache", create_if_missing=True)
media_cache = modal.Volume.from_name("clipper-media-cache", create_if_missing=True)
if modal.is_local():
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
else:
    hf_secret = modal.Secret.from_dict({})

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "TRANSFORMERS_CACHE": HF_CACHE,
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
    "sentence-transformers>=5.6,<6",
    "sentencepiece>=0.2,<1",
    "tiktoken>=0.11,<1",
    "bitsandbytes>=0.47,<1",
    "pillow>=11,<13",
)
speech_image = base_image.uv_pip_install(
    "torch>=2.8,<3",
    "faster-whisper>=1.2.1,<2",
    "whisperx>=3.8.6,<4",
    "pyannote.audio>=4.0.7,<5",
)

app = modal.App(APP_NAME)
_embedding_model: Any | None = None
_editorial_tokenizer: Any | None = None
_editorial_model: Any | None = None
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


def _usage(
    started: float, gpu: str, *, input_units: int = 0, output_units: int = 0
) -> dict[str, Any]:
    import torch

    duration = max(0.0, time.perf_counter() - started)
    rate = _gpu_rate(gpu)
    peak = (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available()
        else None
    )
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "gpu_type": gpu,
        "gpu_seconds": duration,
        "peak_vram_mb": peak,
        "input_units": input_units,
        "output_units": output_units,
        "estimated_cost_usd": duration * rate,
    }


class EditorialOutputTruncated(ValueError):
    """Editorial generation exhausted its output budget before valid JSON completed."""


def _json_text(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


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
    task = str(payload.get("task") or "")
    if task == "episode_editorial_profile" or task == "global_concept_comparison":
        base_budget = 1024
    elif task.startswith("story_moments:") or task.startswith("hook_variants:"):
        base_budget = 1536
    else:
        base_budget = 2048
    return min(4096, base_budget * _editorial_recovery_attempt(payload))


def _editorial_contract(task: str) -> str:
    common = (
        "Output exactly one compact JSON object, no markdown and no extra keys. "
        "Keep prose fields concise. For range fields copy the supplied short word_ref values, "
        "never reconstruct or abbreviate word_id values yourself. "
    )
    if task == "episode_editorial_profile":
        return common + (
            'Schema: {"summary":"<=60 words",'
            '"valuable_moment_characteristics":["3-5 short strings"],'
            '"avoid_characteristics":["0-4 short strings"],"confidence":0.0}. '
        )
    if task.startswith("story_moments:"):
        return common + (
            'Schema: {"moments":[{"moment_id":"unique",'
            '"start_word_id":"first word_ref","end_word_id":"last word_ref",'
            '"semantic_summary":"<=24 words","narrative_structure":"short label",'
            '"required_prior_context":"<=16 words or empty",'
            '"required_followup_context":"<=16 words or empty",'
            '"editorial_reason":"<=20 words","confidence":0.0}]}. '
            "Return at most 8 non-overlapping meaningful moments. Do not copy full word-ID lists. "
        )
    if task == "clip_concepts":
        return common + (
            'Schema: {"concepts":[{"concept_id":"unique","story_moment_ids":["ids"],'
            '"start_word_id":"first word_ref","end_word_id":"last word_ref",'
            '"semantic_summary":"<=24 words","standalone_context":"<=16 words or empty",'
            '"narrative_structure":"short label","recommended_duration":20.0,'
            '"visual_dependencies":["short labels"],"confidence":0.0}]}. '
            "Return at most 12 materially distinct contiguous concepts. "
            "Do not copy full word-ID lists. "
        )
    if task == "global_concept_comparison":
        return common + 'Schema: {"concept_ids":["best-first supplied concept IDs"]}. '
    if task.startswith("hook_variants:"):
        return common + (
            'Schema: {"variants":[{"variant_id":"unique","strategy_label":"<=8 words",'
            '"source_start_word_id":"first word_ref","source_end_word_id":"last word_ref",'
            '"overlay_text":null,"rationale":"<=16 words","confidence":0.0}]}. '
            "Return at most 4 materially different truthful hooks. Do not copy full word-ID lists. "
        )
    if task.startswith("edit_plans:"):
        return common + (
            'Schema: {"plans":[{"plan_id":"unique","video_id":"supplied video ID",'
            '"concept_id":"supplied concept ID","variant_id":"supplied hook ID",'
            '"source_start_word_id":"first edit word_ref",'
            '"source_end_word_id":"last edit word_ref",'
            '"hook_start_word_id":"first hook word_ref",'
            '"hook_end_word_id":"last hook word_ref",'
            '"overlay_text":null,"strategy_label":"<=8 words",'
            '"caption_platform":"tiktok","confidence":0.0}]}. '
            "Return at most 4 contiguous chronological plans. Do not copy full word-ID lists. "
        )
    return common + "Follow the task payload exactly."


def _transport_error(exc: Exception, *, context: str | None = None) -> dict[str, Any]:
    message = str(exc)
    if context:
        message = f"{context}: {message}"
    return {"error": {"type": type(exc).__name__, "message": message}}


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
    gpu="L4",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1200,
    memory=16384,
    scaledown_window=2,
)
def embedding(payload: dict[str, Any]) -> dict[str, Any]:
    global _embedding_model
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    texts = payload.get("texts")
    if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
        raise ValueError("embedding payload requires string texts")
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    vectors = _embedding_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    value = [[float(item) for item in row] for row in vectors]
    return {
        "vectors": value,
        "usage": _usage(started, "L4", input_units=len(texts), output_units=len(value)),
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
    global _editorial_model, _editorial_tokenizer
    task = str(payload.get("task") or "")
    recovery_attempt = _editorial_recovery_attempt(payload)
    try:
        import torch
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
        system_content = (
            "You are a source-grounded podcast editor. Never invent spoken words or IDs. "
            + _editorial_contract(task)
        )
        if recovery_attempt > 1:
            system_content += (
                " This is a JSON recovery generation after an invalid or truncated response. "
                "Return the complete JSON object and close every string, array, and object. "
                "If needed, return fewer valid items with shorter prose rather than truncating."
            )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        rendered = _editorial_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _editorial_tokenizer(rendered, return_tensors="pt").to(_editorial_model.device)
        output_budget = _editorial_output_budget(payload)
        output = _editorial_model.generate(
            **inputs,
            max_new_tokens=output_budget,
            do_sample=False,
            use_cache=True,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = _editorial_tokenizer.decode(generated, skip_special_tokens=True)
        try:
            value = _json_text(text)
        except json.JSONDecodeError as exc:
            if int(generated.numel()) >= output_budget:
                raise EditorialOutputTruncated(
                    f"task={task} attempt={recovery_attempt} exhausted "
                    f"max_new_tokens={output_budget}: {exc}"
                ) from exc
            raise
        return {
            "value": value,
            "model": _model_evidence(EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION),
            "usage": _usage(
                started,
                "L4:2",
                input_units=int(inputs["input_ids"].numel()),
                output_units=int(generated.numel()),
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
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        _vision_models[model_id] = (processor, model)
    else:
        processor, model = cached
    content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
    content.append(
        {
            "type": "text",
            "text": "Return only valid JSON. Do not retranscribe audio. "
            + json.dumps(
                {"task": payload.get("task"), "context": payload.get("context")}, ensure_ascii=False
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(generated, skip_special_tokens=True)
    return {
        "value": _json_text(text),
        "model": _model_evidence(model_id),
        "usage": _usage(
            started,
            gpu,
            input_units=int(inputs["input_ids"].numel()),
            output_units=int(generated.numel()),
        ),
    }


@app.function(
    image=text_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1200,
    memory=24576,
    scaledown_window=2,
)
def vision(payload: dict[str, Any]) -> dict[str, Any]:
    return _vision_infer(payload, "Qwen/Qwen3-VL-8B-Instruct", "L4")


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
    try:
        import torch
        from pyannote.audio import Pipeline

        started = time.perf_counter()
        source_path = str(payload["source_path"])
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for pyannote community-1")
        if _diarization_pipeline is None:
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=token
            )
            _diarization_pipeline.to(torch.device("cuda"))
        result = _diarization_pipeline(source_path)
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


@app.function(image=modal.Image.debian_slim(), timeout=120, scaledown_window=2)
def credential_smoke() -> dict[str, Any]:
    return {"app": APP_NAME, "ok": True}


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
