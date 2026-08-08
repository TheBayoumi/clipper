from __future__ import annotations

import base64
import io
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

import modal

APP_NAME = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
HF_CACHE = "/model-cache"
MEDIA_ROOT = "/media"
L4_USD_PER_SECOND = 0.000222
L40S_USD_PER_SECOND = 0.000542

model_cache = modal.Volume.from_name("clipper-hf-cache", create_if_missing=True)
media_cache = modal.Volume.from_name("clipper-media-cache", create_if_missing=True)
if modal.is_local():
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
else:
    hf_secret = modal.Secret.from_dict({})

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .env({"HF_HOME": HF_CACHE, "TRANSFORMERS_CACHE": HF_CACHE})
)
text_image = base_image.uv_pip_install(
    "torch>=2.8,<3",
    "transformers>=5.14,<6",
    "accelerate>=1.14,<2",
    "sentence-transformers>=5.6,<6",
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


def _usage(
    started: float, gpu: str, *, input_units: int = 0, output_units: int = 0
) -> dict[str, Any]:
    import torch

    duration = max(0.0, time.perf_counter() - started)
    rate = L40S_USD_PER_SECOND if gpu == "L40S" else L4_USD_PER_SECOND
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


def _json_text(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


@app.function(
    image=text_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1200,
    memory=16384,
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
    gpu="L4",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
)
def editorial(payload: dict[str, Any]) -> dict[str, Any]:
    global _editorial_model, _editorial_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.perf_counter()
    if _editorial_model is None or _editorial_tokenizer is None:
        model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        _editorial_tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        editorial_offload = f"{HF_CACHE}/offload/editorial"
        os.makedirs(editorial_offload, exist_ok=True)
        _editorial_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            max_memory={0: "20GiB", "cpu": "28GiB"},
            offload_folder=editorial_offload,
            offload_state_dict=True,
            dtype=torch.bfloat16,
            quantization_config=quantization,
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Return only valid JSON. Spoken content must reference canonical word IDs. "
                "Never invent spoken words."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    rendered = _editorial_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _editorial_tokenizer(rendered, return_tensors="pt").to(_editorial_model.device)
    output = _editorial_model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    text = _editorial_tokenizer.decode(generated, skip_special_tokens=True)
    value = _json_text(text)
    return {
        "value": value,
        "usage": _usage(
            started,
            "L4",
            input_units=int(inputs["input_ids"].numel()),
            output_units=int(generated.numel()),
        ),
    }


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
                llm_int8_enable_fp32_cpu_offload=True,
            )
            kwargs["max_memory"] = {0: "20GiB", "cpu": "28GiB"}
            vlm_offload = f"{HF_CACHE}/offload/vision-large"
            os.makedirs(vlm_offload, exist_ok=True)
            kwargs["offload_folder"] = vlm_offload
            kwargs["offload_state_dict"] = True
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
)
def vision(payload: dict[str, Any]) -> dict[str, Any]:
    return _vision_infer(payload, "Qwen/Qwen3-VL-8B-Instruct", "L4")


@app.function(
    image=text_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
)
def vision_large(payload: dict[str, Any]) -> dict[str, Any]:
    return _vision_infer(payload, "Qwen/Qwen3-VL-30B-A3B-Instruct", "L4")


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
    return {"words": words, "usage": _usage(started, "L4", output_units=len(words))}


@app.function(
    image=speech_image,
    gpu="L4",
    volumes={HF_CACHE: model_cache, MEDIA_ROOT: media_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=24576,
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
)
def diarize(payload: dict[str, Any]) -> dict[str, Any]:
    global _diarization_pipeline
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
    return {"turns": turns, "usage": _usage(started, "L4", output_units=len(turns))}


@app.function(image=modal.Image.debian_slim(), timeout=120)
def credential_smoke() -> dict[str, Any]:
    return {"app": APP_NAME, "ok": True}


@app.function(
    image=base_image.uv_pip_install("huggingface_hub>=0.35,<2"),
    secrets=[hf_secret],
    timeout=180,
)
def hf_access_smoke() -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not available inside Modal")
    info = HfApi(token=token).model_info("pyannote/speaker-diarization-community-1")
    return {"ok": True, "model_id": info.id}
