from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_region(text: str, start: str, end: str, replacement: str, label: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeError(f"{label}: start marker missing")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise RuntimeError(f"{label}: end marker missing")
    return text[:start_index] + replacement + text[end_index:]


def patch_modal_workers() -> None:
    path = "scripts/modal_open_models.py"
    text = read(path)
    text = replace_once(text, "import contextlib\n", "import contextlib\nimport gc\n", "gc import")
    text = replace_once(text, "import traceback\n", "import traceback\nimport uuid\n", "uuid import")
    text = replace_once(
        text,
        "VISION_MAX_PIXELS_PER_FRAME = 512 * 28 * 28\n",
        "",
        "fixed vision pixel cap",
    )
    text = replace_once(
        text,
        """_editorial_tokenizer: Any | None = None
_editorial_model: Any | None = None
_editorial_structured_model: Any | None = None
_vision_models: dict[str, tuple[Any, Any]] = {}
""",
        "",
        "process-local editorial/vision caches",
    )

    editorial_marker = text.rfind("@app.function(", 0, text.index("def editorial("))
    speech_marker = text.find("@app.function(", text.index("def transcribe(") - 200)
    if editorial_marker < 0 or speech_marker <= editorial_marker:
        raise RuntimeError("model worker region markers not found")

    replacement = '''def _cuda_memory_snapshot() -> dict[str, dict[str, float]]:
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
            "peak_allocated_mb": float(
                torch.cuda.max_memory_allocated(index) / (1024 * 1024)
            ),
        }
    return snapshot


def _worker_runtime(
    lifecycle_id: str,
    *,
    model_load_count: int,
    batch_frame_count: int | None = None,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "worker_lifecycle_id": lifecycle_id,
        "model_load_count": model_load_count,
        "cuda_memory_by_device": _cuda_memory_snapshot(),
    }
    if batch_frame_count is not None:
        runtime["batch_frame_count"] = batch_frame_count
    return runtime


def _load_editorial_model() -> tuple[Any, Any, Any]:
    import torch
    from outlines import from_transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        EDITORIAL_MODEL_ID,
        revision=EDITORIAL_MODEL_REVISION,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model, from_transformers(model, tokenizer)


def _editorial_infer(
    payload: dict[str, Any],
    tokenizer: Any,
    structured_model: Any,
    *,
    lifecycle_id: str,
) -> dict[str, Any]:
    import torch
    from outlines.types import JsonSchema

    from clipper.providers.editorial_prompt import editorial_contract, editorial_json_schema

    started = time.perf_counter()
    task = str(payload.get("task") or "")
    recovery_attempt = _editorial_recovery_attempt(payload)
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
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    schema = JsonSchema(editorial_json_schema(task))
    output_budget = _editorial_output_budget(payload)
    input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    input_units = len(input_ids)
    try:
        with torch.inference_mode():
            generated_text = structured_model(
                rendered,
                schema,
                max_new_tokens=output_budget,
                do_sample=False,
                use_cache=True,
            )
        if not isinstance(generated_text, str):
            raise TypeError(
                "Outlines transformers generation returned a non-string response: "
                f"{type(generated_text).__name__}"
            )
        output_ids = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
        output_units = len(output_ids)
        try:
            value = _json_text(generated_text)
        except json.JSONDecodeError as exc:
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
            "model": _model_evidence(
                EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION
            ),
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
            "runtime": _worker_runtime(lifecycle_id, model_load_count=1),
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.cls(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
)
class EditorialModel:
    @modal.enter()
    def load_model(self) -> None:
        self.lifecycle_id = uuid.uuid4().hex
        self.tokenizer, self.model, self.structured_model = _load_editorial_model()
        print(
            json.dumps(
                {
                    "event": "editorial_model_ready",
                    "worker_lifecycle_id": self.lifecycle_id,
                    "model": _model_evidence(
                        EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION
                    ),
                    "cuda_memory_by_device": _cuda_memory_snapshot(),
                },
                sort_keys=True,
            )
        )

    @modal.method()
    def ready(self) -> dict[str, Any]:
        return {
            "value": {"ready": True},
            "model": _model_evidence(
                EDITORIAL_MODEL_ID, revision=EDITORIAL_MODEL_REVISION
            ),
            "runtime": _worker_runtime(self.lifecycle_id, model_load_count=1),
        }

    @modal.method()
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task") or "")
        recovery_attempt = _editorial_recovery_attempt(payload)
        try:
            return _editorial_infer(
                payload,
                self.tokenizer,
                self.structured_model,
                lifecycle_id=self.lifecycle_id,
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return _transport_error(
                exc, context=f"task={task or '<missing>'} attempt={recovery_attempt}"
            )


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
    frames = [
        Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB")
        for item in raw_frames
    ]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)

    total_input_units = 0
    total_output_units = 0
    try:
        for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
            inputs: Any | None = None
            output: Any | None = None
            generated: Any | None = None
            try:
                content: list[dict[str, Any]] = [
                    {"type": "image", "image": frame} for frame in frames
                ]
                content.append(
                    {"type": "text", "text": _vision_prompt(payload, attempt)}
                )
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
                    getattr(
                        text_config,
                        "max_position_embeddings",
                        VISION_FALLBACK_CONTEXT_LIMIT,
                    )
                )
                if input_units + VISION_MAX_NEW_TOKENS > context_limit:
                    raise ValueError(
                        "vision request exceeds model context: "
                        f"input_tokens={input_units} "
                        f"output_reserve={VISION_MAX_NEW_TOKENS} "
                        f"context_limit={context_limit} frames={len(frames)}"
                    )
                total_input_units += input_units
                inputs = inputs.to(model.device)
                with torch.inference_mode():
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
                generated_text = processor.decode(generated, skip_special_tokens=True)
                try:
                    value = _json_text(generated_text)
                except (json.JSONDecodeError, ValueError) as exc:
                    digest = hashlib.sha256(
                        generated_text.encode("utf-8")
                    ).hexdigest()[:16]
                    print(
                        "vision JSON validation failed: "
                        f"task={payload.get('task')!s} attempt={attempt} "
                        f"tokens={output_units} chars={len(generated_text)} "
                        f"sha256={digest} error={type(exc).__name__}: {exc}"
                    )
                    if attempt < VISION_MAX_ATTEMPTS:
                        continue
                    raise ValueError(
                        "vision model did not return valid JSON after recovery: "
                        f"task={payload.get('task')!s} attempts={attempt} "
                        f"tokens={output_units} chars={len(generated_text)} "
                        f"sha256={digest}"
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
                    "runtime": _worker_runtime(
                        lifecycle_id,
                        model_load_count=1,
                        batch_frame_count=len(frames),
                    ),
                }
            finally:
                inputs = None
                output = None
                generated = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise AssertionError("vision recovery loop exhausted without returning")
    finally:
        frames.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.cls(
    image=text_image,
    gpu="L4:2",
    volumes={HF_CACHE: model_cache},
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
)
class VisionModel:
    model_id: str = modal.parameter()

    @modal.enter()
    def load_model(self) -> None:
        if not self.model_id.strip():
            raise ValueError("vision model_id cannot be empty")
        self.lifecycle_id = uuid.uuid4().hex
        self.processor, self.model = _load_vision_model(self.model_id)
        print(
            json.dumps(
                {
                    "event": "vision_model_ready",
                    "worker_lifecycle_id": self.lifecycle_id,
                    "model": _model_evidence(self.model_id),
                    "cuda_memory_by_device": _cuda_memory_snapshot(),
                },
                sort_keys=True,
            )
        )

    @modal.method()
    def ready(self) -> dict[str, Any]:
        return {
            "value": {"ready": True},
            "model": _model_evidence(self.model_id),
            "runtime": _worker_runtime(self.lifecycle_id, model_load_count=1),
        }

    @modal.method()
    def inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return _vision_infer(
                payload,
                self.model_id,
                "L4:2",
                self.processor,
                self.model,
                lifecycle_id=self.lifecycle_id,
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                import torch

                if torch.cuda.is_available():
                    print(
                        json.dumps(
                            {
                                "event": "vision_inference_error",
                                "worker_lifecycle_id": self.lifecycle_id,
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


'''
    text = text[:editorial_marker] + replacement + text[speech_marker:]
    write(path, text)


def patch_provider_base() -> None:
    path = "src/clipper/providers/base.py"
    text = read(path)
    text = replace_once(
        text,
        "from dataclasses import asdict, dataclass\n",
        "from dataclasses import asdict, dataclass, field\n",
        "InferenceUsage field import",
    )
    text = replace_once(
        text,
        """    estimated_cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class ProviderResult""",
        """    estimated_cost_usd: float = 0.0
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderResult""",
        "InferenceUsage runtime",
    )
    write(path, text)


def patch_modal_provider() -> None:
    path = "src/clipper/providers/modal.py"
    text = read(path)
    text = replace_once(
        text,
        """class ModalJSONProvider:
    def __init__(self, *, app_name: str, function_name: str, identity: ModelIdentity) -> None:
        self.app_name = app_name
        self.function_name = function_name
        self.identity = identity
""",
        """class ModalJSONProvider:
    def __init__(
        self,
        *,
        app_name: str,
        identity: ModelIdentity,
        function_name: str | None = None,
        class_name: str | None = None,
        method_name: str | None = None,
        class_parameters: dict[str, Any] | None = None,
    ) -> None:
        if (function_name is None) == (class_name is None):
            raise ValueError("Modal provider requires exactly one function_name or class_name")
        if class_name is not None and not method_name:
            raise ValueError("class-backed Modal provider requires method_name")
        self.app_name = app_name
        self.class_name = class_name
        self.method_name = method_name
        self.class_parameters = dict(class_parameters or {})
        self.function_name = function_name or f"{class_name}.{method_name}"
        self.identity = identity
        self._instance_handle: Any | None = None
""",
        "Modal class provider init",
    )
    text = replace_once(
        text,
        """    def _function(self) -> Any:
        try:
            modal = importlib.import_module("modal")
        except ImportError as exc:
            raise ProviderUnavailable("install clipper[modal]") from exc
        return modal.Function.from_name(self.app_name, self.function_name)
""",
        """    def _modal(self) -> Any:
        try:
            return importlib.import_module("modal")
        except ImportError as exc:
            raise ProviderUnavailable("install clipper[modal]") from exc

    def _class_instance(self) -> Any:
        if self.class_name is None:
            raise RuntimeError("function-backed provider has no class instance")
        if self._instance_handle is None:
            cls = self._modal().Cls.from_name(self.app_name, self.class_name)
            self._instance_handle = cls(**self.class_parameters)
        return self._instance_handle

    def _function(self) -> Any:
        if self.class_name is None:
            return self._modal().Function.from_name(self.app_name, self.function_name)
        return getattr(self._class_instance(), str(self.method_name))

    def warm(self) -> dict[str, Any]:
        if self.class_name is None:
            return {}
        ready = getattr(self._class_instance(), "ready", None)
        if ready is None:
            return {}
        response = ready.remote()
        if not isinstance(response, dict):
            raise ValueError("Modal class warmup returned an invalid response")
        self._raise_remote_error(response)
        self.identity = self._resolved_identity(response)
        raw_runtime = response.get("runtime")
        return dict(raw_runtime) if isinstance(raw_runtime, dict) else {}
""",
        "Modal class lookup",
    )
    text = replace_once(
        text,
        """        raw_usage = response.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return ProviderResult(
""",
        """        raw_usage = response.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        raw_runtime = response.get("runtime")
        runtime: dict[str, Any] = dict(raw_runtime) if isinstance(raw_runtime, dict) else {}
        peak_by_device = usage.get("peak_vram_mb_by_device")
        if isinstance(peak_by_device, dict):
            runtime["peak_vram_mb_by_device"] = dict(peak_by_device)
        return ProviderResult(
""",
        "Modal runtime parsing",
    )
    text = replace_once(
        text,
        """                estimated_cost_usd=float(usage.get("estimated_cost_usd") or 0.0),
            ),
""",
        """                estimated_cost_usd=float(usage.get("estimated_cost_usd") or 0.0),
                runtime=runtime,
            ),
""",
        "Modal runtime usage",
    )
    write(path, text)


def patch_provider_factory() -> None:
    path = "src/clipper/providers/factory.py"
    text = read(path)
    text = replace_once(
        text,
        '            function_name="editorial",\n            identity=ModelIdentity(',
        '            class_name="EditorialModel",\n'
        '            method_name="complete",\n'
        "            identity=ModelIdentity(",
        "editorial class factory",
    )
    text = replace_once(
        text,
        '        function_name="vision_large" if large else "vision",\n'
        "        identity=ModelIdentity(",
        '        class_name="VisionModel",\n'
        '        method_name="inspect",\n'
        '        class_parameters={"model_id": model_id},\n'
        "        identity=ModelIdentity(",
        "vision class factory",
    )
    write(path, text)


def patch_visual_ai() -> None:
    path = "src/clipper/visual_ai.py"
    text = read(path)
    text = replace_once(
        text,
        "from typing import Any, Literal, cast\n",
        "from typing import Any, Callable, Literal, cast\n",
        "Callable import",
    )
    text = replace_once(
        text,
        "from .providers.base import InferenceUsage, ProviderResult, VisionProvider\n",
        "from .cache import FileCache\n"
        "from .providers.base import InferenceUsage, ModelIdentity, ProviderResult, VisionProvider\n"
        "from .stage_contracts import content_fingerprint\n",
        "visual cache imports",
    )
    text = replace_once(
        text, "SOURCE_POLICY_BATCH_SIZE = 24\n", "", "fixed source-policy batch size"
    )
    text = replace_once(
        text,
        """    spans: tuple[VisualEvidenceSpan, ...],
) -> tuple[tuple[VisualEvent, ...], list[ProviderResult[dict[str, Any]]]]:
""",
        """    spans: tuple[VisualEvidenceSpan, ...],
    on_observations: Callable[
        [tuple[VisualEvent, ...], tuple[float, ...], ModelIdentity], None
    ]
    | None = None,
) -> tuple[tuple[VisualEvent, ...], list[ProviderResult[dict[str, Any]]]]:
""",
        "source policy checkpoint callback",
    )
    text = replace_once(
        text,
        """        accepted.extend(parsed)
        if not missing:
            return
""",
        """        if parsed and on_observations is not None:
            missing_keys = {round(value, 3) for value in missing}
            observed_times = tuple(
                value for value in subset_times if round(value, 3) not in missing_keys
            )
            on_observations(parsed, observed_times, result.model)
        accepted.extend(parsed)
        if not missing:
            return
""",
        "source policy immediate checkpoint callback",
    )

    start = text.find("def _aggregate_vision_results(")
    end = text.find("def _needs_escalation(", start)
    if start < 0 or end <= start:
        raise RuntimeError("visual scout replacement region missing")

    replacement = '''def _model_identity_from_payload(payload: object) -> ModelIdentity | None:
    if not isinstance(payload, dict):
        return None
    fields = (
        "model_id",
        "revision",
        "quantization",
        "inference_engine",
        "prompt_version",
        "schema_version",
    )
    if not all(payload.get(field) is not None for field in fields):
        return None
    return ModelIdentity(*(str(payload[field]) for field in fields))


def _source_policy_cache_namespace(
    *,
    source_hash: str,
    requested_identity: ModelIdentity,
) -> str:
    instruction = str(
        _source_policy_context(
            video_id="cache-contract",
            source_hash=source_hash,
            frame_timestamps=(),
            recovery_attempt=0,
        )["instruction"]
    )
    return content_fingerprint(
        {
            "stage": "source_policy_visual_frame",
            "source_hash": source_hash,
            "model_identity": requested_identity.to_dict(),
            "inspection_contract": instruction,
            "frame_contract": {"max_edge": VISUAL_SAMPLE_MAX_EDGE},
        }
    )


def _source_policy_frame_cache_key(namespace: str, timestamp: float) -> str:
    return content_fingerprint(
        {"namespace": namespace, "timestamp": round(timestamp, 3)}
    )


def _read_source_policy_checkpoint(
    cache: FileCache,
    *,
    namespace: str,
    timestamp: float,
    span: VisualEvidenceSpan,
) -> tuple[VisualEvent, ModelIdentity] | None:
    payload = cache.read(
        _source_policy_frame_cache_key(namespace, timestamp), "observation"
    )
    if not isinstance(payload, dict):
        return None
    model = _model_identity_from_payload(payload.get("model"))
    observation = payload.get("observation")
    if model is None or not isinstance(observation, dict):
        return None
    try:
        events, missing = _parse_source_policy_events(
            {"observations": [observation]},
            frame_timestamps=(timestamp,),
            spans=(span,),
        )
    except (TypeError, ValueError):
        return None
    if missing or len(events) != 1:
        return None
    return events[0], model


def _write_source_policy_checkpoint(
    cache: FileCache,
    *,
    namespace: str,
    timestamp: float,
    event: VisualEvent,
    model: ModelIdentity,
) -> None:
    cache.write(
        _source_policy_frame_cache_key(namespace, timestamp),
        "observation",
        {
            "timestamp": round(timestamp, 3),
            "model": model.to_dict(),
            "observation": {
                "timestamp": round(timestamp, 3),
                "scene_id": event.scene_id,
                "summary": event.summary,
                "visible_speakers": list(event.visible_speakers),
                "event_labels": list(event.event_labels),
                "confidence": event.confidence,
            },
        },
    )


def _is_vision_capacity_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "outofmemoryerror",
            "out of memory",
            "exceeds model context",
            "request too large",
            "payload too large",
        )
    )


def _capacity_cache_key(requested_identity: ModelIdentity) -> str:
    return content_fingerprint(
        {
            "stage": "source_policy_visual_capacity",
            "model_identity": requested_identity.to_dict(),
            "frame_contract": {"max_edge": VISUAL_SAMPLE_MAX_EDGE},
        }
    )


def _load_capacity_state(
    cache: FileCache | None,
    *,
    requested_identity: ModelIdentity,
) -> tuple[str | None, int, int | None]:
    if cache is None:
        return None, 0, None
    key = _capacity_cache_key(requested_identity)
    payload = cache.read(key, "capacity")
    if not isinstance(payload, dict):
        return key, 0, None
    raw_good = payload.get("largest_good")
    raw_bad = payload.get("smallest_bad")
    good = int(raw_good) if isinstance(raw_good, int) and raw_good > 0 else 0
    bad = int(raw_bad) if isinstance(raw_bad, int) and raw_bad > good else None
    return key, good, bad


def _persist_capacity_state(
    cache: FileCache | None,
    key: str | None,
    *,
    largest_good: int,
    smallest_bad: int | None,
    checkpoint_commit: Callable[[], None] | None,
) -> None:
    if cache is None or key is None:
        return
    cache.write(
        key,
        "capacity",
        {
            "largest_good": largest_good,
            "smallest_bad": smallest_bad,
        },
    )
    if checkpoint_commit is not None:
        checkpoint_commit()


def _next_batch_after_success(
    current: int,
    *,
    largest_good: int,
    smallest_bad: int | None,
    remaining: int,
) -> int:
    if remaining <= 0:
        return 0
    if smallest_bad is None:
        return min(remaining, max(current + 1, current * 2))
    if largest_good + 1 < smallest_bad:
        return min(remaining, largest_good + (smallest_bad - largest_good) // 2)
    return min(remaining, max(1, largest_good))


def _next_batch_after_capacity_failure(
    current: int,
    *,
    largest_good: int,
    smallest_bad: int,
) -> int:
    if current <= 1:
        return 1
    if largest_good > 0 and largest_good + 1 < smallest_bad:
        candidate = largest_good + (smallest_bad - largest_good) // 2
    elif largest_good > 0:
        candidate = largest_good
    else:
        candidate = current // 2
    return max(1, min(current - 1, candidate))


def _aggregate_vision_results(
    results: list[ProviderResult[dict[str, Any]]],
    timeline: VisualTimeline,
    *,
    cached_model: ModelIdentity | None = None,
    cache_hits: int = 0,
    requested_frames: int = 0,
) -> ProviderResult[dict[str, Any]]:
    model = results[0].model if results else cached_model
    if model is None:
        raise ValueError(
            "source-policy visual scout produced no inference or cached evidence"
        )
    if cached_model is not None and model != cached_model:
        raise RuntimeError(
            "source-policy cache model identity differs from active vision model"
        )
    if any(result.model != model for result in results):
        raise RuntimeError(
            "source-policy visual scout changed model identity between batches"
        )

    usages = [result.usage for result in results]
    gpu_types = {usage.gpu_type for usage in usages if usage.gpu_type}
    peaks = [
        usage.peak_vram_mb
        for usage in usages
        if usage.peak_vram_mb is not None
    ]
    lifecycle_loads: dict[str, int] = {}
    peak_by_device: dict[str, float] = {}
    for usage in usages:
        lifecycle_id = usage.runtime.get("worker_lifecycle_id")
        load_count = usage.runtime.get("model_load_count")
        if isinstance(lifecycle_id, str) and isinstance(load_count, int):
            lifecycle_loads[lifecycle_id] = max(
                lifecycle_loads.get(lifecycle_id, 0), load_count
            )
        raw_peaks = usage.runtime.get("peak_vram_mb_by_device")
        if isinstance(raw_peaks, dict):
            for device, value in raw_peaks.items():
                if isinstance(value, int | float):
                    peak_by_device[str(device)] = max(
                        peak_by_device.get(str(device), 0.0), float(value)
                    )
    runtime: dict[str, Any] = {
        "source_policy_cache_hits": cache_hits,
        "source_policy_requested_frames": requested_frames,
        "source_policy_provider_calls": len(results),
        "worker_lifecycle_model_loads": lifecycle_loads,
        "peak_vram_mb_by_device": peak_by_device,
    }
    usage = InferenceUsage(
        provider=usages[0].provider if usages else "cache",
        started_at=usages[0].started_at if usages else "cache",
        duration_seconds=sum(item.duration_seconds for item in usages),
        gpu_type=next(iter(gpu_types)) if len(gpu_types) == 1 else None,
        gpu_seconds=sum(item.gpu_seconds for item in usages),
        peak_vram_mb=max(peaks) if peaks else None,
        input_units=sum(item.input_units for item in usages),
        output_units=sum(item.output_units for item in usages),
        estimated_cost_usd=sum(item.estimated_cost_usd for item in usages),
        runtime=runtime,
    )
    return ProviderResult(
        timeline.to_dict(),
        model,
        usage,
        degraded=any(result.degraded for result in results),
    )


def scout_visual_timeline(
    video_path: Path,
    provider: VisionProvider,
    *,
    video_id: str,
    source_hash: str,
    duration: float,
    output_dir: Path,
    scene_cuts: tuple[float, ...] = (),
    candidate_ranges: tuple[tuple[float, float], ...] = (),
    checkpoint_dir: Path | None = None,
    checkpoint_commit: Callable[[], None] | None = None,
) -> tuple[VisualTimeline, ProviderResult[dict[str, Any]]]:
    """Build source-wide policy evidence with resumable, runtime-learned capacity."""
    media_duration = media_duration_seconds(video_path)
    effective_duration = min(duration, media_duration)
    times = source_policy_sample_times(effective_duration, scene_cuts=scene_cuts)
    if candidate_ranges:
        dense_candidate_times = adaptive_sample_times(
            effective_duration,
            candidate_ranges=candidate_ranges,
            base_interval=SOURCE_POLICY_SAMPLE_INTERVAL_SECONDS,
        )
        times = tuple(sorted(set(times) | set(dense_candidate_times)))
    spans = visual_evidence_spans_from_samples(
        times, effective_duration, scope="source_policy"
    )
    spans_by_sample = {round(span.sample_time, 3): span for span in spans}
    requested_identity = provider.identity
    cache = FileCache(checkpoint_dir) if checkpoint_dir is not None else None
    namespace = _source_policy_cache_namespace(
        source_hash=source_hash,
        requested_identity=requested_identity,
    )

    events: list[VisualEvent] = []
    cached_model: ModelIdentity | None = None
    cached_times: set[float] = set()
    if cache is not None:
        for timestamp in times:
            hit = _read_source_policy_checkpoint(
                cache,
                namespace=namespace,
                timestamp=timestamp,
                span=spans_by_sample[round(timestamp, 3)],
            )
            if hit is None:
                continue
            event, model = hit
            if cached_model is not None and model != cached_model:
                continue
            cached_model = model
            events.append(event)
            cached_times.add(round(timestamp, 3))

    pending_times = tuple(
        timestamp for timestamp in times if round(timestamp, 3) not in cached_times
    )
    results: list[ProviderResult[dict[str, Any]]] = []
    if pending_times:
        prepared_dir = output_dir / "source-policy-pending"
        prepared_frames = extract_video_frames(
            video_path, pending_times, prepared_dir
        )
        if len(prepared_frames) != len(pending_times):
            raise RuntimeError(
                "source-policy frame extraction returned incomplete prepared jobs"
            )

        warm = getattr(provider, "warm", None)
        if callable(warm):
            warm()
        if cached_model is not None and provider.identity != cached_model:
            events.clear()
            cached_times.clear()
            cached_model = None
            pending_times = times
            prepared_dir = output_dir / "source-policy-revalidated"
            prepared_frames = extract_video_frames(
                video_path, pending_times, prepared_dir
            )
            if callable(warm):
                warm()

        work = [
            (timestamp, frame, spans_by_sample[round(timestamp, 3)])
            for timestamp, frame in zip(
                pending_times, prepared_frames, strict=True
            )
        ]

        def persist_observations(
            parsed: tuple[VisualEvent, ...],
            observed_times: tuple[float, ...],
            model: ModelIdentity,
        ) -> None:
            nonlocal cached_model
            if cache is None:
                return
            if len(parsed) != len(observed_times):
                raise RuntimeError(
                    "source-policy checkpoint lost observation identity"
                )
            for timestamp, event in zip(
                observed_times, parsed, strict=True
            ):
                _write_source_policy_checkpoint(
                    cache,
                    namespace=namespace,
                    timestamp=timestamp,
                    event=event,
                    model=model,
                )
            cached_model = model
            if checkpoint_commit is not None:
                checkpoint_commit()

        capacity_key, largest_good, smallest_bad = _load_capacity_state(
            cache, requested_identity=requested_identity
        )
        batch_size = largest_good if largest_good > 0 else 1

        while work:
            size = min(batch_size, len(work))
            subset = tuple(work[:size])
            subset_times = tuple(item[0] for item in subset)
            subset_frames = [item[1] for item in subset]
            subset_spans = tuple(item[2] for item in subset)
            try:
                batch_events, batch_results = _inspect_source_policy_batch(
                    provider,
                    video_id=video_id,
                    source_hash=source_hash,
                    frame_timestamps=subset_times,
                    frames=subset_frames,
                    spans=subset_spans,
                    on_observations=persist_observations,
                )
            except Exception as exc:
                if not _is_vision_capacity_error(exc):
                    raise

                if cache is not None:
                    retained: list[
                        tuple[float, Path, VisualEvidenceSpan]
                    ] = []
                    completed_events: list[VisualEvent] = []
                    for item in subset:
                        hit = _read_source_policy_checkpoint(
                            cache,
                            namespace=namespace,
                            timestamp=item[0],
                            span=item[2],
                        )
                        if hit is None:
                            retained.append(item)
                        else:
                            completed_events.append(hit[0])
                    if completed_events:
                        events.extend(completed_events)
                        work = retained + work[size:]
                        if not retained:
                            batch_size = (
                                min(max(1, batch_size), len(work))
                                if work
                                else 0
                            )
                            continue

                if size <= 1:
                    raise RuntimeError(
                        "vision capacity exhausted for an indivisible "
                        "single-frame inspection"
                    ) from exc
                smallest_bad = (
                    size
                    if smallest_bad is None
                    else min(smallest_bad, size)
                )
                batch_size = _next_batch_after_capacity_failure(
                    size,
                    largest_good=largest_good,
                    smallest_bad=smallest_bad,
                )
                _persist_capacity_state(
                    cache,
                    capacity_key,
                    largest_good=largest_good,
                    smallest_bad=smallest_bad,
                    checkpoint_commit=checkpoint_commit,
                )
                continue

            events.extend(batch_events)
            results.extend(batch_results)
            work = work[size:]
            largest_good = max(largest_good, size)
            batch_size = _next_batch_after_success(
                size,
                largest_good=largest_good,
                smallest_bad=smallest_bad,
                remaining=len(work),
            )
            _persist_capacity_state(
                cache,
                capacity_key,
                largest_good=largest_good,
                smallest_bad=smallest_bad,
                checkpoint_commit=checkpoint_commit,
            )

    timeline = VisualTimeline(
        video_id,
        source_hash,
        tuple(
            sorted(
                events,
                key=lambda event: (event.start, event.end, event.scene_id),
            )
        ),
        coverage_spans=spans,
        source_duration=effective_duration,
    )
    return timeline, _aggregate_vision_results(
        results,
        timeline,
        cached_model=cached_model,
        cache_hits=len(cached_times),
        requested_frames=len(times),
    )


'''
    text = text[:start] + replacement + text[end:]
    write(path, text)


def patch_pipeline() -> None:
    path = "src/clipper/pipeline.py"
    text = read(path)
    text = replace_once(
        text,
        "from typing import Protocol\n",
        "from typing import Callable, Protocol\n",
        "pipeline Callable",
    )
    text = replace_once(
        text,
        """def _visual_timeline(
    media_path: Path,
    video: VideoCandidate,
    timeline: CanonicalTimeline,
    provider: VisionProvider,
    run_dir: Path,
) -> tuple[VisualTimeline, dict[str, object]]:
""",
        """def _visual_timeline(
    media_path: Path,
    video: VideoCandidate,
    timeline: CanonicalTimeline,
    provider: VisionProvider,
    run_dir: Path,
    *,
    cache_root: Path,
    checkpoint_commit: Callable[[], None] | None,
) -> tuple[VisualTimeline, dict[str, object]]:
""",
        "visual timeline signature",
    )
    text = replace_once(
        text,
        '        output_dir=run_dir / "visual-scout" / video.video_id / "frames",\n'
        "    )\n",
        '        output_dir=run_dir / "visual-scout" / video.video_id / "frames",\n'
        '        checkpoint_dir=cache_root / "source-policy-vision",\n'
        "        checkpoint_commit=checkpoint_commit,\n"
        "    )\n",
        "visual checkpoint args",
    )
    text = replace_once(
        text,
        """    diarization_provider: DiarizationProvider | None = None,
    render: bool = True,
) -> Path:
""",
        """    diarization_provider: DiarizationProvider | None = None,
    render: bool = True,
    checkpoint_commit: Callable[[], None] | None = None,
) -> Path:
""",
        "run_pipeline checkpoint callback",
    )
    text = replace_once(
        text,
        """    journal = StageJournal(run_dir / "progress.json")
    cache = FileCache(cfg.cache_root or (cfg.artifact_root / "_cache"))
""",
        """    journal = StageJournal(run_dir / "progress.json")
    cache_root = cfg.cache_root or (cfg.artifact_root / "_cache")
    cache = FileCache(cache_root)
""",
        "pipeline cache root",
    )
    text = replace_once(
        text,
        """            visual, visual_meta = _visual_timeline(media_path, video, timeline, scout, run_dir)
""",
        """            visual, visual_meta = _visual_timeline(
                media_path,
                video,
                timeline,
                scout,
                run_dir,
                cache_root=cache_root,
                checkpoint_commit=checkpoint_commit,
            )
""",
        "pipeline visual call",
    )
    write(path, text)

    path = "scripts/modal_pipeline.py"
    text = read(path)
    text = replace_once(
        text,
        """            diarization_provider=diarization,
            render=render,
        )
""",
        """            diarization_provider=diarization,
            render=render,
            checkpoint_commit=artifact_volume.commit,
        )
""",
        "Modal durable checkpoint callback",
    )
    write(path, text)


def patch_smoke_and_workflow() -> None:
    path = "scripts/modal_vision_smoke.py"
    text = read(path)
    text = replace_once(
        text,
        '    parser.add_argument("--function", default="vision")\n',
        '    parser.add_argument("--class-name", default="VisionModel")\n'
        '    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")\n',
        "vision smoke args",
    )
    text = replace_once(
        text,
        "    result = modal.Function.from_name(args.app, args.function).remote(payload)\n",
        "    worker = modal.Cls.from_name(args.app, args.class_name)(model_id=args.model_id)\n"
        "    worker.ready.remote()\n"
        "    result = worker.inspect.remote(payload)\n",
        "vision smoke class",
    )
    write(path, text)

    path = ".github/workflows/modal-workers-deploy.yml"
    text = read(path)
    text = replace_once(
        text,
        """                      "model_functions_expected": [
                          "transcribe",
                          "align",
                          "diarize",
                          "editorial",
                          "vision",
                          "vision_large",
                          "editorial_schema_smoke",
                          "hf_access_smoke",
                      ],
""",
        """                      "model_functions_expected": [
                          "transcribe",
                          "align",
                          "diarize",
                          "editorial_schema_smoke",
                          "hf_access_smoke",
                      ],
                      "model_classes_expected": [
                          "EditorialModel",
                          "VisionModel",
                      ],
""",
        "deployment class evidence",
    )
    write(path, text)


def patch_tests() -> None:
    path = "tests/test_visual_ai.py"
    text = read(path)
    text = replace_once(
        text,
        """    assert summary["sample_count"] > 24
    assert len(provider.calls) == 2
    assert all(call[0] == "source_policy_visual_scout" for call in provider.calls)
    assert all(len(call[1]) <= 24 for call in provider.calls)
""",
        """    assert summary["sample_count"] > 1
    assert len(provider.calls) > 1
    assert all(call[0] == "source_policy_visual_scout" for call in provider.calls)
    call_sizes = [len(call[1]) for call in provider.calls]
    assert call_sizes[0] == 1
    assert max(call_sizes) > call_sizes[0]
""",
        "adaptive batch assertions",
    )
    text = replace_once(
        text,
        """    assert len(provider.calls) == 2
    assert len(provider.calls[0][1]) == sample_count
    assert len(provider.calls[1][1]) == 1
    assert provider.calls[1][2]["source_policy_recovery_attempt"] == 1
""",
        """    recovery_calls = [
        call
        for call in provider.calls
        if call[2]["source_policy_recovery_attempt"] == 1
    ]
    assert recovery_calls
    assert any(len(call[1]) == 1 for call in recovery_calls)
""",
        "recovery assertions",
    )

    text += '''

class CapacityPolicyVision(PolicyVision):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity
        self.attempted_sizes: list[int] = []

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        self.attempted_sizes.append(len(frames))
        if len(frames) > self.capacity:
            raise RuntimeError("CUDA out of memory")
        return super().inspect(task=task, frames=frames, context=context)


class InterruptingPolicyVision(PolicyVision):
    def __init__(self, fail_after: int | None) -> None:
        super().__init__()
        self.fail_after = fail_after

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated external interruption")
        return super().inspect(task=task, frames=frames, context=context)


def _prepared_source_policy_frames(
    _source: Path, times: tuple[float, ...], output: Path
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, timestamp in enumerate(times):
        frame = output / f"{index:03d}-{timestamp:.3f}.jpg"
        frame.write_bytes(b"frame")
        frames.append(frame)
    return frames


def test_source_policy_capacity_is_learned_from_runtime_failures(
    tmp_path: Path,
) -> None:
    provider = CapacityPolicyVision(capacity=3)
    commits: list[None] = []
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=40.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=40.0,
            output_dir=tmp_path / "frames",
            checkpoint_dir=tmp_path / "cache",
            checkpoint_commit=lambda: commits.append(None),
        )
    sample_count = int(
        timeline.coverage_summary("source_policy")["sample_count"]
    )
    assert len(timeline.events) == sample_count
    assert any(size > provider.capacity for size in provider.attempted_sizes)
    assert (
        max(
            size
            for size in provider.attempted_sizes
            if size <= provider.capacity
        )
        == provider.capacity
    )
    assert result.usage.input_units == sample_count
    assert commits


def test_source_policy_resume_reuses_durable_frame_checkpoints(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    first = InterruptingPolicyVision(fail_after=1)
    commits: list[None] = []
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=20.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
        pytest.raises(RuntimeError, match="simulated external interruption"),
    ):
        scout_visual_timeline(
            tmp_path / "source.mp4",
            first,
            video_id="v",
            source_hash="h",
            duration=20.0,
            output_dir=tmp_path / "first",
            checkpoint_dir=cache,
            checkpoint_commit=lambda: commits.append(None),
        )
    assert commits
    completed_timestamps = {
        round(float(timestamp), 3)
        for _, _, context in first.calls
        for timestamp in context["frame_timestamps"]  # type: ignore[index]
    }

    resumed = InterruptingPolicyVision(fail_after=None)
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=20.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            resumed,
            video_id="v",
            source_hash="h",
            duration=20.0,
            output_dir=tmp_path / "second",
            checkpoint_dir=cache,
            checkpoint_commit=lambda: commits.append(None),
        )
    resumed_timestamps = {
        round(float(timestamp), 3)
        for _, _, context in resumed.calls
        for timestamp in context["frame_timestamps"]  # type: ignore[index]
    }
    assert completed_timestamps
    assert completed_timestamps.isdisjoint(resumed_timestamps)
    assert len(timeline.events) == int(
        timeline.coverage_summary("source_policy")["sample_count"]
    )
    assert result.usage.runtime["source_policy_cache_hits"] >= len(
        completed_timestamps
    )


def test_source_policy_fully_cached_resume_performs_no_inference(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    provider = PolicyVision()
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=12.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
    ):
        scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=12.0,
            output_dir=tmp_path / "first",
            checkpoint_dir=cache,
        )
    cached = PolicyVision()
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=12.0),
        patch("clipper.visual_ai.extract_video_frames") as extract,
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            cached,
            video_id="v",
            source_hash="h",
            duration=12.0,
            output_dir=tmp_path / "second",
            checkpoint_dir=cache,
        )
    assert not cached.calls
    extract.assert_not_called()
    assert result.usage.provider == "cache"
    assert result.usage.runtime["source_policy_cache_hits"] == int(
        timeline.coverage_summary("source_policy")["sample_count"]
    )
'''
    write(path, text)

    lifecycle = '''from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from clipper.providers.base import ModelIdentity
from clipper.providers.modal import ModalEditorialProvider, ModalVisionProvider


def _identity(model_id: str) -> ModelIdentity:
    return ModelIdentity(
        model_id,
        "requested",
        "none",
        "modal-transformers",
        "prompt",
        "schema",
    )


def test_modal_vision_provider_reuses_class_handle_and_surfaces_runtime(
    tmp_path: Path,
) -> None:
    ready = SimpleNamespace(
        remote=Mock(
            return_value={
                "value": {"ready": True},
                "model": {"model_id": "vision", "revision": "actual"},
                "runtime": {
                    "worker_lifecycle_id": "worker-a",
                    "model_load_count": 1,
                },
            }
        )
    )
    inspect = SimpleNamespace(
        remote=Mock(
            return_value={
                "value": {"observations": []},
                "model": {"model_id": "vision", "revision": "actual"},
                "usage": {
                    "duration_seconds": 1.0,
                    "peak_vram_mb_by_device": {"0": 10.0, "1": 11.0},
                },
                "runtime": {
                    "worker_lifecycle_id": "worker-a",
                    "model_load_count": 1,
                },
            }
        )
    )
    instance = SimpleNamespace(ready=ready, inspect=inspect)
    class_handle = Mock(return_value=instance)
    modal = SimpleNamespace(
        Cls=SimpleNamespace(from_name=Mock(return_value=class_handle))
    )
    provider = ModalVisionProvider(
        app_name="app",
        class_name="VisionModel",
        method_name="inspect",
        class_parameters={"model_id": "vision"},
        identity=_identity("vision"),
    )
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")

    with patch(
        "clipper.providers.modal.importlib.import_module",
        return_value=modal,
    ):
        runtime = provider.warm()
        result = provider.inspect(
            task="source_policy_visual_scout",
            frames=[frame],
            context={},
        )
        provider.inspect(
            task="source_policy_visual_scout",
            frames=[frame],
            context={},
        )

    assert runtime["worker_lifecycle_id"] == "worker-a"
    assert modal.Cls.from_name.call_count == 1
    class_handle.assert_called_once_with(model_id="vision")
    assert inspect.remote.call_count == 2
    assert result.usage.runtime["worker_lifecycle_id"] == "worker-a"
    assert result.usage.runtime["peak_vram_mb_by_device"] == {
        "0": 10.0,
        "1": 11.0,
    }


def test_modal_editorial_provider_can_use_persistent_class_method() -> None:
    complete = SimpleNamespace(
        remote=Mock(
            return_value={
                "value": {"ok": True},
                "usage": {},
                "runtime": {
                    "worker_lifecycle_id": "editor-a",
                    "model_load_count": 1,
                },
            }
        )
    )
    instance = SimpleNamespace(complete=complete)
    class_handle = Mock(return_value=instance)
    modal = SimpleNamespace(
        Cls=SimpleNamespace(from_name=Mock(return_value=class_handle))
    )
    provider = ModalEditorialProvider(
        app_name="app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity("editor"),
    )
    with patch(
        "clipper.providers.modal.importlib.import_module",
        return_value=modal,
    ):
        assert provider.complete_json(task="task", payload={}).value == {"ok": True}
        assert provider.complete_json(task="task", payload={}).value == {"ok": True}
    assert modal.Cls.from_name.call_count == 1
    assert complete.remote.call_count == 2


def test_modal_worker_source_uses_enter_loaded_classes_and_dynamic_capacity() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    visual_source = Path("src/clipper/visual_ai.py").read_text(encoding="utf-8")
    assert "class EditorialModel:" in source
    assert "class VisionModel:" in source
    assert source.count("@modal.enter()") >= 2
    assert "def vision(" not in source
    assert "def vision_large(" not in source
    assert '"20GiB"' not in source
    assert '"22GiB"' not in source
    assert "SOURCE_POLICY_BATCH_SIZE" not in visual_source
    assert "_is_vision_capacity_error" in visual_source
    assert "checkpoint_commit" in visual_source
'''
    write("tests/test_modal_model_lifecycle.py", lifecycle)


def patch_contract() -> None:
    path = "plans/autonomous-multimodal-editor-contract.md"
    text = read(path)
    text = replace_once(
        text,
        "Prefer warm persistent Modal model services for repeated structured "
        "editorial/vision calls to avoid repeated large checkpoint initialization.\n",
        "Repeated structured editorial and vision inference MUST use persistent "
        "model-worker lifecycles where technically feasible. Ordinary logical "
        "batch/task boundaries MUST NOT intentionally reload the same checkpoint. "
        "Model initialization count, worker lifecycle identity, per-device peak VRAM "
        "and runtime capacity recovery must be observable in live acceptance evidence.\n\n"
        "Operational inference capacity MUST be learned from runtime evidence rather "
        "than encoded as fixed absolute batch-size or VRAM-headroom numbers. The "
        "vision path must adapt to successful capacity and recover from capacity "
        "failures by reducing work until the indivisible single-frame unit; a "
        "single-frame capacity failure must fail closed rather than silently reduce "
        "evidence quality.\n\n"
        "Every successfully validated expensive source-policy visual observation MUST "
        "be content-addressed and durably checkpointed before subsequent paid work "
        "continues. An interrupted run MUST resume from missing observations and MUST "
        "NOT repeat already checkpointed visual inference. Cache reuse is keyed by "
        "source/model/inspection contract rather than by transient batch numbering.\n",
        "warm worker contract",
    )
    text = replace_once(
        text,
        "- cost accounting\n- adaptive visual strategy\n",
        "- cost accounting\n"
        "- persistent warm editorial/vision worker lifecycle with model-load/container-reuse evidence\n"
        "- runtime-derived vision batch capacity with recoverable OOM/context/payload splitting and fail-closed single-frame exhaustion\n"
        "- per-device VRAM telemetry\n"
        "- durable per-observation source-policy vision checkpoints and interrupted-run resume without replay\n"
        "- adaptive visual strategy\n",
        "Final DOD operational gates",
    )
    write(path, text)

    path = "plans/autonomous-multimodal-editor-implementation-matrix.md"
    text = read(path)
    text = replace_once(
        text,
        "| I | Downstream change does not rerun upstream paid work | dependency/output "
        "fingerprints | interrupted/resume regression | PASS |\n"
        "| J | Dynamic acceptance derives expectations from evidence |",
        "| I | Downstream change does not rerun upstream paid work | dependency/output "
        "fingerprints | interrupted/resume regression | PASS |\n"
        "| I | Source-policy vision resumes at completed observation granularity | "
        "content-addressed per-frame checkpoints + explicit durable commit hook | "
        "interruption/resume and fully-cached zero-inference regressions | "
        "IMPLEMENTED_PENDING_LIVE |\n"
        "| I | Editorial/vision repeated inference uses persistent model-worker "
        "lifecycle | Modal class workers with enter-time loading; no ordinary-batch "
        "checkpoint reload | local lifecycle contract tests; live lifecycle IDs/load "
        "counts required | IMPLEMENTED_PENDING_LIVE |\n"
        "| I | Vision capacity is runtime-derived rather than a fixed production "
        "batch/VRAM threshold | learned good/bad capacity with adaptive recovery | "
        "forced-capacity regression + live per-device VRAM evidence required | "
        "IMPLEMENTED_PENDING_LIVE |\n"
        "| J | Dynamic acceptance derives expectations from evidence |",
        "matrix phase rows",
    )
    text = replace_once(
        text,
        "| Compute/cost accounting | telemetry implemented; current production "
        "measurements required | IMPLEMENTED_PENDING_LIVE |\n"
        "| Adaptive visual strategy |",
        "| Compute/cost accounting | telemetry implemented; current production "
        "measurements required | IMPLEMENTED_PENDING_LIVE |\n"
        "| Warm editorial/vision worker reuse | class lifecycle implemented; exact "
        "live worker lifecycle/model-load evidence required | IMPLEMENTED_PENDING_LIVE |\n"
        "| Runtime-derived vision capacity and OOM recovery | adaptive capacity "
        "implementation/tests; live VRAM evidence required | IMPLEMENTED_PENDING_LIVE |\n"
        "| Durable per-observation visual resume without replay | implementation + "
        "interruption/cache regressions; live interrupted/resume evidence required | "
        "IMPLEMENTED_PENDING_LIVE |\n"
        "| Per-device vision VRAM telemetry | worker/provider telemetry implemented; "
        "live evidence required | IMPLEMENTED_PENDING_LIVE |\n"
        "| Adaptive visual strategy |",
        "matrix final rows",
    )
    write(path, text)


def main() -> None:
    patch_modal_workers()
    patch_provider_base()
    patch_modal_provider()
    patch_provider_factory()
    patch_visual_ai()
    patch_pipeline()
    patch_smoke_and_workflow()
    patch_tests()
    patch_contract()


if __name__ == "__main__":
    main()
