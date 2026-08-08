from __future__ import annotations

import importlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import InferenceUsage, ModelIdentity, ProviderResult


class ProviderUnavailable(RuntimeError):
    pass


def _started() -> tuple[str, float]:
    return datetime.now(UTC).isoformat(), time.perf_counter()


def _usage(
    started_at: str, started: float, *, provider: str, input_units: int = 0, output_units: int = 0
) -> InferenceUsage:
    return InferenceUsage(
        provider=provider,
        started_at=started_at,
        duration_seconds=max(0.0, time.perf_counter() - started),
        input_units=input_units,
        output_units=output_units,
    )


class LocalEmbeddingProvider:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        revision: str = "main",
        *,
        device: str | None = None,
    ) -> None:
        self.identity = ModelIdentity(model_id, revision, "none", "sentence-transformers")
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                module = importlib.import_module("sentence_transformers")
            except ImportError as exc:
                raise ProviderUnavailable("install clipper[embedding]") from exc
            kwargs: dict[str, Any] = {"revision": self.identity.revision}
            if self.device:
                kwargs["device"] = self.device
            self._model = module.SentenceTransformer(self.identity.model_id, **kwargs)
        return self._model

    def embed(self, texts: list[str]) -> ProviderResult[list[list[float]]]:
        started_at, started = _started()
        vectors = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        value = [[float(item) for item in row] for row in vectors]
        return ProviderResult(
            value,
            self.identity,
            _usage(started_at, started, provider="local", input_units=len(texts)),
        )


class LocalEditorialProvider:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        revision: str = "main",
        *,
        quantization: str = "none",
        device_map: str = "auto",
    ) -> None:
        self.identity = ModelIdentity(
            model_id, revision, quantization, "transformers", "editor-v1", "editorial-json-v1"
        )
        self.device_map = device_map
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                transformers = importlib.import_module("transformers")
            except ImportError as exc:
                raise ProviderUnavailable("install clipper[editorial]") from exc
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.identity.model_id, revision=self.identity.revision
            )
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.identity.model_id,
                revision=self.identity.revision,
                device_map=self.device_map,
                torch_dtype="auto",
            )
        return self._tokenizer, self._model

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        started_at, started = _started()
        tokenizer, model = self._load()
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON. Never invent spoken words; "
                    "reference canonical word IDs."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"task": task, "payload": payload}, ensure_ascii=False),
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("editorial model did not return valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("editorial model JSON must be an object")
        return ProviderResult(
            value,
            self.identity,
            _usage(
                started_at,
                started,
                provider="local",
                input_units=int(inputs["input_ids"].numel()),
                output_units=int(generated.numel()),
            ),
        )


class LocalVisionProvider:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        revision: str = "main",
        *,
        device_map: str = "auto",
    ) -> None:
        self.identity = ModelIdentity(
            model_id, revision, "none", "transformers", "vision-v1", "vision-json-v1"
        )
        self.device_map = device_map
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            try:
                transformers = importlib.import_module("transformers")
            except ImportError as exc:
                raise ProviderUnavailable("install clipper[vision]") from exc
            self._processor = transformers.AutoProcessor.from_pretrained(
                self.identity.model_id, revision=self.identity.revision
            )
            self._model = transformers.AutoModelForMultimodalLM.from_pretrained(
                self.identity.model_id,
                revision=self.identity.revision,
                device_map=self.device_map,
                dtype="auto",
            )
        return self._processor, self._model

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        if not frames:
            raise ValueError("vision inspection requires at least one frame")
        started_at, started = _started()
        processor, model = self._load()
        content: list[dict[str, str]] = [
            {"type": "image", "url": frame.resolve().as_uri()} for frame in frames
        ]
        content.append(
            {
                "type": "text",
                "text": (
                    "Return only valid JSON. Do not retranscribe audio. "
                    + json.dumps({"task": task, "context": context}, ensure_ascii=False)
                ),
            }
        )
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        text = processor.decode(generated, skip_special_tokens=True).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("vision model did not return valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("vision model JSON must be an object")
        return ProviderResult(
            value,
            self.identity,
            _usage(
                started_at,
                started,
                provider="local",
                input_units=int(inputs["input_ids"].numel()),
                output_units=int(generated.numel()),
            ),
        )
