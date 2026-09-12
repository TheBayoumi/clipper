from __future__ import annotations

import importlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import EditorialCapacityError, InferenceUsage, ModelIdentity, ProviderResult
from .editorial_prompt import (
    EDITORIAL_IDENTITY,
    EDITORIAL_SCHEMA_IDENTITY,
    editorial_contract,
)


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


class LocalEditorialProvider:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        revision: str = "f50518eb58dfc750271b273fc113bdfc16ec2280",
        *,
        quantization: str = "none",
        device_map: str = "auto",
    ) -> None:
        self.identity = ModelIdentity(
            model_id,
            revision,
            quantization,
            "transformers",
            EDITORIAL_IDENTITY,
            EDITORIAL_SCHEMA_IDENTITY,
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
                    "You are a source-grounded multimodal short-form editor. "
                    "Infer editorial value from the supplied evidence rather than fixed domain "
                    "templates. Never invent source words, facts, or IDs. "
                    + editorial_contract(task)
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
        model_config = getattr(model, "config", None)
        text_config = getattr(model_config, "text_config", model_config)
        candidates = (
            getattr(text_config, "max_position_embeddings", None),
            getattr(tokenizer, "model_max_length", None),
        )
        context_limit: int | None = None
        for raw_context in candidates:
            if isinstance(raw_context, bool) or not isinstance(raw_context, int | str):
                continue
            try:
                parsed = int(raw_context)
            except (ValueError, OverflowError):
                continue
            if parsed > 0:
                context_limit = parsed
                break
        if context_limit is None:
            raise EditorialCapacityError(
                "local editorial model does not expose a usable context limit"
            )
        input_units = int(inputs["input_ids"].numel())
        available_output = context_limit - input_units
        if available_output <= 0:
            raise EditorialCapacityError(
                "local editorial request exceeds model context",
                details={
                    "input_tokens": input_units,
                    "context_limit_tokens": context_limit,
                },
            )
        output = model.generate(
            **inputs,
            max_new_tokens=available_output,
            do_sample=False,
            use_cache=True,
            logits_to_keep=1,
        )
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
                input_units=input_units,
                output_units=int(generated.numel()),
            ),
        )


class LocalVisionProvider:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        revision: str = "main",
        *,
        quantization: str = "none",
        device_map: str = "auto",
    ) -> None:
        self.identity = ModelIdentity(
            model_id,
            revision,
            quantization,
            "transformers",
            "vision",
            "structured-json",
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
                    "Return only valid JSON. Describe only evidence visible in the supplied "
                    "frames and do not infer spoken words from pixels. "
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
