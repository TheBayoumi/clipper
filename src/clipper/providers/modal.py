from __future__ import annotations

import base64
import importlib
from pathlib import Path
from typing import Any

from .base import InferenceUsage, ModelIdentity, ProviderResult
from .local import ProviderUnavailable


class ModalJSONProvider:
    def __init__(self, *, app_name: str, function_name: str, identity: ModelIdentity) -> None:
        self.app_name = app_name
        self.function_name = function_name
        self.identity = identity

    def _resolved_identity(self, response: dict[str, Any]) -> ModelIdentity:
        raw = response.get("model")
        if not isinstance(raw, dict):
            return self.identity
        return ModelIdentity(
            str(raw.get("model_id") or self.identity.model_id),
            str(raw.get("revision") or self.identity.revision),
            self.identity.quantization,
            self.identity.inference_engine,
            self.identity.prompt_version,
            self.identity.schema_version,
        )

    def _function(self) -> Any:
        try:
            modal = importlib.import_module("modal")
        except ImportError as exc:
            raise ProviderUnavailable("install clipper[modal]") from exc
        return modal.Function.from_name(self.app_name, self.function_name)

    def invoke(self, payload: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
        response = self._function().remote(payload)
        if not isinstance(response, dict) or not isinstance(response.get("value"), dict):
            raise ValueError("Modal provider returned an invalid response")
        raw_usage = response.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return ProviderResult(
            response["value"],
            self._resolved_identity(response),
            InferenceUsage(
                provider="modal",
                started_at=str(usage.get("started_at") or "unknown"),
                duration_seconds=float(usage.get("duration_seconds") or 0.0),
                gpu_type=str(usage["gpu_type"]) if usage.get("gpu_type") else None,
                gpu_seconds=float(usage.get("gpu_seconds") or 0.0),
                peak_vram_mb=float(usage["peak_vram_mb"])
                if usage.get("peak_vram_mb") is not None
                else None,
                input_units=int(usage.get("input_units") or 0),
                output_units=int(usage.get("output_units") or 0),
                estimated_cost_usd=float(usage.get("estimated_cost_usd") or 0.0),
            ),
        )


class ModalEditorialProvider(ModalJSONProvider):
    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        return self.invoke({"task": task, "payload": payload})


class ModalVisionProvider(ModalJSONProvider):
    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        encoded_frames = [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in frames]
        return self.invoke({"task": task, "frames_base64": encoded_frames, "context": context})


class ModalEmbeddingProvider(ModalJSONProvider):
    def embed(self, texts: list[str]) -> ProviderResult[list[list[float]]]:
        response = self._function().remote({"texts": texts})
        if not isinstance(response, dict) or not isinstance(response.get("vectors"), list):
            raise ValueError("Modal embedding provider returned an invalid response")
        vectors: list[list[float]] = []
        for row in response["vectors"]:
            if not isinstance(row, list):
                raise ValueError("Modal embedding vector must be a list")
            vectors.append([float(value) for value in row])
        raw_usage = response.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        raw_model = response.get("model")
        identity = self.identity
        if isinstance(raw_model, dict):
            identity = ModelIdentity(
                str(raw_model.get("model_id") or self.identity.model_id),
                str(raw_model.get("revision") or self.identity.revision),
                self.identity.quantization,
                self.identity.inference_engine,
                self.identity.prompt_version,
                self.identity.schema_version,
            )
        return ProviderResult(
            vectors,
            identity,
            InferenceUsage(
                provider="modal",
                started_at=str(usage.get("started_at") or "unknown"),
                duration_seconds=float(usage.get("duration_seconds") or 0.0),
                gpu_type=str(usage["gpu_type"]) if usage.get("gpu_type") else None,
                gpu_seconds=float(usage.get("gpu_seconds") or 0.0),
                peak_vram_mb=(
                    float(usage["peak_vram_mb"]) if usage.get("peak_vram_mb") is not None else None
                ),
                input_units=int(usage.get("input_units") or len(texts)),
                output_units=int(usage.get("output_units") or len(vectors)),
                estimated_cost_usd=float(usage.get("estimated_cost_usd") or 0.0),
            ),
        )
