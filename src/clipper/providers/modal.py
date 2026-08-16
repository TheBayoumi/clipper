from __future__ import annotations

import base64
import importlib
from pathlib import Path
from typing import Any

from .base import InferenceUsage, ModelIdentity, ProviderResult
from .local import ProviderUnavailable


class ModalRemoteError(RuntimeError):
    """Structured error returned by a remote Modal inference function."""

    def __init__(self, *, function_name: str, error_type: str, message: str) -> None:
        self.function_name = function_name
        self.error_type = error_type
        self.remote_message = message
        super().__init__(f"Modal {function_name} failed: {error_type}: {message}")


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

    def _raise_remote_error(self, response: dict[str, Any]) -> None:
        raw_error = response.get("error")
        if not isinstance(raw_error, dict):
            return
        error_type = str(raw_error.get("type") or "RemoteError")
        message = str(raw_error.get("message") or "remote inference failed")
        raise ModalRemoteError(
            function_name=self.function_name,
            error_type=error_type,
            message=message,
        )

    def invoke(self, payload: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
        response = self._function().remote(payload)
        if not isinstance(response, dict):
            raise ValueError("Modal provider returned an invalid response")
        self._raise_remote_error(response)
        if not isinstance(response.get("value"), dict):
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
    _MAX_OUTPUT_RECOVERY_ATTEMPTS = 3

    @staticmethod
    def _is_output_contract_error(exc: ModalRemoteError) -> bool:
        if exc.error_type in {"JSONDecodeError", "EditorialOutputTruncated"}:
            return True
        return (
            exc.error_type == "ValueError"
            and "model output must be a JSON object" in exc.remote_message
        )

    @staticmethod
    def _edit_plan_duration_feedback(
        task: str,
        value: dict[str, Any],
        payload: dict[str, Any],
    ) -> str | None:
        """Return recovery feedback when every proposed EditPlan violates duration bounds."""

        if not task.startswith("edit_plans:"):
            return None
        raw_plans = value.get("plans")
        campaign = payload.get("campaign")
        context_words = payload.get("source_context_words")
        if (
            not isinstance(raw_plans, list)
            or not raw_plans
            or not isinstance(campaign, dict)
            or not isinstance(context_words, list)
        ):
            return None

        try:
            minimum = float(campaign["min_clip_seconds"])
            maximum = float(campaign["max_clip_seconds"])
        except (KeyError, TypeError, ValueError):
            return None
        if minimum <= 0 or maximum < minimum:
            return None

        timings: dict[str, tuple[float, float]] = {}
        for item in context_words:
            if not isinstance(item, dict):
                continue
            word_ref = item.get("word_ref")
            source_start = item.get("source_start")
            source_end = item.get("source_end")
            if (
                isinstance(word_ref, str)
                and word_ref
                and isinstance(source_start, int | float)
                and isinstance(source_end, int | float)
            ):
                timings[word_ref] = (float(source_start), float(source_end))

        if not timings:
            return None

        violations: list[str] = []
        valid_count = 0
        for index, raw_plan in enumerate(raw_plans):
            if not isinstance(raw_plan, dict):
                violations.append(f"plan[{index}] is not an object")
                continue
            start_ref = raw_plan.get("source_start_word_id")
            end_ref = raw_plan.get("source_end_word_id")
            if not isinstance(start_ref, str) or not isinstance(end_ref, str):
                violations.append(f"plan[{index}] is missing source range references")
                continue
            start_timing = timings.get(start_ref)
            end_timing = timings.get(end_ref)
            if start_timing is None or end_timing is None:
                violations.append(
                    f"plan[{index}] references a range outside source_context_words"
                )
                continue
            duration = end_timing[1] - start_timing[0]
            if minimum <= duration <= maximum:
                valid_count += 1
                continue
            violations.append(f"plan[{index}] duration={duration:.3f}s")

        if valid_count > 0 or not violations:
            return None

        observed = "; ".join(violations[:8])
        return (
            "Every proposed EditPlan violated the campaign duration contract. "
            f"The campaign requires {minimum:g}-{maximum:g} seconds. {observed}. "
            "Regenerate from the original task. For each plan, locate "
            "source_start_word_id and source_end_word_id in source_context_words and compute "
            "duration = end.source_end - start.source_start before emitting the plan. "
            "The selected source range may extend outside the short concept start/end when "
            "needed, but it must remain one coherent chronological source story, contain its "
            "spoken hook, and stay within the campaign bounds. If no coherent range satisfies "
            "the bounds, return an empty plans array instead of an invalid plan."
        )

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        request: dict[str, Any] = {"task": task, "payload": payload}
        for attempt in range(1, self._MAX_OUTPUT_RECOVERY_ATTEMPTS + 1):
            try:
                result = self.invoke(request)
            except ModalRemoteError as exc:
                if (
                    not self._is_output_contract_error(exc)
                    or attempt >= self._MAX_OUTPUT_RECOVERY_ATTEMPTS
                ):
                    raise
                request = {
                    "task": task,
                    "payload": payload,
                    "generation_recovery_attempt": attempt + 1,
                    "generation_recovery_instruction": (
                        "The previous generation violated the JSON output contract. Regenerate the "
                        "complete answer from the original task as exactly one strict JSON object. "
                        "Use valid JSON syntax with double-quoted keys and strings, no Markdown, "
                        "no comments, no prose outside the object, and no truncated fields."
                    ),
                }
                continue

            duration_feedback = self._edit_plan_duration_feedback(task, result.value, payload)
            if duration_feedback is None or attempt >= self._MAX_OUTPUT_RECOVERY_ATTEMPTS:
                return result
            request = {
                "task": task,
                "payload": payload,
                "generation_recovery_attempt": attempt + 1,
                "generation_recovery_instruction": duration_feedback,
            }
        raise AssertionError("unreachable editorial recovery loop")


class ModalVisionProvider(ModalJSONProvider):
    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        encoded_frames = [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in frames]
        return self.invoke({"task": task, "frames_base64": encoded_frames, "context": context})


class ModalEmbeddingProvider(ModalJSONProvider):
    def embed(self, texts: list[str]) -> ProviderResult[list[list[float]]]:
        response = self._function().remote({"texts": texts})
        if not isinstance(response, dict):
            raise ValueError("Modal embedding provider returned an invalid response")
        self._raise_remote_error(response)
        if not isinstance(response.get("vectors"), list):
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
