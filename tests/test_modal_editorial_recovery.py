from pathlib import Path
from typing import Any

import pytest

from clipper.providers.base import (
    EditorialCapacityError,
    InferenceUsage,
    ModelIdentity,
    ProviderResult,
)
from clipper.providers.editorial_prompt import (
    EDITORIAL_IDENTITY,
    EDITORIAL_SCHEMA_IDENTITY,
    editorial_contract,
    editorial_contract_fingerprint,
    editorial_json_schema,
    editorial_task_family,
)
from clipper.providers.modal import ModalEditorialProvider, ModalRemoteError


class SequenceEditorialProvider(ModalEditorialProvider):
    def __init__(self, outcomes: list[object]) -> None:
        identity = ModelIdentity("test/editorial", "rev", "none", "test", "prompt", "schema")
        super().__init__(app_name="test", function_name="editorial", identity=identity)
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
        self.requests.append(payload)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, ProviderResult)
        return outcome


def _result(value: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
    identity = ModelIdentity("test/editorial", "rev", "none", "test", "prompt", "schema")
    return ProviderResult(
        value,
        identity,
        InferenceUsage(provider="modal", started_at="now", duration_seconds=0.0),
    )


def test_editorial_provider_recovers_only_when_remote_capacity_expands_monotonically() -> None:
    provider = SequenceEditorialProvider(
        [
            ModalRemoteError(
                function_name="editorial",
                error_type="EditorialOutputTruncated",
                message="output exhausted runtime-derived budget",
                details={
                    "generation_budget_tokens": 100,
                    "next_output_budget_tokens": 200,
                    "generated_sha256": "first",
                },
            ),
            ModalRemoteError(
                function_name="editorial",
                error_type="EditorialOutputTruncated",
                message="output exhausted expanded runtime-derived budget",
                details={
                    "generation_budget_tokens": 200,
                    "next_output_budget_tokens": 400,
                    "generated_sha256": "second",
                },
            ),
            _result({"cores": []}),
        ]
    )
    result = provider.complete_json(task="semantic_cores:0", payload={"words": []})
    assert result.value == {"cores": []}
    assert [item.get("generation_minimum_output_tokens") for item in provider.requests] == [
        None,
        200,
        400,
    ]
    assert (
        "runtime-derived output capacity" in provider.requests[1]["generation_recovery_instruction"]
    )


def test_editorial_provider_does_not_retry_unrelated_remote_error() -> None:
    provider = SequenceEditorialProvider(
        [
            ModalRemoteError(
                function_name="editorial", error_type="RuntimeError", message="gpu failed"
            )
        ]
    )
    with pytest.raises(ModalRemoteError, match="gpu failed"):
        provider.complete_json(task="semantic_cores:0", payload={"words": []})
    assert len(provider.requests) == 1


def test_editorial_provider_stops_when_output_capacity_cannot_expand() -> None:
    provider = SequenceEditorialProvider(
        [
            ModalRemoteError(
                function_name="editorial",
                error_type="EditorialOutputTruncated",
                message="no further output headroom",
                details={
                    "generation_budget_tokens": 200,
                    "next_output_budget_tokens": 200,
                    "generated_sha256": "same",
                },
            )
        ]
    )
    with pytest.raises(ModalRemoteError, match="no further output headroom"):
        provider.complete_json(task="quality_windows:core", payload={})
    assert len(provider.requests) == 1


def test_editorial_provider_maps_remote_oom_to_capacity_signal() -> None:
    provider = SequenceEditorialProvider(
        [
            ModalRemoteError(
                function_name="editorial",
                error_type="OutOfMemoryError",
                message="CUDA out of memory",
                details={"input_tokens": 1234, "reason": "cuda_oom_after_offloaded_cache"},
            )
        ]
    )
    with pytest.raises(EditorialCapacityError, match="CUDA out of memory") as caught:
        provider.complete_json(task="semantic_cores:range", payload={})
    assert caught.value.details["input_tokens"] == 1234
    assert caught.value.details["remote_error_type"] == "OutOfMemoryError"
    assert len(provider.requests) == 1


def test_editorial_contract_exposes_only_active_content_addressed_task_families() -> None:
    tasks = {
        "source_hazards:0": "source_hazards",
        "semantic_cores:0": "semantic_cores",
        "narrative_envelope:core": "narrative_envelope",
        "quality_windows:core": "quality_windows",
    }
    assert EDITORIAL_IDENTITY == "editor"
    assert EDITORIAL_SCHEMA_IDENTITY == "structured-json"
    for task, family in tasks.items():
        assert editorial_task_family(task) == family
        schema = editorial_json_schema(task)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert editorial_contract(task)
        assert len(editorial_contract_fingerprint(task)) == 64
    with pytest.raises(ValueError, match="unsupported production editorial task"):
        editorial_task_family("edit_plans:legacy")


def test_modal_editorial_runtime_is_model_and_history_derived() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "def _editorial_context_limit(" in source
    assert "def _editorial_generation_plan(" in source
    assert "def _load_editorial_capacity_state(" in source
    assert "def _persist_editorial_capacity_state(" in source
    assert 'device_map="balanced_low_0"' in source
    assert '"logits_to_keep": 1' in source
    assert '"cache_implementation" = "offloaded"' not in source
    assert 'kwargs["cache_implementation"] = "offloaded"' in source
    assert '"event": "editorial_oom"' in source
    assert '"event": "editorial_capacity_fallback"' in source
    assert "editorial_output_budget" not in source
