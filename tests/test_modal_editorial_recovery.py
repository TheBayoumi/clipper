from pathlib import Path
from typing import Any

import pytest

from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.providers.editorial_prompt import (
    EDITORIAL_IDENTITY,
    EDITORIAL_SCHEMA_IDENTITY,
    editorial_contract,
    editorial_contract_fingerprint,
    editorial_json_schema,
    editorial_output_budget,
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


def test_editorial_provider_recovers_from_strict_json_contract_errors() -> None:
    provider = SequenceEditorialProvider(
        [
            ModalRemoteError(
                function_name="editorial",
                error_type="EditorialOutputTruncated",
                message="output exhausted generation budget",
            ),
            ModalRemoteError(
                function_name="editorial",
                error_type="JSONDecodeError",
                message="unterminated string",
            ),
            _result({"cores": []}),
        ]
    )
    result = provider.complete_json(task="semantic_cores:0", payload={"words": []})
    assert result.value == {"cores": []}
    assert [item.get("generation_recovery_attempt") for item in provider.requests] == [None, 2, 3]
    assert "strict JSON object" in provider.requests[1]["generation_recovery_instruction"]


def test_editorial_provider_does_not_retry_unrelated_remote_error() -> None:
    provider = SequenceEditorialProvider(
        [ModalRemoteError(function_name="editorial", error_type="RuntimeError", message="gpu failed")]
    )
    with pytest.raises(ModalRemoteError, match="gpu failed"):
        provider.complete_json(task="semantic_cores:0", payload={"words": []})
    assert len(provider.requests) == 1


def test_editorial_provider_stops_after_bounded_recovery_attempts() -> None:
    failures = [
        ModalRemoteError(function_name="editorial", error_type="JSONDecodeError", message="bad json")
        for _ in range(3)
    ]
    provider = SequenceEditorialProvider(failures)
    with pytest.raises(ModalRemoteError, match="bad json"):
        provider.complete_json(task="quality_windows:core", payload={})
    assert len(provider.requests) == 3


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
    assert editorial_output_budget({"task": "source_hazards:0"}) == 2048
    assert editorial_output_budget({"task": "semantic_cores:0"}) == 2048
    assert editorial_output_budget({"task": "narrative_envelope:core"}) == 1536
    assert editorial_output_budget({"task": "quality_windows:core"}) == 1536
    with pytest.raises(ValueError, match="unsupported production editorial task"):
        editorial_task_family("edit_plans:legacy")


def test_modal_editorial_runtime_has_bounded_expanding_recovery_budget() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "class EditorialOutputTruncated(ValueError)" in source
    assert "base_budget * _editorial_recovery_attempt(payload)" in source
    assert "return min(4096," in source
    assert "exhausting the output budget" in source
