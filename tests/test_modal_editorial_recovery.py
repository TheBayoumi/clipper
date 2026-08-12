from pathlib import Path
from typing import Any

from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
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


def _success() -> ProviderResult[dict[str, Any]]:
    identity = ModelIdentity("test/editorial", "rev", "none", "test", "prompt", "schema")
    return ProviderResult(
        {"moments": []},
        identity,
        InferenceUsage(provider="modal", started_at="now", duration_seconds=0.0),
    )


def test_editorial_provider_recovers_from_truncation_then_json_error() -> None:
    provider = SequenceEditorialProvider(
        [
            ModalRemoteError(
                function_name="editorial",
                error_type="EditorialOutputTruncated",
                message="task=story_moments:21 attempt=1 exhausted max_new_tokens=1536",
            ),
            ModalRemoteError(
                function_name="editorial",
                error_type="JSONDecodeError",
                message="task=story_moments:21 attempt=2 unterminated string",
            ),
            _success(),
        ]
    )
    result = provider.complete_json(task="story_moments:21", payload={"words": []})
    assert result.value == {"moments": []}
    assert [item.get("generation_recovery_attempt") for item in provider.requests] == [None, 2, 3]


def test_modal_editorial_runtime_has_bounded_expanding_recovery_budget() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "class EditorialOutputTruncated(ValueError)" in source
    assert "base_budget * _editorial_recovery_attempt(payload)" in source
    assert "return min(4096," in source
    assert "return fewer valid items with shorter prose rather than truncating" in source
    assert "context=f\"task={task or '<missing>'} attempt={recovery_attempt}\"" in source
