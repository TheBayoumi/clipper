from pathlib import Path
from typing import Any

from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.providers.editorial_prompt import EDITORIAL_PROMPT_VERSION, editorial_contract
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


def _success() -> ProviderResult[dict[str, Any]]:
    return _result({"moments": []})


def _edit_plan_payload() -> dict[str, Any]:
    return {
        "campaign": {"min_clip_seconds": 20, "max_clip_seconds": 45},
        "source_context_words": [
            {"word_ref": "w1", "source_start": 0.0, "source_end": 0.5},
            {"word_ref": "w2", "source_start": 10.0, "source_end": 10.5},
            {"word_ref": "w3", "source_start": 21.0, "source_end": 21.5},
            {"word_ref": "w4", "source_start": 30.0, "source_end": 30.5},
        ],
    }


def _plan(start_ref: str, end_ref: str, *, plan_id: str = "p1") -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "video_id": "video",
        "concept_id": "c1",
        "variant_id": "v1",
        "source_start_word_id": start_ref,
        "source_end_word_id": end_ref,
        "hook_start_word_id": start_ref,
        "hook_end_word_id": start_ref,
        "overlay_text": None,
        "strategy_label": "direct",
        "caption_platform": "tiktok",
        "confidence": 0.9,
    }


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


def test_editorial_provider_replans_when_all_edit_plans_are_outside_duration_bounds() -> None:
    provider = SequenceEditorialProvider(
        [
            _result({"plans": [_plan("w1", "w2")]}),
            _result({"plans": [_plan("w1", "w3")]}),
        ]
    )

    result = provider.complete_json(task="edit_plans:c1", payload=_edit_plan_payload())

    assert result.value["plans"][0]["source_end_word_id"] == "w3"
    assert len(provider.requests) == 2
    recovery = provider.requests[1]
    assert recovery["generation_recovery_attempt"] == 2
    instruction = str(recovery["generation_recovery_instruction"])
    assert "duration=10.500s" in instruction
    assert "requires 20-45 seconds" in instruction
    assert "may extend outside the short concept start/end" in instruction


def test_editorial_provider_keeps_batch_when_at_least_one_edit_plan_has_valid_duration() -> None:
    first = _result(
        {
            "plans": [
                _plan("w1", "w2", plan_id="short"),
                _plan("w1", "w3", plan_id="valid"),
            ]
        }
    )
    provider = SequenceEditorialProvider([first])

    result = provider.complete_json(task="edit_plans:c1", payload=_edit_plan_payload())

    assert result is first
    assert len(provider.requests) == 1


def test_editor_v3_contract_requires_measured_duration_from_timestamped_context() -> None:
    contract = editorial_contract("edit_plans:c1")

    assert EDITORIAL_PROMPT_VERSION == "editor-v3"
    assert "Campaign min_clip_seconds is a hard floor" in contract
    assert "duration = end.source_end - start.source_start" in contract
    assert "may extend before or after the concept start/end" in contract
    assert "omit that plan instead of returning an out-of-bounds range" in contract


def test_modal_editorial_runtime_has_bounded_expanding_recovery_budget() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "class EditorialOutputTruncated(ValueError)" in source
    assert "base_budget * _editorial_recovery_attempt(payload)" in source
    assert "return min(4096," in source
    assert "return fewer valid items with shorter prose rather than truncating" in source
    assert "context=f\"task={task or '<missing>'} attempt={recovery_attempt}\"" in source
