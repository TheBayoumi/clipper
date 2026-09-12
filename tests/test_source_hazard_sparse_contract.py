from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.editorial_integrity import HazardClassification
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.providers.editorial_prompt import editorial_contract, editorial_json_schema
from clipper.source_hazards import SourceHazardClassifier


class _HazardEditorial:
    identity = ModelIdentity(
        "hazard-test-model",
        "hazard-test-revision",
        "none",
        "test",
        "editor",
        "editorial-json",
    )

    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value
        self.payloads: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        assert task.startswith("source_hazards:")
        self.payloads.append(payload)
        return ProviderResult(
            self.value,
            self.identity,
            InferenceUsage(
                provider="test",
                started_at=datetime.now(UTC).isoformat(),
                duration_seconds=0.01,
                input_units=10,
                output_units=5,
            ),
        )


def _timeline(count: int = 4) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:x",
                f"word-{index}",
                float(index),
                float(index + 1),
                "speaker",
                0.99,
                "word_exact",
                "test",
            )
            for index in range(count)
        ),
    )


def _refs(timeline: CanonicalTimeline) -> tuple[str, str]:
    return (
        timeline.word_ref(timeline.words[0].word_id),
        timeline.word_ref(timeline.words[-1].word_id),
    )


def test_source_hazard_schema_is_sparse_and_requires_exact_coverage_attestation() -> None:
    schema = editorial_json_schema("source_hazards:test")
    properties = schema["properties"]
    assert schema["required"] == [
        "coverage_start_word_id",
        "coverage_end_word_id",
        "coverage_complete",
        "segments",
    ]
    classifications = properties["segments"]["items"]["properties"]["classification"]["enum"]
    assert "editorial_content" not in classifications
    assert "unknown" in classifications

    contract = editorial_contract("source_hazards:test")
    assert "Do not emit ordinary editorial_content spans" in contract
    assert "coverage_complete" in contract
    assert "empty segments array" in contract


def test_exact_complete_coverage_allows_empty_exception_list(tmp_path: Path) -> None:
    timeline = _timeline()
    start_ref, end_ref = _refs(timeline)
    provider = _HazardEditorial(
        {
            "coverage_start_word_id": start_ref,
            "coverage_end_word_id": end_ref,
            "coverage_complete": True,
            "segments": [],
        }
    )
    result = SourceHazardClassifier(provider, DagStore(tmp_path / "dag")).classify(
        load_brief("campaigns/lovable-clipping.yaml"),
        timeline,
        multimodal=None,
    )

    assert result.hazards == ()
    assert result.rejections == ()
    assert result.stage_executions == 1
    assert "do not emit ordinary editorial_content" in provider.payloads[0]["instruction"]


def test_missing_coverage_attestation_fails_closed_to_unknown(tmp_path: Path) -> None:
    timeline = _timeline()
    provider = _HazardEditorial({"segments": []})
    result = SourceHazardClassifier(provider, DagStore(tmp_path / "dag")).classify(
        load_brief("campaigns/lovable-clipping.yaml"),
        timeline,
        multimodal=None,
    )

    assert len(result.hazards) == 1
    assert result.hazards[0].classification == HazardClassification.UNKNOWN
    assert result.hazards[0].source_word_ids == tuple(word.word_id for word in timeline.words)
    assert len(result.rejections) == 1
    assert result.rejections[0]["decision"] == "ESCALATE"
    assert "missing complete coverage attestation" in str(result.rejections[0]["error"])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("coverage_complete", False, "coverage attestation is incomplete"),
        ("coverage_start_word_id", "wrong-start", "coverage start"),
        ("coverage_end_word_id", "wrong-end", "coverage end"),
    ],
)
def test_inexact_coverage_attestation_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    timeline = _timeline()
    start_ref, end_ref = _refs(timeline)
    payload: dict[str, Any] = {
        "coverage_start_word_id": start_ref,
        "coverage_end_word_id": end_ref,
        "coverage_complete": True,
        "segments": [],
    }
    payload[field] = value
    result = SourceHazardClassifier(_HazardEditorial(payload), DagStore(tmp_path / field)).classify(
        load_brief("campaigns/lovable-clipping.yaml"),
        timeline,
        multimodal=None,
    )

    assert len(result.hazards) == 1
    assert result.hazards[0].classification == HazardClassification.UNKNOWN
    assert error in str(result.rejections[0]["error"])


def test_sparse_attestation_rejects_ordinary_editorial_content(tmp_path: Path) -> None:
    timeline = _timeline()
    start_ref, end_ref = _refs(timeline)
    provider = _HazardEditorial(
        {
            "coverage_start_word_id": start_ref,
            "coverage_end_word_id": end_ref,
            "coverage_complete": True,
            "segments": [
                {
                    "start_word_id": start_ref,
                    "end_word_id": end_ref,
                    "classification": "editorial_content",
                    "confidence": 0.99,
                    "evidence": ["ordinary source content"],
                }
            ],
        }
    )
    result = SourceHazardClassifier(provider, DagStore(tmp_path / "dag")).classify(
        load_brief("campaigns/lovable-clipping.yaml"),
        timeline,
        multimodal=None,
    )

    assert len(result.hazards) == 1
    assert result.hazards[0].classification == HazardClassification.UNKNOWN
    assert "must omit ordinary editorial_content" in str(result.rejections[0]["error"])


def test_legacy_exhaustive_result_is_accepted_only_when_it_covers_every_word(
    tmp_path: Path,
) -> None:
    timeline = _timeline()
    start_ref, end_ref = _refs(timeline)
    exhaustive = _HazardEditorial(
        {
            "segments": [
                {
                    "start_word_id": start_ref,
                    "end_word_id": end_ref,
                    "classification": "editorial_content",
                    "confidence": 0.99,
                    "evidence": ["legacy exhaustive classification"],
                }
            ]
        }
    )
    result = SourceHazardClassifier(exhaustive, DagStore(tmp_path / "exhaustive")).classify(
        load_brief("campaigns/lovable-clipping.yaml"),
        timeline,
        multimodal=None,
    )
    assert len(result.hazards) == 1
    assert result.hazards[0].classification == HazardClassification.EDITORIAL_CONTENT
    assert result.rejections == ()

    partial = _HazardEditorial(
        {
            "segments": [
                {
                    "start_word_id": start_ref,
                    "end_word_id": timeline.word_ref(timeline.words[1].word_id),
                    "classification": "editorial_content",
                    "confidence": 0.99,
                    "evidence": ["legacy partial classification"],
                }
            ]
        }
    )
    failed = SourceHazardClassifier(partial, DagStore(tmp_path / "partial")).classify(
        load_brief("campaigns/lovable-clipping.yaml"),
        timeline,
        multimodal=None,
    )
    assert len(failed.hazards) == 1
    assert failed.hazards[0].classification == HazardClassification.UNKNOWN
    assert "missing complete coverage attestation" in str(failed.rejections[0]["error"])
