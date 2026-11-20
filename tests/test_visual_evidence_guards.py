"""Preserve fail-closed checks in the active visual-evidence path."""

from __future__ import annotations

import pytest

from clipper.visual_ai import (
    adaptive_sample_times,
    parse_visual_review,
    visual_evidence_spans_from_samples,
)


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_visual_evidence_rejects_nonpositive_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be positive"):
        visual_evidence_spans_from_samples((0.0,), duration)


def test_visual_evidence_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        visual_evidence_spans_from_samples((), 5.0)


@pytest.mark.parametrize("sample", [-0.01, 5.0, 5.01])
def test_visual_evidence_rejects_samples_outside_source(sample: float) -> None:
    with pytest.raises(ValueError, match="outside source duration"):
        visual_evidence_spans_from_samples((sample,), 5.0)


def test_visual_evidence_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="scope is invalid"):
        visual_evidence_spans_from_samples((1.0,), 5.0, scope="unknown")  # type: ignore[arg-type]


def test_visual_review_rejects_unknown_decision() -> None:
    with pytest.raises(ValueError, match="decision is invalid"):
        parse_visual_review(
            {"decision": "UNKNOWN", "summary": "Invalid review", "overall_confidence": 1.0}
        )


def test_visual_sampling_skips_empty_candidate_ranges() -> None:
    baseline = adaptive_sample_times(5.0)
    assert adaptive_sample_times(5.0, candidate_ranges=((2.0, 2.0), (3.0, 1.0))) == baseline
