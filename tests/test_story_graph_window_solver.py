from dataclasses import replace

import pytest

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.models import SourceSpan
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.window_solver import enumerate_feasible_windows, validate_feasible_window


def _timeline(word_count: int = 50) -> CanonicalTimeline:
    words = tuple(
        CanonicalWord(
            f"video:w{index:07d}:x",
            f"word-{index}",
            float(index),
            float(index + 1),
            "speaker-a",
            1.0,
            "word_exact",
            "test",
        )
        for index in range(word_count)
    )
    return CanonicalTimeline("video", "source", words)


def _core(timeline: CanonicalTimeline) -> SemanticCore:
    return SemanticCore.from_word_ids(
        timeline,
        core_id="core-1",
        source_word_ids=tuple(word.word_id for word in timeline.words[10:13]),
        semantic_summary="small interesting nucleus",
        editorial_reason="worthwhile source-grounded event",
        confidence=0.9,
    )


def _envelope(
    timeline: CanonicalTimeline,
    core: SemanticCore,
    start_index: int,
    end_index: int,
) -> NarrativeEnvelope:
    return NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id=f"envelope-{start_index}-{end_index}",
        source_word_ids=tuple(word.word_id for word in timeline.words[start_index:end_index]),
        required_prior_context="setup",
        required_followup_context="payoff",
        setup_resolved=True,
        payoff_resolved=True,
        reference_resolution=("reference resolved",),
        confidence=0.9,
    )


def test_short_semantic_core_can_own_long_complete_narrative_envelope() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core, 2, 35)
    assert core.duration == 3.0
    assert envelope.duration == 33.0
    assert envelope.contains(core)
    assert envelope.complete


def test_historical_subminimum_edit_plan_failure_is_impossible_by_construction() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core, 8, 18)
    assert core.duration == 3.0
    assert envelope.duration == 10.0

    windows = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20.0,
        max_seconds=25.0,
    )

    assert windows
    assert all(20.0 <= window.duration <= 25.0 for window in windows)
    assert all(set(envelope.source_word_ids).issubset(window.source_word_ids) for window in windows)
    assert all(set(core.source_word_ids).issubset(window.source_word_ids) for window in windows)
    assert all(window.source_start <= envelope.source_start for window in windows)
    assert all(window.source_end >= envelope.source_end for window in windows)


def test_campaign_valid_complete_envelope_is_the_canonical_window() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core, 5, 28)
    assert envelope.duration == 23.0

    windows = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20.0,
        max_seconds=25.0,
    )

    assert len(windows) == 1
    window = windows[0]
    assert window.source_start == envelope.source_start
    assert window.source_end == envelope.source_end
    assert window.source_word_ids == envelope.source_word_ids


def test_overlong_complete_envelope_cannot_be_amputated_to_hit_campaign_maximum() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core, 2, 35)
    assert envelope.duration == 33.0

    assert (
        enumerate_feasible_windows(
            timeline,
            core,
            envelope,
            min_seconds=20.0,
            max_seconds=25.0,
        )
        == ()
    )


def test_forbidden_source_spans_are_removed_before_model_ranking() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core, 8, 18)
    forbidden = (SourceSpan(0.0, 4.0),)

    baseline = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20.0,
        max_seconds=25.0,
    )
    windows = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20.0,
        max_seconds=25.0,
        forbidden_spans=forbidden,
    )

    assert baseline
    assert any(window.overlaps(forbidden[0]) for window in baseline)
    assert windows
    assert all(not window.overlaps(forbidden[0]) for window in windows)


def test_incomplete_envelope_emits_no_delivery_window() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core, 8, 18)
    incomplete = replace(envelope, payoff_resolved=False)

    assert (
        enumerate_feasible_windows(
            timeline,
            core,
            incomplete,
            min_seconds=20.0,
            max_seconds=25.0,
        )
        == ()
    )


def test_window_validation_is_fail_closed_for_duration_envelope_and_policy() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core, 8, 18)
    window = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20.0,
        max_seconds=25.0,
    )[0]

    validate_feasible_window(
        window,
        core,
        envelope,
        min_seconds=20.0,
        max_seconds=25.0,
    )

    with pytest.raises(ValueError, match="duration"):
        validate_feasible_window(
            window,
            core,
            envelope,
            min_seconds=24.0,
            max_seconds=25.0,
        )

    amputated = replace(
        window,
        source_start=envelope.source_start + 1.0,
        source_word_ids=window.source_word_ids[1:],
    )
    with pytest.raises(ValueError, match="amputates the start"):
        validate_feasible_window(
            amputated,
            core,
            envelope,
            min_seconds=1.0,
            max_seconds=25.0,
        )

    with pytest.raises(ValueError, match="forbidden"):
        validate_feasible_window(
            window,
            core,
            envelope,
            min_seconds=20.0,
            max_seconds=25.0,
            forbidden_spans=(SourceSpan(window.source_start, window.source_start + 0.5),),
        )


def test_story_graph_rejects_noncontiguous_or_noncontaining_provenance() -> None:
    timeline = _timeline()
    with pytest.raises(ValueError, match="contiguous"):
        SemanticCore.from_word_ids(
            timeline,
            core_id="core",
            source_word_ids=(timeline.words[1].word_id, timeline.words[3].word_id),
            semantic_summary="summary",
            editorial_reason="reason",
            confidence=0.5,
        )

    core = _core(timeline)
    with pytest.raises(ValueError, match="contain"):
        NarrativeEnvelope.from_word_ids(
            timeline,
            core,
            envelope_id="bad",
            source_word_ids=tuple(word.word_id for word in timeline.words[20:30]),
            setup_resolved=True,
            payoff_resolved=True,
            confidence=0.5,
        )
