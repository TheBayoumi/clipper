import pytest

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.multimodal_timeline import MultimodalEvent, MultimodalTimeline
from clipper.quality_moments import QualityMoment, WindowQualityAssessment
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.visual_strategy import VisualBeat, VisualStrategy, derive_visual_strategy
from clipper.window_solver import enumerate_feasible_windows


def _quality_moment() -> QualityMoment:
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
        for index in range(30)
    )
    timeline = CanonicalTimeline("video", "source", words)
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=tuple(word.word_id for word in words[10:13]),
        semantic_summary="source event",
        editorial_reason="worthwhile",
        confidence=0.9,
    )
    envelope = NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id="envelope",
        source_word_ids=tuple(word.word_id for word in words[5:25]),
        setup_resolved=True,
        payoff_resolved=True,
        confidence=0.9,
    )
    window = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20,
        max_seconds=20,
    )[0]
    assessment = WindowQualityAssessment(
        core.core_id,
        window.window_id,
        "PASS",
        0.9,
        "open on the first complete source-grounded statement",
        "strong complete moment",
        0.9,
    )
    return QualityMoment("quality:core", core, envelope, window, assessment)


def test_visual_strategy_prefers_relevant_original_source_evidence() -> None:
    moment = _quality_moment()
    window = moment.delivery_window
    timeline = MultimodalTimeline(
        "video",
        "source",
        30.0,
        (
            MultimodalEvent(
                window.source_start,
                window.source_end,
                scene_ids=("scene-1",),
                actions=("demonstration",),
                visual_summaries=("relevant source action",),
                visual_salience=0.9,
                confidence=0.9,
            ),
        ),
    )
    strategy = derive_visual_strategy(moment, timeline)
    assert strategy.source_first
    assert len(strategy.beats) == 1
    assert strategy.beats[0].source == "original_source"
    assert strategy.beats[0].synthetic_request is None
    assert strategy.beats[0].source_event_range == (
        window.source_start,
        window.source_end,
    )
    assert strategy.to_dict()["quality_moment_id"] == "quality:core"


def test_visual_strategy_stays_on_original_source_when_enrichment_is_unnecessary() -> None:
    moment = _quality_moment()
    timeline = MultimodalTimeline("video", "source", 30.0, ())
    strategy = derive_visual_strategy(moment, timeline)
    assert strategy.beats[0].source == "original_source"
    assert "remain on original source" in strategy.beats[0].rationale


def test_visual_strategy_rejects_source_mismatch() -> None:
    moment = _quality_moment()
    timeline = MultimodalTimeline("video", "other", 30.0, ())
    with pytest.raises(ValueError, match="different sources"):
        derive_visual_strategy(moment, timeline)


def test_visual_beat_synthetic_contract_is_explicit_and_fail_closed() -> None:
    with pytest.raises(ValueError, match="timestamps"):
        VisualBeat(1.0, 1.0, "original_source", "bad")
    with pytest.raises(ValueError, match="rationale"):
        VisualBeat(0.0, 1.0, "original_source", "")
    with pytest.raises(ValueError, match="illustration request"):
        VisualBeat(0.0, 1.0, "synthetic_illustration", "illustrate")
    with pytest.raises(ValueError, match="cannot carry"):
        VisualBeat(
            0.0,
            1.0,
            "original_source",
            "source",
            synthetic_request="not allowed here",
        )


def test_visual_strategy_requires_non_overlapping_chronological_beats() -> None:
    first = VisualBeat(0.0, 2.0, "original_source", "first")
    second = VisualBeat(1.0, 3.0, "original_source", "second")
    with pytest.raises(ValueError, match="non-overlapping"):
        VisualStrategy("quality:core", (first, second))
    with pytest.raises(ValueError, match="at least one beat"):
        VisualStrategy("quality:core", ())
    with pytest.raises(ValueError, match="quality_moment_id"):
        VisualStrategy("", (first,))
