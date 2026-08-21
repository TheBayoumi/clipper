import pytest

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.quality_moments import WindowQualityAssessment, choose_quality_moments
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.window_solver import FeasibleDeliveryWindow


def _graph(
    core_id: str,
    offset: int = 0,
) -> tuple[CanonicalTimeline, SemanticCore, NarrativeEnvelope]:
    words = tuple(
        CanonicalWord(
            f"video:w{index + offset:07d}:x",
            f"w{index}",
            float(index + offset),
            float(index + offset + 1),
            None,
            1.0,
            "word_exact",
            "test",
        )
        for index in range(30)
    )
    timeline = CanonicalTimeline("video", "source", words)
    core = SemanticCore.from_word_ids(
        timeline,
        core_id=core_id,
        source_word_ids=tuple(word.word_id for word in words[10:13]),
        semantic_summary=f"summary {core_id}",
        editorial_reason="quality source moment",
        confidence=0.9,
    )
    envelope = NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id=f"env-{core_id}",
        source_word_ids=tuple(word.word_id for word in words[8:18]),
        setup_resolved=True,
        payoff_resolved=True,
        confidence=0.9,
    )
    return timeline, core, envelope


def _window(
    timeline: CanonicalTimeline,
    core: SemanticCore,
    envelope: NarrativeEnvelope,
    suffix: str,
) -> FeasibleDeliveryWindow:
    start_index, end_index = (0, 20) if suffix in {"a", "first"} else (5, 25)
    words = timeline.words[start_index:end_index]
    return FeasibleDeliveryWindow(
        f"window-{core.core_id}-{suffix}",
        core.core_id,
        envelope.envelope_id,
        core.video_id,
        core.source_hash,
        words[0].source_start,
        words[-1].source_end,
        tuple(word.word_id for word in words),
    )


@pytest.mark.parametrize("count", [0, 1, 2, 7])
def test_quality_yield_equals_number_of_independent_passing_cores(count: int) -> None:
    graph = [_graph(f"core-{index}", index * 40) for index in range(count)]
    cores = tuple(item[1] for item in graph)
    envelopes = tuple(item[2] for item in graph)
    windows = tuple(_window(timeline, core, envelope, "a") for timeline, core, envelope in graph)
    assessments = tuple(
        WindowQualityAssessment(core.core_id, window.window_id, "PASS", 0.8, "worthwhile", 0.9)
        for core, window in zip(cores, windows, strict=True)
    )
    moments = choose_quality_moments(cores, envelopes, windows, assessments)
    assert len(moments) == count
    assert len({moment.core.core_id for moment in moments}) == count


def test_multiple_passing_windows_for_one_core_still_produce_one_quality_moment() -> None:
    timeline, core, envelope = _graph("core")
    first = _window(timeline, core, envelope, "first")
    second = _window(timeline, core, envelope, "second")
    moments = choose_quality_moments(
        (core,),
        (envelope,),
        (first, second),
        (
            WindowQualityAssessment(core.core_id, first.window_id, "PASS", 0.7, "good", 0.8),
            WindowQualityAssessment(core.core_id, second.window_id, "PASS", 0.9, "better", 0.9),
        ),
    )
    assert len(moments) == 1
    assert moments[0].delivery_window.window_id == second.window_id


def test_rejected_or_escalated_windows_do_not_become_quality_moments() -> None:
    timeline, core, envelope = _graph("core")
    window = _window(timeline, core, envelope, "a")
    for decision in ("REJECT", "ESCALATE"):
        assessment = WindowQualityAssessment(
            core.core_id,
            window.window_id,
            decision,
            0.9,
            "not publishable",
            0.9,
        )
        assert choose_quality_moments((core,), (envelope,), (window,), (assessment,)) == ()


def test_quality_assessment_cannot_reference_illegal_or_wrong_core_window() -> None:
    timeline, core, envelope = _graph("core")
    window = _window(timeline, core, envelope, "a")
    unknown = WindowQualityAssessment(core.core_id, "missing", "PASS", 0.9, "good", 0.9)
    with pytest.raises(ValueError, match="unknown window"):
        choose_quality_moments((core,), (envelope,), (window,), (unknown,))

    wrong = WindowQualityAssessment("other", window.window_id, "PASS", 0.9, "good", 0.9)
    with pytest.raises(ValueError, match="identity mismatch"):
        choose_quality_moments((core,), (envelope,), (window,), (wrong,))


def test_quality_moment_rejects_window_that_amputates_complete_envelope() -> None:
    timeline, core, envelope = _graph("core")
    words = timeline.words[10:30]
    amputated = FeasibleDeliveryWindow(
        "window-amputated",
        core.core_id,
        envelope.envelope_id,
        core.video_id,
        core.source_hash,
        words[0].source_start,
        words[-1].source_end,
        tuple(word.word_id for word in words),
    )
    assessment = WindowQualityAssessment(
        core.core_id,
        amputated.window_id,
        "PASS",
        0.9,
        "looks good but is incomplete",
        0.9,
    )
    with pytest.raises(ValueError, match="amputates narrative setup"):
        choose_quality_moments((core,), (envelope,), (amputated,), (assessment,))


def test_quality_assessment_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        WindowQualityAssessment("core", "window", "PASS", 1.2, "reason", 0.5)
    with pytest.raises(ValueError, match="rationale"):
        WindowQualityAssessment("core", "window", "PASS", 0.5, "", 0.5)
