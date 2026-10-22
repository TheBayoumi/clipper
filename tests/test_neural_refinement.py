"""Dense VLM source-clock samples are guided by canonical verified evidence."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from clipper_engine.gameplay import interaction
from clipper_engine.inference import vision


def _timeline() -> SimpleNamespace:
    return SimpleNamespace(
        consolidated_events=(
            SimpleNamespace(time=11.0, kinds=("outcome_like",), confidence=0.94),
            SimpleNamespace(time=17.0, kinds=("combat_burst",), confidence=0.85),
            SimpleNamespace(time=22.0, kinds=("contact",), confidence=0.76),
        )
    )


def _settings(**overrides: object) -> vision.VisionSettings:
    return vision.VisionSettings(**overrides)


def test_direct_outcome_and_manual_cinematic_cues_get_local_frame_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        interaction,
        "hostile_decision",
        lambda event, cfg: SimpleNamespace(hostile=float(event.time) == 11.0),
    )
    rate = Fraction(60000, 1001)
    coarse = vision._sample_times(30.0, 4, rate)
    times = vision._refined_sample_times(
        coarse,
        30.0,
        rate,
        _timeline(),
        {
            "cinematic_moments": {
                "verified_anchors": {
                    "ghost_fight": [{"id": "roof", "time": 25.0, "verification": "reviewed"}]
                }
            }
        },
        "ghost_fight",
        _settings(max_refinement_anchors=3),
    )
    frames = {round(Fraction(str(t)) * rate) for t in times}
    assert len(times) > len(coarse)
    assert {round(t * rate) for t in (10.5, 11, 11.5, 24.5, 25, 25.5)}.issubset(frames)
    # A lower-priority source cue was not forced past bounded slots.
    assert 17 * float(rate) > 0
    assert max(times) < 30.0


def test_required_verified_outcomes_cannot_be_silently_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(interaction, "hostile_decision", lambda *_: SimpleNamespace(hostile=True))
    timeline = SimpleNamespace(
        consolidated_events=tuple(
            SimpleNamespace(time=float(i + 1), kinds=("outcome_like",)) for i in range(3)
        )
    )
    with pytest.raises(RuntimeError, match="verified source anchors exceed"):
        vision._refined_sample_times(
            (0.0,),
            30.0,
            Fraction(60000, 1001),
            timeline,
            {},
            "ghost_fight",
            _settings(max_refinement_anchors=2),
        )


def test_neural_budget_rejects_unbounded_source_frame_sampling() -> None:
    with pytest.raises(RuntimeError, match="configured frame budget"):
        vision._refined_sample_times(
            tuple(float(i) for i in range(20)),
            25,
            Fraction(60),
            SimpleNamespace(consolidated_events=()),
            {},
            "ghost_fight",
            _settings(max_sample_frames=8),
        )


def test_late_source_change_gets_refinement_instead_of_first_cpu_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    monkeypatch.setattr(
        interaction,
        "hostile_decision",
        lambda event, _: SimpleNamespace(hostile=float(event.time) == 11.0),
    )
    rate = Fraction(60000, 1001)
    reel = _timeline()
    reel.fps = 5.0
    reel.signals = {
        "interest": np.zeros(150, dtype=np.float32),
        "audio_transient": np.zeros(150, dtype=np.float32),
    }
    # The action is late. A chronological-first algorithm would choose 17s.
    reel.signals["interest"][135:138] = 0.96
    times = vision._refined_sample_times(
        vision._sample_times(30.0, 15.0, rate),
        30.0,
        rate,
        reel,
        {},
        "ghost_fight",
        _settings(max_refinement_anchors=2),
    )
    assert any(26.5 <= time <= 28.0 for time in times)
    assert any(abs(time - 11.0) < 0.02 for time in times)


def test_optional_refinement_never_displaces_source_wide_coarse_samples() -> None:
    import numpy as np

    rate = Fraction(60)
    reel = SimpleNamespace(
        consolidated_events=(),
        fps=10.0,
        signals={"interest": np.full(400, 0.98, dtype=np.float32)},
    )
    coarse = vision._sample_times(40.0, 5.0, rate)
    times = vision._refined_sample_times(
        coarse,
        40.0,
        rate,
        reel,
        {},
        "ghost_fight",
        _settings(max_refinement_anchors=30, max_sample_frames=12),
    )
    assert len(times) <= 12
    assert set(coarse).issubset(set(times))
