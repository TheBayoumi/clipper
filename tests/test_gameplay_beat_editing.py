"""Source-safe Clipper two-beat editing: do not require 10 contiguous source seconds."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clipper_engine.gameplay import combat, contract, editing, interaction, planning
from clipper_engine.gameplay import semantics as core
from clipper_engine.profiles import load_profile


def _example(monkeypatch: pytest.MonkeyPatch, *, death: bool = False):
    config = load_profile("mw4").config
    events = (
        core.ConsolidatedEvent(2.0, ("combat_burst",), 0.96, {}),
        core.ConsolidatedEvent(16.5, ("outcome_like",), 0.98, {}),
    )
    frames = 250
    signals = {
        "interest": np.full(frames, 0.75, dtype=np.float32),
        "player_death": np.zeros(frames, dtype=np.float32),
    }
    if death:
        signals["player_death"][112:125] = 1.0
    timeline = SimpleNamespace(
        consolidated_events=events,
        shots=(core.ShotSpan(0.0, 9.0), core.ShotSpan(9.0, 25.0)),
        signals=signals,
        fps=10.0,
        times=np.arange(frames) / 10.0,
        base=SimpleNamespace(duration=25.0, events=()),
        duration=25.0,
        engagements=(),
        finishing_moves=(),
    )
    monkeypatch.setattr(
        interaction,
        "hostile_decision",
        lambda _event, _config: interaction.HostileDecision(True, 0.95, "direct_confirmation"),
    )
    monkeypatch.setattr(combat, "segment_crosses_gap", lambda *_args: False)
    return config, timeline


def test_independent_payoff_search_composes_discontinuous_source_beats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, timeline = _example(monkeypatch)
    montage, reasons = editing.montage_for_anchor(
        timeline, config, [], "ghost_fight", 16.5, (0.0, 25.0)
    )
    assert montage, reasons
    assert all(len(plan.segments) == 2 for plan in montage)
    for plan in montage:
        first, second = plan.segments
        assert first.end < second.start
        assert second.start <= 16.5 <= second.end
        assert 10.0 <= plan.output_duration <= 12.0
        assert plan.output_duration == pytest.approx(
            (first.end - first.start) + (second.end - second.start)
        )
        assert plan.story_type == "combat_montage"
        assert plan.finishing_move is None
        assert plan.effect_profile == "source_native_combat_hard_cuts"
        assert plan.proposal_anchor_time == 16.5
        assert plan.opening_quality > 0.0 and plan.ending_quality > 0.0
        assert plan.engagements[0].events[0].time == 2.0
        assert plan.engagements[1].events[-1].time == 16.5


def test_anchor_search_keeps_legal_montage_even_if_contiguous_search_finds_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, timeline = _example(monkeypatch)
    monkeypatch.setattr(planning, "_terminal_aware_variants", lambda *_: ())
    monkeypatch.setattr(planning.canonical, "finishing_open_plans", lambda *_: [])
    plans = planning.build_plans_for_source(timeline, config, [], "ghost_fight")
    assert plans
    assert all(p.story_type == "combat_montage" for p in plans)
    diag = timeline._proposal_diagnostics
    assert diag["payoff_anchor_qualified_anchor_count"] == 1
    assert diag["payoff_anchor_independent_search"] is True
    assert diag["payoff_anchor_diagnostics"][0]["beat_edit_qualified_count"] > 0


def test_rejects_dead_footage_even_when_it_would_reach_ten_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, timeline = _example(monkeypatch, death=True)
    plans, rejected = editing.montage_for_anchor(
        timeline, config, [], "ghost_fight", 16.5, (0.0, 25.0)
    )
    assert not plans
    assert rejected.get("confirmed_death_inside_beat", 0) > 0


def test_never_crosses_a_certified_source_cut_inside_a_retained_beat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, timeline = _example(monkeypatch)
    config["source_integrity"]["verified_cut_windows"]["ghost_fight"] = [[2.0, 2.2]]
    plans, failures = editing.montage_for_anchor(
        timeline, config, [], "ghost_fight", 16.5, (0.0, 25.0)
    )
    assert not plans
    assert failures.get("certified_stringout_cut", 0) > 0


def test_neural_models_cannot_supply_unverified_hook_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, timeline = _example(monkeypatch)
    monkeypatch.setattr(
        interaction,
        "hostile_decision",
        lambda event, _cfg: interaction.HostileDecision(
            event.time == 16.5, 0.95, "only_payoff_verified"
        ),
    )
    plans, _ = editing.montage_for_anchor(timeline, config, [], "ghost_fight", 16.5, (0.0, 25.0))
    assert not plans


def test_bounded_source_gap_preserves_context(monkeypatch: pytest.MonkeyPatch) -> None:
    config, timeline = _example(monkeypatch)
    config["semantic_editor"]["beat_editing"]["max_source_gap_seconds"] = 2.0
    plans, reasons = editing.montage_for_anchor(
        timeline, config, [], "ghost_fight", 16.5, (0.0, 25.0)
    )
    assert not plans
    assert reasons.get("invalid_source_cut_distance", 0) > 0


def test_canonical_contract_accepts_verified_multi_segment_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, timeline = _example(monkeypatch)
    plan, _ = editing.montage_for_anchor(timeline, config, [], "ghost_fight", 16.5, (0.0, 25.0))
    assert plan
    assert contract.validate_plan("ghost_fight", 1, asdict(plan[0]), config) == []


def test_canonical_contract_rejects_forged_edit_duration_or_unverified_beat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, timeline = _example(monkeypatch)
    plan, _ = editing.montage_for_anchor(timeline, config, [], "ghost_fight", 16.5, (0.0, 25.0))
    assert plan
    payload = asdict(plan[0])
    payload["output_duration"] = 11.999
    failures = contract.validate_plan("ghost_fight", 1, payload, config)
    assert any("duration does not equal retained footage" in f for f in failures)
    payload = asdict(plan[0])
    payload["engagements"][0]["events"][0]["evidence"] = {}
    monkeypatch.setattr(
        interaction,
        "hostile_decision",
        lambda event, _cfg: interaction.HostileDecision(
            bool(getattr(event, "evidence", {}).get("direct")), 0.95, "strict"
        ),
    )
    failures = contract.validate_plan("ghost_fight", 1, payload, config)
    assert any("lacks directly verified hostile" in f for f in failures)


def test_analysis_manifest_clearly_reports_cpu_vs_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from clipper_engine.gameplay import candidates
    from clipper_engine.inference import vision

    config, timeline = _example(monkeypatch)
    config["capabilities"]["neural_visual_events"]["enabled"] = False
    monkeypatch.setattr(candidates.analysis, "analyze_source", lambda *_: timeline)
    monkeypatch.setattr(candidates.analysis, "diagnose_source", lambda *_: {})
    monkeypatch.setattr(candidates.analysis, "build_plans_for_source", lambda *_: [])
    monkeypatch.setattr(candidates.cinematic, "analyze", lambda *_: ([], []))
    monkeypatch.setattr(candidates, "_automatic_finishing_candidates", lambda *_: [])
    monkeypatch.setattr(vision, "analyze_source", lambda *_args, **_kwargs: None)
    source = tmp_path / "ghost_fight.mp4"
    result = candidates.analyze_source_file("ghost_fight", source, config, tmp_path / "analysis")
    assert result["analysis_compute"]["feature_extraction"] == "cpu_ffmpeg_numpy"
    assert result["analysis_compute"]["neural_visual"] == "disabled"
    assert result["analysis_compute"]["neural_gpu_seconds_this_run"] == 0.0


def test_public_clipper_analysis_route_uses_only_canonical_montage_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from clipper_engine.gameplay import analysis, anchor_search, candidates
    from clipper_engine.inference import vision

    config, timeline = _example(monkeypatch)
    config["capabilities"]["neural_visual_events"]["enabled"] = False
    # The old duplicate payoff planner must not reappear as an inactive authority.
    assert not hasattr(anchor_search, "_anchor_plans")
    monkeypatch.setattr(analysis, "analyze_source", lambda *_: timeline)
    monkeypatch.setattr(analysis, "diagnose_source", lambda *_: {"canonical": True})
    monkeypatch.setattr(planning, "_terminal_aware_variants", lambda *_: ())
    monkeypatch.setattr(planning.canonical, "finishing_open_plans", lambda *_: [])
    monkeypatch.setattr(candidates.cinematic, "analyze", lambda *_: ([], []))
    monkeypatch.setattr(candidates, "_automatic_finishing_candidates", lambda *_: [])
    monkeypatch.setattr(vision, "analyze_source", lambda *_args, **_kwargs: None)
    source = tmp_path / "ghost_fight.mp4"
    source.write_bytes(b"independently certified test source stub")
    out = tmp_path / "analysis"
    report = candidates.analyze_source_file("ghost_fight", source, config, out)
    montages = [p for p in report["candidate_pool"] if p["story_type"] == "combat_montage"]
    assert montages
    assert all(len(p["segments"]) == 2 for p in montages)
    assert all(p["covered_outcome_anchor_times"] == [16.5] for p in montages)
    diag = timeline._proposal_diagnostics
    assert diag["payoff_anchor_diagnostics"][0]["beat_edit_qualified_count"] > 0
    assert diag["semantic_montage_enabled"] is True
    assert (out / "ghost_fight_analysis.json").is_file()
