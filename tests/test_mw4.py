import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper import mw4
from clipper_mw4 import mw4_v3_1_contract as mw4_contract
from clipper_mw4 import mw4_v3_1_dynamic_workflow_support as workflow_support
from clipper_mw4 import orchestrator as mw4_orchestrator


def test_mw4_bridge_lazy_loads_installed_orchestrator() -> None:
    args = argparse.Namespace(mw4_command="self-test")
    runner = Mock(return_value=7)
    module = SimpleNamespace(run=runner)

    with patch("clipper.mw4.importlib.import_module", return_value=module) as import_module:
        assert mw4.run(args) == 7

    import_module.assert_called_once_with("clipper_mw4.orchestrator")
    runner.assert_called_once_with(args)


def test_mw4_bridge_rejects_invalid_runtime() -> None:
    module = SimpleNamespace(run=None)
    with (
        patch("clipper.mw4.importlib.import_module", return_value=module),
        pytest.raises(RuntimeError, match="callable orchestrator"),
    ):
        mw4.run(argparse.Namespace(mw4_command="self-test"))


def test_mw4_scene_derived_selection_keeps_every_distinct_fighting_scene() -> None:
    config = {
        "batch_selection": {"require_verified_finishing_move_when_available": True},
        "semantic_editor": {"selection": {"finishing_move_bonus": 0.14}},
    }
    candidates = [
        {
            "plan_key": "scene-a-low",
            "score": 0.60,
            "effect_events": [{"kind": "outcome_like", "time": 10.0}],
        },
        {
            "plan_key": "scene-a-best",
            "score": 0.95,
            "effect_events": [{"kind": "outcome_like", "time": 10.0}],
        },
        {
            "plan_key": "scene-b",
            "score": 0.70,
            "effect_events": [{"kind": "outcome_like", "time": 30.0}],
        },
        {
            "plan_key": "scene-c",
            "score": 0.65,
            "effect_events": [{"kind": "outcome_like", "time": 50.0}],
        },
    ]

    selected = mw4_contract._adaptive_source_selection("test", candidates, 0, config)

    assert {plan["plan_key"] for plan in selected} == {
        "scene-a-best",
        "scene-b",
        "scene-c",
    }


def test_mw4_orchestrator_enforces_scene_derived_count(tmp_path: Path) -> None:
    allocation = tmp_path / "allocation.json"
    allocation.write_text(
        json.dumps(
            {
                "target_count": 3,
                "selected_count": 3,
                "duration_contract": {
                    "minimum_seconds": 10.0,
                    "maximum_seconds": 12.0,
                },
                "source_allocations": {
                    "r1": {"count": 2, "distinct_fighting_scene_count": 2},
                    "batch2": {"count": 1, "distinct_fighting_scene_count": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    mw4_orchestrator._assert_scene_derived_allocation(allocation)

    bad = json.loads(allocation.read_text(encoding="utf-8"))
    bad["source_allocations"]["r1"]["count"] = 1
    allocation.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="distinct qualified fighting scenes"):
        mw4_orchestrator._assert_scene_derived_allocation(allocation)


def test_mw4_render_matrix_matches_orchestrator_cardinality(tmp_path: Path) -> None:
    allocation = tmp_path / "allocation.json"
    github_output = tmp_path / "github_output.txt"
    allocation.write_text(
        json.dumps(
            {
                "target_count": 3,
                "selected_count": 3,
                "source_order": ["r1", "batch2"],
                "source_allocations": {
                    "r1": {
                        "count": 2,
                        "plan_keys": ["r1-a", "r1-b"],
                    },
                    "batch2": {
                        "count": 1,
                        "plan_keys": ["b2-a"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    matrix = workflow_support.render_matrix(allocation, github_output)

    assert len(matrix) == 3
    assert [item["source"] for item in matrix] == ["r1", "r1", "batch2"]
    outputs = github_output.read_text(encoding="utf-8")
    assert "derived_clip_count=3" in outputs
    assert 'source_distribution={"r1":2,"batch2":1}' in outputs


def test_mw4_render_matrix_rejects_cardinality_mismatch(tmp_path: Path) -> None:
    allocation = tmp_path / "allocation.json"
    github_output = tmp_path / "github_output.txt"
    allocation.write_text(
        json.dumps(
            {
                "target_count": 3,
                "selected_count": 3,
                "source_order": ["r1"],
                "source_allocations": {
                    "r1": {
                        "count": 2,
                        "plan_keys": ["r1-a", "r1-b"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="render matrix does not match"):
        workflow_support.render_matrix(allocation, github_output)


def test_mw4_scene_identity_wins_over_terminal_payoff() -> None:
    config = {
        "batch_selection": {"require_verified_finishing_move_when_available": True},
        "semantic_editor": {"selection": {"finishing_move_bonus": 0.14}},
    }
    candidates = [
        {
            "plan_key": "same-scene-earlier",
            "combat_scene_id": "combat:3:10.000:14.000",
            "score": 0.70,
            "effect_events": [{"kind": "outcome_like", "time": 12.0}],
        },
        {
            "plan_key": "same-scene-best",
            "combat_scene_id": "combat:3:10.000:14.000",
            "score": 0.95,
            "effect_events": [{"kind": "outcome_like", "time": 13.5}],
        },
        {
            "plan_key": "different-scene",
            "combat_scene_id": "combat:3:20.000:24.000",
            "score": 0.80,
            "effect_events": [{"kind": "outcome_like", "time": 23.0}],
        },
    ]

    selected = mw4_contract._adaptive_source_selection("test", candidates, 0, config)

    assert {plan["plan_key"] for plan in selected} == {
        "same-scene-best",
        "different-scene",
    }
