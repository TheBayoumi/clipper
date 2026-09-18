from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Protocol, cast

_PACKAGE = "clipper_mw4"

_FORBIDDEN_MODULES = (
    "mw4_semantic_gameplay_v3_1_refined",
    "mw4_semantic_gameplay_v3_1_final_legacy",
    "mw4_semantic_gameplay_v3_1_finishing_body",
    "mw4_semantic_gameplay_v3_1_no_fallback",
)

_REQUIRED_MODULES = (
    "mw4_semantic_gameplay_v3_1_payoff_complete",
    "mw4_semantic_gameplay_v3_1_payoff_terminal",
    "mw4_semantic_gameplay_v3_1_final",
    "mw4_v3_1_source_catalog",
    "mw4_v3_1_dynamic_contract",
    "mw4_v3_1_dynamic_workflow_support",
    "mw4_fullframe_retention_v3_1",
    "mw4_v3_1_render_one",
)


class _Main(Protocol):
    def __call__(self) -> None: ...


def _qualified(module: str) -> str:
    return f"{_PACKAGE}.{module}"


def _invoke(module: str, *arguments: str) -> None:
    imported = importlib.import_module(_qualified(module))
    entrypoint = getattr(imported, "main", None)
    if not callable(entrypoint):
        raise RuntimeError(f"MW4 module has no callable main(): {_qualified(module)}")
    main = cast(_Main, entrypoint)
    previous_argv = sys.argv
    try:
        sys.argv = [_qualified(module), *arguments]
        main()
    finally:
        sys.argv = previous_argv


def _assert_architecture() -> None:
    missing = [
        module
        for module in _REQUIRED_MODULES
        if importlib.util.find_spec(_qualified(module)) is None
    ]
    if missing:
        raise RuntimeError(f"required installed MW4 modules are missing: {missing}")

    forbidden = [
        module
        for module in _FORBIDDEN_MODULES
        if importlib.util.find_spec(_qualified(module)) is not None
    ]
    if forbidden:
        raise RuntimeError(f"forbidden MW4 planner modules are installed: {forbidden}")


def _print_diagnostics(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    local = (data.get("diagnostics") or {}).get("local_interaction_verifier") or {}
    events = local.get("events") or []
    confirmed = [event for event in events if event.get("confirmed")]
    kill_like = [event for event in confirmed if "outcome_like" in (event.get("kinds") or [])]
    pool = data.get("candidate_pool") or []
    terminals: set[float] = set()
    for plan in pool:
        anchors: list[float] = []
        for event in plan.get("effect_events") or []:
            if event.get("kind") in {"impact", "outcome_like"}:
                anchors.append(float(event["time"]))
        for engagement in plan.get("engagements") or []:
            for event in engagement.get("events") or []:
                if set(event.get("kinds") or []).intersection({"impact", "outcome_like"}):
                    anchors.append(float(event["time"]))
        if anchors:
            terminals.add(round(max(anchors), 3))

    print(
        json.dumps(
            {
                "source": data.get("source_key"),
                "verified_local_interactions": len(confirmed),
                "kill_like_outcome_events": len(kill_like),
                "kill_like_times": [round(float(event["time"]), 3) for event in kill_like],
                "qualified_candidate_count": len(pool),
                "distinct_terminal_payoff_scenes_in_pool": len(terminals),
                "terminal_payoff_times": sorted(terminals),
            },
            indent=2,
        )
    )


def _assert_scene_derived_allocation(path: Path) -> None:
    allocation = json.loads(path.read_text(encoding="utf-8"))
    source_allocations = dict(allocation.get("source_allocations") or {})
    derived_total = 0
    for source, raw in source_allocations.items():
        item = dict(raw)
        scene_count = int(item.get("distinct_fighting_scene_count", -1))
        selected_count = int(item.get("count", -1))
        if scene_count < 0:
            raise RuntimeError(f"{source}: allocation is missing distinct fighting-scene count")
        if selected_count != scene_count:
            raise RuntimeError(
                f"{source}: orchestrator allocation selected {selected_count} clips for "
                f"{scene_count} distinct qualified fighting scenes"
            )
        derived_total += scene_count

    target_count = int(allocation.get("target_count", -1))
    selected_total = int(allocation.get("selected_count", -1))
    if target_count != derived_total or selected_total != derived_total:
        raise RuntimeError(
            "orchestrator scene-derived allocation invariant failed: "
            f"target={target_count} selected={selected_total} derived={derived_total}"
        )

    duration = dict(allocation.get("duration_contract") or {})
    if (
        float(duration.get("minimum_seconds", -1.0)) != 10.0
        or float(duration.get("maximum_seconds", -1.0)) != 12.0
    ):
        raise RuntimeError(f"orchestrator allocation has wrong duration contract: {duration}")

    print(
        json.dumps(
            {
                "orchestrator_scene_derived_allocation": "PASS",
                "derived_clip_count": derived_total,
                "duration_contract_seconds": [10.0, 12.0],
            }
        )
    )


def _discover(args: argparse.Namespace) -> int:
    arguments = [
        "discover",
        "--review-url",
        args.review_url,
        "--output",
        str(args.output),
        "--expected-count",
        str(args.expected_count),
    ]
    if args.github_output is not None:
        arguments.extend(("--github-output", str(args.github_output)))
    _invoke("mw4_v3_1_source_catalog", *arguments)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.source_dir / f"{args.source_key}.mp4"
    qa_path = args.source_dir / f"{args.source_key}.source_qa.json"

    with tempfile.TemporaryDirectory(prefix="clipper-mw4-") as temp_dir:
        catalog = Path(temp_dir) / "source_catalog.json"
        resolved = Path(temp_dir) / "resolved_source.json"
        _invoke(
            "mw4_v3_1_source_catalog",
            "discover",
            "--review-url",
            args.review_url,
            "--output",
            str(catalog),
            "--expected-count",
            str(args.expected_count),
        )
        _invoke(
            "mw4_v3_1_source_catalog",
            "select",
            "--catalog",
            str(catalog),
            "--source-key",
            args.source_key,
            "--output",
            str(resolved),
        )
        _invoke(
            "mw4_v3_1_mediasilo_source",
            "download",
            "--resolved",
            str(resolved),
            "--target",
            str(source_path),
        )
        _invoke(
            "mw4_v3_1_source_qa",
            "certify",
            "--source",
            str(source_path),
            "--source-key",
            args.source_key,
            "--resolved",
            str(resolved),
            "--output",
            str(qa_path),
        )

    _invoke(
        "mw4_fullframe_retention_v3_1",
        "--source-key",
        args.source_key,
        "--source",
        str(source_path),
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
        "--mode",
        "analysis",
    )
    _print_diagnostics(args.output_dir / f"{args.source_key}_analysis_v3_1.json")
    return 0


def _allocate(args: argparse.Namespace) -> int:
    _invoke(
        "mw4_v3_1_dynamic_contract",
        "--config",
        str(args.config),
        "--allocate-from",
        str(args.analysis_root),
        "--allocation-out",
        str(args.allocation_out),
        "--rejection-out",
        str(args.rejection_out),
    )
    _assert_scene_derived_allocation(args.allocation_out)
    _invoke(
        "mw4_v3_1_dynamic_workflow_support",
        "render-matrix",
        "--allocation",
        str(args.allocation_out),
        "--github-output",
        str(args.github_output),
    )
    return 0


def _render_one(args: argparse.Namespace) -> int:
    source_path = args.source_dir / f"{args.source_key}.mp4"
    qa_path = args.source_dir / f"{args.source_key}.source_qa.json"
    _invoke(
        "mw4_v3_1_source_qa",
        "verify",
        "--source",
        str(source_path),
        "--manifest",
        str(qa_path),
    )
    _invoke(
        "mw4_v3_1_render_one",
        "--source-key",
        args.source_key,
        "--source",
        str(source_path),
        "--config",
        str(args.config),
        "--allocation",
        str(args.allocation),
        "--plan-key",
        args.plan_key,
        "--ordinal",
        str(args.ordinal),
        "--output-dir",
        str(args.output_dir),
        "--mode",
        args.mode,
    )
    return 0


def _batch_contract(args: argparse.Namespace) -> int:
    _invoke(
        "mw4_v3_1_dynamic_workflow_support",
        "batch-summaries",
        "--allocation",
        str(args.allocation),
        "--config",
        str(args.config),
        "--clip-meta",
        str(args.clip_meta),
        "--output-dir",
        str(args.output_dir),
    )
    _invoke(
        "mw4_v3_1_dynamic_contract",
        "--config",
        str(args.config),
        "--batch-root",
        str(args.output_dir),
        "--allocation",
        str(args.allocation),
    )
    return 0


def run(args: argparse.Namespace) -> int:
    command = args.mw4_command
    if command == "self-test":
        _assert_architecture()
        _invoke("mw4_semantic_gameplay_v3_1_final", "--self-test")
        _invoke("mw4_v3_1_dynamic_contract", "--self-test")
        return 0
    if command == "discover":
        return _discover(args)
    if command == "analyze":
        return _analyze(args)
    if command == "allocate":
        return _allocate(args)
    if command == "render-one":
        return _render_one(args)
    if command == "batch-contract":
        return _batch_contract(args)
    raise RuntimeError(f"unsupported MW4 command: {command}")
