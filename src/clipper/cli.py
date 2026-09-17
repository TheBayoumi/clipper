from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .brief import load_brief
from .pipeline import PipelineSettings, run_pipeline
from .rights import assert_campaign_authorized
from .youtube import YouTubeClient


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _add_mw4_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    mw4 = subparsers.add_parser(
        "mw4",
        help="execute the canonical Modern Warfare 4 v3.1 campaign pipeline",
    )
    stages = mw4.add_subparsers(dest="mw4_command", required=True)

    stages.add_parser("self-test", help="validate canonical MW4 planner/contract architecture")

    discover = stages.add_parser("discover", help="discover original MediaSilo source masters")
    discover.add_argument("--review-url", required=True)
    discover.add_argument("--output", type=Path, required=True)
    discover.add_argument("--github-output", type=Path)
    discover.add_argument("--expected-count", type=int, required=True)

    analyze = stages.add_parser(
        "analyze",
        help="resolve, download, certify, and semantically analyze one MediaSilo source",
    )
    analyze.add_argument("--review-url", required=True)
    analyze.add_argument("--expected-count", type=int, required=True)
    analyze.add_argument("--source-key", required=True)
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--source-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)

    allocate = stages.add_parser(
        "allocate",
        help="perform adaptive allocation and emit the render matrix",
    )
    allocate.add_argument("--config", type=Path, required=True)
    allocate.add_argument("--analysis-root", type=Path, required=True)
    allocate.add_argument("--allocation-out", type=Path, required=True)
    allocate.add_argument("--rejection-out", type=Path, required=True)
    allocate.add_argument("--github-output", type=Path, required=True)

    render = stages.add_parser("render-one", help="verify source and render one allocated clip")
    render.add_argument("--source-key", required=True)
    render.add_argument("--source-dir", type=Path, required=True)
    render.add_argument("--config", type=Path, required=True)
    render.add_argument("--allocation", type=Path, required=True)
    render.add_argument("--plan-key", required=True)
    render.add_argument("--ordinal", type=int, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--mode", choices=("shadow", "production"), default="production")

    batch = stages.add_parser(
        "batch-contract",
        help="aggregate rendered clip metadata and enforce the adaptive batch contract",
    )
    batch.add_argument("--config", type=Path, required=True)
    batch.add_argument("--allocation", type=Path, required=True)
    batch.add_argument("--clip-meta", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Rights-gated campaign clipping pipeline.",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a campaign brief")
    validate.add_argument("--brief", required=True, type=Path)

    discover = subparsers.add_parser("discover", help="discover authorized source videos")
    discover.add_argument("--brief", required=True, type=Path)

    run = subparsers.add_parser("run", help="execute transcription, planning, and rendering")
    run.add_argument("--brief", required=True, type=Path)
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    run.add_argument(
        "--no-render",
        action="store_true",
        help="stop after timestamped clip planning",
    )

    _add_mw4_parser(subparsers)
    return parser


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "scripts").is_dir():
        raise RuntimeError(
            "MW4 commands require a repository checkout containing the canonical scripts directory"
        )
    return root


def _run_script(root: Path, name: str, *arguments: str) -> None:
    script = root / "scripts" / name
    if not script.is_file():
        raise RuntimeError(f"canonical MW4 implementation is missing: {script}")
    subprocess.run([sys.executable, str(script), *arguments], check=True, cwd=root)


def _print_mw4_diagnostics(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    local = (data.get("diagnostics") or {}).get("local_interaction_verifier") or {}
    events = local.get("events") or []
    confirmed = [event for event in events if event.get("confirmed")]
    kill_like = [
        event
        for event in confirmed
        if "outcome_like" in (event.get("kinds") or [])
    ]
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


def _run_mw4(args: argparse.Namespace) -> int:
    root = _repo_root()
    command = args.mw4_command
    if command == "self-test":
        forbidden = (
            "mw4_semantic_gameplay_v3_1_refined.py",
            "mw4_semantic_gameplay_v3_1_final_legacy.py",
            "mw4_semantic_gameplay_v3_1_finishing_body.py",
            "mw4_semantic_gameplay_v3_1_no_fallback.py",
        )
        present = [name for name in forbidden if (root / "scripts" / name).exists()]
        if present:
            raise RuntimeError(f"forbidden MW4 planner modules are present: {present}")
        compile_targets = (
            "mw4_semantic_gameplay_v3_1_payoff_complete.py",
            "mw4_semantic_gameplay_v3_1_payoff_terminal.py",
            "mw4_semantic_gameplay_v3_1_final.py",
            "mw4_v3_1_source_catalog.py",
            "mw4_v3_1_dynamic_contract.py",
            "mw4_v3_1_dynamic_workflow_support.py",
            "mw4_fullframe_retention_v3_1.py",
            "mw4_v3_1_render_one.py",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                *(str(root / "scripts" / name) for name in compile_targets),
            ],
            check=True,
            cwd=root,
        )
        _run_script(root, "mw4_semantic_gameplay_v3_1_final.py", "--self-test")
        _run_script(root, "mw4_v3_1_dynamic_contract.py", "--self-test")
        return 0

    if command == "discover":
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
        _run_script(root, "mw4_v3_1_source_catalog.py", *arguments)
        return 0

    if command == "analyze":
        args.source_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        source_path = args.source_dir / f"{args.source_key}.mp4"
        qa_path = args.source_dir / f"{args.source_key}.source_qa.json"
        with tempfile.TemporaryDirectory(prefix="clipper-mw4-") as temp_dir:
            catalog = Path(temp_dir) / "source_catalog.json"
            resolved = Path(temp_dir) / "resolved_source.json"
            _run_script(
                root,
                "mw4_v3_1_source_catalog.py",
                "discover",
                "--review-url",
                args.review_url,
                "--output",
                str(catalog),
                "--expected-count",
                str(args.expected_count),
            )
            _run_script(
                root,
                "mw4_v3_1_source_catalog.py",
                "select",
                "--catalog",
                str(catalog),
                "--source-key",
                args.source_key,
                "--output",
                str(resolved),
            )
            _run_script(
                root,
                "mw4_v3_1_mediasilo_source.py",
                "download",
                "--resolved",
                str(resolved),
                "--target",
                str(source_path),
            )
            _run_script(
                root,
                "mw4_v3_1_source_qa.py",
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
        _run_script(
            root,
            "mw4_fullframe_retention_v3_1.py",
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
        _print_mw4_diagnostics(args.output_dir / f"{args.source_key}_analysis_v3_1.json")
        return 0

    if command == "allocate":
        _run_script(
            root,
            "mw4_v3_1_dynamic_contract.py",
            "--config",
            str(args.config),
            "--allocate-from",
            str(args.analysis_root),
            "--allocation-out",
            str(args.allocation_out),
            "--rejection-out",
            str(args.rejection_out),
        )
        _run_script(
            root,
            "mw4_v3_1_dynamic_workflow_support.py",
            "render-matrix",
            "--allocation",
            str(args.allocation_out),
            "--github-output",
            str(args.github_output),
        )
        return 0

    if command == "render-one":
        source_path = args.source_dir / f"{args.source_key}.mp4"
        qa_path = args.source_dir / f"{args.source_key}.source_qa.json"
        _run_script(
            root,
            "mw4_v3_1_source_qa.py",
            "verify",
            "--source",
            str(source_path),
            "--manifest",
            str(qa_path),
        )
        _run_script(
            root,
            "mw4_v3_1_render_one.py",
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

    if command == "batch-contract":
        _run_script(
            root,
            "mw4_v3_1_dynamic_workflow_support.py",
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
        _run_script(
            root,
            "mw4_v3_1_dynamic_contract.py",
            "--config",
            str(args.config),
            "--batch-root",
            str(args.output_dir),
            "--allocation",
            str(args.allocation),
        )
        return 0

    raise RuntimeError(f"unsupported MW4 command: {command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        if args.command == "validate":
            brief = load_brief(args.brief)
            assert_campaign_authorized(brief)
            print(json.dumps(brief.to_dict(), indent=2))
            return 0
        if args.command == "discover":
            brief = load_brief(args.brief)
            assert_campaign_authorized(brief)
            videos = YouTubeClient().discover(brief)
            print(json.dumps([video.to_dict() for video in videos], indent=2))
            return 0
        if args.command == "run":
            settings = replace(PipelineSettings.from_env(), artifact_root=args.artifact_root)
            run_dir = run_pipeline(args.brief, settings=settings, render=not args.no_render)
            print(run_dir)
            return 0
        if args.command == "mw4":
            return _run_mw4(args)
    except Exception as exc:
        logging.getLogger("clipper").error("%s", exc)
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
