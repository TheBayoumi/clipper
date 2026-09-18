from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .brief import load_brief
from .mw4 import run as run_mw4
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
            return run_mw4(args)
    except Exception as exc:
        logging.getLogger("clipper").error("%s", exc)
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
