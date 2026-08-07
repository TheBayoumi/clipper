from __future__ import annotations

import argparse
import json
import logging
import sys
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Rights-gated Whop campaign to YouTube clip pipeline.",
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
    except Exception as exc:
        logging.getLogger("clipper").error("%s", exc)
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
