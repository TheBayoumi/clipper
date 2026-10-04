from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path

from .brief import load_brief
from .editorial_run import EditorialRunConfig, load_editorial_run_config, write_editorial_run_config
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
    editorial = subparsers.add_parser(
        "editorial", help="run the contextual podcast editor from a YAML/JSON run config"
    )
    editorial.add_argument("--config", required=True, type=Path)
    editorial.add_argument(
        "--check-config", action="store_true", help="validate inputs without acquiring media"
    )
    export = subparsers.add_parser(
        "editorial-config", help="write validated workflow inputs to one editorial JSON run file"
    )
    export.add_argument("--brief", required=True, type=Path)
    export.add_argument("--artifact-root", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    relation = subparsers.add_parser(
        "relation-benchmark", help="diagnose pinned source-to-relation entailment without media"
    )
    relation.add_argument("--fixture", required=True, type=Path)
    relation.add_argument("--proof", required=True, type=Path)
    relation.add_argument("--transcript", required=True, type=Path)
    relation.add_argument("--provenance", required=True, type=Path)
    relation.add_argument("--output", required=True, type=Path)
    syntax = subparsers.add_parser(
        "syntax-inventory-benchmark",
        help="diagnose parser-owned headline obligations on pinned controls without media",
    )
    syntax.add_argument("--fixture", required=True, type=Path)
    syntax.add_argument("--proof", required=True, type=Path)
    syntax.add_argument("--transcript", required=True, type=Path)
    syntax.add_argument("--provenance", required=True, type=Path)
    syntax.add_argument("--heldout", required=True, type=Path)
    syntax.add_argument("--output", required=True, type=Path)
    audio_review = subparsers.add_parser(
        "audio-review-manifest",
        help="prepare a blind, hash-verified held-out MP4 review without provisional labels",
    )
    audio_review.add_argument("--fixture", required=True, type=Path)
    audio_review.add_argument("--transcript", required=True, type=Path)
    audio_review.add_argument("--provenance", required=True, type=Path)
    audio_review.add_argument("--clips-dir", required=True, type=Path)
    audio_review.add_argument("--output", required=True, type=Path)
    audio_check = subparsers.add_parser(
        "audio-review-check",
        help="check a completed blind review against pinned MP4s without promoting it to gold",
    )
    audio_check.add_argument("--submitted", required=True, type=Path)
    audio_check.add_argument("--fixture", required=True, type=Path)
    audio_check.add_argument("--transcript", required=True, type=Path)
    audio_check.add_argument("--provenance", required=True, type=Path)
    audio_check.add_argument("--clips-dir", required=True, type=Path)
    audio_check.add_argument("--output", required=True, type=Path)
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
        if args.command == "editorial":
            config = load_editorial_run_config(args.config)
            if args.check_config:
                print(json.dumps(asdict(config), indent=2, default=str))
                return 0
            render_youtube_previews = import_module(
                "scripts.tjr_youtube_preview"
            ).render_youtube_previews
            print(render_youtube_previews(config.artifact_root, config.brief, run_config=config))
            return 0
        if args.command == "editorial-config":
            config = EditorialRunConfig.from_legacy_environment(args.brief, args.artifact_root)
            print(write_editorial_run_config(config, args.output))
            return 0
        if args.command == "relation-benchmark":
            from .editorial_relation_probe import run_relation_probe

            return run_relation_probe(
                args.fixture,
                args.proof,
                args.transcript,
                args.provenance,
                args.output,
            )
        if args.command == "syntax-inventory-benchmark":
            from .editorial_syntax_probe import run_syntax_inventory_probe

            return run_syntax_inventory_probe(
                args.fixture,
                args.proof,
                args.transcript,
                args.provenance,
                args.heldout,
                args.output,
            )
        if args.command == "audio-review-manifest":
            from .editorial_benchmark import prepare_blind_audio_review

            print(
                prepare_blind_audio_review(
                    args.fixture,
                    args.transcript,
                    args.provenance,
                    args.clips_dir,
                    args.output,
                )
            )
            return 0
        if args.command == "audio-review-check":
            from .editorial_benchmark import assess_completed_audio_review

            result = assess_completed_audio_review(
                args.submitted,
                args.fixture,
                args.transcript,
                args.provenance,
                args.clips_dir,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(args.output)
            return 0
    except Exception as exc:
        logging.getLogger("clipper").error("%s", exc)
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
