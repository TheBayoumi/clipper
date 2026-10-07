from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .. import qualification
from ..profiles import GameplayProfile, load_profile
from ..rendering import worker
from ..sources import catalog, mediasilo
from ..sources import qa as source_qa
from . import allocation, analysis, candidates, contract, planning, workflow_support


def _profile_from_args(args: argparse.Namespace) -> GameplayProfile:
    override = getattr(args, "config", None)
    return load_profile("mw4", override)


def _source_settings(
    profile: GameplayProfile,
    args: argparse.Namespace,
) -> tuple[str, int]:
    review_url = str(getattr(args, "review_url", None) or profile.source_url)
    expected_count = int(getattr(args, "expected_count", None) or profile.expected_source_count)
    if not review_url:
        raise RuntimeError("MW4 profile has no MediaSilo source URL")
    if expected_count <= 0:
        raise RuntimeError("MW4 profile has no positive expected source count")
    return review_url, expected_count


def discover(profile: GameplayProfile, args: argparse.Namespace) -> dict[str, Any]:
    review_url, expected_count = _source_settings(profile, args)
    return catalog.discover(
        review_url,
        args.output,
        getattr(args, "github_output", None),
        expected_count,
        dict(profile.config.get("source_profile") or {}),
    )


def analyze(profile: GameplayProfile, args: argparse.Namespace) -> dict[str, Any]:
    review_url, expected_count = _source_settings(profile, args)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.source_dir / f"{args.source_key}.mp4"
    qa_path = args.source_dir / f"{args.source_key}.source_qa.json"

    with tempfile.TemporaryDirectory(prefix="clipper-source-") as temp_dir:
        root = Path(temp_dir)
        catalog_path = root / "source_catalog.json"
        resolved_path = root / "resolved_source.json"
        source_profile = dict(profile.config.get("source_profile") or {})
        catalog.discover(review_url, catalog_path, None, expected_count, source_profile)
        catalog.select(catalog_path, args.source_key, resolved_path)
        mediasilo.download(resolved_path, source_path)
        source_qa.certify(
            source_path,
            args.source_key,
            resolved_path,
            qa_path,
            allow_proxy=bool(source_profile.get("allow_proxy_for_analysis", False)),
        )

    return candidates.analyze_source_file(
        args.source_key,
        source_path,
        profile.config,
        args.output_dir,
    )


def allocate(profile: GameplayProfile, args: argparse.Namespace) -> dict[str, Any]:
    try:
        result = allocation.allocate(args.analysis_root, profile.config)
    except AssertionError as exc:
        rejection = allocation.allocation_rejection_diagnostics(args.analysis_root)
        rejection["reason"] = str(exc)
        args.rejection_out.parent.mkdir(parents=True, exist_ok=True)
        args.rejection_out.write_text(json.dumps(rejection, indent=2), encoding="utf-8")
        raise
    args.allocation_out.parent.mkdir(parents=True, exist_ok=True)
    args.allocation_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    workflow_support.render_matrix(args.allocation_out, args.github_output)
    print(
        json.dumps(
            {
                "coverage_conflict_allocation": "PASS",
                "selected_count": result["selected_count"],
                "duration_contract_seconds": [10.0, 12.0],
            }
        )
    )
    return result


def _promote_analysis_source_for_production(
    profile: GameplayProfile,
    *,
    source_key: str,
    source_path: Path,
    qa_path: Path,
    alignment_path: Path,
) -> dict[str, Any]:
    analysis_manifest = source_qa.verify(
        source_path,
        qa_path,
        require_original=False,
    )
    if analysis_manifest.get("derivative_type") == "source":
        return source_qa.verify(source_path, qa_path, require_original=True)

    if analysis_manifest.get("derivative_type") != "proxy":
        raise RuntimeError(
            f"{source_key}: unsupported analysis derivative "
            f"{analysis_manifest.get('derivative_type')!r}"
        )

    review_url = profile.source_url
    expected_count = profile.expected_source_count
    if not review_url or expected_count <= 0:
        raise RuntimeError(f"{source_key}: profile cannot resolve production source")

    source_profile = dict(profile.config.get("source_profile") or {})
    if str(source_profile.get("analysis_derivative") or "source").lower() != "proxy":
        raise RuntimeError(
            f"{source_key}: proxy analysis artifact is inconsistent with source profile"
        )

    original_profile = dict(source_profile)
    original_profile["analysis_derivative"] = "source"
    with tempfile.TemporaryDirectory(prefix="clipper-production-source-") as temp_dir:
        root = Path(temp_dir)
        catalog_path = root / "source_catalog.json"
        resolved_path = root / "resolved_source.json"
        downloaded_path = root / f"{source_key}.mp4"
        original_qa_path = root / f"{source_key}.source_qa.json"

        catalog.discover(
            review_url,
            catalog_path,
            None,
            expected_count,
            original_profile,
        )
        catalog.select(catalog_path, source_key, resolved_path)
        mediasilo.download(resolved_path, downloaded_path)
        source_qa.certify(
            downloaded_path,
            source_key,
            resolved_path,
            original_qa_path,
            allow_proxy=False,
        )
        source_qa.certify_analysis_alignment(
            qa_path,
            original_qa_path,
            alignment_path,
        )

        downloaded_path.replace(source_path)
        original_qa_path.replace(qa_path)

    return source_qa.verify(
        source_path,
        qa_path,
        require_original=True,
        require_analysis_alignment=True,
    )


def render_one(profile: GameplayProfile, args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_dir / f"{args.source_key}.mp4"
    qa_path = args.source_dir / f"{args.source_key}.source_qa.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_path = args.output_dir / f"{args.source_key}_analysis_alignment.json"

    if args.mode == "production":
        source_manifest = _promote_analysis_source_for_production(
            profile,
            source_key=args.source_key,
            source_path=source_path,
            qa_path=qa_path,
            alignment_path=alignment_path,
        )
    else:
        source_manifest = source_qa.verify(
            source_path,
            qa_path,
            require_original=False,
        )

    return worker.render_one(
        source_key=args.source_key,
        source=source_path,
        config=profile.config,
        allocation_path=args.allocation,
        plan_key=args.plan_key,
        ordinal=args.ordinal,
        output_dir=args.output_dir,
        mode=args.mode,
        source_manifest=source_manifest,
    )


def batch_contract(profile: GameplayProfile, args: argparse.Namespace) -> dict[str, Any]:
    workflow_support.batch_summaries(
        args.allocation,
        profile.config,
        args.clip_meta,
        args.output_dir,
    )
    allocation_payload = json.loads(args.allocation.read_text(encoding="utf-8"))
    result = qualification.validate_batch(
        args.output_dir,
        profile.config,
        allocation_payload,
    )
    print(json.dumps(result, indent=2))
    return result


def self_test(profile: GameplayProfile) -> None:
    contract.validate_configuration(profile.config)
    if str(profile.config.get("content_strategy", {}).get("mode") or "") == "spoken_content":
        from .. import spoken

        spoken.validate_configuration(profile.config)
        spoken.self_test()
    else:
        analysis.validate_configuration(profile.config)
    analysis.self_test()
    planning.self_test()
    candidates.self_test()
    allocation.self_test()
    mediasilo.self_test()
    print(
        json.dumps(
            {
                "self_test": "PASS",
                "engine_owner": "clipper",
                "profile": profile.name,
                "monkey_patching": False,
                "versioned_runtime_chain": False,
            }
        )
    )


def run_mw4(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    command = args.mw4_command
    if command == "self-test":
        self_test(profile)
        return 0
    if command == "discover":
        discover(profile, args)
        return 0
    if command == "analyze":
        analyze(profile, args)
        return 0
    if command == "allocate":
        allocate(profile, args)
        return 0
    if command == "render-one":
        render_one(profile, args)
        return 0
    if command == "batch-contract":
        batch_contract(profile, args)
        return 0
    raise RuntimeError(f"unsupported MW4 command: {command}")
