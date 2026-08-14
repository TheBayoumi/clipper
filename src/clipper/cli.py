from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from .benchmark import evaluate_corpus_manifest
from .brief import load_brief
from .modal_execution import ensure_modal_runtime, run_modal_pipeline
from .pipeline import PipelineSettings, run_pipeline
from .providers.factory import editorial_and_embedding_providers, speech_providers
from .rights import RightsError, assert_campaign_authorized, assert_video_allowed
from .source_cache import PersistentYouTubeClient
from .youtube import YouTubeClient

LOGGER = logging.getLogger("clipper")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Rights-gated campaign-to-short-form clipping pipeline.",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a campaign brief")
    validate.add_argument("--brief", required=True, type=Path)

    discover = subparsers.add_parser("discover", help="discover authorized source videos")
    discover.add_argument("--brief", required=True, type=Path)

    benchmark = subparsers.add_parser(
        "benchmark", help="evaluate a private multi-domain acceptance corpus"
    )
    benchmark.add_argument("--manifest", required=True, type=Path)
    benchmark.add_argument("--output", type=Path)

    run = subparsers.add_parser("run", help="execute transcription, planning, and rendering")
    run.add_argument("--brief", required=True, type=Path)
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    run.add_argument(
        "--resume",
        metavar="RUN_ID",
        help=(
            "continue from a previous interrupted run. In the default Modal V10 path the old "
            "run is provenance only: local source masters are ignored and the canonical source "
            "is acquired or reused directly inside Modal"
        ),
    )
    run.add_argument(
        "--no-render",
        action="store_true",
        help="stop after timestamped clip planning",
    )
    run.add_argument(
        "--fresh-inference",
        action="store_true",
        help="use a new empty cache so model/grounding inference cannot be satisfied by cache hits",
    )
    run.add_argument(
        "--allow-legacy",
        action="store_true",
        help="explicitly permit heuristic/legacy compatibility engines",
    )
    run.add_argument(
        "--allow-local-lite",
        action="store_true",
        help="explicitly permit the smaller local-lite model profile",
    )
    return parser


def _v10_settings(artifact_root: Path) -> PipelineSettings:
    """Use the open-weight V10 path unless the caller explicitly overrides it."""
    base = PipelineSettings.from_env()
    return replace(
        base,
        artifact_root=artifact_root,
        editorial_engine=os.getenv("CLIPPER_EDITORIAL_ENGINE", "open").strip().lower(),
        grounding_engine=os.getenv("CLIPPER_GROUNDING_ENGINE", "open").strip().lower(),
        compute_profile=os.getenv("CLIPPER_COMPUTE_PROFILE", "balanced").strip().lower(),
    )


def _assert_v10_execution(settings: PipelineSettings, args: argparse.Namespace) -> None:
    if (
        settings.editorial_engine != "open" or settings.grounding_engine != "open"
    ) and not args.allow_legacy:
        raise RuntimeError(
            "refusing non-V10 execution: editorial_engine and grounding_engine must both be open; "
            "remove legacy CLIPPER_* overrides or pass --allow-legacy explicitly"
        )
    if settings.compute_profile == "local-lite" and not args.allow_local_lite:
        raise RuntimeError(
            "refusing implicit local-lite model downgrade; use balanced/quality or pass "
            "--allow-local-lite explicitly"
        )


def _resolved_model_plan(settings: PipelineSettings) -> dict[str, object]:
    plan: dict[str, object] = {
        "editorial_engine": settings.editorial_engine,
        "grounding_engine": settings.grounding_engine,
        "compute_profile": settings.compute_profile,
    }
    if settings.editorial_engine == "open":
        editorial, embedding = editorial_and_embedding_providers(settings.compute_profile)
        plan["editorial"] = editorial.identity.to_dict()
        plan["embedding"] = embedding.identity.to_dict()
    if settings.grounding_engine == "open":
        transcription, alignment, diarization = speech_providers(settings.compute_profile)
        plan["transcription"] = transcription.identity.to_dict()
        plan["alignment"] = alignment.identity.to_dict()
        plan["diarization"] = diarization.identity.to_dict()
    return plan


def _requires_modal(plan: dict[str, object]) -> bool:
    return any(
        isinstance(value, dict)
        and str(value.get("inference_engine", "")).strip().lower().startswith("modal-")
        for value in plan.values()
    )


def _assert_runtime_dependencies(plan: dict[str, object]) -> None:
    """Fail before source acquisition when the resolved model backend is not runnable."""
    if not _requires_modal(plan):
        return
    try:
        modal_spec = importlib.util.find_spec("modal")
    except (ImportError, ValueError):
        modal_spec = None
    if modal_spec is None:
        raise RuntimeError(
            "resolved V10 model plan requires the Modal Python SDK, but 'modal' is not installed. "
            "Install the balanced runtime before source acquisition with: "
            'python -m pip install -e ".[modal]"'
        )


def _assert_modal_functions_available(plan: dict[str, object]) -> None:
    """Hydrate or deploy the complete default Modal runtime before source acquisition."""
    if not _requires_modal(plan):
        return
    try:
        ensure_modal_runtime()
    except Exception as exc:
        raise RuntimeError(
            "required Modal V10 runtime is unavailable and automatic exact-checkout deployment "
            "did not recover it"
        ) from exc


def _source_client_for_run(settings: PipelineSettings) -> PersistentYouTubeClient | None:
    """Keep authorized YouTube masters in a cache for explicit local/legacy execution only."""
    if os.getenv("CLIPPER_SOURCE_FIXTURE_DIR"):
        return None
    configured = os.getenv("CLIPPER_SOURCE_MEDIA_CACHE_ROOT")
    cache_root = Path(configured) if configured else settings.artifact_root / "_source-media-cache"
    return PersistentYouTubeClient(
        max_height=settings.source_max_height,
        media_cache_root=cache_root,
    )


def _resolve_resume_run(artifact_root: Path, resume: str) -> Path:
    run_id = resume.strip()
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise RuntimeError("--resume must be a run ID under the selected artifact root")
    root = artifact_root.resolve()
    run_dir = (root / run_id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("--resume resolved outside the selected artifact root") from exc
    if not run_dir.is_dir():
        raise RuntimeError(f"resume run does not exist: {run_dir}")
    return run_dir


def _validate_resume_run(settings: PipelineSettings, resume: str, *, campaign_id: str) -> Path:
    """Validate continuation provenance without importing any previous source media."""
    run_dir = _resolve_resume_run(settings.artifact_root, resume)
    if not run_dir.name.startswith(f"{campaign_id}-"):
        raise RuntimeError(f"resume run {run_dir.name} does not belong to campaign {campaign_id}")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict) and manifest.get("status") == "SUCCESS":
            raise RuntimeError("refusing to resume a run that already completed successfully")
    return run_dir


def _seed_resume_source_cache(settings: PipelineSettings, resume: str, *, campaign_id: str) -> Path:
    """Promote source masters only for explicit local/legacy execution."""
    run_dir = _validate_resume_run(settings, resume, campaign_id=campaign_id)

    configured = os.getenv("CLIPPER_SOURCE_MEDIA_CACHE_ROOT")
    cache_root = Path(configured) if configured else settings.artifact_root / "_source-media-cache"
    imported: list[Path] = []
    work_dir = run_dir / "work"
    for source in sorted(work_dir.glob("*/*.mkv")):
        if not source.is_file() or source.stat().st_size <= 0:
            continue
        video_id = source.parent.name
        if source.name != f"{video_id}.mkv":
            continue
        target_dir = cache_root / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{video_id}.mkv"
        if not target.is_file() or target.stat().st_size <= 0:
            target.unlink(missing_ok=True)
            try:
                os.link(source, target)
                transfer = "hard-linked"
            except OSError:
                shutil.copy2(source, target)
                transfer = "copied"
            LOGGER.info(
                "resume source cache %s %s -> %s",
                transfer,
                source,
                target,
            )
        else:
            LOGGER.info("resume source cache already contains %s", target)
        sidecar = source.with_suffix(".source.json")
        if sidecar.is_file():
            shutil.copy2(sidecar, target.with_suffix(".source.json"))
        imported.append(target)

    if not imported:
        raise RuntimeError(f"resume run contains no reusable YouTube MKV masters under {work_dir}")
    LOGGER.info(
        "resume recovered %d source master(s) from %s; continuing in a new auditable run",
        len(imported),
        run_dir,
    )
    return run_dir


def _model_id(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    model_id = value.get("model_id")
    return str(model_id) if model_id else None


def _audit_model_evidence(
    run_dir: Path,
    settings: PipelineSettings,
    plan: dict[str, object],
) -> dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("successful run returned without manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("run_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("manifest is missing run_metadata model evidence")

    audit: dict[str, object] = {
        "schema_version": "clipper-model-execution-v1",
        "status": "PASS",
        "resolved_plan": plan,
    }

    if settings.editorial_engine == "open":
        editorial_meta = metadata.get("editorial_inference")
        invocations = (
            editorial_meta.get("model_invocations") if isinstance(editorial_meta, dict) else None
        )
        if not isinstance(invocations, list) or not invocations:
            raise RuntimeError(
                "open editorial run produced no model invocation evidence; refusing success"
            )
        expected_editorial = _model_id(plan.get("editorial"))
        actual_model_ids = {
            model_id
            for item in invocations
            if isinstance(item, dict)
            for model_id in [_model_id(item.get("model"))]
            if model_id
        }
        if expected_editorial and expected_editorial not in actual_model_ids:
            raise RuntimeError(
                "open editorial evidence does not contain the resolved editorial model "
                f"{expected_editorial}"
            )
        live = sum(
            1 for item in invocations if isinstance(item, dict) and item.get("cache_hit") is False
        )
        cached = sum(
            1 for item in invocations if isinstance(item, dict) and item.get("cache_hit") is True
        )
        audit["editorial"] = {
            "expected_model": expected_editorial,
            "observed_models": sorted(actual_model_ids),
            "invocations": len(invocations),
            "live_invocations": live,
            "cache_hits": cached,
        }

    if settings.grounding_engine == "open":
        grounding_meta = metadata.get("grounding_inference")
        models = grounding_meta.get("models") if isinstance(grounding_meta, dict) else None
        if not isinstance(models, list) or not models:
            raise RuntimeError("open grounding run produced no model evidence; refusing success")
        observed: set[str] = set()
        live = 0
        cached = 0
        evidence_count = 0
        for source in models:
            if not isinstance(source, dict):
                continue
            for key in ("transcription", "alignment", "diarization"):
                evidence = source.get(key)
                if not isinstance(evidence, dict):
                    continue
                evidence_count += 1
                model_id = _model_id(evidence.get("model"))
                if model_id:
                    observed.add(model_id)
                if evidence.get("cache_hit") is True:
                    cached += 1
                else:
                    live += 1
        expected = {
            model_id
            for key in ("transcription", "alignment", "diarization")
            for model_id in [_model_id(plan.get(key))]
            if model_id
        }
        missing = sorted(expected - observed)
        if missing:
            raise RuntimeError(
                "open grounding evidence is missing resolved models: " + ", ".join(missing)
            )
        audit["grounding"] = {
            "expected_models": sorted(expected),
            "observed_models": sorted(observed),
            "evidence_records": evidence_count,
            "live_invocations": live,
            "cache_hits": cached,
        }

    (run_dir / "model-execution.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _log_model_summary(plan: dict[str, object], audit: dict[str, object] | None = None) -> None:
    LOGGER.info("resolved model execution plan: %s", json.dumps(plan, sort_keys=True))
    if audit is not None:
        LOGGER.info("model execution evidence: %s", json.dumps(audit, sort_keys=True))


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
            allowed = []
            for video in videos:
                try:
                    assert_video_allowed(brief, video)
                except RightsError:
                    continue
                allowed.append(video)
            print(json.dumps([video.to_dict() for video in allowed], indent=2))
            return 0
        if args.command == "benchmark":
            result = evaluate_corpus_manifest(args.manifest)
            output = json.dumps(result.to_dict(), indent=2) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output, encoding="utf-8")
            print(output, end="")
            return 0 if result.status == "PASS" else 1
        if args.command == "run":
            settings = _v10_settings(args.artifact_root)
            if args.fresh_inference:
                settings = replace(
                    settings,
                    cache_root=args.artifact_root / "_fresh-cache" / uuid.uuid4().hex,
                )
            _assert_v10_execution(settings, args)
            plan = _resolved_model_plan(settings)
            _log_model_summary(plan)
            _assert_runtime_dependencies(plan)
            _assert_modal_functions_available(plan)

            brief = load_brief(args.brief)
            assert_campaign_authorized(brief)
            should_render = not args.no_render
            default_modal_pipeline = _requires_modal(plan) and not os.getenv(
                "CLIPPER_SOURCE_FIXTURE_DIR"
            )
            if default_modal_pipeline:
                resume_run: Path | None = None
                if args.resume:
                    resume_run = _validate_resume_run(
                        settings, args.resume, campaign_id=brief.campaign_id
                    )
                    LOGGER.info(
                        "resume provenance accepted from %s; local source masters "
                        "are intentionally ignored because default V10 execution "
                        "acquires/reuses canonical media in Modal",
                        resume_run,
                    )
                run_dir = run_modal_pipeline(
                    args.brief,
                    artifact_root=args.artifact_root,
                    resume_from_run_id=resume_run.name if resume_run is not None else None,
                    render=should_render,
                    fresh_inference=args.fresh_inference,
                )
            else:
                if args.resume:
                    _seed_resume_source_cache(
                        settings, args.resume, campaign_id=brief.campaign_id
                    )
                source_client = _source_client_for_run(settings)
                run_dir = run_pipeline(
                    args.brief,
                    settings=settings,
                    source_client=source_client,
                    render=should_render,
                )

            audit = _audit_model_evidence(run_dir, settings, plan)
            _log_model_summary(plan, audit)
            print(run_dir)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if should_render and manifest.get("status") == "FAILED":
                return 1
            return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
