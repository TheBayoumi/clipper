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
from .providers.factory import editorial_provider, speech_providers
from .rights import assert_campaign_authorized
from .source_cache import PersistentYouTubeClient
from .stage_contracts import content_fingerprint
from .youtube import DiscoveryRequest, YouTubeClient

LOGGER = logging.getLogger("clipper")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Rights-gated autonomous multimodal short-form clipping pipeline.",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a campaign brief")
    validate.add_argument("--brief", required=True, type=Path)

    discover = subparsers.add_parser(
        "discover", help="discover source candidates outside production execution"
    )
    discover.add_argument("--query", default="")
    discover.add_argument("--channel-id", action="append", default=[])
    discover.add_argument("--video-id", action="append", default=[])
    discover.add_argument("--limit", type=int, default=10)
    discover.add_argument("--language", default="en")
    discover.add_argument("--region-code", default="US")
    discover.add_argument("--published-after")

    benchmark = subparsers.add_parser(
        "benchmark", help="evaluate a private multi-domain acceptance corpus"
    )
    benchmark.add_argument("--manifest", required=True, type=Path)
    benchmark.add_argument("--output", type=Path)

    preflight = subparsers.add_parser(
        "preflight",
        help="validate model-plan runtime dependencies without starting inference",
    )
    preflight.add_argument(
        "--profile",
        choices=("balanced", "quality", "local-lite"),
        default=os.getenv("CLIPPER_COMPUTE_PROFILE", "balanced"),
    )
    preflight.add_argument(
        "--allow-local-lite",
        action="store_true",
        help="explicitly permit the local-lite runtime dependency preflight",
    )

    run = subparsers.add_parser("run", help="execute transcription, planning, and rendering")
    run.add_argument("--brief", required=True, type=Path)
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    run.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="reuse matching content-addressed stage artifacts from an interrupted run",
    )
    run.add_argument(
        "--no-render",
        action="store_true",
        help="stop after timestamped quality-moment planning",
    )
    run.add_argument(
        "--fresh-inference",
        action="store_true",
        help="use a new empty cache so inference cannot be satisfied by existing cache entries",
    )
    run.add_argument(
        "--allow-local-lite",
        action="store_true",
        help="explicitly permit the smaller local-lite model profile",
    )
    run.add_argument(
        "--max-gpu-seconds",
        type=float,
        default=os.getenv("CLIPPER_MAX_GPU_SECONDS"),
        help="hard Modal production GPU-seconds budget (default: env or 21600)",
    )
    run.add_argument(
        "--max-estimated-usd",
        type=float,
        default=os.getenv("CLIPPER_MAX_ESTIMATED_USD"),
        help="hard Modal production estimated-cost budget in USD (default: env or 10)",
    )
    return parser


def _modal_budget_limits(args: argparse.Namespace) -> tuple[float, float]:
    return (
        float(args.max_gpu_seconds) if args.max_gpu_seconds is not None else 21600.0,
        float(args.max_estimated_usd) if args.max_estimated_usd is not None else 10.0,
    )


def _production_settings(artifact_root: Path) -> PipelineSettings:
    base = PipelineSettings.from_env()
    return replace(
        base,
        artifact_root=artifact_root,
        compute_profile=os.getenv("CLIPPER_COMPUTE_PROFILE", "balanced").strip().lower(),
    )


def _assert_production_execution(settings: PipelineSettings, args: argparse.Namespace) -> None:
    if settings.compute_profile == "local-lite" and not args.allow_local_lite:
        raise RuntimeError(
            "refusing implicit local-lite model downgrade; use balanced/quality or pass "
            "--allow-local-lite explicitly"
        )


def _resolved_model_plan(settings: PipelineSettings) -> dict[str, object]:
    plan: dict[str, object] = {
        "architecture": "autonomous-multimodal-quality-graph",
        "compute_profile": settings.compute_profile,
    }
    editorial = editorial_provider(settings.compute_profile)
    plan["editorial"] = editorial.identity.to_dict()
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


def _required_runtime_modules(plan: dict[str, object]) -> tuple[str, ...]:
    modules: set[str] = set()
    for value in plan.values():
        if not isinstance(value, dict):
            continue
        engine = str(value.get("inference_engine") or "").strip().lower()
        if engine.startswith("modal-"):
            modules.add("modal")
        elif engine == "transformers":
            modules.add("transformers")
        elif engine == "faster-whisper":
            modules.add("faster_whisper")
        elif engine == "whisperx":
            modules.add("whisperx")
        elif engine == "pyannote.audio":
            modules.add("pyannote.audio")
    return tuple(sorted(modules))


def _assert_runtime_dependencies(plan: dict[str, object]) -> None:
    missing: list[str] = []
    for module_name in _required_runtime_modules(plan):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "resolved production model plan is missing runtime module(s): "
            + ", ".join(missing)
            + '. Install the complete runtime with: python -m pip install -e ".[open-models]"'
        )


def _assert_modal_functions_available(plan: dict[str, object]) -> None:
    if not _requires_modal(plan):
        return
    try:
        ensure_modal_runtime()
    except Exception as exc:
        raise RuntimeError(
            "required Modal production runtime is unavailable and automatic exact-checkout "
            "deployment did not recover it"
        ) from exc


def _source_client_for_run(settings: PipelineSettings) -> PersistentYouTubeClient | None:
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
            LOGGER.info("resume source cache %s %s -> %s", transfer, source, target)
        sidecar = source.with_suffix(".source.json")
        if sidecar.is_file():
            shutil.copy2(sidecar, target.with_suffix(".source.json"))
        imported.append(target)
    if not imported:
        raise RuntimeError(f"resume run contains no reusable source masters under {work_dir}")
    LOGGER.info(
        "resume recovered %d source master(s) from %s; continuing by content identity",
        len(imported),
        run_dir,
    )
    return run_dir


def _model_id(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    model_id = value.get("model_id")
    return str(model_id) if model_id else None


def _model_cache_fingerprint(value: object) -> str | None:
    if not isinstance(value, dict) or not value:
        return None
    return content_fingerprint({**value, "sampling": {}})


def _audit_model_evidence(
    run_dir: Path,
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
        "contract_hash": content_fingerprint(
            {
                "artifact": "model-execution",
                "required": ["resolved_plan", "editorial", "grounding"],
            }
        ),
        "status": "PASS",
        "resolved_plan": plan,
    }

    editorial_meta = metadata.get("editorial_inference")
    if not isinstance(editorial_meta, dict):
        raise RuntimeError("autonomous editorial run produced no model invocation evidence")
    invocations = editorial_meta.get("model_invocations")
    if not isinstance(invocations, list):
        raise RuntimeError("editorial model invocation evidence must be an array")

    expected_editorial = _model_id(plan.get("editorial"))
    expected_fingerprint = _model_cache_fingerprint(plan.get("editorial"))
    cache_summary = editorial_meta.get("cache_summary")
    fully_cached = False
    stage_cache_hits = 0
    stage_executions = 0
    if isinstance(cache_summary, dict):
        raw_hits = cache_summary.get("stage_cache_hits")
        raw_executions = cache_summary.get("stage_executions")
        if (
            isinstance(raw_hits, int)
            and not isinstance(raw_hits, bool)
            and raw_hits >= 0
            and isinstance(raw_executions, int)
            and not isinstance(raw_executions, bool)
            and raw_executions >= 0
        ):
            stage_cache_hits = raw_hits
            stage_executions = raw_executions
            fully_cached = stage_executions == 0 and stage_cache_hits > 0

    actual_model_ids = {
        model_id
        for item in invocations
        if isinstance(item, dict)
        for model_id in [_model_id(item.get("model"))]
        if model_id
    }
    if invocations:
        if expected_editorial and expected_editorial not in actual_model_ids:
            raise RuntimeError(
                "editorial evidence does not contain the resolved editorial model "
                f"{expected_editorial}"
            )
    elif fully_cached:
        if not isinstance(cache_summary, dict):
            raise RuntimeError("fully cached editorial resume is missing cache identity evidence")
        observed_fingerprint = str(cache_summary.get("editorial_model_fingerprint") or "")
        observed_identity = cache_summary.get("editorial_model")
        if (
            not expected_fingerprint
            or observed_fingerprint != expected_fingerprint
            or observed_identity != plan.get("editorial")
        ):
            raise RuntimeError(
                "fully cached editorial resume is not bound to the resolved model identity"
            )
        if expected_editorial:
            actual_model_ids.add(expected_editorial)
    else:
        raise RuntimeError("autonomous editorial run produced no model invocation evidence")

    audit["editorial"] = {
        "expected_model": expected_editorial,
        "observed_models": sorted(actual_model_ids),
        "invocations": len(invocations),
        "live_invocations": sum(
            1 for item in invocations if isinstance(item, dict) and item.get("cache_hit") is False
        ),
        "cache_hits": (
            stage_cache_hits
            if fully_cached
            else sum(
                1
                for item in invocations
                if isinstance(item, dict) and item.get("cache_hit") is True
            )
        ),
        "stage_executions": stage_executions,
        "fully_cached_resume": fully_cached,
    }

    grounding_meta = metadata.get("grounding_inference")
    models = grounding_meta.get("models") if isinstance(grounding_meta, dict) else None
    if not isinstance(models, list) or not models:
        raise RuntimeError("canonical grounding run produced no model evidence")
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
            "canonical grounding evidence is missing resolved models: " + ", ".join(missing)
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
            request = DiscoveryRequest(
                query=args.query,
                channel_ids=tuple(args.channel_id),
                video_ids=tuple(args.video_id),
                limit=args.limit,
                language=args.language,
                region_code=args.region_code,
                published_after=args.published_after,
            )
            videos = YouTubeClient().discover(request)
            print(json.dumps([video.to_dict() for video in videos], indent=2))
            return 0
        if args.command == "benchmark":
            result = evaluate_corpus_manifest(args.manifest)
            output = json.dumps(result.to_dict(), indent=2) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output, encoding="utf-8")
            print(output, end="")
            return 0 if result.status == "PASS" else 1
        if args.command == "preflight":
            settings = replace(PipelineSettings.from_env(), compute_profile=args.profile)
            _assert_production_execution(settings, args)
            plan = _resolved_model_plan(settings)
            _assert_runtime_dependencies(plan)
            print(
                json.dumps(
                    {
                        "status": "READY",
                        "compute_profile": settings.compute_profile,
                        "required_modules": list(_required_runtime_modules(plan)),
                        "model_plan": plan,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            settings = _production_settings(args.artifact_root)
            if args.fresh_inference:
                settings = replace(
                    settings,
                    cache_root=args.artifact_root / "_fresh-cache" / uuid.uuid4().hex,
                )
            _assert_production_execution(settings, args)
            plan = _resolved_model_plan(settings)
            _log_model_summary(plan)
            _assert_runtime_dependencies(plan)

            brief = load_brief(args.brief)
            assert_campaign_authorized(brief)
            should_render = not args.no_render
            default_modal_pipeline = _requires_modal(plan) and not os.getenv(
                "CLIPPER_SOURCE_FIXTURE_DIR"
            )
            if default_modal_pipeline:
                max_gpu_seconds, max_estimated_usd = _modal_budget_limits(args)
                resume_run: Path | None = None
                if args.resume:
                    resume_run = _validate_resume_run(
                        settings, args.resume, campaign_id=brief.campaign_id
                    )
                run_dir = run_modal_pipeline(
                    args.brief,
                    artifact_root=args.artifact_root,
                    resume_from_run_id=resume_run.name if resume_run is not None else None,
                    render=should_render,
                    fresh_inference=args.fresh_inference,
                    max_gpu_seconds=max_gpu_seconds,
                    max_estimated_usd=max_estimated_usd,
                )
            else:
                if args.max_gpu_seconds is not None or args.max_estimated_usd is not None:
                    raise RuntimeError(
                        "this execution path cannot enforce hard in-flight GPU/cost budgets; "
                        "omit the budget flags for local/fixture diagnostics or use the "
                        "cancellable Modal production pipeline"
                    )
                _assert_modal_functions_available(plan)
                if args.resume:
                    _seed_resume_source_cache(settings, args.resume, campaign_id=brief.campaign_id)
                source_client = _source_client_for_run(settings)
                run_dir = run_pipeline(
                    args.brief,
                    settings=settings,
                    source_client=source_client,
                    render=should_render,
                )

            audit = _audit_model_evidence(run_dir, plan)
            _log_model_summary(plan, audit)
            print(run_dir)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") == "FAILED":
                return 1
            return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
