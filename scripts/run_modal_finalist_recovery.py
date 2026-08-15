from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clipper.modal_execution import _materialize_remote_run

PLAN_KEYS = (
    {"concept_id": "c14", "plan_id": "p3"},
    {"concept_id": "c5", "plan_id": "p1"},
)
LEGACY_FINALIST_KEYS = (
    {"concept_id": "c3", "plan_id": "p3"},
    {"concept_id": "c6", "plan_id": "p1"},
    {"concept_id": "c11", "plan_id": "p1"},
    {"concept_id": "c2", "plan_id": "p4"},
)
AUTOFRAME_REPAIR_KEYS = ({"concept_id": "c3", "plan_id": "p3"},)
_ARTIFACT_PATH_MARKERS = (
    "/clips/",
    "/captions/",
    "/tracking/",
    "/visual-review/",
    "/qc/",
    "/assets/",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _modal_function(app_name: str, function_name: str) -> Any:
    try:
        import modal
    except ImportError as exc:  # pragma: no cover - exercised only without the Modal extra
        raise RuntimeError("install clipper[modal] before running targeted recovery") from exc
    return modal.Function.from_name(app_name, function_name)


def normalize_materialized_recovery_paths(run_dir: Path, serialized_root: str) -> int:
    """Rewrite Modal artifact paths for the environment consuming the manifest."""

    manifest = _load_object(run_dir / "manifest.json")
    if manifest.get("status") != "SUCCESS" or not isinstance(
        manifest.get("run_metadata", {}).get("targeted_recovery"), dict
    ):
        raise RuntimeError("refusing to normalize a non-recovery manifest")

    root = serialized_root.rstrip("/\\")

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            for marker in _ARTIFACT_PATH_MARKERS:
                marker_index = normalized.find(marker)
                if marker_index >= 0:
                    relative = normalized[marker_index + 1 :]
                    if root.startswith("/"):
                        return f"{root}/{relative}"
                    return str(Path(root).joinpath(*relative.split("/")))
            return value
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    paths = [run_dir / "manifest.json", run_dir / "editorial-review.json"]
    paths.extend(sorted((run_dir / "qc").glob("*.json")))
    rewritten = 0
    for path in paths:
        before = _load_object(path)
        after = rewrite(before)
        if after != before:
            path.write_text(
                json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            rewritten += 1
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair explicitly selected finalists inside Modal, review only newly rendered "
            "frames, revalidate every reused clip, and materialize the six-finalist run."
        )
    )
    parser.add_argument("source_run_id")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--app",
        default=os.getenv("CLIPPER_MODAL_PIPELINE_APP", "clipper-production-pipeline"),
    )
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-c14-from-run",
        help="Reuse an already-passed c14/p3 result and rerender only c5/p1.",
    )
    reuse_group.add_argument(
        "--repair-legacy-captions-from-run",
        metavar="RUN_ID",
        help=(
            "Reuse clean c14/p3 and c5/p1 from RUN_ID, then rerender and review only "
            "the four legacy finalists with overlapping opening captions."
        ),
    )
    reuse_group.add_argument(
        "--repair-autoframe-from-run",
        metavar="RUN_ID",
        help=(
            "Reuse five passed finalists from RUN_ID, then rerender and review only "
            "c3/p3 with the current autoframing implementation."
        ),
    )
    args = parser.parse_args()

    source_run = args.artifact_root / args.source_run_id
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    manifest = _load_object(source_run / "manifest.json")
    if manifest.get("status") != "FAILED":
        raise RuntimeError("the source run is not the expected failed manifest")
    plan_keys = {
        (str(item.get("concept_id") or ""), str(item.get("plan_id") or ""))
        for item in manifest.get("edit_plans", [])
        if isinstance(item, dict)
    }
    expected = {
        (item["concept_id"], item["plan_id"]) for item in (*LEGACY_FINALIST_KEYS, *PLAN_KEYS)
    }
    if not expected.issubset(plan_keys):
        raise RuntimeError("the source run does not contain all six recovery plans")

    prior_review = _load_object(source_run / "visual-review-recovery.json")
    recovery_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    request = {
        "source_run_id": args.source_run_id,
        "recovery_id": recovery_id,
        "plan_keys": list(PLAN_KEYS),
        "prior_review_recovery": prior_review,
    }
    if args.reuse_c14_from_run:
        request["base_run_id"] = args.reuse_c14_from_run
        request["reuse_plan_keys"] = [PLAN_KEYS[0]]
    elif args.repair_legacy_captions_from_run:
        request["base_run_id"] = args.repair_legacy_captions_from_run
        request["reuse_plan_keys"] = list(PLAN_KEYS)
        request["rerender_plan_keys"] = list(LEGACY_FINALIST_KEYS)
    elif args.repair_autoframe_from_run:
        request["base_run_id"] = args.repair_autoframe_from_run
        request["reuse_plan_keys"] = list(PLAN_KEYS)
        request["rerender_plan_keys"] = list(AUTOFRAME_REPAIR_KEYS)
    try:
        response = _modal_function(args.app, "recover_finalists").remote(request)
    except Exception:
        failed_run_path = f"/{args.source_run_id}-targeted-{recovery_id}-failed"
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "source_run_id": args.source_run_id,
                    "recovery_id": recovery_id,
                    "failure_evidence_volume": "clipper-production-artifacts",
                    "expected_failure_evidence_path": failed_run_path,
                },
                indent=2,
            )
        )
        raise
    if not isinstance(response, dict) or response.get("status") != "PASS":
        raise RuntimeError(f"targeted Modal recovery returned an invalid response: {response!r}")
    remote_run_path = str(response.get("run_path") or "")
    volume_name = str(response.get("run_volume") or "clipper-production-artifacts")
    if not remote_run_path:
        raise RuntimeError("targeted Modal recovery returned no run path")
    local_run = _materialize_remote_run(
        artifact_root=args.artifact_root,
        volume_name=volume_name,
        remote_run_path=remote_run_path,
    )
    normalize_materialized_recovery_paths(local_run, str(local_run.resolve()))
    print(json.dumps({**response, "local_run": str(local_run.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
