"""Replay only failed visual-review calls from a materialized Clipper run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clipper.providers.factory import vision_provider
from clipper.visual_ai import _review_context, parse_visual_review


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _frame_time(path: Path) -> float:
    try:
        return float(path.stem.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"visual review frame has no timestamp: {path.name}") from exc


def main() -> int:
    args = _parser().parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    attempts = manifest.get("render_attempts")
    plans = manifest.get("edit_plans")
    if not isinstance(attempts, list) or not isinstance(plans, list):
        raise ValueError("manifest is missing render attempts or edit plans")
    provider = vision_provider("balanced")
    results: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("status") != "RENDER_FAILED":
            continue
        attempt_number = int(attempt["attempt"])
        matches = sorted((run_dir / "clips").glob(f"attempt-{attempt_number:02d}-*.mp4"))
        if len(matches) != 1:
            raise RuntimeError(
                f"attempt {attempt_number} expected one rendered clip, found {len(matches)}"
            )
        clip_path = matches[0]
        qc = _read_json(run_dir / "qc" / f"{clip_path.stem}.json")
        if qc.get("status") != "PASS":
            continue
        plan = next(
            (
                item
                for item in plans
                if isinstance(item, dict)
                and item.get("plan_id") == attempt.get("plan_id")
                and item.get("concept_id") == attempt.get("concept_id")
            ),
            None,
        )
        if not isinstance(plan, dict):
            raise RuntimeError(f"attempt {attempt_number} has no matching edit plan")
        frame_dir = run_dir / "visual-review" / clip_path.stem / "frames"
        frames = sorted(frame_dir.glob("frame-*.jpg"))
        if not frames:
            raise RuntimeError(f"attempt {attempt_number} has no visual-review frames")
        frame_times = tuple(_frame_time(path) for path in frames)
        spans = plan.get("source_spans")
        span = spans[0] if isinstance(spans, list) and spans else {}
        duration = float(qc.get("video", {}).get("expected_duration_seconds") or 0.0)
        context = _review_context(
            duration=duration,
            frame_times=frame_times,
            context={
                "plan_id": attempt.get("plan_id"),
                "concept_id": attempt.get("concept_id"),
                "source_start": span.get("start") if isinstance(span, dict) else None,
                "source_end": span.get("end") if isinstance(span, dict) else None,
                "hook_mode": plan.get("hook_mode"),
                "technical_qc": qc,
            },
        )
        try:
            response = provider.inspect(
                task="rendered_clip_review",
                frames=frames,
                context=context,
            )
            report = parse_visual_review(response.value)
            results.append(
                {
                    "attempt": attempt_number,
                    "plan_id": attempt.get("plan_id"),
                    "concept_id": attempt.get("concept_id"),
                    "clip": clip_path.name,
                    "status": "REVIEWED",
                    "report": report.to_dict(),
                    "model": response.model.to_dict(),
                    "usage": asdict(response.usage),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "attempt": attempt_number,
                    "plan_id": attempt.get("plan_id"),
                    "concept_id": attempt.get("concept_id"),
                    "clip": clip_path.name,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    output = args.output or run_dir / "visual-review-recovery.json"
    payload = {
        "schema_version": "clipper-visual-review-recovery-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_manifest_status": manifest.get("status"),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if results and all(item["status"] == "REVIEWED" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
