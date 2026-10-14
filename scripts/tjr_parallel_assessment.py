"""Resume frozen Double Coverage candidate work in independent GitHub Actions shards.

Workers consume only the previously validated transcript, proposal and source evidence.
No worker downloads video or runs FFmpeg. All candidates are indexed against the
same complete proposal list; placeholders are never interpreted as assessments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from clipper.brief import load_brief
from clipper.models import ClipCandidate, TranscriptSegment, WordTiming

SHARD_WIDTH = 8
SOURCE_KEYS = (
    "version",
    "source_sha256",
    "transcript_sha256",
    "proposal_sha256",
    "model_sha256",
    "model_revision",
    "duration_bounds",
    "seed",
    "temperature",
    "selector_contract_version",
    "selector_sha256",
    "boundary_contract_version",
    "boundary_sha256",
    "reviewer_sha256",
    "reviewer_model_profile",
)


def unique(root: Path, basename: str) -> Path:
    matches = sorted(root.rglob(basename))
    if len(matches) != 1:
        raise RuntimeError(
            f"PARALLEL_ARCHIVE_MISSING_OR_AMBIGUOUS: {basename} count={len(matches)}"
        )
    return matches[0]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assessment_data(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cache = read(unique(root, "editorial-cache.json"))
    proposal = read(unique(root, "proposal-cache.json"))
    transcript = read(unique(root, "transcript.json"))
    staged = read(unique(root, "staged-original.json"))
    if not isinstance(cache, dict) or not isinstance(proposal, dict):
        raise RuntimeError("PARALLEL_INVALID_CACHE_ENVELOPE")
    candidates = proposal.get("candidates")
    if (
        proposal.get("complete") is not True
        or not isinstance(candidates, list)
        or not candidates
        or not isinstance(transcript, list)
        or not transcript
    ):
        raise RuntimeError("PARALLEL_INCOMPLETE_PROPOSAL_OR_TRANSCRIPT")
    identity = cache.get("identity", {})
    if (
        staged.get("video_id") != candidates[0].get("video_id")
        or identity.get("source_sha256") != staged.get("source_sha256")
        or identity.get("transcript_sha256") != digest(transcript)
        or identity.get("proposal_sha256") != digest(candidates)
        or any(c.get("video_id") != staged["video_id"] for c in candidates)
    ):
        raise RuntimeError("PARALLEL_ARCHIVE_SOURCE_IDENTITY_MISMATCH")
    count = cache.get("processed_count")
    decisions = cache.get("decisions", [])
    if (
        cache.get("complete") is not False
        or type(count) is not int
        or not 0 < count <= len(candidates)
        or not isinstance(decisions, list)
        or len(decisions) != count
        or not isinstance(cache.get("candidates"), list)
    ):
        raise RuntimeError("PARALLEL_BASELINE_CHECKPOINT_INVALID")
    for i, assessment in enumerate(decisions):
        if not isinstance(assessment, dict) or (
            assessment.get("proposal_start"),
            assessment.get("proposal_end"),
        ) != (candidates[i]["start"], candidates[i]["end"]):
            raise RuntimeError(f"PARALLEL_BASELINE_ORDER_MISMATCH: {i}")
    return cache, proposal, transcript, candidates


def plan(root: Path, output: Path, *, width: int = SHARD_WIDTH) -> list[dict[str, int]]:
    if not 1 <= width <= 12:
        raise ValueError("shard width must be between 1 and 12")
    cache, _, _, candidates = assessment_data(root)
    start = cache["processed_count"]
    if start == len(candidates):
        raise RuntimeError("PARALLEL_NOT_NEEDED: baseline is already complete")
    shards = [
        {"start": start_index, "stop": min(start_index + width, len(candidates))}
        for start_index in range(start, len(candidates), width)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"shards": shards, "total": len(candidates), "cached": start}) + "\n"
    )
    print(
        f"PARALLEL_PLAN: preserved={start}/{len(candidates)} "
        f"remaining={len(candidates) - start} shards={len(shards)}",
        flush=True,
    )
    return shards


def transcript_objects(items: list[dict[str, Any]]) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=item["start"],
            end=item["end"],
            text=item["text"],
            words=tuple(WordTiming(**word) for word in item.get("words", [])),
        )
        for item in items
    ]


def candidate_objects(items: list[dict[str, Any]]) -> list[ClipCandidate]:
    return [
        ClipCandidate(
            video_id=item["video_id"],
            start=item["start"],
            end=item["end"],
            text=item["text"],
            score=item["score"],
            reasons=tuple(item["reasons"]),
        )
        for item in items
    ]


def shard(root: Path, output: Path, *, start: int, stop: int) -> None:
    from scripts.tjr_semantic_editor import refine_contextual_candidates

    cache, _, transcript, proposals = assessment_data(root)
    count = cache["processed_count"]
    if not count <= start < stop <= len(proposals) or stop - start > 12:
        raise RuntimeError("PARALLEL_SHARD_RANGE_INVALID")
    output.mkdir(parents=True, exist_ok=True)
    result, _audit = refine_contextual_candidates(
        load_brief(Path("campaigns/reach-double-coverage-dedicated.yaml")),
        candidate_objects(proposals),
        transcript_objects(transcript),
        source_sha256=cache["identity"]["source_sha256"],
        cache_path=output / "editorial-cache.json",
        reuse_path=None,
        require_reusable_editorial_cache=False,
        force_full_candidate_reassessment=True,
        only_indices=frozenset(range(start, stop)),
    )
    saved = read(output / "editorial-cache.json")
    if saved.get("complete") is not False or len(saved["audit"]["assessments"]) != len(proposals):
        raise RuntimeError("PARALLEL_SHARD_UNEXPECTED_COMPLETION")
    for idx, decision in enumerate(saved["audit"]["assessments"]):
        if start <= idx < stop:
            if decision.get("rejection") == "SHARD_NOT_ASSIGNED":
                raise RuntimeError(f"PARALLEL_SHARD_MISSING_CANDIDATE: {idx}")
        elif decision.get("rejection") != "SHARD_NOT_ASSIGNED":
            raise RuntimeError(f"PARALLEL_SHARD_OVERREACH: {idx}")
    (output / "shard-manifest.json").write_text(
        json.dumps(
            {
                "start": start,
                "stop": stop,
                "total": len(proposals),
                "source_sha256": cache["identity"]["source_sha256"],
                "proposal_sha256": cache["identity"]["proposal_sha256"],
                "accepted": len(result),
                "assessment_count": stop - start,
                "complete": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PARALLEL_SHARD_COMPLETE: indices={start}:{stop} accepted={len(result)}", flush=True)


def merge(root: Path, parts: Path, output: Path) -> None:
    baseline, _proposal, _transcript, candidates = assessment_data(root)
    first = baseline["processed_count"]
    all_decisions: list[dict[str, Any] | None] = [None] * len(candidates)
    all_decisions[:first] = baseline["decisions"]
    all_clips = list(baseline["candidates"])
    seen: set[int] = set(range(first))
    shard_paths = sorted(parts.rglob("shard-manifest.json"))
    if not shard_paths:
        raise RuntimeError("PARALLEL_NO_COMPLETED_SHARDS")
    merged_identity: dict[str, Any] | None = None
    for path in shard_paths:
        manifest = read(path)
        start, stop = manifest["start"], manifest["stop"]
        saved = read(path.with_name("editorial-cache.json"))
        identity = saved["identity"]
        if (
            not manifest.get("complete")
            or manifest.get("total") != len(candidates)
            or manifest.get("assessment_count") != stop - start
            or manifest.get("source_sha256") != baseline["identity"]["source_sha256"]
            or manifest.get("proposal_sha256") != baseline["identity"]["proposal_sha256"]
            or saved.get("complete") is not False
            or len(saved["audit"]["assessments"]) != len(candidates)
        ):
            raise RuntimeError(f"PARALLEL_SHARD_CONTRACT_FAILED: {path}")
        if merged_identity is None:
            merged_identity = identity
            if any(identity.get(key) != baseline["identity"].get(key) for key in SOURCE_KEYS):
                raise RuntimeError("PARALLEL_BASELINE_REVIEW_CONTRACT_MISMATCH")
        elif identity != merged_identity:
            raise RuntimeError("PARALLEL_SHARD_IDENTITY_DRIFT")
        if not first <= start < stop <= len(candidates) or seen.intersection(range(start, stop)):
            raise RuntimeError("PARALLEL_SHARD_OVERLAP_OR_RANGE")
        for i, decision in enumerate(saved["audit"]["assessments"]):
            if start <= i < stop:
                if decision.get("rejection") == "SHARD_NOT_ASSIGNED" or (
                    decision.get("proposal_start"),
                    decision.get("proposal_end"),
                ) != (candidates[i]["start"], candidates[i]["end"]):
                    raise RuntimeError(f"PARALLEL_MISSING_OR_WRONG_DECISION: {i}")
                all_decisions[i] = decision
            elif decision.get("rejection") != "SHARD_NOT_ASSIGNED":
                raise RuntimeError(f"PARALLEL_UNASSIGNED_DECISION_PRESENT: {i}")
        all_clips.extend(saved["candidates"])
        seen.update(range(start, stop))
    if seen != set(range(len(candidates))) or any(item is None for item in all_decisions):
        raise RuntimeError(f"PARALLEL_AGGREGATION_INCOMPLETE: {len(seen)}/{len(candidates)}")
    if merged_identity is None:
        raise RuntimeError("PARALLEL_NO_EDITORIAL_IDENTITY")
    for clip in all_clips:
        if clip.get("video_id") != candidates[0]["video_id"]:
            raise RuntimeError("PARALLEL_RENDER_CANDIDATE_SOURCE_MISMATCH")
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "architecture": merged_identity["version"],
        "assessments": all_decisions,
        "cache_reused": False,
        "resumed_assessments": first,
        "selector_cache_hits": first,
        "reviewer_cache_hits": 0,
        "candidate_count": len(all_clips),
        "human_review_required": True,
    }
    (output / "editorial-cache.json").write_text(
        json.dumps(
            {
                "identity": merged_identity,
                "complete": True,
                "audit": audit,
                "candidates": all_clips,
            },
            indent=2,
        )
        + "\n"
    )
    shutil.copyfile(unique(root, "proposal-cache.json"), output / "proposal-cache.json")
    # Staged-original publication provenance must remain available offline.
    # The renderer MUST NOT query YouTube again when the official feed omits a date.
    shutil.copyfile(unique(root, "staged-original.json"), output / "staged-original.json")
    shutil.copyfile(
        unique(root, "official-source-candidates.json"),
        output / "official-source-candidates.json",
    )
    (output / "merge-proof.json").write_text(
        json.dumps(
            {
                "source_sha256": merged_identity["source_sha256"],
                "proposal_sha256": merged_identity["proposal_sha256"],
                "transcript_sha256": merged_identity["transcript_sha256"],
                "assessment_count": len(all_decisions),
                "reused_baseline": first,
                "shard_count": len(shard_paths),
                "accepted_candidate_count": len(all_clips),
                "all_indices_unique": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"PARALLEL_MERGE_VERIFIED: {len(all_decisions)}/{len(candidates)} "
        f"baseline={first} accepted={len(all_clips)} shards={len(shard_paths)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "shard", "merge"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parts", type=Path, default=Path("editorial-parts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    if args.action == "plan":
        plan(args.baseline, args.output)
    elif args.action == "shard":
        if args.start is None or args.stop is None:
            parser.error("shard needs --start and --stop")
        shard(args.baseline, args.output, start=args.start, stop=args.stop)
    else:
        merge(args.baseline, args.parts, args.output)


if __name__ == "__main__":
    main()
