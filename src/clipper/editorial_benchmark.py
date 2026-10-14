"""Provenance checks for transcript-grounded headline evaluation cases.

These checks establish source identity and exact text windows. They do not
establish that an annotator's semantic label is correct or audio-verified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


@dataclass(frozen=True)
class HeldoutClaim:
    case_id: str
    headline: str
    expected_supported: bool
    annotation_reason: str
    annotation_status: str
    source_start: float
    source_end: float
    source_clip: str
    source_units: tuple[str, ...]


@dataclass(frozen=True)
class FrozenRelation:
    case_id: str
    fixture_index: int
    headline: str
    dimension: str
    question: str
    answer: str
    expected_supported: bool
    annotation_reason: str
    evidence_first_unit: int
    evidence_last_unit: int
    evidence_quote: str
    source_units: tuple[str, ...]


def load_frozen_relations(
    fixture_path: Path, proof_path: Path, transcript_path: Path, provenance_path: Path
) -> list[FrozenRelation]:
    """Bind relation-level diagnostics to the frozen proof and original transcript.

    Labels are derived from the existing transcript controls, not independent
    audio-reviewed gold. Neither a valid fixture nor a high score qualifies a
    production reviewer.
    """
    fixture: Any = json.loads(fixture_path.read_text(encoding="utf-8"))
    proof: Any = json.loads(proof_path.read_text(encoding="utf-8"))
    transcript: Any = json.loads(transcript_path.read_text(encoding="utf-8"))
    provenance: Any = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict) or set(fixture) != {
        "schema",
        "annotation_status",
        "source_video_id",
        "source_sha256",
        "transcript_sha256",
        "baseline_proof_sha256",
        "cases",
    }:
        raise ValueError("frozen relation fixture has missing or unknown fields")
    if (
        fixture["schema"] != "clipper-frozen-relations-v1"
        or fixture["annotation_status"]
        != "derived_from_frozen_transcript_controls_not_independent_audio_gold"
        or not isinstance(fixture["source_video_id"], str)
        or not _VIDEO_ID.fullmatch(fixture["source_video_id"])
        or not isinstance(fixture["source_sha256"], str)
        or not _SHA256.fullmatch(fixture["source_sha256"])
        or not isinstance(fixture["transcript_sha256"], str)
        or not _SHA256.fullmatch(fixture["transcript_sha256"])
        or not isinstance(fixture["baseline_proof_sha256"], str)
        or not _SHA256.fullmatch(fixture["baseline_proof_sha256"])
    ):
        raise ValueError("frozen relation identity or annotation status is invalid")
    proof_hash = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    transcript_hash = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
    identity = provenance.get("identity") if isinstance(provenance, dict) else None
    if (
        proof_hash != fixture["baseline_proof_sha256"]
        or not isinstance(proof, dict)
        or proof.get("experiment") != "evidence_preserving_source_qa"
        or proof.get("experiment_complete") is not True
        or proof.get("transcript_sha256") != fixture["transcript_sha256"]
        or not isinstance(proof.get("model_profile"), dict)
        or proof["model_profile"].get("source_sha256") != fixture["source_sha256"]
        or not isinstance(identity, dict)
        or identity.get("source_sha256") != fixture["source_sha256"]
        or identity.get("transcript_sha256") != fixture["transcript_sha256"]
        or transcript_hash != fixture["transcript_sha256"]
    ):
        raise ValueError("frozen relation proof/source/transcript identity mismatch")
    frozen = proof.get("annotated_fixtures")
    if not isinstance(frozen, list) or len(frozen) != 12:
        raise ValueError("frozen relations require twelve original headline controls")
    if (
        not isinstance(transcript, list)
        or not transcript
        or any(
            not isinstance(segment, dict) or not isinstance(segment.get("text"), str)
            for segment in transcript
        )
    ):
        raise ValueError("frozen relations require the original full transcript")
    full_text = " ".join(" ".join(segment["text"] for segment in transcript).split()).casefold()
    sources: list[tuple[str, ...]] = []
    for item in frozen:
        try:
            if not isinstance(item, dict):
                raise ValueError("frozen relation proof has invalid fixture")
            payload = json.loads(item["request"]["messages"][1]["content"])
            units = tuple(unit["text"] for unit in payload["source_units"])
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("frozen relation proof has invalid source units") from error
        if not units or any(not isinstance(unit, str) or not unit.strip() for unit in units):
            raise ValueError("frozen relation proof has empty source units")
        if " ".join(" ".join(units).split()).casefold() not in full_text:
            raise ValueError("frozen relation source units are absent from the full transcript")
        sources.append(units)
    raw_cases = fixture["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("frozen relation fixture has no cases")
    seen: set[str] = set()
    covered: set[int] = set()
    result: list[FrozenRelation] = []
    for raw in raw_cases:
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "fixture_index",
            "dimension",
            "question",
            "answer",
            "expected_supported",
            "evidence",
            "annotation_reason",
        }:
            raise ValueError("frozen relation case has missing or unknown fields")
        index, case_id = raw["fixture_index"], raw["id"]
        if (
            type(index) is not int
            or not 0 <= index < 12
            or not isinstance(case_id, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", case_id)
            or case_id in seen
            or not isinstance(raw["dimension"], str)
            or raw["dimension"]
            not in {
                "actor_action",
                "attribution",
                "event_modality",
                "condition",
                "relationship_role",
                "quantity_outcome",
                "setting_time",
            }
            or not isinstance(raw["question"], str)
            or not raw["question"].strip().endswith("?")
            or not isinstance(raw["answer"], str)
            or not raw["answer"].strip()
            or type(raw["expected_supported"]) is not bool
            or not isinstance(raw["annotation_reason"], str)
            or not raw["annotation_reason"].strip()
        ):
            raise ValueError("frozen relation case has invalid annotation")
        headline = frozen[index].get("headline")
        if not isinstance(headline, str) or raw["answer"].casefold() not in headline.casefold():
            raise ValueError("frozen relation answer is absent from its original headline")
        evidence = raw["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {
            "first_unit",
            "last_unit",
            "quote",
        }:
            raise ValueError("frozen relation evidence is incomplete")
        first, last, quote = (
            evidence["first_unit"],
            evidence["last_unit"],
            evidence["quote"],
        )
        units = sources[index]
        if (
            type(first) is not int
            or type(last) is not int
            or not 0 <= first <= last < len(units)
            or not isinstance(quote, str)
            or not quote.strip()
            or quote not in " ".join(units[first : last + 1])
        ):
            raise ValueError("frozen relation quote is absent from cited source units")
        result.append(
            FrozenRelation(
                case_id=case_id,
                fixture_index=index,
                headline=headline,
                dimension=raw["dimension"],
                question=raw["question"],
                answer=raw["answer"],
                expected_supported=raw["expected_supported"],
                annotation_reason=raw["annotation_reason"],
                evidence_first_unit=first,
                evidence_last_unit=last,
                evidence_quote=quote,
                source_units=units,
            )
        )
        seen.add(case_id)
        covered.add(index)
    if covered != set(range(12)):
        raise ValueError("frozen relations must cover all twelve original controls")
    return result


def qualification_pass(
    exchange_rows: list[dict[str, Any]],
    factual_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    *,
    expected_heldout: int,
) -> bool:
    """Fail closed over both frozen controls and the unmodified factual gate.

    Experimental backends are scored separately and must never make the
    production reviewer appear qualified. A held-out mismatch is a failed
    qualification, including a false rejection or a contract error.
    """
    if len(exchange_rows) != 6 or len(factual_rows) != 12 or len(heldout_rows) != expected_heldout:
        return False
    if not all(
        row.get("contract_valid") is True and row.get("passed") is True
        for row in (*exchange_rows, *factual_rows)
    ):
        return False
    return all(
        isinstance(row.get("existing"), dict)
        and row["existing"].get("contract_valid") is True
        and row["existing"].get("passed") is True
        for row in heldout_rows
    )


def load_heldout_claims(
    fixture_path: Path, transcript_path: Path, provenance_path: Path
) -> list[HeldoutClaim]:
    """Bind provisional controls to the preserved full transcript and source SHA."""
    fixture: Any = json.loads(fixture_path.read_text(encoding="utf-8"))
    transcript: Any = json.loads(transcript_path.read_text(encoding="utf-8"))
    provenance: Any = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict) or fixture.get("schema") != "clipper-heldout-headlines-v1":
        raise ValueError("unknown held-out headline fixture schema")
    if fixture.get("annotation_status") != "transcript_only_pending_audio_review":
        raise ValueError("held-out labels need an explicit provisional annotation status")
    video_id, source_sha, transcript_sha = (
        fixture.get("source_video_id"),
        fixture.get("source_sha256"),
        fixture.get("transcript_sha256"),
    )
    if (
        not isinstance(video_id, str)
        or not _VIDEO_ID.fullmatch(video_id)
        or not isinstance(source_sha, str)
        or not _SHA256.fullmatch(source_sha)
        or not isinstance(transcript_sha, str)
        or not _SHA256.fullmatch(transcript_sha)
        or not isinstance(fixture.get("source_artifact_run_id"), str)
        or not fixture["source_artifact_run_id"].isdecimal()
    ):
        raise ValueError("held-out source identity is incomplete")
    actual_transcript_sha = hashlib.sha256(
        json.dumps(transcript, sort_keys=True).encode()
    ).hexdigest()
    identity = provenance.get("identity") if isinstance(provenance, dict) else None
    if (
        not isinstance(identity, dict)
        or identity.get("source_sha256") != source_sha
        or identity.get("transcript_sha256") != transcript_sha
        or actual_transcript_sha != transcript_sha
    ):
        raise ValueError("held-out source/transcript provenance mismatch")
    if not isinstance(transcript, list) or not transcript:
        raise ValueError("held-out benchmark needs the full transcript")
    for segment in transcript:
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("text"), str)
            or not segment["text"].strip()
            or type(segment.get("start")) not in (float, int)
            or type(segment.get("end")) not in (float, int)
            or not math.isfinite(segment["start"])
            or not math.isfinite(segment["end"])
            or segment["start"] < 0
            or segment["end"] <= segment["start"]
        ):
            raise ValueError("held-out transcript contains an invalid segment")
    raw_cases = fixture.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("held-out benchmark has no claims")
    cases: list[HeldoutClaim] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "source_start",
            "source_end",
            "source_clip",
            "headline",
            "expected_supported",
            "annotation_reason",
        }:
            raise ValueError("held-out claim has missing or unknown fields")
        case_id, headline, reason, clip = (
            raw["id"],
            raw["headline"],
            raw["annotation_reason"],
            raw["source_clip"],
        )
        start, end = raw["source_start"], raw["source_end"]
        if (
            not isinstance(case_id, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", case_id)
            or case_id in seen
            or not isinstance(headline, str)
            or not headline.strip()
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(clip, str)
            or not re.fullmatch(rf"\d{{2}}-double-coverage-{re.escape(video_id)}\.mp4", clip)
            or type(raw["expected_supported"]) is not bool
            or type(start) not in (float, int)
            or type(end) not in (float, int)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end
        ):
            raise ValueError("held-out claim has an invalid annotation or source range")
        units = tuple(
            segment["text"]
            for segment in transcript
            if segment["start"] >= start - 0.01 and segment["end"] <= end + 0.01
        )
        if not units:
            raise ValueError(f"held-out claim {case_id} has no exact transcript window")
        cases.append(
            HeldoutClaim(
                case_id=case_id,
                headline=headline,
                expected_supported=raw["expected_supported"],
                annotation_reason=reason,
                annotation_status=fixture["annotation_status"],
                source_start=float(start),
                source_end=float(end),
                source_clip=clip,
                source_units=units,
            )
        )
        seen.add(case_id)
    return cases


def _blind_audio_review_payload(
    fixture_path: Path,
    transcript_path: Path,
    provenance_path: Path,
    clips_dir: Path,
) -> dict[str, Any]:
    """Reconstruct a blind review from pinned transcript and actual MP4 bytes.

    Provisional transcript labels and reasons must never enter this manifest.
    """
    cases = load_heldout_claims(fixture_path, transcript_path, provenance_path)
    fixture: Any = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected: Any = fixture.get("source_clip_sha256")
    names = {case.source_clip for case in cases}
    if (
        not isinstance(expected, dict)
        or set(expected) != names
        or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in expected.values()
        )
    ):
        raise ValueError("blind audio review requires exact pinned hashes for every source clip")
    clips: dict[str, dict[str, str]] = {}
    for name in sorted(names):
        target = clips_dir / name
        if not target.is_file():
            raise ValueError(f"blind audio review source clip is missing: {name}")
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[name]:
            raise ValueError(f"blind audio review source clip hash differs: {name}")
        clips[name] = {"path": str(target.resolve()), "sha256": digest}
    ordered = sorted(
        cases,
        key=lambda case: hashlib.sha256(
            (fixture["source_sha256"] + ":" + case.case_id).encode()
        ).hexdigest(),
    )
    return {
        "schema": "clipper-blind-audio-review-v1",
        "annotation_status": "pending_independent_audio_review",
        "source_video_id": fixture["source_video_id"],
        "source_sha256": fixture["source_sha256"],
        "transcript_sha256": fixture["transcript_sha256"],
        "source_artifact_run_id": fixture["source_artifact_run_id"],
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "instructions": (
            "Listen to the complete linked MP4 before deciding whether every factual "
            "part of the headline is established by audible speech. Distinguish actors, "
            "roles, reported or quoted speech, actual versus hypothetical outcomes, "
            "conditions, negation and setting. Do not consult transcript-derived labels, "
            "model scores or case IDs. Mark uncertain if the audio is insufficient. "
            "This is factual review, not publication or creative-quality approval."
        ),
        "cases": [
            {
                "blind_id": f"A{index:03d}",
                "headline": case.headline,
                "source_clip": case.source_clip,
                "source_clip_path": clips[case.source_clip]["path"],
                "source_clip_sha256": clips[case.source_clip]["sha256"],
                "source_start": case.source_start,
                "source_end": case.source_end,
                "audible_claim_verdict": None,
                "reviewer_id": None,
                "review_notes": None,
            }
            for index, case in enumerate(ordered, start=1)
        ],
    }


def prepare_blind_audio_review(
    fixture_path: Path,
    transcript_path: Path,
    provenance_path: Path,
    clips_dir: Path,
    output_path: Path,
) -> Path:
    """Write the unlabelled audio manifest, not a review result or qualification."""
    manifest = _blind_audio_review_payload(
        fixture_path, transcript_path, provenance_path, clips_dir
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def assess_completed_audio_review(
    submitted_path: Path,
    fixture_path: Path,
    transcript_path: Path,
    provenance_path: Path,
    clips_dir: Path,
) -> dict[str, Any]:
    """Compare a complete audio attestation to provisional transcript labels.

    A self-reported reviewer name cannot authenticate a human or promote labels
    to gold. This check only proves the submitted fields are complete and the
    immutable MP4/source/headline payload has not changed.
    """
    canonical = _blind_audio_review_payload(
        fixture_path, transcript_path, provenance_path, clips_dir
    )
    submitted: Any = json.loads(submitted_path.read_text(encoding="utf-8"))
    if not isinstance(submitted, dict) or set(submitted) != set(canonical):
        raise ValueError("submitted audio review has missing or unknown fields")
    immutable = set(canonical) - {"cases"}
    if any(submitted[key] != canonical[key] for key in immutable):
        raise ValueError("submitted audio review changed pinned source or instructions")
    rows = submitted["cases"]
    expected_rows = canonical["cases"]
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise ValueError("submitted audio review omitted or added cases")
    editable = {"audible_claim_verdict", "reviewer_id", "review_notes"}
    reviewer_ids: set[str] = set()
    for row, original in zip(rows, expected_rows, strict=True):
        if not isinstance(row, dict) or set(row) != set(original):
            raise ValueError("submitted audio review changed case fields")
        if any(row[key] != original[key] for key in set(original) - editable):
            raise ValueError("submitted audio review changed pinned clip or headline")
        if not isinstance(row["audible_claim_verdict"], str) or row[
            "audible_claim_verdict"
        ] not in {"supported", "unsupported", "uncertain", "inaudible"}:
            raise ValueError("submitted audio review has a missing or invalid verdict")
        for key in ("reviewer_id", "review_notes"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("submitted audio review needs reviewer identity and notes")
        reviewer_ids.add(row["reviewer_id"].strip())
    if len(reviewer_ids) != 1:
        raise ValueError("submitted audio review requires one consistent reviewer identity")
    claims = sorted(
        load_heldout_claims(fixture_path, transcript_path, provenance_path),
        key=lambda case: hashlib.sha256(
            (canonical["source_sha256"] + ":" + case.case_id).encode()
        ).hexdigest(),
    )
    comparisons = [
        {
            "blind_id": row["blind_id"],
            "audible_claim_verdict": row["audible_claim_verdict"],
            "provisional_transcript_label": claim.expected_supported,
            "matches_provisional_label": (
                row["audible_claim_verdict"] == "supported"
                if claim.expected_supported
                else row["audible_claim_verdict"] == "unsupported"
            ),
        }
        for row, claim in zip(rows, claims, strict=True)
    ]
    return {
        "schema": "clipper-audio-review-assessment-v1",
        "submitted_review_sha256": hashlib.sha256(submitted_path.read_bytes()).hexdigest(),
        "fixture_sha256": canonical["fixture_sha256"],
        "source_sha256": canonical["source_sha256"],
        "reviewer_id_self_reported": next(iter(reviewer_ids)),
        "reviewer_identity_verified": False,
        "audio_gold_qualified": False,
        "production_approved": False,
        "total": len(comparisons),
        "provisional_label_disagreements": sum(
            not row["matches_provisional_label"] for row in comparisons
        ),
        "comparisons": comparisons,
    }
