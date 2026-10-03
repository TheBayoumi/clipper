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
