"""Pure, model-free contracts for source-driven hybrid chapter editorial evidence.

No model, tokenizer, host, or file I/O is invoked by this module. The caller
must persist the immutable source ledger and bind request and producer IDs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from . import constrained, scripting

LEDGER_SCHEMA = "clipper-chapter-hybrid-source-ledger-v1"
ARGUMENT_SCHEMA = "clipper-chapter-hybrid-argument-v1"
AUDIT_SCHEMA = "clipper-chapter-hybrid-local-audit-v1"
BRIEF_SCHEMA = "clipper-chapter-hybrid-brief-v1"
REVIEW_SCHEMA = "clipper-chapter-hybrid-human-review-v1"
MAX_ENTRIES = 128
MAX_SOURCE_BYTES = 512_000
MAX_AUDITS = 16
MAX_CORE_UNITS = 12
MAX_BEATS = 128
MAX_SPANS_PER_BEAT = 24
MAX_AUDIT_RAW_BYTES = 100_000
SAFE_CONTEXT_TOKENS = 32_768
MAX_REQUEST_BYTES = 280_000
SPAN_ROLES = frozenset(
    {"ASSERTION", "AUTHOR_EXAMPLE", "COUNTEREXAMPLE", "QUALIFICATION"}
)
DECISIONS = frozenset({"keep", "compress", "omit"})


class HybridStop(ValueError):
    """A sanitized, fail-closed source or model-contract error."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _object(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise HybridStop("hybrid response has unexpected fields")
    return value


def _text(value: object, limit: int = 2_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise HybridStop("hybrid response has missing or oversized editorial text")
    return value


def _list(value: object, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise HybridStop("hybrid response exceeds bounded list contract")
    return value


def freeze(
    units: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    reading_order: list[str],
    identity: dict[str, object],
) -> dict[str, Any]:
    """Keep canonical bytes/order and reversible IDs outside model requests."""
    sources = [*units, *gaps]
    if not 1 <= len(sources) <= MAX_ENTRIES or len(sources) != len(reading_order):
        raise HybridStop("selected chapter work/gap cardinality exceeds finite cap")
    by_id: dict[str, dict[str, Any]] = {}
    for item in sources:
        sid = item.get("source_id")
        if not isinstance(sid, str) or not sid or sid in by_id:
            raise HybridStop("selected chapter original source IDs are not unique")
        if not isinstance(item.get("text"), str):
            raise HybridStop("selected chapter original text is malformed")
        if item.get("kind") == "CANONICAL_WORK_UNIT":
            if (
                hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                != item.get("text_sha256")
            ):
                raise HybridStop("selected chapter original source text digest changed")
        elif (
            item.get("kind") != "UNRECOVERED_OR_EMPTY_SOURCE_MARKER"
            or item["text"]
        ):
            raise HybridStop("selected chapter gap has unsupported source content")
        by_id[sid] = item
    if set(reading_order) != set(by_id) or len(set(reading_order)) != len(reading_order):
        raise HybridStop("selected chapter source traversal is incomplete or reordered")
    entries: list[dict[str, Any]] = []
    for ordinal, sid in enumerate(reading_order, 1):
        unit = by_id[sid]
        short_id = f"s{ordinal:04d}"
        entries.append(
            {
                "id": short_id,
                "source_id": sid,
                "order": ordinal,
                "kind": unit["kind"],
                "origin": unit.get("origin"),
                "review_status": unit.get("review_status"),
                "research_link_status": unit.get("research_link_status"),
                "anchor": unit.get("anchor"),
                "section_id": unit.get("section_id"),
                "source_path": unit.get("source_path"),
                "unit_fingerprint": unit.get("unit_fingerprint"),
                "original_parent_ordinal": unit.get("original_parent_ordinal"),
                "text_sha256": hashlib.sha256(unit["text"].encode("utf-8")).hexdigest(),
                "text": unit["text"],
            }
        )
    ledger = {"schema": LEDGER_SCHEMA, "identity": identity, "entries": entries}
    if len(_canonical(ledger)) > MAX_SOURCE_BYTES:
        raise HybridStop("selected chapter immutable ledger exceeds finite byte cap")
    ledger["ledger_sha256"] = digest(ledger)
    return ledger


def validate_ledger(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Recheck immutable source, map and order on every replay."""
    checked = _object(ledger, {"schema", "identity", "entries", "ledger_sha256"})
    if checked["schema"] != LEDGER_SCHEMA:
        raise HybridStop("hybrid source ledger contract mismatch")
    entries = _list(checked["entries"], MAX_ENTRIES)
    if not entries:
        raise HybridStop("empty hybrid source ledger")
    if not isinstance(checked["identity"], dict):
        raise HybridStop("hybrid source identity malformed")
    if digest({k: v for k, v in checked.items() if k != "ledger_sha256"}) != checked[
        "ledger_sha256"
    ]:
        raise HybridStop("hybrid source ledger digest changed")
    seen: set[str] = set()
    for index, entry in enumerate(entries, 1):
        if (
            not isinstance(entry, dict)
            or entry.get("id") != f"s{index:04d}"
            or not isinstance(entry.get("source_id"), str)
            or entry["source_id"] in seen
            or entry.get("order") != index
            or not isinstance(entry.get("text"), str)
            or hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
            != entry.get("text_sha256")
        ):
            raise HybridStop("hybrid source map, order, or text has changed")
        seen.add(entry["source_id"])
    return entries


def projection(entries: list[dict[str, Any]]) -> list[list[object]]:
    """Each exact original text occurs once in this deterministic request projection."""
    return [
        [
            entry["id"],
            "gap" if entry["kind"] != "CANONICAL_WORK_UNIT" else "text",
            entry["origin"],
            entry["review_status"],
            entry["research_link_status"],
            entry["text"],
        ]
        for entry in entries
    ]


def request(
    stage: str,
    payload: dict[str, object],
    *,
    output_tokens: int,
    token_count: Callable[[list[dict[str, str]]], int],
    producer_identity: dict[str, object],
) -> tuple[dict[str, object], str, int]:
    """Exact pinned-template preflight supplied by caller, never a byte proxy."""
    if (
        stage not in {"argument", "audit", "reconcile"}
        or type(output_tokens) is not int
        or not 1024 <= output_tokens <= 8192
        or not producer_identity
    ):
        raise HybridStop("hybrid stage capacity or producer identity is invalid")
    message = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    messages = [
        {
            "role": "system",
            "content": (
                "You are preparing source-driven chapter evidence, not extracting lessons. "
                "An argument draft is revisable. Preserve omissions, contradictions, "
                "counterexamples, causal links, provenance and author qualifications. "
                "Return one strict JSON object; never invent original-source quotes. "
                f"Stage: {stage}. No conclusions are approved without human review."
            ),
        },
        {"role": "user", "content": message},
    ]
    if len(_canonical(messages)) > MAX_REQUEST_BYTES:
        raise HybridStop("hybrid complete request exceeds finite byte cap")
    try:
        count = token_count(messages)
    except Exception as exc:
        raise HybridStop("exact pinned tokenizer preflight unavailable") from exc
    if (
        type(count) is not int
        or count < 1
        or count + output_tokens > SAFE_CONTEXT_TOKENS
    ):
        raise HybridStop("hybrid exact prompt and reserved output exceed safe context")
    result: dict[str, object] = {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": output_tokens,
        "messages": messages,
    }
    key = digest(
        {
            "contract": f"hybrid-{stage}-v1",
            "request": result,
            "tokenizer_sha256": constrained.PINNED_TOKENIZER_SHA256,
            "template": "pinned-qwen-chat-enable-thinking-false-v1",
            "safe_context": SAFE_CONTEXT_TOKENS,
            "producer_identity": producer_identity,
            "generation_settings": scripting.QWEN_GENERATION_SETTINGS,
        }
    )
    return result, key, count


def audit_windows(ledger: dict[str, Any]) -> list[dict[str, object]]:
    """Disjoint contiguous core, single-neighbor halo, finite output sizing."""
    entries = validate_ledger(ledger)
    windows: list[dict[str, object]] = []
    for offset in range(0, len(entries), MAX_CORE_UNITS):
        core = entries[offset : offset + MAX_CORE_UNITS]
        halo = entries[max(0, offset - 1) : offset] + entries[
            offset + len(core) : offset + len(core) + 1
        ]
        windows.append(
            {
                "audit_id": f"a{len(windows) + 1:03d}",
                "core_ids": [entry["id"] for entry in core],
                "core": projection(core),
                "halo_context_only": projection(halo),
            }
        )
    if len(windows) > MAX_AUDITS:
        raise HybridStop("hybrid audit count exceeds finite budget")
    return windows


def _span(
    raw: object, entries: dict[str, dict[str, Any]], *, core: set[str] | None = None
) -> dict[str, Any]:
    item = _object(raw, {"id", "start", "end", "role"})
    sid = item["id"]
    if sid not in entries or (core is not None and sid not in core):
        raise HybridStop("hybrid source span outside allowed original units")
    original = entries[sid]
    start, end = item["start"], item["end"]
    if (
        original["kind"] != "CANONICAL_WORK_UNIT"
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(original["text"])
        or end - start > 700
        or item["role"] not in SPAN_ROLES
    ):
        raise HybridStop("hybrid source character span or role is invalid")
    return {
        **item,
        "quote": original["text"][start:end],
        "original_source_id": original["source_id"],
    }


def validate_argument(raw: object, ledger: dict[str, Any]) -> dict[str, Any]:
    data = _object(raw, {"hypothesis", "questions", "source_links"})
    _text(data["hypothesis"], 3000)
    _list(data["questions"], 24)
    for question in data["questions"]:
        _text(question, 500)
    source = {entry["id"]: entry for entry in validate_ledger(ledger)}
    links = _list(data["source_links"], MAX_ENTRIES)
    if not links:
        raise HybridStop("revisable argument has no source-linked hypothesis")
    for link in links:
        _span(link, source)
    return data


def validate_audit(
    raw: object, ledger: dict[str, Any], window: dict[str, object]
) -> dict[str, Any]:
    source = {entry["id"]: entry for entry in validate_ledger(ledger)}
    core = window["core_ids"]
    assert isinstance(core, list)
    expected = set(core)
    data = _object(raw, {"audit_id", "core_ids", "decisions", "disagreements"})
    if data["audit_id"] != window["audit_id"] or data["core_ids"] != core:
        raise HybridStop("hybrid audit identity or core order changed")
    decisions = _list(data["decisions"], len(core))
    seen: set[str] = set()
    for value in decisions:
        item = _object(value, {"id", "decision", "reason", "evidence"})
        sid = item["id"]
        if sid not in expected or sid in seen or item["decision"] not in DECISIONS:
            raise HybridStop("hybrid audit source decision missing or duplicated")
        seen.add(sid)
        _text(item["reason"], 800)
        if source[sid]["kind"] != "CANONICAL_WORK_UNIT" and item["decision"] != "omit":
            raise HybridStop("hybrid unresolved original gap cannot be retained")
        evidence = _list(item["evidence"], 12)
        for span in evidence:
            _span(span, source, core=expected)
    if seen != expected:
        raise HybridStop("hybrid audit silently omitted original units")
    disagreements = _list(data["disagreements"], 24)
    for index, value in enumerate(disagreements, 1):
        item = _object(value, {"id", "issue", "evidence"})
        if item["id"] != f"{window['audit_id']}-d{index:02d}":
            raise HybridStop("hybrid disagreement identity changed")
        _text(item["issue"], 800)
        if not _list(item["evidence"], 12):
            raise HybridStop("hybrid disagreement has no source support")
        for span in item["evidence"]:
            _span(span, source, core=expected)
    return data


def reconcile(
    raw: object,
    ledger: dict[str, Any],
    argument: dict[str, Any],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use only actual model decisions and explicit grounded amendments."""
    entries = validate_ledger(ledger)
    source = {entry["id"]: entry for entry in entries}
    if not isinstance(argument, dict):
        raise HybridStop("hybrid initial argument is absent")
    windows = audit_windows(ledger)
    if len(audits) != len(windows):
        raise HybridStop("hybrid audits are incomplete")
    original: dict[str, dict[str, Any]] = {}
    disputes: set[str] = set()
    for audit, window in zip(audits, windows, strict=True):
        checked = validate_audit(audit, ledger, window)
        for decision in checked["decisions"]:
            sid = decision["id"]
            if sid in original:
                raise HybridStop("hybrid overlapping local audit coverage")
            original[sid] = decision
        disputes.update(disagreement["id"] for disagreement in checked["disagreements"])
    if set(original) != set(source):
        raise HybridStop("hybrid original source decisions incomplete")
    data = _object(raw, {"argument", "amendments", "resolutions", "beats"})
    _text(data["argument"], 3500)
    decisions = {sid: dict(value) for sid, value in original.items()}
    edited: set[str] = set()
    for value in _list(data["amendments"], MAX_ENTRIES):
        amendment = _object(
            value, {"id", "from", "to", "reason", "evidence"}
        )
        sid = amendment["id"]
        if (
            sid not in decisions
            or sid in edited
            or amendment["from"] != decisions[sid]["decision"]
            or amendment["to"] not in DECISIONS
            or amendment["to"] == amendment["from"]
        ):
            raise HybridStop("hybrid amendment not an explicit grounded change")
        if source[sid]["kind"] != "CANONICAL_WORK_UNIT" and amendment["to"] != "omit":
            raise HybridStop("hybrid amendment cannot reintroduce unresolved source")
        edited.add(sid)
        _text(amendment["reason"], 800)
        if not _list(amendment["evidence"], 12):
            raise HybridStop("hybrid amendment has no exact supporting spans")
        for span in amendment["evidence"]:
            _span(span, source)
        decisions[sid] = {
            **decisions[sid],
            "decision": amendment["to"],
            "reason": amendment["reason"],
            "evidence": amendment["evidence"],
        }
    resolved: set[str] = set()
    for value in _list(data["resolutions"], MAX_AUDITS * 24):
        item = _object(value, {"id", "resolution", "reason", "evidence"})
        if item["id"] not in disputes or item["id"] in resolved:
            raise HybridStop("hybrid duplicate or unknown audit disagreement")
        resolved.add(item["id"])
        if item["resolution"] not in {"accepted", "rejected", "qualified"}:
            raise HybridStop("unresolved hybrid argument disagreement")
        _text(item["reason"], 800)
        if not _list(item["evidence"], 12):
            raise HybridStop("hybrid disagreement was resolved without original evidence")
        for span in item["evidence"]:
            _span(span, source)
    if resolved != disputes:
        raise HybridStop("hybrid global reconciliation left audit disagreements unresolved")
    beats: list[dict[str, Any]] = []
    linked: set[str] = set()
    for index, value in enumerate(_list(data["beats"], MAX_BEATS), 1):
        beat = _object(
            value, {"beat_id", "spans", "meaning", "qualification", "causal_link", "essential"}
        )
        if beat["beat_id"] != f"beat-{index:03d}" or type(beat["essential"]) is not bool:
            raise HybridStop("hybrid beat identity or essential flag invalid")
        for key in ("meaning", "causal_link"):
            _text(beat[key], 1000)
        if not isinstance(beat["qualification"], str) or len(beat["qualification"]) > 1000:
            raise HybridStop("hybrid beat qualification malformed")
        spans = _list(beat["spans"], MAX_SPANS_PER_BEAT)
        if not spans:
            raise HybridStop("hybrid beat cannot lack original source evidence")
        expanded = []
        for item in spans:
            span = _span(item, source)
            if decisions[span["id"]]["decision"] == "omit":
                raise HybridStop("hybrid beat cites editorially omitted source")
            linked.add(span["id"])
            expanded.append(span)
        beats.append({**beat, "spans": expanded, "unit_id": expanded[0]["original_source_id"]})
    if len(beats) < 2:
        raise HybridStop("hybrid coherent chapter requires multiple source-linked beats")
    for sid, decision in decisions.items():
        if (decision["decision"] != "omit") != (sid in linked):
            raise HybridStop("hybrid final beat grouping contradicts complete source coverage")
    brief: dict[str, Any] = {
        "schema": BRIEF_SCHEMA,
        "ledger_sha256": ledger["ledger_sha256"],
        "initial_argument_sha256": digest(argument),
        "audit_sha256": [digest(audit) for audit in audits],
        "reconciliation_sha256": digest(data),
        "argument": data["argument"],
        "beats": beats,
        "coverage": [
            {
                **decisions[entry["id"]],
                "source_id": entry["source_id"],
                "short_id": entry["id"],
            }
            for entry in entries
        ],
        "resolutions": data["resolutions"],
        "publish_eligible": False,
    }
    brief["brief_sha256"] = digest(brief)
    return brief


def human_review_packet(brief: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Full original text and immediate neighbors, not a lexical fidelity certificate."""
    entries = validate_ledger(ledger)
    if brief.get("ledger_sha256") != ledger["ledger_sha256"]:
        raise HybridStop("hybrid brief is not bound to the immutable source ledger")
    return {
        "schema": REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "argument": brief["argument"],
        "beats": brief["beats"],
        "resolutions": brief["resolutions"],
        "units": [
            {
                "id": entry["id"],
                "original_source_id": entry["source_id"],
                "original_text": entry["text"],
                "previous_text": entries[index - 1]["text"] if index else "",
                "next_text": entries[index + 1]["text"]
                if index + 1 < len(entries)
                else "",
                "decision": brief["coverage"][index],
            }
            for index, entry in enumerate(entries)
        ],
        "lexical_flags_not_semantic_proof": True,
        "review_status": "PENDING_ACTUAL_HUMAN_ATTESTATION",
        "publish_eligible": False,
    }


def validate_human_review(
    raw: object, brief: dict[str, Any], ledger: dict[str, Any]
) -> dict[str, Any]:
    """Mandatory explicit per-beat AND per-source semantic approval."""
    entries = validate_ledger(ledger)
    doc = _object(
        raw, {"schema", "brief_sha256", "reviewer", "unit_decisions", "beat_decisions"}
    )
    if doc["schema"] != REVIEW_SCHEMA or doc["brief_sha256"] != brief["brief_sha256"]:
        raise HybridStop("hybrid human approval was not for the exact coherent brief")
    _text(doc["reviewer"], 100)
    units = _list(doc["unit_decisions"], len(entries))
    seen: set[str] = set()
    for value in units:
        item = _object(value, {"id", "review", "reason"})
        if item["id"] not in {entry["id"] for entry in entries} or item["id"] in seen:
            raise HybridStop("hybrid human review omitted original source units")
        if item["review"] != "accept":
            raise HybridStop("hybrid original source decision needs human revision")
        _text(item["reason"], 1000)
        seen.add(item["id"])
    if seen != {entry["id"] for entry in entries}:
        raise HybridStop("hybrid original source decisions not individually reviewed")
    beat_ids = {beat["beat_id"] for beat in brief["beats"]}
    decided: set[str] = set()
    for value in _list(doc["beat_decisions"], len(beat_ids)):
        item = _object(value, {"beat_id", "review", "reason"})
        if item["beat_id"] not in beat_ids or item["beat_id"] in decided:
            raise HybridStop("hybrid human beat decisions incomplete")
        if item["review"] != "support_attested":
            raise HybridStop("hybrid semantic beat review rejected source interpretation")
        _text(item["reason"], 1000)
        decided.add(item["beat_id"])
    if decided != beat_ids:
        raise HybridStop("hybrid meaning and qualifications have not been human attested")
    return doc
