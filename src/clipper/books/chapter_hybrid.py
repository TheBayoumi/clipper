"""Pure, model-free contracts for source-driven hybrid chapter editorial evidence.

No model, tokenizer, host, or file I/O is invoked by this module. The caller
must persist the immutable source ledger and bind request and producer IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from . import constrained, model_profile, scripting

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
CHAT_TEMPLATE_CONTRACT = "pinned-qwen-chat-enable-thinking-false-v1"
MAX_REQUEST_BYTES = 280_000
SPAN_ROLES = frozenset({"ASSERTION", "AUTHOR_EXAMPLE", "COUNTEREXAMPLE", "QUALIFICATION"})
DECISIONS = frozenset({"keep", "compress", "omit"})


class HybridStop(ValueError):
    """A sanitized, fail-closed source or model-contract error.

    ``diagnostics`` optionally lists source-free per-citation failures
    (where, index, id or None, code) for bounded retry feedback. It never
    carries model text and is not part of the error message.
    """

    def __init__(self, message: str, *, diagnostics: list[dict[str, object]] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


MAX_DIAGNOSTICS = 24
MIN_SPAN_CHARS = 3
MIN_KEPT_SHARE = 0.05
FIRST_ATTEMPT_SEED = 42
RETRY_ATTEMPT_SEED = 43
ALLOWED_SEEDS = (FIRST_ATTEMPT_SEED, RETRY_ATTEMPT_SEED)
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_:.-]{1,40}")
_CODES = frozenset(
    {
        "BAD_FORM",
        "BAD_TYPE",
        "UNKNOWN_ID",
        "GAP_UNIT",
        "BAD_ROLE",
        "QUOTE_INVALID",
        "QUOTE_ABSENT",
        "QUOTE_AMBIGUOUS",
        "SPAN_TOO_LONG",
        "BAD_OFFSETS",
        "DEGENERATE_LINKS",
    }
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


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
            if hashlib.sha256(item["text"].encode("utf-8")).hexdigest() != item.get("text_sha256"):
                raise HybridStop("selected chapter original source text digest changed")
        elif item.get("kind") != "UNRECOVERED_OR_EMPTY_SOURCE_MARKER" or item["text"]:
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
                "unit_word_offset": unit.get("unit_word_offset"),
                "evidence_field": unit.get("evidence_field"),
                "asset_sha256": unit.get("asset_sha256"),
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
    if len(_canonical(checked)) > MAX_SOURCE_BYTES + 256:
        raise HybridStop("hybrid source ledger replay exceeds finite byte cap")
    if (
        digest({k: v for k, v in checked.items() if k != "ledger_sha256"})
        != checked["ledger_sha256"]
    ):
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
            or hashlib.sha256(entry["text"].encode("utf-8")).hexdigest() != entry.get("text_sha256")
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


def producer_identity(
    *,
    ledger_sha256: str,
    chapter_scope_sha256: str,
    selected_bundle_sha256: str,
    plan_fingerprint: str,
    model_producer: dict[str, object],
) -> dict[str, object]:
    """Bind each private chapter attempt to exact producer/runtime material."""
    identifiers = (
        ledger_sha256,
        chapter_scope_sha256,
        selected_bundle_sha256,
        plan_fingerprint,
    )
    if (
        any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in identifiers
        )
        or not isinstance(model_producer, dict)
        or not model_producer
    ):
        raise HybridStop("hybrid source/producer identity is malformed")
    return {
        "source_ledger_sha256": ledger_sha256,
        "chapter_scope_sha256": chapter_scope_sha256,
        "selected_source_bundle_sha256": selected_bundle_sha256,
        "chapter_plan_fingerprint": plan_fingerprint,
        "model_producer": model_producer,
        "source_mode": "DIRECT_SOURCE_HYBRID",
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "tokenizer_sha256": constrained.PINNED_TOKENIZER_SHA256,
        "chat_template_contract": CHAT_TEMPLATE_CONTRACT,
        "generation_settings": scripting.QWEN_GENERATION_SETTINGS,
        "thinking_enabled": False,
        "combined_context_limit": model_profile.active().safe_context_tokens,
        "stage_schemas": [ARGUMENT_SCHEMA, AUDIT_SCHEMA, BRIEF_SCHEMA],
    }


_STAGE_GUIDANCE = {
    "argument": (
        "Each source_links item is one object with exactly id and role, plus an optional "
        "quote. id is an id taken from original_source and cites that whole unit. Do not "
        "write character offsets. Add quote only to cite part of a unit, and then it must "
        "be copied exactly, once, from that unit's text. role is exactly one of ASSERTION, "
        "AUTHOR_EXAMPLE, COUNTEREXAMPLE or QUALIFICATION, chosen per unit from what that "
        "unit does in the argument. Link counterexamples and author qualifications "
        "whenever the source contains them, not only assertions."
    ),
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_INPUT_MARK = "INPUT (JSON, read-only source material):\n"
_TASK_MARK = "\n\nTASK:"


def input_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    """The input object of a hybrid request (the response contract is stated in prose)."""
    content = messages[1]["content"]
    start = content.index(_INPUT_MARK) + len(_INPUT_MARK)
    value = json.loads(content[start : content.index(_TASK_MARK, start)])
    if not isinstance(value, dict):
        raise HybridStop("hybrid request input is malformed")
    return value


def _messages(stage: str, payload: dict[str, object]) -> list[dict[str, str]]:
    """Source first, then a plain-language task with the answer shape read last.

    The shape is not a JSON key of the input, so a small model has no slot to fill in
    except its own answer. Top-level key names are stated in prose.
    """
    data = dict(payload)
    contract = data.pop("response_contract", None)
    feedback = data.pop("retry_feedback", None)
    system = (
        "You are preparing source-driven chapter evidence, not extracting lessons. "
        "An argument draft is revisable. Preserve omissions, contradictions, "
        "counterexamples, causal links, provenance and author qualifications. "
        "Return one strict JSON object; never invent original-source quotes. "
        f"This request is for the {stage} step. "
        "No conclusions are approved without human review."
    )
    if isinstance(contract, dict):
        keys = ", ".join(sorted(contract))
        shape = (
            f"Reply with ONE JSON object whose top-level keys are exactly: {keys}. "
            "Put your answer directly at the top level. Do not wrap it in another "
            "object and do not repeat the input or these instructions. Shape of each "
            "key (angle-bracket text is a description to replace, never to copy): "
            f"{_json(contract)}"
        )
    else:
        shape = (
            "Reply with ONE JSON object and nothing else. Do not repeat the input or "
            f"these instructions. Required content: {contract}"
        )
    guidance = _STAGE_GUIDANCE.get(stage, "")
    if feedback is not None:
        guidance += (
            "\nYour previous answer was rejected by the source check. These citations were "
            "invalid (where, index, id, code; no text is shown): "
            f"{_json(feedback)}. Write the whole answer again and cite only ids that appear "
            "in the input above."
        )
    user = (
        f"{_INPUT_MARK}{_json(data)}{_TASK_MARK} write the {stage} answer for the input above.\n"
        f"{shape}\n{guidance}\nNow write the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_LOCATION = re.compile(
    r"source_links|decision:[A-Za-z0-9_.-]{1,32}|disagreement:[0-9]{1,3}|"
    r"amendment:[A-Za-z0-9_.-]{1,32}|resolution:[A-Za-z0-9_.-]{1,32}|beat:beat-[0-9]{3}"
)


def retry_feedback(error: BaseException) -> dict[str, object] | None:
    """Bounded, source-free feedback for one retry, or None when there is none.

    Only validator diagnostics are used (where, index, a ledger id that exists in the
    request, a fixed code), and only from a HybridStop. Anything else is refused.
    No model output or source text is ever forwarded.
    """
    if not isinstance(error, HybridStop):
        return None
    diagnostics = error.diagnostics
    if not isinstance(diagnostics, list) or not diagnostics:
        return None
    links: list[dict[str, object]] = []
    for item in diagnostics[:MAX_DIAGNOSTICS]:
        if not isinstance(item, dict) or set(item) != {"where", "index", "id", "code"}:
            return None
        where, index, sid, code = item["where"], item["index"], item["id"], item["code"]
        if (
            not isinstance(code, str)
            or code not in _CODES
            or type(index) is not int
            or not 0 <= index < MAX_ENTRIES
            or not isinstance(where, str)
            or _LOCATION.fullmatch(where) is None
            or not (sid is None or (isinstance(sid, str) and _SAFE_LABEL.fullmatch(sid)))
        ):
            return None
        links.append({"where": where, "index": index, "id": sid, "code": code})
    return {"reason": "INVALID_CITATIONS", "invalid_links": links}


def request(
    stage: str,
    payload: dict[str, object],
    *,
    output_tokens: int,
    token_count: Callable[[list[dict[str, str]]], int | dict[str, object]],
    producer_identity: dict[str, object],
    auto_output: bool = False,
    seed: int = FIRST_ATTEMPT_SEED,
) -> tuple[dict[str, object], str, int]:
    """Exact pinned-template preflight supplied by caller, never a byte proxy.

    With auto_output the reserved output is resolved here, from the measured prompt and
    before the request is hashed; output_tokens is then only the profile ceiling.
    """
    if (
        stage
        not in {
            "argument",
            "audit",
            "reconcile",
            "adaptation",
            "chapter_script",
            "chapter_direction",
        }
        or type(output_tokens) is not int
        or not model_profile.output_bounds()[0] <= output_tokens <= model_profile.output_bounds()[1]
        or not producer_identity
    ):
        raise HybridStop("hybrid stage capacity or producer identity is invalid")
    messages = _messages(stage, payload)
    if len(_canonical(messages)) > MAX_REQUEST_BYTES:
        raise HybridStop("hybrid complete request exceeds finite byte cap")
    try:
        counted = token_count(messages)
    except Exception as exc:
        raise HybridStop("exact pinned tokenizer preflight unavailable") from exc
    # Authentic CPU requests carry ALL rendered input token IDs' digest and
    # effective template/config provenance. A count-only callback is retained
    # solely for explicitly injected, model-free legacy test fixtures.
    encoding: dict[str, object] | None = None
    if isinstance(counted, dict):
        ids = counted.get("token_ids")
        if not isinstance(ids, list):
            raise HybridStop("hybrid effective token encoding provenance invalid")
        if (
            set(counted)
            != {
                "token_ids",
                "tokenizer_config_sha256",
                "chat_template_sha256",
                "tokenizer_json_sha256",
            }
            or not ids
            or len(ids) > model_profile.active().safe_context_tokens
            or any(type(value) is not int or value < 0 for value in ids)
            or counted["tokenizer_json_sha256"] != constrained.PINNED_TOKENIZER_SHA256
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in (
                    counted["tokenizer_config_sha256"],
                    counted["chat_template_sha256"],
                )
            )
        ):
            raise HybridStop("hybrid effective token encoding provenance invalid")
        count = len(ids)
        encoding = {
            "input_token_ids_sha256": digest(ids),
            "tokenizer_config_sha256": counted["tokenizer_config_sha256"],
            "chat_template_sha256": counted["chat_template_sha256"],
            "tokenizer_json_sha256": counted["tokenizer_json_sha256"],
        }
    else:
        count = counted
    if auto_output and type(count) is int and count >= 1:
        try:
            output_tokens = model_profile.active().auto_output_tokens(count)
        except model_profile.ModelProfileError as exc:
            raise HybridStop("hybrid prompt leaves too little room for the minimum output") from exc
    if seed not in ALLOWED_SEEDS or type(seed) is not int:
        raise HybridStop("hybrid request seed is not a pinned attempt seed")
    if (
        type(count) is not int
        or count < 1
        or count + output_tokens > model_profile.active().safe_context_tokens
    ):
        raise HybridStop("hybrid exact prompt and reserved output exceed safe context")
    result: dict[str, object] = {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": seed,
        "max_new_tokens": output_tokens,
        "messages": messages,
        **({"verified_input_encoding": encoding} if encoding is not None else {}),
    }
    key = digest(
        {
            "contract": f"hybrid-{stage}-v1",
            "request": result,
            "tokenizer_sha256": constrained.PINNED_TOKENIZER_SHA256,
            "template": CHAT_TEMPLATE_CONTRACT,
            "safe_context": model_profile.active().safe_context_tokens,
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
        halo = (
            entries[max(0, offset - 1) : offset]
            + entries[offset + len(core) : offset + len(core) + 1]
        )
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


def _resolve_reference(item: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    """Resolve an ID reference to exact offsets in code, never from model arithmetic.

    {"id", "role"} cites the whole unit. {"id", "role", "quote"} cites one exact,
    unambiguous substring of the unit. Offsets and the stored quote come from the
    immutable ledger text, so a model can only choose what to cite, not where it sits.
    """
    text = original["text"]
    if "quote" not in item:
        start, end = 0, len(text)
    else:
        quote = item["quote"]
        if not isinstance(quote, str) or not quote.strip():
            raise HybridStop(_QUOTE_MESSAGE, diagnostics=[{"code": "QUOTE_INVALID"}])
        start = text.find(quote)
        if start == -1:
            raise HybridStop(_QUOTE_MESSAGE, diagnostics=[{"code": "QUOTE_ABSENT"}])
        # find() from start + 1 also sees overlapping matches that count() misses.
        if text.find(quote, start + 1) != -1:
            raise HybridStop(_QUOTE_MESSAGE, diagnostics=[{"code": "QUOTE_AMBIGUOUS"}])
        end = start + len(quote)
    return {"id": item["id"], "start": start, "end": end, "role": item["role"]}


_QUOTE_MESSAGE = "hybrid source quote is absent, empty or ambiguous in its unit"


def _stop(message: str, code: str) -> HybridStop:
    return HybridStop(message, diagnostics=[{"code": code}])


def _span(
    raw: object, entries: dict[str, dict[str, Any]], *, core: set[str] | None = None
) -> dict[str, Any]:
    outside = "hybrid source span outside allowed original units"
    invalid = "hybrid source character span or role is invalid"
    if isinstance(raw, dict) and set(raw) in ({"id", "role"}, {"id", "role", "quote"}):
        form = "ref"
    elif isinstance(raw, dict) and set(raw) == {"id", "start", "end", "role"}:
        form = "offsets"
    else:
        # _object raises the established contract error for any other shape.
        try:
            _object(raw, {"id", "start", "end", "role"})
        except HybridStop as error:
            raise _stop(str(error), "BAD_FORM") from None
        raise _stop(invalid, "BAD_FORM")
    sid, role = raw["id"], raw["role"]
    # A list or dict here would be unhashable in the lookups below.
    if not isinstance(sid, str) or not isinstance(role, str):
        raise _stop(invalid, "BAD_TYPE")
    if sid not in entries or (core is not None and sid not in core):
        raise _stop(outside, "UNKNOWN_ID")
    original = entries[sid]
    if original["kind"] != "CANONICAL_WORK_UNIT":
        raise _stop(invalid, "GAP_UNIT")
    if role not in SPAN_ROLES:
        raise _stop(invalid, "BAD_ROLE")
    item = _resolve_reference(raw, original) if form == "ref" else dict(raw)
    start, end = item["start"], item["end"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(original["text"])
        or end - start < MIN_SPAN_CHARS
    ):
        raise _stop(invalid, "BAD_OFFSETS")
    # One bound for every citation form: whole-unit, quote and legacy offsets.
    if end - start > 700:
        raise _stop(invalid, "SPAN_TOO_LONG")
    return {
        **item,
        "quote": original["text"][start:end],
        "original_source_id": original["source_id"],
    }


def _canonical_span(
    raw: object, entries: dict[str, dict[str, Any]], *, core: set[str] | None = None
) -> dict[str, Any]:
    """Validate a cited span and return it as the stored id/start/end/role form."""
    span = _span(raw, entries, core=core)
    return {key: span[key] for key in ("id", "start", "end", "role")}


def _canonical_spans(
    spans: list[Any],
    entries: dict[str, dict[str, Any]],
    where: str,
    *,
    core: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Canonicalize a citation list; collect every failure as source-free diagnostics.

    The raised error keeps the first failure's message, so its rejection digest is
    unchanged. Repeated identical id-reference citations are kept once, first-seen order.
    """
    out: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    first: HybridStop | None = None
    for index, raw in enumerate(spans):
        try:
            span = _canonical_span(raw, entries, core=core)
        except HybridStop as error:
            first = first or error
            if len(failures) < MAX_DIAGNOSTICS:
                code = error.diagnostics[0]["code"] if error.diagnostics else "BAD_FORM"
                sid = raw.get("id") if isinstance(raw, dict) else None
                failures.append(
                    {
                        "where": where,
                        "index": index,
                        "id": sid if isinstance(sid, str) and sid in entries else None,
                        "code": code if code in _CODES else "BAD_FORM",
                    }
                )
            continue
        # Only repeated id-reference citations are collapsed; legacy offset artifacts
        # keep their historical normalized form byte for byte.
        is_ref = isinstance(raw, dict) and "start" not in raw
        if not (is_ref and span in out):
            out.append(span)
    if first is not None:
        raise HybridStop(str(first), diagnostics=failures)
    return out


_ECHO_KEYS = frozenset({"response_contract", "schema", "stage", "input", "original_source"})
HYPOTHESIS_PLACEHOLDER = "<your own revisable argument of this chapter, as text>"
QUESTION_PLACEHOLDER = "<one disputed, source-grounded question per item>"


def _is_placeholder(value: object) -> bool:
    """True for a whole angle-bracket contract token or a known placeholder prefix."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if text.startswith((HYPOTHESIS_PLACEHOLDER, QUESTION_PLACEHOLDER)):
        return True
    return text.startswith("<") and text.endswith(">") and ">" not in text[1:-1]


def reject_template_echo(raw: object) -> None:
    """Reject a response that repeats the request/contract instead of answering."""
    if not isinstance(raw, dict):
        return
    if _ECHO_KEYS & set(raw):
        raise HybridStop("hybrid response echoed the request contract instead of answering")
    texts = [raw.get("hypothesis"), raw.get("argument")]
    questions = raw.get("questions")
    if isinstance(questions, list):
        texts.extend(questions)
    if any(_is_placeholder(value) for value in texts):
        raise HybridStop("hybrid response repeated a contract placeholder")


MIN_LINKS_FOR_DEGENERACY = 6
MAX_TOP_UNIT_SHARE = 0.5
MAX_TOP_ROLE_SHARE = 0.8


def reject_degenerate_links(links: list[dict[str, Any]], where: str) -> None:
    """Fail closed when most citations hit one source unit or all share one role.

    A real argument over a chapter cites several units with a mix of roles. Counts
    only; no text is read or reported.
    """
    if len(links) < MIN_LINKS_FOR_DEGENERACY:
        return
    units = Counter(link["id"] for link in links)
    roles = Counter(link["role"] for link in links)
    if (
        max(units.values()) / len(links) > MAX_TOP_UNIT_SHARE
        or max(roles.values()) / len(links) > MAX_TOP_ROLE_SHARE
    ):
        raise HybridStop(
            "hybrid citations are degenerate: most links cite one unit or one role",
            diagnostics=[{"where": where, "index": 0, "id": None, "code": "DEGENERATE_LINKS"}],
        )


def validate_argument(raw: object, ledger: dict[str, Any]) -> dict[str, Any]:
    reject_template_echo(raw)
    data = _object(raw, {"hypothesis", "questions", "source_links"})
    _text(data["hypothesis"], 3000)
    _list(data["questions"], 24)
    for question in data["questions"]:
        _text(question, 500)
    source = {entry["id"]: entry for entry in validate_ledger(ledger)}
    links = _list(data["source_links"], MAX_ENTRIES)
    if not links:
        raise HybridStop("revisable argument has no source-linked hypothesis")
    links[:] = _canonical_spans(links, source, "source_links")
    reject_degenerate_links(links, "source_links")
    return data


def validate_audit(
    raw: object, ledger: dict[str, Any], window: dict[str, object]
) -> dict[str, Any]:
    source = {entry["id"]: entry for entry in validate_ledger(ledger)}
    core = window["core_ids"]
    if not isinstance(core, list) or not all(isinstance(sid, str) for sid in core):
        raise HybridStop("hybrid audit core identity malformed")
    expected = set(core)
    data = _object(raw, {"audit_id", "core_ids", "decisions", "disagreements"})
    if data["audit_id"] != window["audit_id"] or data["core_ids"] != core:
        raise HybridStop("hybrid audit identity or core order changed")
    decisions = _list(data["decisions"], len(core))
    seen: set[str] = set()
    for value in decisions:
        item = _object(value, {"id", "decision", "reason", "evidence"})
        sid = item["id"]
        if (
            not isinstance(sid, str)
            or not isinstance(item["decision"], str)
            or sid not in expected
            or sid in seen
            or item["decision"] not in DECISIONS
        ):
            raise HybridStop("hybrid audit source decision missing or duplicated")
        seen.add(sid)
        _text(item["reason"], 800)
        if source[sid]["kind"] != "CANONICAL_WORK_UNIT" and item["decision"] != "omit":
            raise HybridStop("hybrid unresolved original gap cannot be retained")
        evidence = _list(item["evidence"], 12)
        evidence[:] = _canonical_spans(evidence, source, f"decision:{sid}", core=expected)
    if seen != expected:
        raise HybridStop("hybrid audit silently omitted original units")
    disagreements = _list(data["disagreements"], 24)
    for index, value in enumerate(disagreements, 1):
        item = _object(value, {"id", "issue", "proposed_change", "evidence"})
        if item["id"] != f"{window['audit_id']}-d{index:02d}":
            raise HybridStop("hybrid disagreement identity changed")
        _text(item["issue"], 800)
        _text(item["proposed_change"], 1000)
        if not _list(item["evidence"], 12):
            raise HybridStop("hybrid disagreement has no source support")
        item["evidence"][:] = _canonical_spans(
            item["evidence"], source, f"disagreement:{index}", core=expected
        )
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
    validate_argument(argument, ledger)
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
    if _is_placeholder(data["argument"]):
        raise HybridStop("hybrid response repeated a contract placeholder")
    decisions = {sid: dict(value) for sid, value in original.items()}
    edited: set[str] = set()
    for value in _list(data["amendments"], MAX_ENTRIES):
        amendment = _object(value, {"id", "from", "to", "reason", "evidence"})
        sid = amendment["id"]
        if (
            not isinstance(sid, str)
            or not isinstance(amendment["from"], str)
            or not isinstance(amendment["to"], str)
            or sid not in decisions
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
        amendment["evidence"][:] = _canonical_spans(
            amendment["evidence"], source, f"amendment:{sid}"
        )
        decisions[sid] = {
            **decisions[sid],
            "decision": amendment["to"],
            "reason": amendment["reason"],
            "evidence": amendment["evidence"],
        }
    resolved: set[str] = set()
    for value in _list(data["resolutions"], MAX_AUDITS * 24):
        item = _object(value, {"id", "resolution", "reason", "evidence"})
        if not isinstance(item["id"], str) or item["id"] not in disputes or item["id"] in resolved:
            raise HybridStop("hybrid duplicate or unknown audit disagreement")
        resolved.add(item["id"])
        if not isinstance(item["resolution"], str) or item["resolution"] not in {
            "accepted",
            "rejected",
            "qualified",
        }:
            raise HybridStop("unresolved hybrid argument disagreement")
        _text(item["reason"], 800)
        if not _list(item["evidence"], 12):
            raise HybridStop("hybrid disagreement was resolved without original evidence")
        item["evidence"][:] = _canonical_spans(item["evidence"], source, f"resolution:{item['id']}")
    if resolved != disputes:
        raise HybridStop("hybrid global reconciliation left audit disagreements unresolved")
    beats: list[dict[str, Any]] = []
    linked: set[str] = set()
    beat_links: dict[str, list[str]] = {sid: [] for sid in source}
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
        _canonical_spans(spans, source, f"beat:{beat['beat_id']}")
        expanded: list[dict[str, Any]] = []
        for item in spans:
            span = _span(item, source)
            if decisions[span["id"]]["decision"] == "omit":
                raise HybridStop("hybrid beat cites editorially omitted source")
            linked.add(span["id"])
            if beat["beat_id"] not in beat_links[span["id"]]:
                beat_links[span["id"]].append(beat["beat_id"])
            if all(
                (prior["id"], prior["start"], prior["end"], prior["role"])
                != (span["id"], span["start"], span["end"], span["role"])
                for prior in expanded
            ):
                expanded.append(span)
        beats.append({**beat, "spans": expanded, "unit_id": expanded[0]["original_source_id"]})
    if len(beats) < 2:
        raise HybridStop("hybrid coherent chapter requires multiple source-linked beats")
    for sid, decision in decisions.items():
        if (decision["decision"] != "omit") != (sid in linked):
            raise HybridStop("hybrid final beat grouping contradicts complete source coverage")
    canonical = [e for e in entries if e["kind"] == "CANONICAL_WORK_UNIT"]
    kept = sum(1 for e in canonical if decisions[e["id"]]["decision"] != "omit")
    if canonical and kept / len(canonical) < MIN_KEPT_SHARE:
        raise HybridStop("hybrid reconciliation omits nearly all source units")
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
                "beat_ids": beat_links[entry["id"]],
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
    if brief.get("ledger_sha256") != ledger["ledger_sha256"] or digest(
        {k: v for k, v in brief.items() if k != "brief_sha256"}
    ) != brief.get("brief_sha256"):
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
                "next_text": entries[index + 1]["text"] if index + 1 < len(entries) else "",
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
    human_review_packet(brief, ledger)
    entries = validate_ledger(ledger)
    doc = _object(raw, {"schema", "brief_sha256", "reviewer", "unit_decisions", "beat_decisions"})
    if doc["schema"] != REVIEW_SCHEMA or doc["brief_sha256"] != brief["brief_sha256"]:
        raise HybridStop("hybrid human approval was not for the exact coherent brief")
    _text(doc["reviewer"], 100)
    units = _list(doc["unit_decisions"], len(entries))
    seen: set[str] = set()
    for value in units:
        item = _object(value, {"id", "review", "reason"})
        if (
            not isinstance(item["id"], str)
            or item["id"] not in {entry["id"] for entry in entries}
            or item["id"] in seen
        ):
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
        if (
            not isinstance(item["beat_id"], str)
            or item["beat_id"] not in beat_ids
            or item["beat_id"] in decided
        ):
            raise HybridStop("hybrid human beat decisions incomplete")
        if item["review"] != "support_attested":
            raise HybridStop("hybrid semantic beat review rejected source interpretation")
        _text(item["reason"], 1000)
        decided.add(item["beat_id"])
    if decided != beat_ids:
        raise HybridStop("hybrid meaning and qualifications have not been human attested")
    return doc


def validate_adaptation(raw: object, brief: dict[str, Any], plan: Any) -> dict[str, Any]:
    """Reuse the existing chapter creative contract without a lexical meaning proxy.

    Exact original spans and qualifications are appended to the validated artifact,
    not regenerated or copied from an invented illustration.
    """
    from . import chapter

    beats = _list(brief["beats"], plan.max_beats)
    if not beats:
        raise HybridStop("hybrid adaptation has no reviewed source beats")
    # Old chapter's lexical containment checks are not semantic verification.
    # The source is bound separately and kept in the validated artifact.
    shape = {
        "beats": [
            {
                "beat_id": beat["beat_id"],
                "unit_id": beat["spans"][0]["original_source_id"],
                "qualification": "",
                "essential": True,
            }
            for beat in beats
        ]
    }
    decisions = {
        beat["beat_id"]: {"decision": "support_attested", "qualifier_quote": ""} for beat in beats
    }
    try:
        adapted = chapter._adapt_validate(raw, shape, decisions, plan)
    except chapter.ChapterStop as error:
        raise HybridStop("hybrid chapter adaptation violated creative contract") from error
    if any(item["decision"] == "omit" for item in adapted["beats"]):
        raise HybridStop("reviewed hybrid beat cannot disappear without new human approval")
    return {
        **adapted,
        "source_spans_by_beat": {beat["beat_id"]: beat["spans"] for beat in beats},
        "reviewed_qualification_by_beat": {
            beat["beat_id"]: beat["qualification"] for beat in beats
        },
        "brief_sha256": brief["brief_sha256"],
    }


def validate_script(
    raw: object, adaptation: dict[str, Any], brief: dict[str, Any], plan: Any
) -> dict[str, Any]:
    """Validate the existing chapter scene graph, retaining source-span metadata."""
    from . import chapter

    base_adaptation = {
        **adaptation,
        "beats": [{**beat, "qualification": ""} for beat in adaptation["beats"]],
    }
    base_brief = {
        "beats": [
            {
                "beat_id": beat["beat_id"],
                "unit_id": beat["spans"][0]["original_source_id"],
                "qualification": "",
            }
            for beat in brief["beats"]
        ]
    }
    decisions = {
        beat["beat_id"]: {"decision": "support_attested", "qualifier_quote": ""}
        for beat in brief["beats"]
    }
    try:
        script = chapter._script_validate(raw, base_adaptation, decisions, plan, brief=base_brief)
    except chapter.ChapterStop as error:
        raise HybridStop("hybrid script violated connected chapter scene contract") from error
    source_beats = {beat["beat_id"]: beat for beat in brief["beats"]}
    annotated: list[dict[str, Any]] = []
    for scene in script["scenes"]:
        spans = [span for bid in scene["beat_ids"] for span in source_beats[bid]["spans"]]
        annotated.append(
            {
                **scene,
                "source_spans": spans,
                "reviewed_qualifications": {
                    bid: source_beats[bid]["qualification"] for bid in scene["beat_ids"]
                },
                "semantic_review": "NO_AUTOMATIC_SEMANTIC_CERTIFICATION",
            }
        )
    return {**script, "scenes": annotated, "brief_sha256": brief["brief_sha256"]}


def __getattr__(name: str) -> int:
    """Historical name for the active profile's combined prompt-plus-output limit."""
    if name == "SAFE_CONTEXT_TOKENS":
        return int(model_profile.active().safe_context_tokens)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
