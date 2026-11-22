"""Book-only, bounded source-span JSON decoding contract; no weights imported."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from . import selection

DECODER_PACKAGE = "lm-format-enforcer"
DECODER_VERSION = "0.11.3"
GRAMMAR_CONTRACT = "book-seven-field-json-schema-v1"
SPAN_POLICY = "bounded-source-clause-spans-v1"
POOL_SCHEMA = "clipper-book-exact-source-spans-v1"
MAX_SOURCE_CHARS = 12_000
MAX_SPANS = 24
MAX_FIELD_CHARS = 400
FIELDS = ("id", "topic", "claim", "hook_angle", "visual_action", "evidence", "evidence_field")
SPLIT = re.compile(r"(?<=[.!?;:])\s+")


class ConstraintError(ValueError):
    """Sanitized, fail-closed book-only decoding contract error."""


class NoEligibleSpans(ConstraintError):
    """No bounded, representable source evidence exists for this unit."""


class DecoderUnavailable(ConstraintError):
    """Exact opt-in grammar runtime absent or incompatible."""


def span_pool(text: str, evidence_field: str) -> dict[str, object]:
    """Keep complete sentence/clause or boundary fragments; never trim long clauses.

    Offsets index the exact supplied Python Unicode string. The 350-word upstream
    split may cut a sentence; boundary fragments are explicitly identified.
    Quotes, backslashes and control characters need JSON escaping; this v1
    enum parser cannot faithfully constrain their escaped spellings, so such
    whole fragments are omitted and counted rather than silently rewritten.
    """
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    spans: list[dict[str, object]] = []
    omitted = {"overlong": 0, "escaping": 0, "pool_cap": 0, "duplicate": 0}
    omitted_chars = 0
    covered_chars = 0
    oversized_source = len(text) > MAX_SOURCE_CHARS
    if not oversized_source:
        boundaries = [(0, len(text))]
        if text:
            boundaries = []
            offset = 0
            for match in SPLIT.finditer(text):
                boundaries.append((offset, match.start()))
                offset = match.end()
            boundaries.append((offset, len(text)))
        seen: set[str] = set()
        for left, right in boundaries:
            fragment = text[left:right]
            start = left + len(fragment) - len(fragment.lstrip())
            end = right - len(fragment) + len(fragment.rstrip())
            value = text[start:end]
            if not value:
                continue
            reason: str | None = None
            if len(value) > MAX_FIELD_CHARS:
                reason = "overlong"
            elif any(c in value for c in ('"', "\\")) or any(ord(c) < 32 for c in value):
                reason = "escaping"
            elif value in seen:
                reason = "duplicate"
            elif len(spans) >= MAX_SPANS:
                reason = "pool_cap"
            if reason:
                omitted[reason] += 1
                omitted_chars += len(value)
                continue
            seen.add(value)
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            words = re.findall(r"[a-z0-9]+", value.casefold())[:5]
            stem = "-".join(words)[:70].strip("-") or "unicode-source-span"
            identifier = f"{stem}-{digest[:12]}"
            if not selection.IDENTIFIER.fullmatch(identifier):
                raise ConstraintError("invalid deterministic evidence identifier")
            boundary = "clause"
            if start == 0:
                boundary = "unit_start_unknown"
            elif end == len(text) and value[-1] not in ".!?;:":
                boundary = "unit_end_may_be_incomplete"
            spans.append(
                {
                    "text": value,
                    "start": start,
                    "end": end,
                    "sha256": digest,
                    "id": identifier,
                    "boundary": boundary,
                }
            )
            covered_chars += len(value)
    else:
        omitted_chars = len(text)
    return {
        "schema": POOL_SCHEMA,
        "policy": SPAN_POLICY,
        "text_sha256": sha,
        "evidence_field": evidence_field,
        "source_chars": len(text),
        "spans": spans,
        "coverage": {
            "covered_chars": covered_chars,
            "omitted_fragment_chars": omitted_chars,
            "omitted": omitted,
            "oversized_source": oversized_source,
        },
    }


def decode_schema(request: dict[str, object]) -> dict[str, object]:
    """Recompute and verify the canonical span pool before enabling a grammar."""
    try:
        metadata = request["constraints"]
        messages = request["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("invalid messages")
        evidence = json.loads(messages[1]["content"])
        if not isinstance(evidence, dict):
            raise ValueError("invalid evidence")
        text = evidence["text"]
        field = evidence["evidence_field"]
        if not isinstance(text, str) or not isinstance(field, str):
            raise ValueError("invalid source fields")
        expected = span_pool(text, field)
        if metadata != expected:
            raise ValueError("source span identity mismatch")
        spans = cast(list[dict[str, object]], expected["spans"])
        if not spans:
            raise NoEligibleSpans("no eligible exact source span for constrained mining")
    except NoEligibleSpans:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, IndexError) as exc:
        raise ConstraintError("invalid canonical source span request") from exc

    values = [str(span["text"]) for span in spans]
    ids = [str(span["id"]) for span in spans]
    fields: dict[str, object] = {
        key: {"type": "string", "minLength": 1, "maxLength": MAX_FIELD_CHARS}
        for key in ("topic", "claim", "hook_angle", "visual_action")
    }
    fields["id"] = {"type": "string", "enum": ids}
    fields["evidence"] = {"type": "string", "enum": values}
    fields["evidence_field"] = {"type": "string", "enum": [field]}
    return {
        "type": "object",
        "properties": {
            "lessons": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": fields,
                    "required": list(FIELDS),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["lessons"],
        "additionalProperties": False,
    }


def _decoder() -> tuple[Any, Any]:
    try:
        installed = version(DECODER_PACKAGE)
    except PackageNotFoundError as exc:
        raise DecoderUnavailable(
            "Install the [book-constrained] extra for CPU book JSON constraints"
        ) from exc
    if installed != DECODER_VERSION:
        raise DecoderUnavailable("local book decoder version does not match its exact pin")
    try:
        module = importlib.import_module("lmformatenforcer")
        integration = importlib.import_module("lmformatenforcer.integrations.transformers")
    except ImportError as exc:
        raise DecoderUnavailable("local book decoder or Transformers integration missing") from exc
    return module, integration


@lru_cache(maxsize=2)
def _tokenizer_data(tokenizer: Any, revision: str) -> Any:
    """Cache tokenizer vocabulary only per tokenizer object and pinned revision."""
    _module, integration = _decoder()
    return integration.build_token_enforcer_tokenizer_data(tokenizer)


def prefix_for(request: dict[str, object], tokenizer: Any, revision: str) -> Any:
    """Fresh per-request parser/enforcer state; scripts never use this callback."""
    schema = decode_schema(request)
    module, integration = _decoder()
    parser = module.JsonSchemaParser(schema)
    tokenizer_data = _tokenizer_data(tokenizer, revision)
    return integration.build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
