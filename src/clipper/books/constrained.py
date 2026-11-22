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
GRAMMAR_CONTRACT = "book-seven-field-json-exact-literal-filter-v2"
SPAN_POLICY = "bounded-source-clause-spans-inclusive-v2"
POOL_SCHEMA = "clipper-book-exact-source-spans-v2"
MAX_SOURCE_CHARS = 12_000
MAX_SPANS = 400
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
    """Bounded complete clause evidence with offsets, duplicates and honest gaps.

    Original string content is never rewritten. JSON-escaped literal choices are
    built only at decoding time; the returned pool contains exact source slices.
    Source lengths/caps that would omit a clause fail closed before inference.
    """
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    spans: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    omitted = {"overlong": 0, "pool_cap": 0, "duplicate": 0}
    omitted_chars = 0
    covered_chars = 0
    separator_chars = 0
    oversized_source = len(text) > MAX_SOURCE_CHARS
    if not oversized_source:
        boundaries = [(0, len(text))]
        if text:
            boundaries = []
            offset = 0
            for match in SPLIT.finditer(text):
                boundaries.append((offset, match.start()))
                separator_chars += match.end() - match.start()
                offset = match.end()
            boundaries.append((offset, len(text)))
        seen: dict[str, int] = {}
        for left, right in boundaries:
            fragment = text[left:right]
            start = left + len(fragment) - len(fragment.lstrip())
            end = right - len(fragment) + len(fragment.rstrip())
            separator_chars += right - left - (end - start)
            value = text[start:end]
            if not value:
                continue
            reason: str | None = None
            if len(value) > MAX_FIELD_CHARS:
                reason = "overlong"
            elif value not in seen and len(spans) >= MAX_SPANS:
                reason = "pool_cap"
            if reason:
                omitted[reason] += 1
                omitted_chars += len(value)
                continue
            covered_chars += len(value)
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if value in seen:
                omitted["duplicate"] += 1
                duplicates.append(
                    {
                        "start": start,
                        "end": end,
                        "sha256": digest,
                        "first_start": seen[value],
                    }
                )
                continue
            seen[value] = start
            words = re.findall(r"[a-z0-9]+", value.casefold())[:5]
            stem = "-".join(words)[:70].strip("-") or "unicode-source-span"
            identifier = f"{stem}-{digest[:12]}"
            if not selection.IDENTIFIER.fullmatch(identifier):
                raise ConstraintError("invalid deterministic evidence identifier")
            boundary = "clause"
            if start == 0:
                boundary = "unit_start_unknown"
            elif end == len(text):
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
    else:
        omitted_chars = len(text)
    return {
        "schema": POOL_SCHEMA,
        "policy": SPAN_POLICY,
        "text_sha256": sha,
        "evidence_field": evidence_field,
        "source_chars": len(text),
        "spans": spans,
        "duplicates": duplicates,
        "coverage": {
            "covered_chars": covered_chars,
            "split_separator_chars": separator_chars,
            "omitted_fragment_chars": omitted_chars,
            "omitted": omitted,
            "oversized_source": oversized_source,
            "fail_closed_for_coverage": bool(omitted_chars or oversized_source),
            "unit_start_context_unknown": bool(text.strip()),
            "unit_end_context_unknown": bool(text.strip()),
        },
    }


def coverage_record(pool: dict[str, object]) -> dict[str, object]:
    """Report all private source offsets/hashes without duplicating passage text."""
    spans = cast(list[dict[str, object]], pool["spans"])
    return {
        "schema": pool["schema"],
        "policy": pool["policy"],
        "text_sha256": pool["text_sha256"],
        "evidence_field": pool["evidence_field"],
        "source_chars": pool["source_chars"],
        "selectable_spans": [
            {key: value for key, value in span.items() if key != "text"} for span in spans
        ],
        "duplicate_occurrences": pool["duplicates"],
        "coverage": pool["coverage"],
    }


def coverage_summary(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate exclusions, separators and unit-boundary uncertainty."""
    return {
        "schema": "clipper-book-private-evidence-coverage-summary-v1",
        "policy": SPAN_POLICY,
        "units": len(records),
        "source_chars": sum(int(cast(int, record["source_chars"])) for record in records),
        "selectable_span_count": sum(
            len(cast(list[object], record["selectable_spans"])) for record in records
        ),
        "covered_fragment_chars": sum(
            int(cast(dict[str, object], record["coverage"])["covered_chars"]) for record in records
        ),
        "split_separator_chars": sum(
            int(cast(dict[str, object], record["coverage"])["split_separator_chars"])
            for record in records
        ),
        "omitted_fragment_chars": sum(
            int(cast(dict[str, object], record["coverage"])["omitted_fragment_chars"])
            for record in records
        ),
        "units_fail_closed_for_coverage": sum(
            bool(cast(dict[str, object], record["coverage"])["fail_closed_for_coverage"])
            for record in records
        ),
        "unit_start_context_unknown": sum(
            bool(cast(dict[str, object], record["coverage"])["unit_start_context_unknown"])
            for record in records
        ),
        "unit_end_context_unknown": sum(
            bool(cast(dict[str, object], record["coverage"])["unit_end_context_unknown"])
            for record in records
        ),
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
        if expected["coverage"]["fail_closed_for_coverage"]:
            raise NoEligibleSpans("source clauses exceed bounded evidence coverage")
        if not spans:
            raise NoEligibleSpans("no eligible exact source span for constrained mining")
    except NoEligibleSpans:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, IndexError) as exc:
        raise ConstraintError("invalid canonical source span request") from exc

    ids = [str(span["id"]) for span in spans]
    fields: dict[str, object] = {
        key: {"type": "string", "minLength": 1, "maxLength": MAX_FIELD_CHARS}
        for key in ("topic", "claim", "hook_angle", "visual_action")
    }
    fields["id"] = {"type": "string", "enum": ids}
    # A literal overlay constrains the raw JSON spelling to exact source
    # values. The bare LMFE enum mishandles quotes/backslashes/UTF-8 escapes.
    fields["evidence"] = {"type": "string"}
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


def _literal_evidence_guard(parser: Any, values: list[str], base_type: Any) -> Any:
    """Fast, immutable public CharacterLevelParser overlay for evidence only.

    JSON Schema enforces the seven-field structure. This small lexer detects
    the evidence value and narrows its legal next characters to a trie of
    JSON-serialized literal interiors, including ASCII Unicode escapes.
    No private library hooks or whole-response SequenceParser are involved.
    """
    terminal = "\x00"
    trie: dict[str, Any] = {}
    for value in values:
        literal = json.dumps(value, ensure_ascii=True)[1:-1]
        node = trie
        for character in literal:
            node = node.setdefault(character, {})
        node[terminal] = True

    class EvidenceGuard(base_type):
        def __init__(
            self,
            inner: Any,
            *,
            in_string: bool = False,
            escaped: bool = False,
            is_key: bool = False,
            key_buffer: str = "",
            last_key: str = "",
            last_sig: str = "",
            active: dict[str, Any] | None = None,
            parser_config: Any = None,
        ) -> None:
            self.inner = inner
            self.in_string = in_string
            self.escaped = escaped
            self.is_key = is_key
            self.key_buffer = key_buffer
            self.last_key = last_key
            self.last_sig = last_sig
            self.active = active
            super().__init__(parser_config)

        @property
        def config(self) -> Any:
            return self._config

        @config.setter
        def config(self, value: Any) -> None:
            self._config = value
            # LMFE's UnionParser does not initialize a config property.
            # Its child JSON parsers retain the configured shared root.
            self.inner.config = value

        def get_allowed_characters(self) -> str:
            allowed: str = self.inner.get_allowed_characters()
            if self.active is None:
                return allowed
            options = set(self.active).difference({terminal})
            if terminal in self.active and not self.escaped:
                options.add('"')
            return "".join(c for c in allowed if c in options)

        def add_character(self, character: str) -> Any:
            if character not in self.get_allowed_characters():
                raise ConstraintError("invalid exact-evidence character continuation")
            inner = self.inner.add_character(character)
            in_string, escaped, is_key = self.in_string, self.escaped, self.is_key
            key_buffer, last_key, last_sig = self.key_buffer, self.last_key, self.last_sig
            active = self.active
            if not in_string:
                if character == '"':
                    is_key = last_sig != ":"
                    key_buffer = ""
                    active = trie if not is_key and last_key == "evidence" else None
                    in_string = True
                    escaped = False
                elif not character.isspace():
                    last_sig = character
            elif escaped:
                if active is not None:
                    active = active[character]
                elif is_key:
                    key_buffer += character
                escaped = False
            elif character == "\\":
                if active is not None:
                    active = active[character]
                escaped = True
            elif character == '"':
                if active is not None and terminal not in active:
                    raise ConstraintError("incomplete exact source evidence literal")
                if is_key:
                    last_key = key_buffer
                active = None
                in_string = False
                last_sig = '"'
            else:
                if active is not None:
                    active = active[character]
                elif is_key:
                    key_buffer += character
            return EvidenceGuard(
                inner,
                in_string=in_string,
                escaped=escaped,
                is_key=is_key,
                key_buffer=key_buffer,
                last_key=last_key,
                last_sig=last_sig,
                active=active,
                parser_config=self.config,
            )

        def can_end(self) -> bool:
            return not self.in_string and self.active is None and self.inner.can_end()

        def shortcut_key(self) -> Any:
            return None if self.active is not None else self.inner.shortcut_key()

        def cache_key(self) -> Any:
            return None if self.active is not None else self.inner.cache_key()

    return EvidenceGuard(parser, parser_config=parser.config)


def validate_local_choice(lessons: list[dict[str, str]], request: dict[str, object]) -> None:
    """Independent raw-derived local evidence choice check, also run on replay."""
    decode_schema(request)
    choices = request["constraints"]["spans"]
    evidence = {span["text"] for span in choices}
    ids = {span["id"] for span in choices}
    for lesson in lessons:
        if lesson["evidence"] not in evidence or lesson["id"] not in ids:
            raise ConstraintError("local reply evidence was not a canonical source choice")


def prefix_for(request: dict[str, object], tokenizer: Any, revision: str) -> Any:
    """Fresh per-request LMFE parser/enforcer; scripts remain unaffected."""
    schema = decode_schema(request)
    module, integration = _decoder()
    values = [str(span["text"]) for span in request["constraints"]["spans"]]
    parser = _literal_evidence_guard(
        module.JsonSchemaParser(schema), values, module.CharacterLevelParser
    )
    tokenizer_data = _tokenizer_data(tokenizer, revision)
    return integration.build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
