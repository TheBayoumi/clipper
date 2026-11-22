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
GRAMMAR_CONTRACT = "book-seven-field-token-mixed-escaped-literal-v4"
PINNED_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
SPAN_POLICY = "bounded-multilingual-clauses-occurrences-v4"
POOL_SCHEMA = "clipper-book-exact-source-spans-v3"
MAX_SOURCE_CHARS = 12_000
MAX_SPANS = 400
MAX_FIELD_CHARS = 400
FIELDS = ("id", "topic", "claim", "hook_angle", "visual_action", "evidence", "evidence_field")
CREATIVE_FIELDS = frozenset({"topic", "claim", "hook_angle", "visual_action"})
SENTENCE_STARTERS = frozenset(
    {
        "then",
        "we",
        "he",
        "she",
        "they",
        "it",
        "the",
        "a",
        "an",
        "but",
        "however",
        "this",
        "that",
        "there",
        "i",
        "you",
        "later",
        "next",
        "finally",
        "also",
        "now",
        "meanwhile",
    }
)
SPLIT = re.compile(
    r"(?<=[.!?;:\u3002\uFF01\uFF1F\u061B\u061F])\s+"
    r"|(?<=[\u3002\uFF01\uFF1F\u061B\u061F])(?=\S)"
)
ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "e.g.",
        "i.e.",
        "vs.",
        "etc.",
        "fig.",
        "vol.",
    }
)
INITIAL = re.compile(r"(?:[A-Za-z]\.){1,6}$")


class ConstraintError(ValueError):
    """Sanitized, fail-closed book-only decoding contract error."""


class NoEligibleSpans(ConstraintError):
    """No bounded, representable source evidence exists for this unit."""


class DecoderUnavailable(ConstraintError):
    """Exact opt-in grammar runtime absent or incompatible."""


def _boundaries(text: str) -> tuple[list[tuple[int, int]], int, int]:
    """Split multilingual clauses without cutting common abbreviations/initials."""
    chunks: list[tuple[int, int]] = []
    previous = 0
    separators = 0
    prevented = 0
    for match in SPLIT.finditer(text):
        left, right = match.span()
        if text[left - 1] == ".":
            previous_token = text[max(0, text.rfind(" ", 0, left) + 1) : left]
            # Standalone "no." is a sentence; "No. 5" is a numbered reference.
            initials = INITIAL.fullmatch(previous_token) is not None
            next_token = text[right:].split(maxsplit=1)
            follows_name = bool(next_token and next_token[0][0].isupper())
            sentence_start = bool(
                next_token and next_token[0].strip(",;:").casefold() in SENTENCE_STARTERS
            )
            numbered = previous_token.casefold() == "no." and bool(
                next_token and re.match(r"[#0-9]", next_token[0])
            )
            if not sentence_start and (
                previous_token.casefold() in ABBREVIATIONS
                or numbered
                or (initials and (previous_token.count(".") > 1 or follows_name))
            ):
                prevented += 1
                continue
        chunks.append((previous, left))
        separators += right - left
        previous = right
    chunks.append((previous, len(text)))
    return chunks, separators, prevented


def span_pool(text: str, evidence_field: str) -> dict[str, object]:
    """Exact source offsets per occurrence; no truncation or hidden exclusion."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    spans: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    omitted = {"overlong": 0, "pool_cap": 0, "duplicate": 0}
    omitted_chars = 0
    covered_chars = 0
    separator_chars = 0
    prevented_breaks = 0
    oversized_source = len(text) > MAX_SOURCE_CHARS
    if not oversized_source:
        boundaries, separator_chars, prevented_breaks = _boundaries(text)
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
            elif len(spans) >= MAX_SPANS:
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
            else:
                seen[value] = start
            words = re.findall(r"[a-z0-9]+", value.casefold())[:5]
            stem = "-".join(words)[:70].strip("-") or "unicode-source-span"
            identifier = f"{stem}-{digest[:12]}-{start}"
            if not selection.IDENTIFIER.fullmatch(identifier):
                raise ConstraintError("invalid deterministic evidence identifier")
            boundary = "clause"
            if start == 0:
                boundary = "unit_start_context_unknown"
            elif end == len(text):
                boundary = "unit_end_context_unknown"
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
            "abbreviation_splits_prevented": prevented_breaks,
            "omitted_fragment_chars": omitted_chars,
            "omitted": omitted,
            "oversized_source": oversized_source,
            "fail_closed_for_coverage": bool(omitted_chars or oversized_source),
            "unit_start_context_unknown": bool(text.strip()),
            "unit_end_context_unknown": bool(text.strip()),
            "source_selectable_span_count": len(spans),
            "verified_feasible_span_count": None,
            "verified_feasible_full_reply_count": None,
            "generation_budget_status": "unverified_per_span_and_full_reply",
            "whole_reply_max_tokens": 960,
            "raw_reply_max_bytes": 16_384,
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
        "verified_feasible_span_count": None,
        "verified_feasible_full_reply_count": None,
        "covered_fragment_chars": sum(
            cast(int, cast(dict[str, object], record["coverage"])["covered_chars"])
            for record in records
        ),
        "split_separator_chars": sum(
            cast(int, cast(dict[str, object], record["coverage"])["split_separator_chars"])
            for record in records
        ),
        "omitted_fragment_chars": sum(
            cast(int, cast(dict[str, object], record["coverage"])["omitted_fragment_chars"])
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
        menu = evidence.get("source_evidence_choices")
        if menu is not None and menu != [
            {
                "id": span["id"],
                "evidence": span["text"],
                "start": span["start"],
                "end": span["end"],
                "boundary": span["boundary"],
            }
            for span in spans
        ]:
            raise ValueError("source evidence choice menu mismatch")
        if cast(dict[str, object], expected["coverage"])["fail_closed_for_coverage"]:
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


def _literal_evidence_guard(parser: Any, spans: list[dict[str, object]], base_type: Any) -> Any:
    """Book-only public LMFE parser overlay enforcing exact source ID/quote pairs.

    Per-character direct and escaped JSON spellings share suffix states,
    allowing the public tokenizer adapter to choose a tokenizable mixed path.
    Every allowed spelling decodes to the identical source occurrence. No
    private hooks, Unicode normalization or post-hoc assembly are used.
    """
    terminal = "\x00"
    pairs = {str(span["id"]): str(span["text"]) for span in spans}
    by_quote: dict[str, set[str]] = {}
    for identifier, quote in pairs.items():
        by_quote.setdefault(quote, set()).add(identifier)

    def trie_for(strings: set[str], *, escaped: bool) -> dict[str, Any]:
        """Build linear-size spelling DAGs instead of 2**N alternatives."""

        def merge(
            left: dict[str, Any],
            right: dict[str, Any],
            memo: dict[tuple[int, int], dict[str, Any]],
        ) -> dict[str, Any]:
            # Persistent union: mutating a shared DAG suffix can introduce
            # cycles and admit spurious paths for overlapping source quotes.
            if left is right:
                return left
            key = (id(left), id(right))
            if key in memo:
                return memo[key]
            result: dict[str, Any] = {}
            memo[key] = result
            for letter in sorted(set(left) | set(right)):
                if letter == terminal:
                    result[letter] = True
                elif letter in left and letter in right:
                    result[letter] = merge(left[letter], right[letter], memo)
                elif letter in left:
                    result[letter] = left[letter]
                else:
                    result[letter] = right[letter]
            return result

        root: dict[str, Any] = {}
        for value in sorted(strings):
            suffix: dict[str, Any] = {terminal: True}
            for character in reversed(value):
                direct = json.dumps(character, ensure_ascii=False)[1:-1]
                ascii_wire = json.dumps(character, ensure_ascii=True)[1:-1]
                spellings = {direct, ascii_wire} if escaped else {character}
                prefix: dict[str, Any] = {}
                for spelling in sorted(spellings):
                    node = prefix
                    for symbol in spelling[:-1]:
                        node = node.setdefault(symbol, {})
                    node[spelling[-1]] = suffix
                suffix = prefix
            root = merge(root, suffix, {})
        return root

    all_ids = trie_for(set(pairs), escaped=False)
    all_quotes = trie_for(set(pairs.values()), escaped=True)
    quote_by_id = {key: trie_for({value}, escaped=True) for key, value in pairs.items()}
    # The released adapter replaces parser.config with a tokenizer-derived
    # alphabet. Extend that PUBLIC config for exact source characters: otherwise
    # its free-string parser excludes direct Unicode even when tokens exist.
    extra_alphabet = "".join(sorted({c for value in pairs.values() for c in value if ord(c) > 127}))

    class EvidenceGuard(base_type):  # type: ignore[misc]
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
            field: str = "",
            value_buffer: str = "",
            in_lessons: bool = False,
            in_candidate: bool = False,
            selected_id: str | None = None,
            selected_quote: str | None = None,
            used_ids: frozenset[str] = frozenset(),
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
            self.field = field
            self.value_buffer = value_buffer
            self.in_lessons = in_lessons
            self.in_candidate = in_candidate
            self.selected_id = selected_id
            self.selected_quote = selected_quote
            self.used_ids = used_ids
            self.active = active
            super().__init__(parser_config)

        @property
        def config(self) -> Any:
            return self._config

        @config.setter
        def config(self, value: Any) -> None:
            if extra_alphabet:
                value.alphabet += "".join(c for c in extra_alphabet if c not in value.alphabet)
            self._config = value
            self.inner.config = value

        def get_allowed_characters(self) -> str:
            allowed: str = self.inner.get_allowed_characters()
            if self.active is None:
                if (
                    self.in_string
                    and not self.is_key
                    and self.in_candidate
                    and self.field in CREATIVE_FIELDS
                    and not self.escaped
                ):
                    try:
                        meaningful = bool(json.loads('"' + self.value_buffer + '"').strip())
                    except ValueError:
                        meaningful = False
                    if not meaningful:
                        allowed = allowed.replace('"', "")
                if self.in_lessons and not self.in_candidate and len(self.used_ids) == len(pairs):
                    allowed = allowed.replace("{", "")
                return allowed
            options = set(self.active).difference({terminal})
            if terminal in self.active and not self.escaped:
                options.add('"')
            return "".join(character for character in allowed if character in options)

        def add_character(self, character: str) -> Any:
            if character not in self.get_allowed_characters():
                raise ConstraintError("invalid paired source evidence character")
            inner = self.inner.add_character(character)
            in_string, escaped, is_key = self.in_string, self.escaped, self.is_key
            key_buffer, last_key, last_sig = self.key_buffer, self.last_key, self.last_sig
            field, value_buffer = self.field, self.value_buffer
            in_lessons, in_candidate = self.in_lessons, self.in_candidate
            selected_id, selected_quote = self.selected_id, self.selected_quote
            used_ids = self.used_ids
            active = self.active

            if not in_string:
                if character == '"':
                    is_key = last_sig != ":"
                    field = "" if is_key else last_key
                    key_buffer, value_buffer = "", ""
                    if in_candidate and field == "id":
                        eligible = (
                            by_quote[selected_quote] if selected_quote is not None else pairs.keys()
                        )
                        active = (
                            all_ids
                            if not used_ids and selected_quote is None
                            else trie_for(set(eligible) - used_ids, escaped=False)
                        )
                    elif in_candidate and field == "evidence":
                        if selected_id is not None:
                            active = quote_by_id[selected_id]
                        elif not used_ids:
                            active = all_quotes
                        else:
                            unused_quotes = {
                                quote
                                for identifier, quote in pairs.items()
                                if identifier not in used_ids
                            }
                            active = trie_for(unused_quotes, escaped=True)
                    else:
                        active = None
                    in_string, escaped = True, False
                elif not character.isspace():
                    if character == "[" and last_key == "lessons":
                        in_lessons = True
                    elif character == "{" and in_lessons:
                        in_candidate = True
                        selected_id, selected_quote = None, None
                    elif character == "}" and in_candidate:
                        if (
                            selected_id is None
                            or pairs.get(selected_id) != selected_quote
                            or selected_id in used_ids
                        ):
                            raise ConstraintError("source ID/evidence pair mismatch")
                        used_ids = used_ids | {selected_id}
                        in_candidate = False
                    elif character == "]" and in_lessons:
                        in_lessons = False
                    last_sig = character
            elif escaped:
                if active is not None:
                    active = active[character]
                if is_key:
                    key_buffer += character
                elif field in FIELDS:
                    value_buffer += character
                escaped = False
            elif character == "\\":
                if active is not None:
                    active = active[character]
                if field in FIELDS and not is_key:
                    value_buffer += character
                escaped = True
            elif character == '"':
                if active is not None and terminal not in active:
                    raise ConstraintError("incomplete source ID/evidence choice")
                if is_key:
                    last_key = key_buffer
                elif in_candidate and field == "id":
                    selected_id = json.loads('"' + value_buffer + '"')
                    if (
                        selected_id not in pairs
                        or selected_id in used_ids
                        or (selected_quote is not None and pairs[selected_id] != selected_quote)
                    ):
                        raise ConstraintError("source ID/evidence pair mismatch")
                elif in_candidate and field == "evidence":
                    selected_quote = json.loads('"' + value_buffer + '"')
                    if selected_quote not in by_quote or (
                        selected_id is not None and pairs[selected_id] != selected_quote
                    ):
                        raise ConstraintError("source ID/evidence pair mismatch")
                elif (
                    in_candidate
                    and field in CREATIVE_FIELDS
                    and not json.loads('"' + value_buffer + '"').strip()
                ):
                    raise ConstraintError("empty creative field")
                active = None
                in_string = False
                field = ""
                last_sig = '"'
            else:
                if active is not None:
                    active = active[character]
                if is_key:
                    key_buffer += character
                elif field in FIELDS:
                    value_buffer += character

            return EvidenceGuard(
                inner,
                in_string=in_string,
                escaped=escaped,
                is_key=is_key,
                key_buffer=key_buffer,
                last_key=last_key,
                last_sig=last_sig,
                field=field,
                value_buffer=value_buffer,
                in_lessons=in_lessons,
                in_candidate=in_candidate,
                selected_id=selected_id,
                selected_quote=selected_quote,
                used_ids=used_ids,
                active=active,
                parser_config=self.config,
            )

        def can_end(self) -> bool:
            return not self.in_string and self.active is None and self.inner.can_end()

        def shortcut_key(self) -> Any:
            return None if self.active is not None else self.inner.shortcut_key()

        def cache_key(self) -> Any:
            return None if self.active is not None else self.inner.cache_key()

    root = EvidenceGuard(parser, parser_config=parser.config)
    root.config = parser.config
    return root


def selected_span(item: dict[str, str], request: dict[str, object]) -> dict[str, object]:
    """Return the one declared occurrence after verifying its raw ID/quote pair."""
    decode_schema(request)
    spans = cast(list[dict[str, object]], cast(dict[str, object], request["constraints"])["spans"])
    for span in spans:
        if span["id"] == item["id"] and span["text"] == item["evidence"]:
            return span
    raise ConstraintError("local reply source ID/evidence pair was not declared")


def validate_local_choice(lessons: list[dict[str, str]], request: dict[str, object]) -> None:
    """Independent actual-raw validation, including every first/replay pair."""
    seen: set[str] = set()
    for lesson in lessons:
        span = selected_span(lesson, request)
        identifier = str(span["id"])
        if identifier in seen:
            raise ConstraintError("duplicate local source ID within one reply")
        seen.add(identifier)


def prefix_for(request: dict[str, object], tokenizer: Any, revision: str) -> Any:
    """Fresh per-request LMFE parser/enforcer; scripts remain unaffected."""
    schema = decode_schema(request)
    module, integration = _decoder()
    spans = cast(list[dict[str, object]], cast(dict[str, object], request["constraints"])["spans"])
    parser = _literal_evidence_guard(
        module.JsonSchemaParser(schema), spans, module.CharacterLevelParser
    )
    tokenizer_data = _tokenizer_data(tokenizer, revision)
    return integration.build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
