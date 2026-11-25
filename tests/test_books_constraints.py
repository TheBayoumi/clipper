"""Real released JSON/token parser tests; strictly no model or weight loading."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

from clipper.books import constrained, mining, scripting, whole_book
from clipper.stage_contracts import content_fingerprint


def _request(text: str, field: str = "native_text") -> dict[str, object]:
    evidence = {"text": text, "evidence_field": field}
    return {
        "messages": [
            {"role": "system", "content": "synthetic schema test"},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ],
        "constraints": constrained.span_pool(text, field),
    }


def _decode_request(text: str) -> tuple[dict[str, object], dict[str, object]]:
    request = _request(text)
    return request, constrained.decode_schema(request)


def _drive(parser: CharacterLevelParser, value: str) -> CharacterLevelParser:
    for c in value:
        assert c in parser.get_allowed_characters(), (repr(c), value)
        parser = parser.add_character(c)
    assert parser.can_end()
    return parser


def _guard(request: dict[str, object]) -> CharacterLevelParser:
    schema = constrained.decode_schema(request)
    parser = JsonSchemaParser(schema)
    spans = request["constraints"]["spans"]
    return constrained._literal_evidence_guard(parser, spans, CharacterLevelParser)


def _synthetic_tokens() -> TokenEnforcerTokenizerData:
    # A fully synthetic Unicode codepoint tokenizer, NOT Qwen or model output.
    alphabet = [chr(i) for i in range(32, 127)]
    tokens = [(i, char, False) for i, char in enumerate(alphabet)]
    return TokenEnforcerTokenizerData(
        tokens,
        lambda ids: "".join(alphabet[i] for i in ids),
        eos_token_id=len(alphabet),
        use_bitmask=False,
        vocab_size=len(alphabet) + 1,
    )


def test_shipped_parser_requires_object_with_lessons_and_all_fields() -> None:
    assert version(constrained.DECODER_PACKAGE) == "0.11.3"
    _request_data, schema = _decode_request("A short example demonstrates a real test.")
    parser = JsonSchemaParser(schema)
    assert "{" in parser.get_allowed_characters()
    assert "[" not in parser.get_allowed_characters()
    _drive(parser, '{"lessons":[]}')
    assert not JsonSchemaParser(schema).can_end()
    fields = list(constrained.FIELDS)
    spans = constrained.span_pool("A short example demonstrates a real test.", "native_text")
    data = {
        "lessons": [
            {
                "id": spans["spans"][0]["id"],
                "topic": "A test",
                "claim": "The example demonstrates a test",
                "hook_angle": "What did the test demonstrate?",
                "visual_action": "An actor marks a check",
                "evidence": spans["spans"][0]["text"],
                "evidence_field": "native_text",
            }
        ]
    }
    assert set(data["lessons"][0]) == set(fields)
    _drive(_guard(_request("A short example demonstrates a real test.")), json.dumps(data))
    with pytest.raises(mining.BookMiningError, match="lessons array"):
        whole_book.validate_reply(data["lessons"], SimpleNamespace())
    # Duplicate keys must be rejected even when plain json.loads would overwrite them.
    with pytest.raises(mining.BookMiningError, match="duplicate JSON"):
        whole_book.load_strict_reply('{"lessons":[],"lessons":[]}')
    with pytest.raises(mining.BookMiningError, match="duplicate JSON"):
        whole_book.load_strict_reply('{"lessons":[{"id":"first","id":"second"}]}')


def test_real_released_token_enforcer_restricts_root_and_terminal_eos() -> None:
    _request_data, _schema = _decode_request("One short source statement.")
    data = _synthetic_tokens()
    enforcer = TokenEnforcer(data, _guard(_request("One short source statement.")))
    initial = enforcer.get_allowed_tokens([999]).allowed_tokens
    open_id = next(i for i, value, _ in data.regular_tokens if value == "{")
    array_id = next(i for i, value, _ in data.regular_tokens if value == "[")
    assert open_id in initial
    assert array_id not in initial
    assert data.eos_token_id not in initial
    # Exercise allowed-token machinery, not a mocked always-accept callback.
    by_char = {value: i for i, value, _ in data.regular_tokens}
    ids = [999]
    for character in '{"lessons":[]}':
        ids.append(by_char[character])
        allowed = enforcer.get_allowed_tokens(ids).allowed_tokens
        if character != "}":
            assert data.eos_token_id not in allowed
    assert data.eos_token_id in enforcer.get_allowed_tokens(ids).allowed_tokens


@pytest.mark.parametrize(
    "source,expected",
    [
        ("First source sentence. Second supported clause!", 2),
        ("Simple evidence with Unicode café and 漢字.", 1),
        ('A quoted "term" is retained with its full caveat.', 1),
        ("An escaped \\ fragment.", 1),
        ("X" * 401, 0),
        (" ", 0),
    ],
)
def test_span_selection_exact_unicode_and_declared_omissions(source: str, expected: int) -> None:
    pool = constrained.span_pool(source, "machine_ocr_text")
    spans = pool["spans"]
    assert len(spans) == expected
    assert pool["text_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    for span in spans:
        assert source[span["start"] : span["end"]] == span["text"]
        assert len(span["text"]) <= 400
        assert span["sha256"] == hashlib.sha256(span["text"].encode()).hexdigest()
        assert constrained.selection.IDENTIFIER.fullmatch(span["id"])
    if not expected:
        with pytest.raises(constrained.NoEligibleSpans):
            constrained.decode_schema(_request(source, "machine_ocr_text"))
    else:
        request = _request(source, "machine_ocr_text")
        schema = constrained.decode_schema(request)
        assert schema["properties"]["lessons"]["items"]["properties"]["evidence"] == {
            "type": "string"
        }
        assert pool["coverage"]["omitted_fragment_chars"] == 0


def test_boundary_cap_duplicate_and_overlong_coverage_are_explicit() -> None:
    pool = constrained.span_pool("Repeated. Repeated. Tail without terminator", "native_text")
    assert len(pool["spans"]) == 3
    assert pool["coverage"]["omitted"]["duplicate"] == 1
    assert pool["spans"][0]["id"] != pool["spans"][1]["id"]
    assert pool["spans"][0]["text"] == pool["spans"][1]["text"]
    assert pool["spans"][-1]["boundary"] == "unit_end_context_unknown"
    assert pool["spans"][0]["boundary"] == "unit_start_context_unknown"
    cap = constrained.span_pool(" ".join(f"Sentence {i}." for i in range(30)), "native_text")
    assert len(cap["spans"]) == 30
    assert cap["coverage"]["omitted"]["pool_cap"] == 0
    assert cap["coverage"]["omitted_fragment_chars"] == 0
    larger = constrained.span_pool(
        " ".join(f"Sentence {i}." for i in range(constrained.MAX_SPANS + 1)),
        "native_text",
    )
    assert larger["coverage"]["omitted"]["pool_cap"] == 1
    with pytest.raises(constrained.NoEligibleSpans, match="coverage"):
        constrained.decode_schema(
            _request(" ".join(f"Sentence {i}." for i in range(constrained.MAX_SPANS + 1)))
        )
    huge = constrained.span_pool("a" * (constrained.MAX_SOURCE_CHARS + 1), "native_text")
    assert huge["coverage"]["oversized_source"] is True
    assert huge["spans"] == []
    assert huge["coverage"]["omitted_fragment_chars"] == constrained.MAX_SOURCE_CHARS + 1


def test_pool_identity_and_nonselected_quote_cannot_pass_decoder_grammar() -> None:
    text = "One supported point. Another exact point."
    req, schema = _decode_request(text)
    original = content_fingerprint(req)
    req["constraints"]["spans"][0]["text"] = "invented quote"
    with pytest.raises(constrained.ConstraintError, match="invalid canonical source span request"):
        constrained.decode_schema(req)
    assert content_fingerprint(req) != original
    _req, schema = _decode_request(text)
    ids = schema["properties"]["lessons"]["items"]["properties"]
    assert ids["evidence"] == {"type": "string"}
    assert constrained.span_pool(text, "native_text")["spans"][1]["text"] == (
        "Another exact point."
    )
    assert ids["evidence_field"]["enum"] == ["native_text"]


def test_unavailable_optional_decoder_and_bad_requests_fail_without_script_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(constrained.ConstraintError):
        constrained.decode_schema({"constraints": {}, "messages": []})
    with pytest.raises(constrained.ConstraintError):
        constrained.decode_schema(_request("Text.") | {"constraints": {}})
    monkeypatch.setattr(
        constrained, "version", Mock(side_effect=constrained.PackageNotFoundError())
    )
    with pytest.raises(constrained.DecoderUnavailable, match="book-constrained"):
        constrained._decoder()
    # Script-only requests do not depend on book-specific decoder imports or fields.
    assert "constraints" not in scripting._script_request(
        scripting.NicheConfig(
            niche="Books",
            audience="Adults",
            brand="Test",
            voice="am_adam",
            facts=(),
            topic="A synthetic example",
        )
    )


def test_released_parser_rejects_nonselected_quotes_extra_keys_and_bounds() -> None:
    text = "A supported sentence. Another supported sentence!"
    _req, _schema = _decode_request(text)
    spans = constrained.span_pool(text, "native_text")["spans"]
    lesson = {
        "id": spans[0]["id"],
        "topic": "A test",
        "claim": "A synthetic finding",
        "hook_angle": "Inspect the steps",
        "visual_action": "An actor points at a card",
        "evidence": spans[0]["text"],
        "evidence_field": "native_text",
    }

    def rejected(raw: str) -> None:
        parser = _guard(_request(text))
        for c in raw:
            if c not in parser.get_allowed_characters():
                return
            parser = parser.add_character(c)
        assert not parser.can_end(), raw

    rejected(json.dumps([lesson]))
    rejected(json.dumps({"wrong": []}))
    rejected(json.dumps({"lessons": [lesson], "extra": 1}))
    rejected(json.dumps({"lessons": [lesson] * 4}))
    rejected(json.dumps({"lessons": [{**lesson, "extra": "bad"}]}))
    rejected(json.dumps({"lessons": [{**lesson, "id": "anchor:0"}]}))
    rejected(json.dumps({"lessons": [{**lesson, "evidence": "A supported"}]}))
    rejected(json.dumps({"lessons": [{**lesson, "evidence": "fabricated"}]}))
    rejected(json.dumps({"lessons": [{**lesson, "evidence_field": "wrong"}]}))
    rejected(json.dumps({"lessons": [{**lesson, "claim": "x" * 401}]}))


def test_tokenizer_vocabulary_cache_is_identity_scoped_and_parser_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = _request("One short source statement.")
    data = _synthetic_tokens()

    class Tokenizer:
        pass

    class FakeIntegration:
        calls: int = 0

        def build_token_enforcer_tokenizer_data(self, _tokenizer: object) -> object:
            self.calls += 1
            return data

        def build_transformers_prefix_allowed_tokens_fn(
            self, tokenizer_data: object, parser: object
        ) -> object:
            enforcer = TokenEnforcer(tokenizer_data, parser)
            return lambda _batch, ids: enforcer.get_allowed_tokens(ids).allowed_tokens

    fake = FakeIntegration()
    monkeypatch.setattr(
        constrained,
        "_decoder",
        lambda: (
            SimpleNamespace(
                JsonSchemaParser=JsonSchemaParser, CharacterLevelParser=CharacterLevelParser
            ),
            fake,
        ),
    )
    constrained._tokenizer_data.cache_clear()
    try:
        tokenizer = Tokenizer()
        first = constrained.prefix_for(req, tokenizer, "pinned-revision")
        second = constrained.prefix_for(req, tokenizer, "pinned-revision")
        assert first is not second
        assert fake.calls == 1
        assert first(0, [999]) == second(0, [999])
        constrained.prefix_for(req, tokenizer, "different-revision")
        assert fake.calls == 2
    finally:
        constrained._tokenizer_data.cache_clear()


def test_decoder_wrong_version_and_missing_transformers_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(constrained, "version", lambda _package: "0.11.2")
        with pytest.raises(constrained.DecoderUnavailable, match="version"):
            constrained._decoder()

    def not_available(_name: str) -> object:
        raise ImportError("synthetic missing optional module")

    with monkeypatch.context() as patch:
        patch.setattr(constrained, "version", lambda _package: "0.11.3")
        patch.setattr(constrained, "importlib", SimpleNamespace(import_module=not_available))
        with pytest.raises(constrained.DecoderUnavailable, match="integration"):
            constrained._decoder()


def test_no_eligible_span_rejects_before_any_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scripting, "_script_pipeline", lambda: pytest.fail("no model may load"))
    with pytest.raises(constrained.NoEligibleSpans):
        scripting._local_book_completion(_request("X" * 401))


@pytest.mark.parametrize(
    "source",
    [
        'The quoted "qualification" is essential.',
        r"The literal backslash \ remains essential.",
        "A control tab\tand newline\nretain exact meaning.",
        "Unicode café 漢字 العربية remains exact.",
        "A curly quote \u2018and\u2019 emoji 🚀 remain verbatim.",
    ],
)
def test_released_json_parser_accepts_exact_escaped_source_literal(
    source: str,
) -> None:
    request = _request(source)
    pool = request["constraints"]
    assert len(pool["spans"]) == 1
    assert pool["coverage"]["omitted_fragment_chars"] == 0
    lesson = {
        "id": pool["spans"][0]["id"],
        "topic": "A synthetic test",
        "claim": "This is a mechanical parser test",
        "hook_angle": "A reviewer checks a quote",
        "visual_action": "Show a source excerpt",
        "evidence": source,
        "evidence_field": "native_text",
    }
    raw = json.dumps({"lessons": [lesson]}, ensure_ascii=True, separators=(",", ":"))
    _drive(_guard(request), raw)
    assert whole_book.load_strict_reply(raw)["lessons"][0]["evidence"] == source
    assert json.dumps(source, ensure_ascii=True)[1:-1] in raw
    assert set(json.loads(raw)) == {"lessons"}
    # A convenient but nonselected quote remains disallowed at character level.
    corrupted = dict(lesson, evidence="made up source evidence")
    mutated = json.dumps({"lessons": [corrupted]})
    parser = _guard(request)
    for character in mutated:
        if character not in parser.get_allowed_characters():
            break
        parser = parser.add_character(character)
    else:
        assert not parser.can_end()


def test_real_token_enforcer_accepts_escaped_literal_with_synthetic_tokens() -> None:
    request = _request('The "critical" note includes \\\\ slash and café.')
    pool = request["constraints"]
    lesson = {
        "id": pool["spans"][0]["id"],
        "topic": "Test",
        "claim": "A test for the parser",
        "hook_angle": "Inspect evidence",
        "visual_action": "A note on screen",
        "evidence": pool["spans"][0]["text"],
        "evidence_field": "native_text",
    }
    raw = json.dumps({"lessons": [lesson]}, ensure_ascii=True, separators=(",", ":"))
    tok_data = _synthetic_tokens()
    enforcer = TokenEnforcer(tok_data, _guard(request))
    by_char = {value: i for i, value, _ in tok_data.regular_tokens}
    sequence = [999]
    for char in raw:
        assert by_char[char] in enforcer.get_allowed_tokens(sequence).allowed_tokens
        sequence.append(by_char[char])
    assert tok_data.eos_token_id in enforcer.get_allowed_tokens(sequence).allowed_tokens


def test_private_coverage_record_reports_separators_and_boundaries() -> None:
    pool = constrained.span_pool('A quoted "caveat". Another precise claim.', "native_text")
    report = constrained.coverage_record(pool)
    assert report["source_chars"] == len('A quoted "caveat". Another precise claim.')
    assert report["coverage"]["split_separator_chars"] == 1
    assert report["coverage"]["omitted_fragment_chars"] == 0
    assert report["selectable_spans"][0]["start"] == 0
    assert report["selectable_spans"][-1]["end"] == report["source_chars"]
    assert all("text" not in span for span in report["selectable_spans"])
    total = constrained.coverage_summary([report])
    assert (
        total["covered_fragment_chars"] + total["split_separator_chars"] == (total["source_chars"])
    )
    assert total["units_fail_closed_for_coverage"] == 0
    assert total["unit_start_context_unknown"] == 1
    assert total["unit_end_context_unknown"] == 1


def test_exact_pairs_rejected_in_both_field_orders_and_raw_validator() -> None:
    request = _request("First source. Second source.")
    spans = request["constraints"]["spans"]
    first, second = spans[0], spans[1]
    assert first["id"] != second["id"]
    selected = {
        "id": first["id"],
        "topic": "A topic",
        "claim": "An authored claim",
        "hook_angle": "A source hook",
        "visual_action": "Draw two cards",
        "evidence": first["text"],
        "evidence_field": "native_text",
    }
    constrained.validate_local_choice([selected], request)
    wrong = {**selected, "evidence": second["text"]}
    with pytest.raises(constrained.ConstraintError, match="pair"):
        constrained.validate_local_choice([wrong], request)
    with pytest.raises(constrained.ConstraintError, match="pair"):
        constrained.selected_span(wrong, request)
    for candidate in (
        wrong,
        {
            "evidence": wrong["evidence"],
            **{key: value for key, value in wrong.items() if key != "evidence"},
        },
    ):
        raw = json.dumps({"lessons": [candidate]}, separators=(",", ":"))
        parser = _guard(request)
        for char in raw:
            if char not in parser.get_allowed_characters():
                break
            parser = parser.add_character(char)
        else:
            assert not parser.can_end(), "mismatched pair must not decode"
    # The same quote occurs twice, with distinct *declared* occurrence IDs.
    duplicated = constrained.span_pool("Repeat. Repeat.", "native_text")
    assert len(duplicated["spans"]) == 2
    assert len(duplicated["duplicates"]) == 1
    assert duplicated["spans"][0]["id"] != duplicated["spans"][1]["id"]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Dr. Ada met Prof. Lin. The result followed.", 2),
        ("U.S. trials succeeded. Dr. Lin agreed.", 2),
        ("A. Smith reported a result. Another result followed.", 2),
        ("第一句。第二句\uff01第三句\uff1f", 3),
        ("الجملة الأولى؟الجملة الثانية؛الجملة الثالثة.", 3),
        ("سؤال؟ جواب.", 2),
    ],
)
def test_multilingual_boundaries_and_abbreviations(source: str, expected: int) -> None:
    pool = constrained.span_pool(source, "native_text")
    assert len(pool["spans"]) == expected
    assert pool["coverage"]["omitted_fragment_chars"] == 0
    assert sum(len(s["text"]) for s in pool["spans"]) + (
        pool["coverage"]["split_separator_chars"]
    ) == len(source)
    assert pool["coverage"]["generation_budget_status"] == "unverified_per_span_and_full_reply"


def test_true_350_word_upper_bound_and_unresolved_long_fragment() -> None:
    # Upstream WorkUnit text is exactly 350 whitespace-delimited words.
    source = " ".join("a." for _ in range(350))
    pool = constrained.span_pool(source, "native_text")
    assert len(pool["spans"]) == 350 < constrained.MAX_SPANS
    assert pool["coverage"]["omitted_fragment_chars"] == 0
    large = constrained.span_pool("x" * 401 + ". Another.", "native_text")
    assert large["coverage"]["omitted"]["overlong"] == 1
    assert large["coverage"]["fail_closed_for_coverage"] is True
    with pytest.raises(constrained.NoEligibleSpans):
        constrained.decode_schema(_request("x" * 401 + ". Another."))


@pytest.mark.parametrize(
    "source",
    [
        'Quoted "qualification" with a slash \\ and an emoji 🧪.',
        "Unicode café 漢字 العربية remains exact.",
        "Control\\tand\\nencoded sequences remain exact.",
    ],
)
def test_direct_and_ascii_wire_representations_decode_to_same_source(source: str) -> None:
    request = _request(source)
    span = request["constraints"]["spans"][0]
    lesson = {
        "id": span["id"],
        "topic": "Test",
        "claim": "An authored claim",
        "hook_angle": "Show source",
        "visual_action": "A source card",
        "evidence": span["text"],
        "evidence_field": "native_text",
    }
    for ascii_only in (False, True):
        raw = json.dumps({"lessons": [lesson]}, ensure_ascii=ascii_only)
        _drive(_guard(request), raw)
        assert whole_book.load_strict_reply(raw)["lessons"][0] == lesson


def test_incomplete_generation_raw_is_not_a_valid_completion() -> None:
    raw = '{"lessons":[]}'
    failure = scripting.IncompleteConstrainedResponse(raw)
    assert failure.raw == raw
    assert "without EOS" in str(failure)
    assert not isinstance(failure, str)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("it was I. Then he left.", 2),
        ("it was I. then he left.", 2),
        ("the plan B. Then we left.", 2),
        ("the plan B. then we left.", 2),
        ("He said no. Then we left.", 2),
        ("He said no. then we left.", 2),
        ("See No. 5 for the details. Then leave.", 2),
        ("Dr. Ada saw A. Smith. Then she left.", 2),
        ("U.S. trials succeeded. Dr. Lin agreed.", 2),
        ("第一句。第二句\uff01第三句\uff1f", 3),
        ("سؤال؟ جواب. ثم غادر.", 3),
    ],
)
def test_sentence_final_initials_no_and_multilingual_boundaries(source: str, expected: int) -> None:
    pool = constrained.span_pool(source, "native_text")
    assert len(pool["spans"]) == expected
    assert pool["coverage"]["omitted_fragment_chars"] == 0
    assert sum(len(span["text"]) for span in pool["spans"]) + (
        pool["coverage"]["split_separator_chars"]
    ) == len(source)


def test_character_mixed_escaped_literal_is_one_unchanged_cjk_source() -> None:
    source = "漢字語彙" * 97 + "."
    request = _request(source)
    assert len(request["constraints"]["spans"]) == 1
    span = request["constraints"]["spans"][0]
    lesson = {
        "id": span["id"],
        "topic": "One test",
        "claim": "This is a synthetic claim",
        "hook_angle": "Display test evidence",
        "visual_action": "Point at a source card",
        "evidence": source,
        "evidence_field": "native_text",
    }
    escaped_character = json.dumps("彙", ensure_ascii=True)[1:-1]
    raw = json.dumps({"lessons": [lesson]}, ensure_ascii=False, separators=(",", ":"))
    mixed = raw.replace("彙", escaped_character)
    assert "漢字語" in mixed
    assert escaped_character in mixed
    assert mixed != raw
    assert whole_book.load_strict_reply(mixed) == {"lessons": [lesson]}
    _drive(_guard(request), mixed)
    constrained.validate_local_choice([lesson], request)
    assert request["constraints"]["coverage"]["verified_feasible_span_count"] is None


@pytest.mark.parametrize("field", ["topic", "claim", "hook_angle", "visual_action"])
@pytest.mark.parametrize("blank", ["   ", "\\u0020\\u0020"])
def test_public_grammar_blocks_creative_whitespace_only(field: str, blank: str) -> None:
    request = _request("A sentence of exact source evidence.")
    span = request["constraints"]["spans"][0]
    lesson = {
        "id": span["id"],
        "topic": "Topic",
        "claim": "A claim",
        "hook_angle": "Inspect the claim",
        "visual_action": "Show two cards",
        "evidence": span["text"],
        "evidence_field": "native_text",
    }
    raw = json.dumps({"lessons": [lesson]}, ensure_ascii=True)
    original = json.dumps(lesson[field])
    rewritten = raw.replace('"' + field + '": ' + original, '"' + field + '": "' + blank + '"')
    assert rewritten != raw
    parser = _guard(request)
    for character in rewritten:
        if character not in parser.get_allowed_characters():
            break
        parser = parser.add_character(character)
    else:
        assert not parser.can_end(), "whitespace-only creative field must not finish"


def test_duplicate_ids_rejected_in_grammar_and_independent_raw_choice() -> None:
    request = _request("Same quote. Same quote.")
    first, second = request["constraints"]["spans"]
    assert first["text"] == second["text"] and first["id"] != second["id"]

    def candidate(span: dict[str, object]) -> dict[str, str]:
        return {
            "id": str(span["id"]),
            "topic": "Test",
            "claim": "Same exact sentence",
            "hook_angle": "Consider the sentence",
            "visual_action": "Show a document",
            "evidence": str(span["text"]),
            "evidence_field": "native_text",
        }

    distinct = [candidate(first), candidate(second)]
    _drive(_guard(request), json.dumps({"lessons": distinct}))
    constrained.validate_local_choice(distinct, request)
    duplicated = [candidate(first), candidate(first)]
    with pytest.raises(constrained.ConstraintError, match="duplicate"):
        constrained.validate_local_choice(duplicated, request)
    raw = json.dumps({"lessons": duplicated})
    parser = _guard(request)
    for char in raw:
        if char not in parser.get_allowed_characters():
            break
        parser = parser.add_character(char)
    else:
        assert not parser.can_end(), "a second use of the same ID must be blocked"


def test_coverage_counts_never_claim_unverified_feasibility() -> None:
    record = constrained.coverage_record(constrained.span_pool("One. Two.", "native_text"))
    assert len(record["selectable_spans"]) == 2
    assert record["coverage"]["source_selectable_span_count"] == 2
    assert record["coverage"]["verified_feasible_span_count"] is None
    assert record["coverage"]["verified_feasible_full_reply_count"] is None
    summary = constrained.coverage_summary([record])
    assert summary["selectable_span_count"] == 2
    assert summary["verified_feasible_span_count"] is None


def test_persistent_mixed_literal_union_on_repeated_cjk_spans() -> None:
    source = " ".join("漢字語彙" * 96 + f"{i:03d}." for i in range(24))
    request = _request(source)
    spans = request["constraints"]["spans"]
    assert len(spans) == 24
    assert request["constraints"]["coverage"]["omitted_fragment_chars"] == 0
    first = spans[0]
    lesson = {
        "id": first["id"],
        "topic": "Test",
        "claim": "Authored synthetic claim",
        "hook_angle": "Check the source",
        "visual_action": "Display evidence",
        "evidence": first["text"],
        "evidence_field": "native_text",
    }
    mixed = json.dumps({"lessons": [lesson]}, ensure_ascii=False).replace(
        "彙", json.dumps("彙", ensure_ascii=True)[1:-1]
    )
    _drive(_guard(request), mixed)
    assert whole_book.load_strict_reply(mixed) == {"lessons": [lesson]}
    # A non-source literal must still fail despite sharing the CJK prefix.
    bad = json.dumps({"lessons": [{**lesson, "evidence": "漢字語彙" * 96 + "999."}]})
    parser = _guard(request)
    for character in bad:
        if character not in parser.get_allowed_characters():
            break
        parser = parser.add_character(character)
    else:
        assert not parser.can_end()


def test_released_token_enforcer_cannot_cache_away_choice_restrictions() -> None:
    request = _request("Repeat. Repeat.")
    first, second = request["constraints"]["spans"]
    assert first["id"] != second["id"]
    tokens = _synthetic_tokens()
    by_character = {char: identifier for identifier, char, _ in tokens.regular_tokens}

    def reachable(raw: str) -> bool:
        enforcer = TokenEnforcer(tokens, _guard(request))
        sequence = [999]
        for char in raw:
            if by_character[char] not in enforcer.get_allowed_tokens(sequence).allowed_tokens:
                return False
            sequence.append(by_character[char])
        return tokens.eos_token_id in enforcer.get_allowed_tokens(sequence).allowed_tokens

    def lesson(span: dict[str, object]) -> dict[str, str]:
        return {
            "id": str(span["id"]),
            "topic": "Test",
            "claim": "Exact quote",
            "hook_angle": "Inspect evidence",
            "visual_action": "Show a card",
            "evidence": str(span["text"]),
            "evidence_field": "native_text",
        }

    first_lesson, second_lesson = lesson(first), lesson(second)
    assert reachable(json.dumps({"lessons": [first_lesson, second_lesson]}))
    assert not reachable(json.dumps({"lessons": [first_lesson, first_lesson]}))
    assert not reachable(json.dumps({"lessons": [{**first_lesson, "topic": "  "}]}))


@pytest.mark.parametrize(
    "source,expected",
    [
        ("it was I. Xylophone arrived.", 2),
        ("it was I. Zelda arrived.", 2),
        ("the plan B. Xylophone arrived.", 2),
        ("the plan B. Zelda arrived.", 2),
        ("Dr. Then reported a result. Zelda left.", 2),
        ("Dr. Ada saw A. Smith. Zoe followed.", 2),
        ("B. Smith went home. Another visitor arrived.", 2),
        ("John B. Smith visited. Another person arrived.", 2),
        ("The U.S. Army gathered. Another person arrived.", 2),
        ("The U.S. economy grew. Another person arrived.", 2),
        ("See No. 5 in the table. Then continue.", 2),
        ("He answered no. Zelda left.", 2),
        ("They left etc. Zelda returned.", 2),
        ('He said "I." Zelda left.', 2),
        ('He said "no." Zelda left.', 2),
        ("第一句。第二句\uff01第三句\uff1f", 3),
        ("السؤال؟ الجواب؛ ثم غادر.", 3),
    ],
)
def test_contextual_initial_and_quoted_punctuation_boundaries(source: str, expected: int) -> None:
    pool = constrained.span_pool(source, "native_text")
    assert len(pool["spans"]) == expected
    coverage = pool["coverage"]
    assert coverage["omitted_fragment_chars"] == 0
    assert sum(len(span["text"]) for span in pool["spans"]) + (
        coverage["split_separator_chars"]
    ) == len(source)
    for span in pool["spans"]:
        assert source[span["start"] : span["end"]] == span["text"]


def test_full_chat_context_budget_fail_closed_with_synthetic_tokenizer() -> None:
    request = _request("Synthetic source.")
    request["max_new_tokens"] = 960

    class SyntheticTokenizer:
        def __init__(self, prompt_length: int) -> None:
            self.prompt_length = prompt_length

        def apply_chat_template(self, messages: object, **options: object) -> list[int]:
            assert messages == request["messages"]
            assert options == {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return [1] * self.prompt_length

    assert constrained.check_prompt_context(request, SyntheticTokenizer(40_000), 40_960) == 40_000
    with pytest.raises(constrained.ConstraintError, match="exceeds context"):
        constrained.check_prompt_context(request, SyntheticTokenizer(40_001), 40_960)
    with pytest.raises(constrained.ConstraintError, match="pinned book prompt budget"):
        constrained.check_prompt_context(
            {**request, "max_new_tokens": 959}, SyntheticTokenizer(1), 40_960
        )


def test_oversized_full_prompt_is_rejected_before_book_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("Synthetic source sentence.")
    request["max_new_tokens"] = 960

    class OversizedPublicTokenizer:
        def apply_chat_template(self, _messages: object, **_kwargs: object) -> list[int]:
            return [1] * 40_001

    monkeypatch.setattr(
        scripting, "_public_book_context", lambda: (OversizedPublicTokenizer(), 40_960)
    )
    monkeypatch.setattr(
        scripting,
        "_script_pipeline",
        lambda: pytest.fail("oversized canonical prompt may not load model weights"),
    )
    with pytest.raises(constrained.ConstraintError, match="exceeds context"):
        scripting._local_book_completion(request)


@pytest.mark.parametrize(
    "fault",
    ["none", "wrong-hash", "wrong-model-limit", "wrong-tokenizer-limit", "missing-public"],
)
def test_public_book_context_preflight_is_pinned_and_weight_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:

    snapshot = tmp_path / "models--Qwen--Qwen3-1.7B" / "snapshots" / scripting.SCRIPT_REVISION
    snapshot.mkdir(parents=True)
    synthetic_tokenizer_bytes = b'{"synthetic":"no model, no weights"}'
    (snapshot / "tokenizer.json").write_bytes(synthetic_tokenizer_bytes)
    model_limit = 40_959 if fault == "wrong-model-limit" else 40_960
    tokenizer_limit = 131_071 if fault == "wrong-tokenizer-limit" else 131_072
    (snapshot / "config.json").write_text(json.dumps({"max_position_embeddings": model_limit}))
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": tokenizer_limit})
    )

    download_calls: list[dict[str, object]] = []

    def public_download(**kwargs: object) -> str:
        download_calls.append(kwargs)
        if fault == "missing-public":
            raise OSError("synthetic public tokenizer unavailable")
        return str(snapshot)

    class FakeTokenizer:
        model_max_length = 131_072
        eos_token_id = 151645

    def fake_tokenizer_load(location: object, *, local_files_only: bool) -> object:
        assert location == snapshot
        assert local_files_only
        return FakeTokenizer()

    original_import = scripting.importlib.import_module

    def injected(name: str) -> object:
        if name == "huggingface_hub":
            return SimpleNamespace(snapshot_download=public_download)
        if name == "transformers":
            return SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=fake_tokenizer_load)
            )
        return original_import(name)

    expected_hash = hashlib.sha256(synthetic_tokenizer_bytes).hexdigest()
    scripting._public_book_context.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setenv("HF_HUB_OFFLINE", "1")
            patch.setattr(scripting.importlib, "import_module", injected)
            patch.setattr(
                constrained,
                "PINNED_TOKENIZER_SHA256",
                "0" * 64 if fault == "wrong-hash" else expected_hash,
            )
            if fault == "none":
                public_tokenizer, limit = scripting._public_book_context()
                assert public_tokenizer.eos_token_id == 151645
                assert limit == 40_960
            else:
                with pytest.raises(scripting.PinnedQwenSnapshotError):
                    scripting._public_book_context()
    finally:
        scripting._public_book_context.cache_clear()

    assert download_calls
    assert all(call["repo_id"] == scripting.SCRIPT_MODEL for call in download_calls)
    assert all(call["revision"] == scripting.SCRIPT_REVISION for call in download_calls)
    assert all(call["local_files_only"] is True for call in download_calls)
    assert all(
        "model.safetensors" not in cast(list[str], call["allow_patterns"])
        for call in download_calls
    )


def test_synthetic_public_book_prefix_wrapper_and_escape_paths() -> None:
    """Exercise production wrapper in ordinary pytest, with no models."""
    source = "A fully synthetic source sentence."
    request = _request(source)
    data = _synthetic_tokens()
    root = _guard(request)

    class Sent:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self) -> list[int]:
            return self.values

    class ReleasedAdapter:
        def __init__(self) -> None:
            self.token_enforcer = TokenEnforcer(data, root)

        def __call__(self, _batch_id: int, sent: Sent) -> list[int]:
            return self.token_enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

    by_char = {value: index for index, value, _new_word in data.regular_tokens}
    assert '"' in by_char
    assert constrained._blank_closing_token_ids(data)
    for topic in ("Visible", "\u00a0Visible"):
        # The second topic is an in-memory synthetic raw escape, not a source
        # rewrite. Its decoded value is nonblank and exactly validated.
        span = cast(list[dict[str, object]], request["constraints"]["spans"])[0]
        lesson: dict[str, str] = {
            "id": str(span["id"]),
            "topic": topic,
            "claim": "An authored synthetic claim",
            "hook_angle": "View the exact quote",
            "visual_action": "Show a synthetic card",
            "evidence": str(span["text"]),
            "evidence_field": "native_text",
        }
        raw = json.dumps({"lessons": [lesson]}, ensure_ascii=True, separators=(",", ":"))
        decoder = constrained._BlankSafePrefixFn(ReleasedAdapter(), root, data)
        state = [999]
        assert data.eos_token_id not in decoder(0, Sent(state))
        for character in raw:
            allowed = decoder(0, Sent(state))
            assert by_char[character] in allowed
            state.append(by_char[character])
        assert data.eos_token_id in decoder(0, Sent(state))
        parsed: Any = whole_book.load_strict_reply(raw)
        assert parsed == {"lessons": [lesson]}
        constrained.validate_local_choice(parsed["lessons"], request)


def test_blank_safe_public_wrapper_rejects_invalid_prefix_states_model_free() -> None:
    """Real public CharacterLevelParser; synthetic tokens and callbacks only."""
    request = _request("A synthetic exact source.")
    root = _guard(request)
    regular_tokens = [
        (9, '"', False),
        (10, '\\"', False),
        (11, '\t"', False),
    ]
    data = SimpleNamespace(regular_tokens=regular_tokens, decoder=lambda _ids: "")
    invalid = constrained._blank_closing_token_ids(data)
    assert {9, 11}.issubset(invalid)
    assert 10 not in invalid

    def make(decoder: Any) -> Any:
        tokenizer_data = SimpleNamespace(regular_tokens=regular_tokens, decoder=decoder)
        return constrained._BlankSafePrefixFn(
            lambda _batch, _tokens: [9, 10, 11], root, tokenizer_data
        )

    changed = make(lambda _ids: "")
    assert changed(0, [999]) == [9, 10, 11]
    with pytest.raises(constrained.ConstraintError, match="prompt changed"):
        changed(0, [998])

    invalid_decode = make(lambda _ids: 123)
    invalid_decode(0, [999])
    with pytest.raises(constrained.ConstraintError, match="decoder returned invalid"):
        invalid_decode(0, [999, 1])

    invalid_prefix = make(lambda _ids: "[")
    invalid_prefix(0, [999])
    with pytest.raises(constrained.ConstraintError, match="invalid generated book JSON prefix"):
        invalid_prefix(0, [999, 1])

    # Contextual decoding can retract a suffix. It must rebuild the public
    # parser rather than trust stale incremental state or repair generated raw.
    retract = make(lambda ids: "{" if ids == [1] else "")
    retract(0, [999])
    assert retract(0, [999, 1]) == [9, 10, 11]
    assert retract(0, [999, 1, 2]) == [9, 10, 11]

    # A candidate token with a quote may finish an illegal pending escape.
    # The filter must inspect that token's contextual decoded characters.
    partial = '{"lessons":[{"topic":"' + "\\"

    def decode_quote(ids: list[int]) -> str:
        if ids == [1]:
            return partial
        if ids == [1, 9]:
            return "not-a-prefix"
        return partial + "x"

    escaped = make(decode_quote)
    escaped(0, [999])
    allowed = escaped(0, [999, 1])
    assert 9 not in allowed
    assert 10 not in allowed


def test_full_context_preflight_rejects_malformed_public_encodings() -> None:
    request = _request("A synthetic exact source.")
    request["max_new_tokens"] = 960

    class Encoded:
        def __init__(self, result: object) -> None:
            self.result = result

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> object:
            return self.result

    for invalid in ("not-a-token-list", [1, "not-an-integer"]):
        with pytest.raises(constrained.ConstraintError, match="invalid pinned tokenizer"):
            constrained.check_prompt_context(request, Encoded(invalid), 40_960)
    with pytest.raises(constrained.ConstraintError, match="model context limit"):
        constrained.check_prompt_context(request, Encoded([1]), 960)
    with pytest.raises(constrained.ConstraintError, match="preflight failed"):
        constrained.check_prompt_context({"max_new_tokens": 960}, Encoded([1]), 40_960)


@pytest.mark.parametrize("fault", ["none", "partial", "missing", "stale"])
def test_explicit_replay_tokenizer_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    snapshot = tmp_path / "models--Qwen--Qwen3-1.7B" / "snapshots" / scripting.SCRIPT_REVISION
    snapshot.mkdir(parents=True)
    tokenizer_bytes = b'{"synthetic":"offline-only"}'
    for name in (
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_bytes(tokenizer_bytes)
    (snapshot / "config.json").write_text('{"max_position_embeddings":40960}')
    (snapshot / "tokenizer_config.json").write_text('{"model_max_length":131072}')
    if fault == "partial":
        (snapshot / "vocab.json").unlink()
    if fault == "stale":
        (snapshot / "tokenizer.json").write_bytes(b"changed")
    calls: list[bool] = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs["local_files_only"])
        assert kwargs["local_files_only"] is True
        if fault == "missing":
            raise OSError("no retained snapshot")
        return str(snapshot)

    tokenizer = SimpleNamespace(model_max_length=131072)
    load = Mock(return_value=tokenizer)
    original = scripting.importlib.import_module

    def imports(name: str) -> Any:
        if name == "huggingface_hub":
            return SimpleNamespace(snapshot_download=download)
        if name == "transformers":
            return SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load))
        return original(name)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(scripting.importlib, "import_module", imports)
    monkeypatch.setattr(
        constrained, "PINNED_TOKENIZER_SHA256", hashlib.sha256(tokenizer_bytes).hexdigest()
    )
    scripting._public_book_context.cache_clear()
    try:
        if fault == "none":
            assert scripting._public_book_context(local_files_only=True) == (tokenizer, 40960)
            load.assert_called_once_with(snapshot, local_files_only=True)
        else:
            with pytest.raises(scripting.PinnedQwenSnapshotError):
                scripting._public_book_context(local_files_only=True)
            load.assert_not_called()
    finally:
        scripting._public_book_context.cache_clear()
    assert calls == [True]


def test_replay_context_token_count_uses_retained_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = Mock(side_effect=AssertionError("network-capable context must not run"))
    monkeypatch.setattr(scripting, "_public_book_context", forbidden)
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [10, 20, 30])
    request = {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 1024,
        "messages": [],
    }
    assert scripting._chapter_context_tokens(request, tokenizer_context=(tokenizer, 40960)) == 3
    forbidden.assert_not_called()
