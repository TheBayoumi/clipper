"""Real released JSON/token parser tests; strictly no model or weight loading."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from types import SimpleNamespace
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
        ("第一句。第二句\uFF01第三句\uFF1F", 3),
        ("سؤال؟ جواب. ثم غادر.", 3),
    ],
)
def test_sentence_final_initials_no_and_multilingual_boundaries(
    source: str, expected: int
) -> None:
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
def test_public_grammar_blocks_creative_whitespace_only(
    field: str, blank: str
) -> None:
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
