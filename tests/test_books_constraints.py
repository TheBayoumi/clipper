"""Real released JSON/token parser tests; strictly no model or weight loading."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lmformatenforcer import JsonSchemaParser
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


def _drive(parser: JsonSchemaParser, value: str) -> JsonSchemaParser:
    for c in value:
        assert c in parser.get_allowed_characters(), (repr(c), value)
        parser = parser.add_character(c)
    assert parser.can_end()
    return parser


def _synthetic_tokens() -> TokenEnforcerTokenizerData:
    # A fully synthetic Unicode codepoint tokenizer, NOT Qwen or model output.
    alphabet = sorted(set('{}[]":, abcdefghijklmnopqrstuvwxyz0123456789-'))
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
    _drive(JsonSchemaParser(schema), json.dumps(data, ensure_ascii=False))
    with pytest.raises(mining.BookMiningError, match="lessons array"):
        whole_book.validate_reply(data["lessons"], SimpleNamespace())
    # Duplicate keys must be rejected even when plain json.loads would overwrite them.
    with pytest.raises(mining.BookMiningError, match="duplicate JSON"):
        whole_book.load_strict_reply('{"lessons":[],"lessons":[]}')
    with pytest.raises(mining.BookMiningError, match="duplicate JSON"):
        whole_book.load_strict_reply('{"lessons":[{"id":"first","id":"second"}]}')


def test_real_released_token_enforcer_restricts_root_and_terminal_eos() -> None:
    _request_data, schema = _decode_request("One short source statement.")
    data = _synthetic_tokens()
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
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
        ('A quoted "term" is unsupported by v1 enum.', 0),
        ("An escaped \\ fragment.", 0),
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
        assert schema["properties"]["lessons"]["items"]["properties"]["evidence"]["enum"] == [
            s["text"] for s in spans
        ]


def test_boundary_cap_duplicate_and_overlong_coverage_are_explicit() -> None:
    pool = constrained.span_pool("Repeated. Repeated. Tail without terminator", "native_text")
    assert len(pool["spans"]) == 2
    assert pool["coverage"]["omitted"]["duplicate"] == 1
    assert pool["spans"][-1]["boundary"] == "unit_end_may_be_incomplete"
    assert pool["spans"][0]["boundary"] == "unit_start_unknown"
    cap = constrained.span_pool(" ".join(f"Sentence {i}." for i in range(30)), "native_text")
    assert len(cap["spans"]) == constrained.MAX_SPANS
    assert cap["coverage"]["omitted"]["pool_cap"] == 6
    assert cap["coverage"]["omitted_fragment_chars"] > 0
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
    assert "invented quote" not in ids["evidence"]["enum"]
    assert "One supported" not in ids["evidence"]["enum"]
    assert "Another exact point." in ids["evidence"]["enum"]
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
    _req, schema = _decode_request(text)
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
        parser = JsonSchemaParser(schema)
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
        lambda: (SimpleNamespace(JsonSchemaParser=JsonSchemaParser), fake),
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


def test_incomplete_generation_raw_is_not_a_valid_completion() -> None:
    raw = '{"lessons":[]}'
    failure = scripting.IncompleteConstrainedResponse(raw)
    assert failure.raw == raw
    assert "without EOS" in str(failure)
    assert not isinstance(failure, str)
