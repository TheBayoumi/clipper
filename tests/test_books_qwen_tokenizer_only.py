"""Explicit pinned public tokenizer-only decoder compatibility test; no weights/models.

Run only with CLIPPER_TOKENIZER_ONLY_PROBE=1 on an ephemeral public CPU runner.
Nothing calls model.from_pretrained or generate; no tokenizer file is committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from clipper.books import constrained, scripting, whole_book
from clipper.books.contracts import BookText

PUBLIC_TOKENIZER_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
PINNED_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

pytestmark = pytest.mark.skipif(
    os.environ.get("CLIPPER_TOKENIZER_ONLY_PROBE") != "1",
    reason="Explicit public tokenizer-only compatibility probe is not enabled",
)


def _request(source: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "Synthetic unit; no generation"},
            {
                "role": "user",
                "content": json.dumps(
                    {"text": source, "evidence_field": "native_text"}, ensure_ascii=False
                ),
            },
        ],
        "constraints": constrained.span_pool(source, "native_text"),
    }


def _canonical_reply(source: str, request: dict[str, object]) -> str:
    spans = request["constraints"]["spans"]
    assert len(spans) == 1
    candidate = {
        "id": spans[0]["id"],
        "topic": "Authored parser test",
        "claim": "This artificial claim is not editorially approved",
        "hook_angle": "An authored visual demonstration",
        "visual_action": "Place a test card in view",
        "evidence": source,
        "evidence_field": "native_text",
    }
    return json.dumps({"lessons": [candidate]}, ensure_ascii=True, separators=(",", ":"))


def _pool_reply(request: dict[str, object], *, count: int, ascii_only: bool) -> str:
    spans = request["constraints"]["spans"]
    assert len(spans) >= count
    lessons = [
        {
            "id": span["id"],
            "topic": "Authored test",
            "claim": "A bounded synthetic claim",
            "hook_angle": "Present one claim",
            "visual_action": "Show source text",
            "evidence": span["text"],
            "evidence_field": "native_text",
        }
        for span in spans[:count]
    ]
    return json.dumps({"lessons": lessons}, ensure_ascii=ascii_only, separators=(",", ":"))


def _bounded_alternative_token_probe(
    callback: Any,
    tokenizer: Any,
    torch: Any,
    target: str,
    accepted: list[int],
) -> dict[str, object]:
    """Depth-two token reachability only; NEVER proof of a complete reply path."""

    def decoded(ids: list[int]) -> str:
        # LMFE's published adapter drops trailing Unicode replacement
        # characters when decoding incomplete byte-level token sequences.
        return tokenizer.decode(
            ids[1:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).rstrip("\ufffd")

    base = decoded(accepted)
    if not target.startswith(base):
        return {"prefix_matches": False}
    first_allowed = callback(0, torch.tensor(accepted))
    explored_first = min(len(first_allowed), 2048)
    candidates: list[tuple[int, int]] = []
    for token in first_allowed[:explored_first]:
        next_text = decoded([*accepted, token])
        if next_text.startswith(base) and target.startswith(next_text):
            candidates.append((token, len(next_text) - len(base)))
    progressed_first = sum(advance > 0 for _, advance in candidates)
    progressed_second = 0
    explored_second = 0
    for first_token, _advance in candidates[:12]:
        extended = [*accepted, first_token]
        allowed_second = callback(0, torch.tensor(extended))
        for token in allowed_second[:512]:
            explored_second += 1
            next_text = decoded([*extended, token])
            if target.startswith(next_text) and len(next_text) > len(base):
                progressed_second += 1
    required_char = target[len(base) : len(base) + 1]
    tokenizer_data = callback.token_enforcer
    regular = tokenizer_data.regular_tokens
    return {
        "prefix_matches": True,
        "required_next_char_codepoint": (ord(required_char) if required_char else None),
        "parser_alphabet_contains_next": (
            required_char in callback.token_enforcer.root_parser.config.alphabet
        ),
        "registered_tokens_starting_with_next_char": sum(
            token_string.startswith(required_char) for _, token_string, _ in regular
        ),
        "first_allowed_tokens": len(first_allowed),
        "first_scanned": explored_first,
        "prefix_compatible_first_choices": len(candidates),
        "first_choices_advancing_text": progressed_first,
        "second_scanned": explored_second,
        "second_choices_advancing_text": progressed_second,
        "max_depth": 2,
        "exhaustive": False,
        "proves_complete_reply_path": False,
    }


def test_real_pinned_qwen_tokenizer_and_prefix_adapter_without_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor

    assert version("torch") == "2.6.0+cpu"
    assert version("transformers") == "4.57.3"
    assert version("lm-format-enforcer") == "0.11.3"

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("model creation, download, inference and generation are forbidden")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(scripting, "_script_pipeline", forbidden)

    with tempfile.TemporaryDirectory(prefix="clipper-tokenizer-only-") as tmp:
        start = time.perf_counter()
        snapshot = Path(
            snapshot_download(
                repo_id=scripting.SCRIPT_MODEL,
                revision=scripting.SCRIPT_REVISION,
                allow_patterns=list(PUBLIC_TOKENIZER_FILES),
                cache_dir=tmp,
                local_files_only=False,
            )
        )
        actual_files = {p.name for p in snapshot.iterdir() if p.is_file()}
        assert actual_files == set(PUBLIC_TOKENIZER_FILES)
        assert not any(p.suffix in {".safetensors", ".bin", ".pt"} for p in snapshot.rglob("*"))
        assert hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest() == (
            PINNED_TOKENIZER_SHA256
        )
        fetched_seconds = time.perf_counter() - start
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        assert tokenizer.eos_token_id == 151645
        model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads(
            (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        generation_config = json.loads(
            (snapshot / "generation_config.json").read_text(encoding="utf-8")
        )
        assert model_config["max_position_embeddings"] == 40960
        assert tokenizer_config["model_max_length"] == 131072
        assert generation_config["eos_token_id"] == [151645, 151643]
        effective_context = min(
            model_config["max_position_embeddings"], tokenizer_config["model_max_length"]
        )
        assert effective_context == 40960
        assert constrained.PINNED_TOKENIZER_SHA256 == PINNED_TOKENIZER_SHA256
        # Exercise the ACTUAL preweights public-only loader with these eight
        # pinned public files, not a fake tokenizer and never a model object.
        original_import = scripting.importlib.import_module
        public_fetch_calls: list[bool] = []

        def public_hub_snapshot(**kwargs: Any) -> str:
            assert kwargs["repo_id"] == scripting.SCRIPT_MODEL
            assert kwargs["revision"] == scripting.SCRIPT_REVISION
            assert set(kwargs["allow_patterns"]) == set(PUBLIC_TOKENIZER_FILES)
            public_fetch_calls.append(kwargs["local_files_only"])
            return str(snapshot)

        def public_modules(name: str, package: str | None = None) -> Any:
            if name == "huggingface_hub":
                from types import SimpleNamespace

                return SimpleNamespace(snapshot_download=public_hub_snapshot)
            return original_import(name, package)

        scripting._public_book_context.cache_clear()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(scripting.importlib, "import_module", public_modules)
                checked_tokenizer, checked_limit = scripting._public_book_context()
                assert checked_limit == effective_context
                assert checked_tokenizer.eos_token_id == tokenizer.eos_token_id
                assert public_fetch_calls == [True]
        finally:
            scripting._public_book_context.cache_clear()

        # Test admission BEFORE consuming a token, not parser rejection after.
        # Token 330 is the pinned tokenizer's merged space + closing quote.
        assert (
            tokenizer.decode([330], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            == ' "'
        )
        blank_request = _request("A synthetic exact source sentence.")
        template = _canonical_reply("A synthetic exact source sentence.", blank_request)

        def reach_public_state(callback: Any, wire: str) -> tuple[list[int], float]:
            """Walk ACTUAL generated tokens from the prompt, never prime with a full prefix."""
            sent = [tokenizer.eos_token_id]
            callback(0, torch.tensor(sent))
            traversal_started = time.perf_counter()
            ids = tokenizer.encode(wire, add_special_tokens=False)
            assert (
                tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                == wire
            )
            for position, token in enumerate(ids):
                allowed = callback(0, torch.tensor(sent))
                assert token in allowed, (position, token, len(sent))
                assert tokenizer.eos_token_id not in allowed
                sent.append(token)
            return sent, time.perf_counter() - traversal_started

        # These twelve states are reached incrementally through the released
        # adapter. A one-shot full-prefix first callback only initializes the
        # adapter's prompt state and does NOT test the blank-field guard.
        for creative in ("topic", "claim", "hook_angle", "visual_action"):
            marker = '"' + creative + '":"'
            assert marker in template
            for blank_count in (0, 1, 2):
                partial = template.split(marker, 1)[0] + marker + " " * blank_count
                setup_started = time.perf_counter()
                fresh = constrained.prefix_for(blank_request, tokenizer, scripting.SCRIPT_REVISION)
                setup_seconds = time.perf_counter() - setup_started
                sent, traversal_seconds = reach_public_state(fresh, partial)
                started = time.perf_counter()
                allowed = fresh(0, torch.tensor(sent))
                callback_seconds = time.perf_counter() - started
                assert 330 not in allowed, (creative, blank_count)
                assert tokenizer.eos_token_id not in allowed
                assert callback_seconds < 5, "reached blank callback latency regression"
                assert not any(
                    quote_token in allowed
                    for quote_token in tokenizer.encode('"', add_special_tokens=False)
                )
                # Scoped: the fast path must remain usable after actual
                # nonblank text, not merely reject bad blank closings.
                if blank_count == 0:
                    positive = constrained.prefix_for(
                        blank_request, tokenizer, scripting.SCRIPT_REVISION
                    )
                    positive_sent, _ = reach_public_state(positive, partial + "meaningful")
                    assert 330 in positive(0, torch.tensor(positive_sent))
                print(
                    "QWEN_REACHED_CREATIVE_BLANK_STATE "
                    + json.dumps(
                        {
                            "field": creative,
                            "blanks": blank_count,
                            "setup_seconds": round(setup_seconds, 4),
                            "prefix_traversal_seconds": round(traversal_seconds, 4),
                            "blank_callback_seconds": round(callback_seconds, 4),
                            "merged_quote_admitted": False,
                            "model_loaded": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        # JSON escaped SPACE and NBSP are semantically whitespace even though
        # their raw spelling contains letters/digits; they may not close empty.
        for encoded_space in (r"\u0020", r"\u00a0"):
            marker = '"topic":"'
            partial = template.split(marker, 1)[0] + marker + encoded_space
            fresh = constrained.prefix_for(blank_request, tokenizer, scripting.SCRIPT_REVISION)
            sent, _ = reach_public_state(fresh, partial)
            started = time.perf_counter()
            allowed = fresh(0, torch.tensor(sent))
            elapsed = time.perf_counter() - started
            assert 330 not in allowed and tokenizer.eos_token_id not in allowed
            print(
                "QWEN_ESCAPED_WHITESPACE_BLANK_STATE "
                + json.dumps(
                    {"space": encoded_space, "callback_seconds": round(elapsed, 4)},
                    sort_keys=True,
                ),
                flush=True,
            )

        constrained._tokenizer_data.cache_clear()
        cases = {
            "ascii": "One concise evidence statement.",
            "quoted": 'The "qualification" must remain within the evidence.',
            "escape_control": "The path C:\\work\\notes includes a tab\tand newline\n.",
            "mixed_script": "Unicode café 漢字 العربية remains exact.",
            "emoji_curly": "Curly \u2018quote\u2019 and emoji 🧪 remain unchanged.",
        }
        for case, source in cases.items():
            request = _request(source)
            raw = _canonical_reply(source, request)
            expected = whole_book.load_strict_reply(raw)
            assert expected["lessons"][0]["evidence"] == source
            assert constrained.decode_schema(request)
            setup = time.perf_counter()
            callback = constrained.prefix_for(request, tokenizer, scripting.SCRIPT_REVISION)
            build_seconds = time.perf_counter() - setup
            assert callback.__class__.__name__ == "_BlankSafePrefixFn"
            # The model prompt is a sentinel, not a generated token.
            prefix_ids = [tokenizer.eos_token_id]
            open_ids = tokenizer.encode("[", add_special_tokens=False)
            initial = callback(0, torch.tensor(prefix_ids))
            assert not any(token in initial for token in open_ids)
            assert tokenizer.eos_token_id not in initial

            tokens = tokenizer.encode(raw, add_special_tokens=False)
            assert (
                tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                == raw
            )
            first_prefix = time.perf_counter()
            for index, token in enumerate(tokens):
                allowed = callback(0, torch.tensor(prefix_ids))
                assert token in allowed, (
                    f"{case}: token {index} id {token} rejected after "
                    f"{len(tokenizer.decode(prefix_ids[1:], skip_special_tokens=True))} chars"
                )
                prefix_ids.append(token)
            elapsed = time.perf_counter() - first_prefix
            assert tokenizer.eos_token_id in callback(0, torch.tensor(prefix_ids))
            assert elapsed < 60, f"{case}: prefix filter took too long: {elapsed:.2f}s"
            raw_candidate = tokenizer.decode(
                prefix_ids[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assert raw_candidate == raw
            constrained.validate_local_choice(expected["lessons"], request)
            # Use the actual Transformers logits processor with fake zero scores only.
            processor = PrefixConstrainedLogitsProcessor(callback, num_beams=1)
            start_logits = torch.zeros((1, len(tokenizer)), dtype=torch.float32)
            masked = processor(torch.tensor([prefix_ids]), start_logits)
            assert torch.isfinite(masked[0, tokenizer.eos_token_id])
            print(
                "QWEN_TOKENIZER_ONLY_PROBE "
                + json.dumps(
                    {
                        "case": case,
                        "model_initialized": False,
                        "weights_loaded": False,
                        "generate_called": False,
                        "real_adapter": True,
                        "revision": scripting.SCRIPT_REVISION,
                        "tokenizer_sha256": PINNED_TOKENIZER_SHA256,
                        "tokens": len(tokens),
                        "initial_tokenizer_fetch_seconds": round(fetched_seconds, 3),
                        "parser_adapter_setup_seconds": round(build_seconds, 3),
                        "prefix_filter_seconds": round(elapsed, 3),
                        "per_token_ms": round(elapsed * 1000 / max(1, len(tokens)), 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        # Diagnostic-only: test the actual pinned adapter with unconstrained
        # schema evidence spelling, not a simulated codepoint tokenizer.
        # This cannot certify that the production exact-choice overlay works.
        _, integration = constrained._decoder()
        import lmformatenforcer

        long_sources = {
            "cjk_400": ("漢字語彙" * 100)[:400],
            "arabic_400": ("العربيةبيانات" * 40)[:400],
        }
        for language, source in long_sources.items():
            request = _request(source)
            schema = constrained.decode_schema(request)
            lesson = _canonical_reply(source, request)
            for spelling, ascii_only in (("direct", False), ("escaped", True)):
                output = json.dumps(
                    json.loads(lesson), ensure_ascii=ascii_only, separators=(",", ":")
                )
                tokens = tokenizer.encode(output, add_special_tokens=False)
                adapter = integration.build_transformers_prefix_allowed_tokens_fn(
                    constrained._tokenizer_data(tokenizer, scripting.SCRIPT_REVISION),
                    lmformatenforcer.JsonSchemaParser(schema),
                )
                prefix = [tokenizer.eos_token_id]
                rejected_at: int | None = None
                t0 = time.perf_counter()
                for i, token in enumerate(tokens[:961]):
                    if token not in adapter(0, torch.tensor(prefix)):
                        rejected_at = i
                        break
                    prefix.append(token)
                seconds = time.perf_counter() - t0
                accepted = (
                    rejected_at is None
                    and len(tokens) <= 960
                    and tokenizer.eos_token_id in adapter(0, torch.tensor(prefix))
                )
                assert (
                    tokenizer.decode(
                        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    == output
                )
                print(
                    "QWEN_TOKENIZER_ONLY_BUDGET_DIAGNOSTIC "
                    + json.dumps(
                        {
                            "source": language,
                            "spelling": spelling,
                            "adapter": "actual pinned tokenizer + released JsonSchemaParser",
                            "evidence_chars": len(source),
                            "wire_bytes": len(output.encode("utf-8")),
                            "whole_reply_tokens": len(tokens),
                            "fits_960": len(tokens) <= 960,
                            "canonical_token_path_accepted": accepted,
                            "first_rejected_token": rejected_at,
                            "prefix_check_seconds": round(seconds, 3),
                            "weights_loaded": False,
                            "generate_called": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        for count, span_length in ((24, 390), (400, 12)):
            fragments = [(f"{i:03d}" + "漢字" * span_length)[:span_length] for i in range(count)]
            pool = constrained.span_pool(
                " ".join(fragment + "." for fragment in fragments), "native_text"
            )
            print(
                "QWEN_TOKENIZER_ONLY_POOL_DIAGNOSTIC "
                + json.dumps(
                    {
                        "source_fragment_count": count,
                        "span_length_target": span_length,
                        "selected_span_count": len(pool["spans"]),
                        "source_chars": pool["source_chars"],
                        "excluded_fragment_chars": pool["coverage"]["omitted_fragment_chars"],
                        "max_spans": constrained.MAX_SPANS,
                        "weights_loaded": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        # Production callback, not a plain-schema proxy: 24 near-max spans,
        # representative 350-word/350-span bound, and one/three lesson replies.
        # No logits are sampled and no model is initialized.
        def near_max_span(language: str, number: int) -> str:
            base = {
                "latin": "reproducibleevidence",
                "cjk": "漢字語彙",
                "arabic": "العربيةبيانات",
            }[language]
            body = (base * 400)[:386]
            return body + f"{number:03d}."

        cases = {
            language: " ".join(near_max_span(language, index) for index in range(24))
            for language in ("latin", "cjk", "arabic")
        }
        cases["bounded_350_spans"] = " ".join(f"authored{index:03d}." for index in range(350))
        cases["bounded_400_spans"] = " ".join(f"authored{index:03d}." for index in range(400))
        cases["mixed_emoji_combining"] = " ".join(
            f'Source {i} combines café é, 🧪, "quotation", and Arabic العربية.' for i in range(24)
        )
        for case_name, source in cases.items():
            request = _request(source)
            pool = request["constraints"]
            expected_span_count = {
                "bounded_400_spans": 400,
                "bounded_350_spans": 350,
            }.get(case_name, 24)
            assert len(pool["spans"]) == expected_span_count
            assert pool["coverage"]["omitted_fragment_chars"] == 0
            # Measure cold vocabulary construction separately from warm/fresh
            # parser construction; never confuse per-prefix cache with cold cost.
            constrained._tokenizer_data.cache_clear()
            cold_start = time.perf_counter()
            cold_callback = constrained.prefix_for(request, tokenizer, scripting.SCRIPT_REVISION)
            cold_time = time.perf_counter() - cold_start
            setup = time.perf_counter()
            callback = constrained.prefix_for(request, tokenizer, scripting.SCRIPT_REVISION)
            build_time = time.perf_counter() - setup
            assert callback is not cold_callback

            # Use the real canonical production prompt, including context and the
            # COMPLETE source-choice menu, not the short synthetic parser request.
            unit = whole_book.WorkUnit(
                ordinal=0,
                anchor="synthetic-only",
                section_id="test",
                source_path="synthetic.txt",
                word_offset=0,
                evidence_field="native_text",
                origin="native_text",
                review_status="unreviewed",
                asset_sha256=None,
                research_link_status="unreviewed",
                text=source,
            )
            book_stub = cast(
                BookText,
                SimpleNamespace(
                    source=SimpleNamespace(title="Synthetic Title", author="Synthetic Author"),
                    edition_identity="synthetic-edition",
                    content_sha256=hashlib.sha256(source.encode()).hexdigest(),
                ),
            )
            full_request = whole_book._request_for_identity(
                book_stub, unit, "synthetic-fingerprint"
            )
            prompt_ids = tokenizer.apply_chat_template(
                full_request["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert len(prompt_ids) + 960 <= effective_context
            assert constrained.check_prompt_context(
                full_request, tokenizer, effective_context
            ) == len(prompt_ids)
            print(
                "QWEN_FULL_CANONICAL_PROMPT_BUDGET "
                + json.dumps(
                    {
                        "case": case_name,
                        "source_chars": len(source),
                        "selectable_spans": expected_span_count,
                        "prompt_tokens": len(prompt_ids),
                        "reserved_reply_tokens": 960,
                        "effective_context": effective_context,
                        "cold_adapter_seconds": round(cold_time, 3),
                        "warm_fresh_adapter_seconds": round(build_time, 3),
                        "model_initialized": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            parser_alphabet = callback.token_enforcer.root_parser.config.alphabet
            if case_name == "cjk":
                assert "漢" in parser_alphabet
            if case_name == "arabic":
                assert "ع" in parser_alphabet
            for count in (1, 3):
                for ascii_only in (False, True):
                    raw = _pool_reply(request, count=count, ascii_only=ascii_only)
                    tokens = tokenizer.encode(raw, add_special_tokens=False)
                    token_count = len(tokens)
                    raw_bytes = len(raw.encode("utf-8"))
                    wire_roundtrip = (
                        tokenizer.decode(
                            tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        == raw
                    )
                    if not wire_roundtrip:
                        # The actual tokenizer NFC-normalizes direct decomposed
                        # accents. Never accept a changed source or post-repair it;
                        # the explicit JSON Unicode-escape spelling stays exact.
                        assert case_name == "mixed_emoji_combining" and not ascii_only
                        print(
                            "QWEN_NONEXACT_CANONICAL_WIRE_REJECTED "
                            + json.dumps(
                                {"case": case_name, "lesson_count": count, "ascii": ascii_only},
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        continue
                    assert (
                        whole_book.load_strict_reply(raw)["lessons"] == (json.loads(raw)["lessons"])
                    )
                    allowed = token_count <= 960 and raw_bytes <= 16_384
                    fresh_start = time.perf_counter()
                    callback = constrained.prefix_for(request, tokenizer, scripting.SCRIPT_REVISION)
                    fresh_path_setup = time.perf_counter() - fresh_start
                    if allowed:
                        prefix = [tokenizer.eos_token_id]
                        tested = 0
                        first_rejected: int | None = None
                        start = time.perf_counter()
                        for index, token in enumerate(tokens):
                            next_allowed = callback(0, torch.tensor(prefix))
                            if token not in next_allowed:
                                first_rejected = index
                                break
                            prefix.append(token)
                            tested += 1
                            if time.perf_counter() - start > 25:
                                break
                        seconds = time.perf_counter() - start
                        complete = (
                            tested == token_count
                            and first_rejected is None
                            and tokenizer.eos_token_id in callback(0, torch.tensor(prefix))
                        )
                    else:
                        tested, first_rejected, seconds, complete = 0, None, 0.0, False
                    print(
                        "QWEN_PRODUCTION_POOL_BUDGET "
                        + json.dumps(
                            {
                                "case": case_name,
                                "pool_spans": expected_span_count,
                                "source_chars": len(source),
                                "lessons": count,
                                "ascii_escaped": ascii_only,
                                "token_count": token_count,
                                "raw_bytes": raw_bytes,
                                "fits_960_tokens": token_count <= 960,
                                "fits_16k_raw": raw_bytes <= 16_384,
                                "adapter_setup_seconds": round(build_time, 3),
                                "fresh_path_setup_seconds": round(fresh_path_setup, 3),
                                "cross_variant_callback_reused": False,
                                "cold_setup_seconds": round(cold_time, 3),
                                "per_checked_token_ms": round(seconds * 1000 / max(tested, 1), 2),
                                "prefix_checked": tested,
                                "prefix_seconds": round(seconds, 3),
                                "first_rejected_token": first_rejected,
                                "complete_path": complete,
                                "model_loaded": False,
                                "generated": False,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if (
                        case_name == "cjk"
                        and count == 1
                        and not ascii_only
                        and first_rejected is not None
                    ):
                        probe = _bounded_alternative_token_probe(
                            callback, tokenizer, torch, raw, prefix
                        )
                        print(
                            "QWEN_PRODUCTION_ALT_PATH_DIAGNOSTIC "
                            + json.dumps(probe, sort_keys=True),
                            flush=True,
                        )
                    if (
                        case_name in {"latin", "arabic", "bounded_350_spans", "bounded_400_spans"}
                        and not ascii_only
                    ) or (case_name == "mixed_emoji_combining" and ascii_only):
                        assert allowed and complete
            if case_name == "cjk":
                # The pinned vocabulary has no complete token containing 彙.
                # Mixing only its JSON Unicode escape with direct 漢字語
                # produces a genuine full seven-field tokenizable reply.
                mixed = _pool_reply(request, count=1, ascii_only=False).replace(
                    "彙", json.dumps("彙", ensure_ascii=True)[1:-1]
                )
                assert "漢字語" in mixed and "\\u" in mixed
                mixed_tokens = tokenizer.encode(mixed, add_special_tokens=False)
                assert len(mixed_tokens) <= 960
                assert len(mixed.encode("utf-8")) <= 16_384
                mixed_callback = constrained.prefix_for(
                    request, tokenizer, scripting.SCRIPT_REVISION
                )
                prefix = [tokenizer.eos_token_id]
                started = time.perf_counter()
                for i, token in enumerate(mixed_tokens):
                    assert token in mixed_callback(0, torch.tensor(prefix)), (
                        "real adapter rejected the mixed CJK token at index " + str(i)
                    )
                    prefix.append(token)
                seconds = time.perf_counter() - started
                assert tokenizer.eos_token_id in mixed_callback(0, torch.tensor(prefix))
                assert whole_book.load_strict_reply(mixed) == json.loads(mixed)
                constrained.validate_local_choice(
                    whole_book.load_strict_reply(mixed)["lessons"], request
                )
                print(
                    "QWEN_PRODUCTION_MIXED_CJK_PROOF "
                    + json.dumps(
                        {
                            "source_chars": len(source),
                            "full_reply_tokens": len(mixed_tokens),
                            "raw_reply_bytes": len(mixed.encode("utf-8")),
                            "prefix_seconds": round(seconds, 3),
                            "full_path_accepted": True,
                            "final_eos_allowed": True,
                            "source_exact": True,
                            "model_loaded": False,
                            "generate_called": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        wide_request = _request("Fully synthetic evidence only.")
        wide_span = wide_request["constraints"]["spans"][0]
        wide_text = ("Detailed fabricated perspective and specific steps. " * 20)[:400]
        assert len(wide_text) == 400
        wide_lesson = {
            "id": wide_span["id"],
            "topic": wide_text,
            "claim": wide_text,
            "hook_angle": wide_text,
            "visual_action": wide_text,
            "evidence": wide_span["text"],
            "evidence_field": "native_text",
        }
        wide_raw = json.dumps({"lessons": [wide_lesson]}, separators=(",", ":"))
        wide_tokens = tokenizer.encode(wide_raw, add_special_tokens=False)
        assert len(wide_tokens) <= 960
        assert len(wide_raw.encode("utf-8")) <= 16_384
        assert (
            tokenizer.decode(
                wide_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            == wide_raw
        )
        # The pinned tokenizer's greedy encoding uses token 15553 for
        # space+quote+comma+quote, which is not a legal single JSON-string
        # token to LMFE. A tokenization path is not an authored raw reply:
        # preserve EVERY raw byte while splitting only this token boundary.
        assert (
            tokenizer.decode([15553], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            == ' ","'
        )
        replacement = [
            330,
            *tokenizer.encode(',"', add_special_tokens=False),
        ]
        assert (
            tokenizer.decode(
                replacement, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            == ' ","'
        )
        assert 15553 in wide_tokens
        alternate_tokens = [
            part for token in wide_tokens for part in (replacement if token == 15553 else [token])
        ]
        assert len(alternate_tokens) <= 960
        assert (
            tokenizer.decode(
                alternate_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            == wide_raw
        )
        wide_callback = constrained.prefix_for(wide_request, tokenizer, scripting.SCRIPT_REVISION)
        wide_prefix = [tokenizer.eos_token_id]
        wide_start = time.perf_counter()
        for index, token in enumerate(alternate_tokens):
            assert token in wide_callback(0, torch.tensor(wide_prefix)), index
            wide_prefix.append(token)
        wide_seconds = time.perf_counter() - wide_start
        assert tokenizer.eos_token_id in wide_callback(0, torch.tensor(wide_prefix))
        assert whole_book.load_strict_reply(wide_raw) == {"lessons": [wide_lesson]}
        constrained.validate_local_choice([wide_lesson], wide_request)
        print(
            "QWEN_FOUR_400_CHAR_FIELDS "
            + json.dumps(
                {
                    "whole_reply_tokens": len(alternate_tokens),
                    "canonical_boundary_token_rejected": 15553,
                    "raw_json_unchanged": True,
                    "prefix_seconds": round(wide_seconds, 3),
                    "complete_path": True,
                    "model_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        oversize = {
            **full_request,
            "messages": [
                {"role": "system", "content": "long synthetic " * 50_000},
                {"role": "user", "content": "unchanged exact source"},
            ],
        }
        with pytest.raises(constrained.ConstraintError, match="exceeds context"):
            constrained.check_prompt_context(oversize, tokenizer, effective_context)

        # Exactly TWO unchanged English synthetic fixture units. This is a
        # model-free local grammar test, NOT a third CPU model run or a book.
        fixture_root = Path(__file__).resolve().parents[1] / "acceptance/book-local-cpu-smoke"
        fixture_bytes = (fixture_root / "public-original-synthetic.txt").read_bytes()
        consumed_bytes = (fixture_root / "run-request.json").read_bytes()

        def git_blob(value: bytes) -> str:
            payload = b"blob " + str(len(value)).encode() + b"\0" + value
            return hashlib.sha1(payload, usedforsecurity=False).hexdigest()

        assert git_blob(fixture_bytes) == "07ff45bac97ebef4c73cc8804db78bd4425aa5c2"
        assert git_blob(consumed_bytes) == "5ba1c40d28833e4993d1f017dfa925ec784bbf19"
        consumed = json.loads(consumed_bytes)
        assert consumed["max_new_units"] == 2
        assert hashlib.sha256(fixture_bytes).hexdigest() == consumed["fixture_sha256"]
        original_sections = fixture_bytes.decode("utf-8").split("\n\n")
        assert original_sections[0] == "PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK"
        assert len(original_sections) == 3
        fixture_book = cast(
            BookText,
            SimpleNamespace(
                source=SimpleNamespace(
                    title="PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK",
                    author="Synthetic test only",
                ),
                edition_identity="synthetic-fixture",
                content_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
            ),
        )
        for index, text_unit in enumerate(original_sections[1:]):
            unit = whole_book.WorkUnit(
                ordinal=index,
                anchor=f"synthetic-unit-{index}",
                section_id=f"synthetic-{index}",
                source_path="public-original-synthetic.txt",
                word_offset=0,
                evidence_field="native_text",
                origin="native_text",
                review_status="unreviewed",
                asset_sha256=None,
                research_link_status="unreviewed",
                text=text_unit,
            )
            actual_request = whole_book._request_for_identity(
                fixture_book, unit, "public-synthetic-fixture"
            )
            pool = actual_request["constraints"]
            assert len(pool["spans"]) > 0
            assert pool["coverage"]["omitted_fragment_chars"] == 0
            assert pool["coverage"]["fail_closed_for_coverage"] is False
            prompt_count = constrained.check_prompt_context(
                actual_request, tokenizer, effective_context
            )
            # This is an explicitly authored seven-field candidate, never
            # host-repaired output or a model claim of editorial support.
            raw = _pool_reply(actual_request, count=1, ascii_only=True)
            raw_bytes = raw.encode("utf-8")
            ids = tokenizer.encode(raw, add_special_tokens=False)
            assert len(ids) <= 960 and len(raw_bytes) <= 16_384
            assert (
                tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                == raw
            )
            callback = constrained.prefix_for(actual_request, tokenizer, scripting.SCRIPT_REVISION)
            candidate_sent = [tokenizer.eos_token_id]
            callback(0, torch.tensor(candidate_sent))
            started = time.perf_counter()
            for pos, token in enumerate(ids):
                allowed = callback(0, torch.tensor(candidate_sent))
                assert token in allowed, (index, pos, token)
                assert tokenizer.eos_token_id not in allowed, (index, pos)
                candidate_sent.append(token)
            full_path_seconds = time.perf_counter() - started
            assert tokenizer.eos_token_id in callback(0, torch.tensor(candidate_sent))
            payload = whole_book.load_strict_reply(raw)
            assert len(payload["lessons"]) == 1
            assert len(payload["lessons"][0]) == 7
            validated = whole_book.validate_reply(payload, unit)
            constrained.validate_local_choice(validated, actual_request)
            assert payload == json.loads(raw)
            print(
                "QWEN_UNCHANGED_TWO_UNIT_FIXTURE_MODEL_FREE "
                + json.dumps(
                    {
                        "unit_index": index,
                        "source_chars": len(text_unit),
                        "eligible_spans": len(pool["spans"]),
                        "omitted_chars": 0,
                        "prompt_tokens": prompt_count,
                        "reply_tokens": len(ids),
                        "raw_reply_bytes": len(raw_bytes),
                        "full_callback_seconds": round(full_path_seconds, 4),
                        "canonical_eos_accepted": True,
                        "seven_fields_unchanged": True,
                        "model_loaded": False,
                        "generate_called": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        constrained._tokenizer_data.cache_clear()
