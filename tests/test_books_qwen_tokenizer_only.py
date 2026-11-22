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
from typing import Any

import pytest

from clipper.books import constrained, scripting, whole_book

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
        assert tokenizer.eos_token_id is not None

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
            assert callback.__class__.__name__ == "TransformersPrefixAllowedTokensFn"
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
        for case_name, source in cases.items():
            request = _request(source)
            pool = request["constraints"]
            expected_span_count = 350 if case_name == "bounded_350_spans" else 24
            assert len(pool["spans"]) == expected_span_count
            assert pool["coverage"]["omitted_fragment_chars"] == 0
            setup = time.perf_counter()
            callback = constrained.prefix_for(request, tokenizer, scripting.SCRIPT_REVISION)
            build_time = time.perf_counter() - setup
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
                    assert (
                        tokenizer.decode(
                            tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        == raw
                    )
                    assert (
                        whole_book.load_strict_reply(raw)["lessons"] == (json.loads(raw)["lessons"])
                    )
                    allowed = token_count <= 960 and raw_bytes <= 16_384
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
                        case_name in {"latin", "arabic", "bounded_350_spans"}
                        and count == 1
                        and not ascii_only
                    ):
                        assert allowed and complete
        constrained._tokenizer_data.cache_clear()
