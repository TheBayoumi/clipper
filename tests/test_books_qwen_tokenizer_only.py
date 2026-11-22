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
PINNED_TOKENIZER_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)

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
            "emoji_curly": "Curly ‘quote’ and emoji 🧪 remain unchanged.",
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
            assert tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ) == raw
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
        constrained._tokenizer_data.cache_clear()
