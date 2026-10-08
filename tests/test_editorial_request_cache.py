"""The packaged request cache preserves exact evidence without model coupling."""

from __future__ import annotations

import hashlib
import json

import pytest

from clipper.editorial_request_cache import EditorialRequestCache


def make_cache(path, factory, *, replay_only=False, recorded_runtime=None):
    return EditorialRequestCache(
        path,
        factory,
        {"source_sha256": "a" * 64},
        draft_completion=lambda editor, prompt, payload, tokens: editor.draft(
            prompt, payload, tokens
        ),
        json_implementation=lambda *_: None,
        stage_fingerprint=lambda *_: "unchanged-stage",
        replay_only=replay_only,
        recorded_runtime=recorded_runtime,
    )


def test_package_cache_persists_and_replays_draft_without_model_weights(tmp_path):
    class Editor:
        def draft(self, prompt, payload, tokens):
            assert (prompt, payload, tokens) == ("Source first", {"units": ["Speech"]}, 32)
            return "Source-grounded notes"

    path = tmp_path / "requests.json"
    first = make_cache(path, Editor)
    assert first.semantic_draft("Source first", {"units": ["Speech"]}, 32) == (
        "Source-grounded notes"
    )
    assert first.metrics["model_calls"] == 1
    warm = make_cache(
        path,
        lambda: pytest.fail("warm cache must not load weights"),
        replay_only=True,
        recorded_runtime=first.identity["runtime"],
    )
    assert warm.semantic_draft("Source first", {"units": ["Speech"]}, 32) == (
        "Source-grounded notes"
    )
    assert warm.metrics["cache_hits"] == 1
    assert warm.metrics["model_calls"] == 0


def test_package_cache_records_failed_inference_without_persisting_false_response(tmp_path):
    class Editor:
        def _review_completion(self, prompt, payload, properties, tokens):
            raise RuntimeError("model assessment failed")

    path = tmp_path / "requests.json"
    cache = make_cache(path, Editor)
    with pytest.raises(RuntimeError, match="model assessment failed"):
        cache._review_completion("Check", {"units": ["Speech"]}, {"ok": {}}, 32)
    assert cache.calls[0]["error"] == "RuntimeError: model assessment failed"
    assert cache.metrics["model_calls"] == 1
    assert not path.exists()


def test_package_cache_move_preserves_legacy_selector_migration_identity():
    from scripts import tjr_semantic_editor as editor

    assert issubclass(editor.ReviewRequestCache, EditorialRequestCache)
    selector = editor._stage_fingerprint(
        editor.LocalContextualEditor.__init__,
        editor.LocalContextualEditor.__call__,
        editor.EDITOR_PROMPT,
        editor._selection_context,
        editor._thought_units,
        editor._closed_ending,
        editor.source_headline_candidates,
    )
    assert selector == editor._LEGACY_SELECTOR_FINGERPRINT


def test_compatibility_constructor_keeps_old_exact_request_key(tmp_path):
    from scripts import tjr_semantic_editor as editor

    class Reviewer:
        def _review_completion(self, prompt, payload, properties, tokens):
            return {"verdict": "uncertain"}

    cache = editor.ReviewRequestCache(
        tmp_path / "requests.json", Reviewer, {"source_sha256": "a" * 64}
    )
    request = {
        "method": "json",
        "prompt": "Check source",
        "payload": {"units": ["Exact speech"]},
        "properties": {"verdict": {"type": "string"}},
        "tokens": 64,
    }
    old_implementation = editor._stage_fingerprint(editor.LocalContextualEditor._review_completion)
    old_key = hashlib.sha256(
        json.dumps(
            {
                "identity": cache.identity,
                "implementation": old_implementation,
                "request": request,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert cache._review_completion(
        request["prompt"], request["payload"], request["properties"], request["tokens"]
    ) == {"verdict": "uncertain"}
    assert old_key in cache.records


def test_resuming_request_cache_prefers_current_checkpoint_over_older_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    class Editor:
        def __init__(self, value):
            self.value = value

        def draft(self, prompt, payload, tokens):
            return self.value

    prior = make_cache(baseline, lambda: Editor("prior evidence"))
    assert prior.semantic_draft("Check", {"source": "same"}, 32) == "prior evidence"
    newer = make_cache(current, lambda: Editor("resumed evidence"))
    assert newer.semantic_draft("Check", {"source": "same"}, 32) == "resumed evidence"
    resumed = EditorialRequestCache(
        current,
        lambda: pytest.fail("existing checkpoint must not load the model"),
        {"source_sha256": "a" * 64},
        draft_completion=lambda editor, prompt, payload, tokens: editor.draft(
            prompt, payload, tokens
        ),
        json_implementation=lambda *_: None,
        stage_fingerprint=lambda *_: "unchanged-stage",
        reuse_path=baseline,
        replay_only=True,
        recorded_runtime=newer.identity["runtime"],
    )
    assert resumed.semantic_draft("Check", {"source": "same"}, 32) == "resumed evidence"
    assert resumed.metrics["cache_hits"] == 1
    assert resumed.metrics["model_calls"] == 0
