from clipper.models import CampaignBrief, TranscriptSegment
from scripts.tjr_semantic_editor import build_semantic_editorial_candidates


def _audit_reply(verdict, reason, unit_ids=None):
    return {
        "reason": reason,
        "evidence_unit_ids": [0] if unit_ids is None else unit_ids,
        "actor_action": verdict,
        "relationship_role": "supported",
        "setting_time": "supported",
        "quantities_outcomes": "supported",
    }


def test_inference_diagnostics_reuses_failures_and_never_approves(tmp_path, monkeypatch):
    import json
    from importlib import metadata

    from scripts import tjr_semantic_editor as editor

    requests = []

    class Model:
        chat_format = "chatml"

        def __init__(self):
            self.metadata = {"tokenizer.chat_template": "recorded template"}

        def create_chat_completion(self, **request):
            requests.append(request)
            return {"choices": [{"finish_reason": "stop", "message": {"content": "evidence"}}]}

    class FakeEditor:
        model = Model()

        def close(self):
            pass

    monkeypatch.setattr(editor, "LocalContextualEditor", FakeEditor)
    monkeypatch.setattr(metadata, "version", lambda name: "0.3.35")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "fixture": str(index),
                    "review_context": {
                        "selected_units": ["A delivered answer."],
                        "after": ["Excluded continuation."],
                    },
                    "review": {"boundary_audit": {"reason": "saved"}},
                }
                for index in range(3)
            ]
        )
    )
    output = tmp_path / "diagnostics.json"
    assert editor.reviewer_inference_diagnostics(baseline, output) == 0
    report = json.loads(output.read_text())
    assert report["production_approved"] is False
    assert report["chat_template"] == "recorded template"
    assert len(requests) == 12
    assert requests[0]["temperature"] == 0.7
    assert requests[0]["response_format"]["schema"]["properties"]["promotion_unit_ids"]
    assert "response_format" not in requests[1]
    assert "schema" in requests[1]["messages"][1]["content"]
    assert "response_format" in requests[2]
    assert "schema" in requests[2]["messages"][1]["content"]
    assert "response_format" not in requests[3]
    assert report["baseline"][0]["review"]["boundary_audit"]["reason"] == "saved"
    assert editor.reviewer_inference_diagnostics(output, tmp_path / "warm.json") == 0
    assert len(requests) == 12
    assert all(
        item["cache_hit"]
        for item in json.loads((tmp_path / "warm.json").read_text())["comparisons"]
    )
    report.pop("llama_cpp_python_version")
    report["comparisons"] = [
        item
        for item in report["comparisons"]
        if item["variant"] != "constrained_greedy_schema_in_prompt"
    ]
    output.write_text(json.dumps(report))
    assert editor.reviewer_inference_diagnostics(output, tmp_path / "extended.json") == 0
    assert len(requests) == 15
    report["comparisons"][0]["request"]["temperature"] = 0.6
    output.write_text(json.dumps(report))
    assert editor.reviewer_inference_diagnostics(output, tmp_path / "changed.json") == 0
    assert len(requests) == 19


def _brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="double-coverage-test",
        title="Double Coverage Podcast",
        objective="Find complete funny, surprising, memorable podcast moments",
        keywords=["podcast", "story", "reaction", "guest"],
        source_channel_ids=["UCf1q6dhccWr6eQEcFFnJSbA"],
        min_clip_seconds=20,
        max_clip_seconds=45,
        rights_confirmed=True,
    )


def test_model_probe_closes_thinking_and_preserves_production_init(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from scripts import tjr_semantic_editor as editor

    monkeypatch.setattr(editor.Path, "home", lambda: tmp_path)
    model_path = tmp_path / ".cache/clipper/editor/Qwen_Qwen3.5-4B-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    monkeypatch.setattr(
        editor.hashlib,
        "file_digest",
        lambda *args: SimpleNamespace(
            hexdigest=lambda: "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983"
        ),
    )
    requests = []
    inferences = []
    probe_prompt = {"value": "same speech-purpose task"}

    class Model:
        def __init__(self, **kwargs):
            requests.append(kwargs)
            self.metadata = {"tokenizer.chat_template": "template"}

        def token_eos(self):
            return 1

        def token_bos(self):
            return 2

        def detokenize(self, tokens, special):
            return b"token"

        def create_chat_completion(self, **kwargs):
            inferences.append(kwargs)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"reason": "Saved source evidence"})},
                    }
                ]
            }

        def close(self):
            pass

    class Formatter:
        def __init__(self, **kwargs):
            self.template = kwargs["template"]
            assert self.template.startswith("{% set enable_thinking = false %}")

        def __call__(self, **kwargs):
            return SimpleNamespace(prompt="assistant\n<think>\n\n</think>\n\n")

        def to_chat_handler(self):
            return "hard-non-thinking"

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(
            Llama=Model, llama_chat_format=SimpleNamespace(Jinja2ChatFormatter=Formatter)
        ),
    )
    monkeypatch.setattr(
        editor.LocalContextualEditor,
        "__init__",
        lambda self: (_ for _ in ()).throw(
            AssertionError("production model must not load during replacement probe")
        ),
    )

    def focused_review(self, context):
        self._review_completion(
            probe_prompt["value"], {"speech": "An answer."}, {"reason": {"type": "string"}}, 32
        )
        return {
            "opening_standalone": True,
            "payoff_complete": True,
            "ending_complete": True,
            "headline_supported": True,
            "headline_self_contained": True,
            "contains_promotion_or_intro": False,
        }

    monkeypatch.setattr(editor, "_focused_span_review", focused_review)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "fixture": str(i),
                    "review_context": {"selected_units": ["An answer."]},
                    "expected_accept": i == 0,
                    "expected_flags": {"payoff_complete": i == 0},
                }
                for i in range(3)
            ]
        )
    )
    output = tmp_path / "probe.json"
    assert editor.reviewer_model_probe(baseline, output) == 1
    report = json.loads(output.read_text())
    assert report["production_approved"] is False
    assert report["semantic_pass"] is False
    assert len(report["cases"]) == 3
    assert len(requests) == 1 and requests[0]["n_ctx"] == 4096
    assert len(inferences) == 3
    warm = tmp_path / "warm-probe.json"
    assert editor.reviewer_model_probe(output, warm) == 1
    assert len(inferences) == 3
    assert json.loads(warm.read_text())["request_cache_hits"] == 3
    probe_prompt["value"] = "changed speech-purpose task"
    assert editor.reviewer_model_probe(warm, tmp_path / "changed-probe.json") == 1
    assert len(inferences) == 6

    expanded = json.loads((tmp_path / "changed-probe.json").read_text())
    expanded["baseline"].append({**expanded["baseline"][0], "fixture": "extra"})
    extra = tmp_path / "expanded.json"
    extra.write_text(json.dumps(expanded))
    assert editor.reviewer_model_probe(extra, tmp_path / "expanded-result.json") == 1
    assert len(inferences) == 6
    assert json.loads((tmp_path / "expanded-result.json").read_text())["request_cache_hits"] == 4


def test_source_quote_provenance_crosses_units_without_changing_words_or_numbers():
    from scripts import tjr_semantic_editor as editor

    units = ["Do you get paid for views?", "Like, how is that?"]
    span = editor._source_quote_span("Do you get paid for views? Like, how is that.", units)
    assert span == {
        "text": "Do you get paid for views? Like, how is that",
        "first_unit": 0,
        "last_unit": 1,
    }
    money = ["I got 60 million views but earned nothing."]
    assert (
        editor._source_quote_span("I got 60 million views.", money)["text"]
        == "I got 60 million views"
    )
    assert editor._source_quote_span("I got 600 million views.", money) is None
    assert editor._source_quote_span("I earned 60 million views.", money) is None
    assert editor._source_quote_span("I got 60 million views.", ["Other source words."]) is None
    profanity = ["He said this was f***ing brilliant."]
    assert (
        editor._source_quote_span("this was f***ing brilliant", profanity)["text"]
        == "this was f***ing brilliant"
    )
    assert editor._source_quote_span("this was fucking brilliant", profanity) is None


def test_focused_review_separates_speech_purpose_and_excluded_payoff():
    import pytest

    from scripts import tjr_semantic_editor as editor

    context = {
        "selected_units": [
            "I had huge views on my social account.",
            "But I did not earn any money from those views.",
            "I used them to drive my podcast instead.",
        ],
        "before": ["An unrelated host ad read."],
        "after": ["Then another topic starts here."],
    }
    replies = [
        {
            "reason": "Personal earnings discussion without an ad read or show introduction.",
            "ad_read_quote": "",
            "show_intro_quote": "",
        },
        {
            "reason": "The clip resolves the views-versus-income contrast.",
            "setup_quote": "huge views on my social account",
            "resolution_quote": "drive my podcast instead",
            "opening_independent": 1,
            "last_thought_finished": 1,
        },
        {
            "reason": "The next topic is not needed for the delivered resolution.",
            "final_point": "Views drive the podcast",
            "final_point_quote": "drive my podcast instead",
            "continuation_quote": "Then another topic starts here",
            "relation": "new_topic",
        },
        {"headline": "Huge social views earned nothing but helped the podcast"},
    ]
    replies.append(_audit_reply("supported", "Views were not paid; they drove the podcast."))
    calls = []

    class FakeEditor:
        def _review_completion(self, prompt, payload, properties, tokens):
            calls.append(payload)
            return replies[len(calls) - 1]

    review = editor._focused_span_review(FakeEditor(), context)
    assert review["payoff_complete"] is True
    assert review["contains_promotion_or_intro"] is False
    assert review["boundary_audit"]["payoff_unit_id"] == 2
    assert set(calls[0]) == {"clip_transcript"}
    assert set(calls[1]) == {"clip_transcript"}
    assert list(calls[2])[-1] == "actual_final_delivered_text"
    assert "source_continuation_not_delivered" not in calls[3]
    assert "Then another topic" not in str(calls[3])
    assert calls[3]["source_passages"] == {
        "setup_quote": "huge views on my social account",
        "resolution_quote": "drive my podcast instead",
    }
    assert len(calls) == 5
    assert review["headline_source_spans"] == review["source_quote_spans"]
    replies[1]["resolution_quote"] = "another topic starts here"
    calls.clear()
    with pytest.raises(RuntimeError, match="not in delivered speech"):
        editor._focused_span_review(FakeEditor(), context)
    replies[1]["resolution_quote"] = "drive my podcast instead"
    replies[2]["relation"] = "missing_contrast"
    calls.clear()
    rejected = editor._focused_span_review(FakeEditor(), context)
    assert rejected["exchange_has_payoff"] is True
    assert rejected["ending_complete"] is False
    assert rejected["payoff_complete"] is False
    assert rejected["headline"] == ""
    assert len(calls) == 3
    assert "delivered_resolution_quote" not in calls[2]
    assert "delivered_setup_quote" not in calls[2]
    replies[2]["continuation_quote"] = "invented missing fact without source"
    calls.clear()
    with pytest.raises(RuntimeError, match="grounded boundary evidence"):
        editor._focused_span_review(FakeEditor(), context)


def _fake_embedder(texts: list[str]) -> list[list[float]]:
    from scripts.tjr_semantic_editor import CONTEXT_RUBRIC, HOOK_RUBRIC, OFF_TOPIC_DESCRIPTIONS

    negative_references = {
        *OFF_TOPIC_DESCRIPTIONS,
        HOOK_RUBRIC[1],
        *(pair[1] for pair in CONTEXT_RUBRIC.values()),
    }
    return [[0.0, 1.0, 0.0] if text in negative_references else [1.0, 0.0, 0.0] for text in texts]


def test_semantic_editor_builds_complete_story_without_sliding_window_quota() -> None:
    segments = [
        TranscriptSegment(0, 6, "The guest told us the wildest story from his first big show."),
        TranscriptSegment(6, 12, "Zach asked what happened when security suddenly stopped him."),
        TranscriptSegment(12, 18, "He admitted he had forgotten the one pass everyone needed."),
        TranscriptSegment(
            18,
            24,
            "Then the whole room laughed when he revealed how he got inside.",
        ),
    ]
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", segments, embedder=_fake_embedder
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert 20 <= candidate.duration <= 42
    assert "candidate_origin=podcast_contextual_editor" in candidate.reasons
    assert any(reason.startswith("semantic_event=") for reason in candidate.reasons)
    hook = next(
        reason.removeprefix("semantic_hook=")
        for reason in candidate.reasons
        if reason.startswith("semantic_hook=")
    )
    assert hook.upper() in candidate.text.upper()
    assert not hook.startswith("THE MOMENT:")
    assert audit["fixed_candidate_or_output_quota"] is False
    assert audit["architecture"] == "podcast_contextual_editor_v5"
    assert audit["campaign_keyword_gate"] is False
    assert any(reason.startswith("context_story_margin=") for reason in candidate.reasons)


def test_semantic_editor_returns_zero_when_source_has_no_units() -> None:
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", [], embedder=_fake_embedder
    )
    assert candidates == []
    assert audit["candidate_count"] == 0


def test_semantic_editor_does_not_reject_topics_absent_from_campaign_keywords() -> None:
    segments = [
        TranscriptSegment(0, 6, "This Chrome Hearts vest is one of my favorite pieces."),
        TranscriptSegment(6, 12, "The bracelet is twenty two carat gold and custom made."),
        TranscriptSegment(12, 18, "This Rolex is another watch from the collection."),
        TranscriptSegment(18, 24, "The leopard bag is the last thing I wanted to show."),
    ]
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", segments, embedder=_fake_embedder
    )
    assert candidates
    assert audit["campaign_keyword_gate"] is False
    assert audit["event_anchor_count"] > 0
    assert audit["fixed_candidate_or_output_quota"] is False


def test_question_window_keeps_setup_and_stops_before_new_question() -> None:
    from scripts.tjr_semantic_editor import SemanticUnit, _anchor_window

    units = [
        SemanticUnit(0, 6, "Yeah, that reminds me of something else."),
        SemanticUnit(6, 12, "How did your first show pay you?"),
        SemanticUnit(12, 20, "They promised a percentage of the ticket revenue."),
        SemanticUnit(20, 28, "The room sold out, but I received nothing."),
        SemanticUnit(28, 36, "What happened at your next show?"),
    ]
    assert _anchor_window(2, units, [[1.0]] * 5, [0] * 5, min_seconds=20, max_seconds=45) == (1, 3)
    assert _anchor_window(2, units, [[1.0]] * 5, [0] * 5, min_seconds=30, max_seconds=45) is None


def test_campaign_title_objective_and_keywords_never_enter_editor_model() -> None:
    from dataclasses import replace

    segments = [
        TranscriptSegment(i * 6, (i + 1) * 6, text)
        for i, text in enumerate(
            [
                "My first attempt reached an unexpected result.",
                "The measurement contradicted our explanation.",
                "We repeated the experiment with a better control.",
                "The result showed why the first assumption failed.",
            ]
        )
    ]
    seen = []

    def embed(texts):
        seen.extend(texts)
        return _fake_embedder(texts)

    brief = _brief()
    first, _ = build_semantic_editorial_candidates(brief, "v", segments, embedder=embed)
    second, _ = build_semantic_editorial_candidates(
        replace(
            brief,
            title="UNRELATED_CAMPAIGN_TITLE",
            objective="UNRELATED_CAMPAIGN_OBJECTIVE",
            keywords=["UNRELATED_CAMPAIGN_KEYWORD"],
        ),
        "v",
        segments,
        embedder=embed,
    )
    assert first == second and first
    assert all("UNRELATED_CAMPAIGN" not in text for text in seen)


def test_boundary_repair_preserves_anchor_and_removes_unresolved_tail() -> None:
    segments = [
        TranscriptSegment(i * 8, (i + 1) * 8, text)
        for i, text in enumerate(
            [
                "Our first measurement changed the entire explanation.",
                "The repeat measurement produced the same result.",
                "That result settled the question we had started with.",
                "Another unrelated question begins without its answer.",
            ]
        )
    ]
    whole = " ".join(segment.text for segment in segments)

    def embed(texts):
        return [
            [0.0, 1.0, 0.0] if text == whole else vector
            for text, vector in zip(texts, _fake_embedder(texts), strict=True)
        ]

    candidates, audit = build_semantic_editorial_candidates(_brief(), "v", segments, embedder=embed)
    assert candidates
    assert all(candidate.text != whole for candidate in candidates)
    assert audit["context_rejected_variants"] > 0
    assert audit["boundary_repaired_candidate_count"] > 0
    assert all(20 <= candidate.duration <= 45 for candidate in candidates)


def test_incomplete_context_batch_raises_instead_of_using_keyword_fallback() -> None:
    import pytest

    calls = 0

    def embed(texts):
        nonlocal calls
        calls += 1
        return _fake_embedder(texts) if calls == 1 else []

    segments = [
        TranscriptSegment(i * 6, (i + 1) * 6, text)
        for i, text in enumerate(
            [
                "The first result contradicted our initial explanation.",
                "A second measurement confirmed the unexpected finding.",
                "We tested the proposed explanation using a control.",
                "The control resolved why the original result was different.",
            ]
        )
    ]
    with pytest.raises(RuntimeError, match=r"context assessment.*incomplete batch"):
        build_semantic_editorial_candidates(_brief(), "v", segments, embedder=embed)


def test_invalid_embedding_vectors_fail_closed() -> None:
    import pytest

    from scripts.tjr_semantic_editor import _embedding_batch

    for vector in ([], [float("nan")], [float("inf")], [0.0]):
        with pytest.raises(RuntimeError, match="invalid embedding vectors"):
            _embedding_batch(
                lambda texts, item=vector: [item for _ in texts], ["source"], "context assessment"
            )


def test_closed_endings_reject_pauses_questions_and_ellipses() -> None:
    from scripts.tjr_semantic_editor import _closed_ending

    assert _closed_ending("That settled the question.")
    assert _closed_ending('He said, "That settled it."')
    assert not _closed_ending("Yeah, the only way is, like,")
    assert not _closed_ending("How does it work out for you?")
    assert not _closed_ending("I was going to say...")


def test_structured_editor_selects_complete_span_and_full_clip_hook(tmp_path) -> None:
    from clipper.models import ClipCandidate
    from scripts.tjr_semantic_editor import refine_contextual_candidates

    segments = [
        TranscriptSegment(0, 7, "How did security stop you at your own show?"),
        TranscriptSegment(7, 14, "I had left my own access pass in the car."),
        TranscriptSegment(14, 22, "The owner came outside and finally let me inside."),
        TranscriptSegment(22, 29, "How much did your next show pay you?"),
    ]
    calls = []

    def assess(context):
        calls.append(context)
        hook_index = next(
            item["id"] for item in context["headlines"] if item["text"] == segments[2].text.upper()
        )
        return dict(
            keep=True,
            start_unit=0,
            end_unit=2,
            hook_index=hook_index,
            opening=4,
            story=4,
            ending=5,
            hook=4,
            reason="Forgotten pass caused rejection; the owner finally let him into his own show.",
        )

    brief = _brief()
    proposal = ClipCandidate("v", 0, 29, " ".join(s.text for s in segments), 10)
    output = tmp_path / "cache.json"
    first, audit = refine_contextual_candidates(
        brief,
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=output,
        assessor=assess,
        reviewer=_review,
    )
    assert len(first) == 1 and first[0].start == 0 and first[0].end == 22
    assert "semantic_hook=SECURITY STOPPED THE PERFORMER AT HIS OWN SHOW" in first[0].reasons
    assert segments[3].text not in first[0].text
    assert audit["hook_scope"] == "entire_selected_exchange"
    second, reused = refine_contextual_candidates(
        brief,
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "reused.json",
        reuse_path=output,
        assessor=assess,
        reviewer=_review,
    )
    assert second == first and reused["cache_reused"] is True and len(calls) == 1


def test_structured_editor_rejects_incomplete_cut_and_unsupported_hook(tmp_path) -> None:
    import pytest

    from clipper.models import ClipCandidate
    from scripts.tjr_semantic_editor import refine_contextual_candidates

    segments = [
        TranscriptSegment(0, 12, "The first measurement contradicted our initial explanation."),
        TranscriptSegment(12, 24, "The repeat experiment confirmed the unexpected finding."),
        TranscriptSegment(24, 30, "The next reason for that was,"),
    ]
    proposal = ClipCandidate("v", 0, 30, " ".join(s.text for s in segments), 10)

    def assess(context):
        return dict(
            keep=True,
            start_unit=0,
            end_unit=2,
            hook_index=0,
            opening=4,
            story=4,
            ending=4,
            hook=4,
            reason="Unfinished proposed end.",
        )

    result, audit = refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "bad.json",
        assessor=assess,
        reviewer=_review,
    )
    assert result == []
    assert audit["assessments"][0]["rejection"] == "UNSUPPORTED_BOUNDARY_OR_HEADLINE"
    with pytest.raises(RuntimeError, match="keep flag"):
        refine_contextual_candidates(
            _brief(),
            [proposal],
            segments,
            source_sha256="a" * 64,
            cache_path=tmp_path / "invalid.json",
            assessor=lambda context: {"keep": "yes"},
        )


def test_local_editor_constrains_ranges_ratings_and_separates_headline_id() -> None:
    import json

    from scripts.tjr_semantic_editor import LocalContextualEditor

    class Model:
        def create_chat_completion(self, **kwargs):
            self.request = kwargs
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "keep": True,
                                    "window_index": 0,
                                    "headline_index": 0,
                                    "opening": 4,
                                    "story": 4,
                                    "ending": 5,
                                    "headline_quality": 4,
                                    "reason": "Missing pass; owner resolved access.",
                                }
                            )
                        },
                    }
                ]
            }

    context = {
        "min_seconds": 20,
        "max_seconds": 45,
        "units": [
            {"id": 101, "start": 100, "end": 107, "text": "Why did security stop you?"},
            {"id": 102, "start": 107, "end": 114, "text": "I forgot my pass in the car."},
            {"id": 103, "start": 114, "end": 122, "text": "The owner let me into my own show."},
            {"id": 104, "start": 122, "end": 129, "text": "What did your next show pay?"},
        ],
        "headlines": [{"id": 0, "text": "THE OWNER LET ME INTO MY OWN SHOW."}],
    }
    editor = object.__new__(LocalContextualEditor)
    editor.model = Model()
    decision = editor(context)
    assert (decision["start_unit"], decision["end_unit"], decision["hook"]) == (101, 103, 4)
    request = editor.model.request
    schema = request["response_format"]["schema"]["properties"]
    assert schema["window_index"]["enum"] == [0]
    assert schema["headline_index"]["enum"] == [0]
    assert all(
        schema[name]["enum"] == list(range(6))
        for name in ("opening", "story", "ending", "headline_quality")
    )
    sent = json.loads(request["messages"][1]["content"])
    assert sent["valid_windows"] == [[0, 101, 103]]
    assert all(set(unit) == {"id", "text"} for unit in sent["units"])
    assert request["temperature"] == 0 and request["seed"] == 0 and request["max_tokens"] == 256
    assert schema["reason"]["maxLength"] == 180


def test_local_editor_rejects_invalid_selection_and_truncated_assessment() -> None:
    import json

    import pytest

    from scripts.tjr_semantic_editor import LocalContextualEditor

    editor = object.__new__(LocalContextualEditor)

    class Model:
        finish = "stop"

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "finish_reason": self.finish,
                        "message": {"content": json.dumps({"window_index": 22})},
                    }
                ]
            }

    editor.model = Model()
    context = {
        "min_seconds": 20,
        "max_seconds": 45,
        "units": [{"id": 8, "start": 100, "end": 122, "text": "The answer settled it."}],
        "headlines": [{"id": 0, "text": "THE ANSWER SETTLED IT."}],
    }
    with pytest.raises(RuntimeError, match="invalid window"):
        editor(context)
    editor.model.finish = "length"
    with pytest.raises(RuntimeError, match="truncated"):
        editor(context)
    context["units"][0]["text"] = "The next question is,"
    assert editor(context)["keep"] is False


def test_editorial_checkpoint_resumes_completed_rejections_and_retains_failure(tmp_path) -> None:
    import json

    import pytest

    from clipper.models import ClipCandidate
    from scripts.tjr_semantic_editor import refine_contextual_candidates

    segments = [TranscriptSegment(0, 22, "The owner finally let me into my own show.")]
    proposals = [ClipCandidate("v", 0, 22, segments[0].text, score) for score in (10, 9)]
    calls = []

    def assess(context):
        calls.append(context)
        if len(calls) == 1:
            return {"keep": False, "reason": "Dependent setup."}
        return dict(
            keep=True,
            start_unit=0,
            end_unit=0,
            hook_index=0,
            opening=17,
            story=4,
            ending=4,
            hook=4,
            reason="Invalid rating.",
        )

    path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="out-of-range"):
        refine_contextual_candidates(
            _brief(), proposals, segments, source_sha256="a" * 64, cache_path=path, assessor=assess
        )
    saved = json.loads(path.read_text())
    assert saved["complete"] is False and saved["processed_count"] == 1
    assert saved["decisions"][1]["decision"]["opening"] == 17

    def repair(context):
        calls.append(context)
        return dict(
            keep=True,
            start_unit=0,
            end_unit=0,
            hook_index=0,
            opening=4,
            story=4,
            ending=4,
            hook=4,
            reason="The owner resolved the access problem.",
        )

    clips, audit = refine_contextual_candidates(
        _brief(),
        proposals,
        segments,
        source_sha256="a" * 64,
        cache_path=path,
        reuse_path=path,
        assessor=repair,
        reviewer=_review,
    )
    assert len(calls) == 3 and len(clips) == 1
    assert audit["resumed_assessments"] == 1 and len(audit["assessments"]) == 2
    assert json.loads(path.read_text())["complete"] is True


def _review(context):
    units = context["selected_units"]
    return dict(
        opening_standalone=True,
        payoff_complete=True,
        ending_complete=True,
        contains_promotion_or_intro=False,
        headline_supported=True,
        headline_self_contained=True,
        headline="Security stopped the performer at his own show",
        setup_quote=" ".join(units[0].split()[:6]),
        payoff_quote=" ".join(units[-1].split()[-6:]),
    )


def test_span_review_rejects_high_ratings_when_payoff_is_outside_clip(tmp_path):
    from clipper.models import ClipCandidate
    from scripts.tjr_semantic_editor import refine_contextual_candidates

    segments = [
        TranscriptSegment(0, 12, "We received sixty million views last month."),
        TranscriptSegment(12, 24, "The channel seemed to be doing very well."),
        TranscriptSegment(24, 31, "But the revenue was zero despite those views."),
    ]
    proposal = ClipCandidate("v", 0, 24, " ".join(s.text for s in segments[:2]), 100)

    def select(context):
        return dict(
            keep=True,
            start_unit=0,
            end_unit=1,
            hook_index=0,
            opening=5,
            story=5,
            ending=5,
            hook=5,
            reason="Views but no income.",
        )

    calls = []

    def review(context):
        calls.append(context)
        return {**_review(context), "payoff_complete": False}

    result, audit = refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "cache.json",
        assessor=select,
        reviewer=review,
    )
    assert result == []
    assert calls[0]["after"] == [segments[2].text]
    assert segments[2].text not in calls[0]["selected_units"]
    assert audit["assessments"][0]["rejection"] == "SPAN_REVIEW_REJECTED"


def test_span_review_rejects_intro_fragment_and_invented_payoff(tmp_path):
    from clipper.models import ClipCandidate
    from scripts.tjr_semantic_editor import refine_contextual_candidates

    segments = [TranscriptSegment(0, 22, "The owner finally let me into my own show.")]
    proposal = ClipCandidate("v", 0, 22, segments[0].text, 100)

    def select(context):
        return dict(
            keep=True,
            start_unit=0,
            end_unit=0,
            hook_index=0,
            opening=5,
            story=5,
            ending=5,
            hook=5,
            reason="Owner resolved access.",
        )

    for field, value, expected in (
        ("contains_promotion_or_intro", True, "SPAN_REVIEW_REJECTED"),
        ("opening_standalone", False, "SPAN_REVIEW_REJECTED"),
        ("ending_complete", False, "SPAN_REVIEW_REJECTED"),
        ("payoff_quote", "He earned a million dollars", "UNSUPPORTED_EXCHANGE_REVIEW_EVIDENCE"),
    ):
        result, audit = refine_contextual_candidates(
            _brief(),
            [proposal],
            segments,
            source_sha256="a" * 64,
            cache_path=tmp_path / "cache.json",
            assessor=select,
            reviewer=lambda context, key=field, item=value: {**_review(context), key: item},
        )
        assert not result
        assert audit["assessments"][0]["rejection"] == expected


def test_local_reviewer_separates_delivered_span_from_excluded_context():
    import json

    from scripts.tjr_semantic_editor import LocalContextualEditor

    context = {
        "selected_units": ["The owner let me into my own show."],
        "before": [],
        "after": ["What happened next?"],
    }

    class Model:
        def __init__(self):
            self.requests = []
            self.boundary = {
                "reason": "The owner resolves access; the next question is unrelated.",
                "promotion_unit_ids": [],
                "opening": "standalone",
                "ending": "closed",
                "payoff_location": "selected",
                "setup_unit_id": 0,
                "payoff_unit_id": 0,
            }
            self.headline = {
                "headline": "The owner rescued access to his own show",
                "setup_quote": "The owner let me",
                "payoff_quote": "into my own show.",
                "reason": "Delivered evidence supports the resolution.",
                "headline_supported": 1,
                "headline_self_contained": 1,
            }

        def create_chat_completion(self, **kwargs):
            self.requests.append(kwargs)
            payload = json.loads(kwargs["messages"][1]["content"])
            value = self.boundary if "excluded_after" in payload else self.headline
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]
            }

    editor = object.__new__(LocalContextualEditor)
    editor.model = Model()
    result = editor.review(context)
    assert result["payoff_complete"] is True
    assert result["headline_supported"] is True
    assert len(editor.model.requests) == 2
    audit_request, headline_request = editor.model.requests
    audit_input = json.loads(audit_request["messages"][1]["content"])
    assert audit_input["output_schema"] == audit_request["response_format"]["schema"]
    assert audit_input["excluded_after"] == context["after"]
    assert audit_input["delivered"] == [{"id": 0, "text": context["selected_units"][0]}]
    headline_input = json.loads(headline_request["messages"][1]["content"])
    assert "excluded_after" not in headline_input
    assert "What happened next?" not in json.dumps(headline_input)
    assert all(r["seed"] == 0 and r["temperature"] == 0 for r in editor.model.requests)

    # A copied excluded quote cannot qualify even if the model says it is supported.
    editor.model.headline["payoff_quote"] = "What happened next?"
    assert editor.review(context)["headline_supported"] is False

    # A missing payoff or mixed intro stops before creative inference.
    for changes in (
        {"payoff_location": "after", "payoff_unit_id": -1},
        {"promotion_unit_ids": [0], "ending": "continues_in_after"},
    ):
        editor.model.requests.clear()
        editor.model.boundary.update(changes)
        rejected = editor.review(context)
        assert len(editor.model.requests) == 1
        assert rejected["headline"] == ""
        assert rejected["headline_supported"] is False

    # Grammar output is also independently checked, including invalid provenance IDs.
    import pytest

    editor.model.boundary["promotion_unit_ids"] = [2]
    with pytest.raises(RuntimeError, match="invalid boundary evidence"):
        editor.review(context)


def test_stage_cache_reuses_selector_when_only_reviewer_changes(tmp_path, monkeypatch):
    import json

    from clipper.models import ClipCandidate
    from scripts import tjr_semantic_editor as editor

    segments = [
        TranscriptSegment(0, 7, "How did security stop you at your own show?"),
        TranscriptSegment(7, 14, "I had left my own access pass in the car."),
        TranscriptSegment(14, 22, "The owner came outside and finally let me inside."),
    ]
    proposal = ClipCandidate("v", 0, 22, " ".join(s.text for s in segments), 10)
    calls = {"selector": 0, "reviewer": 0}

    def assess(context):
        calls["selector"] += 1
        return dict(
            keep=True,
            start_unit=0,
            end_unit=2,
            hook_index=0,
            opening=4,
            story=4,
            ending=5,
            hook=4,
            reason="Missing pass; owner let him inside.",
        )

    def review(context):
        calls["reviewer"] += 1
        return _review(context)

    first = tmp_path / "first.json"
    editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=first,
        assessor=assess,
        reviewer=review,
    )
    monkeypatch.setattr(
        editor, "EXCHANGE_REVIEW_PROMPT", editor.EXCHANGE_REVIEW_PROMPT + " Revised review."
    )
    second = tmp_path / "second.json"
    result, audit = editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=second,
        reuse_path=first,
        assessor=assess,
        reviewer=review,
    )
    assert len(result) == 1
    assert calls == {"selector": 1, "reviewer": 2}
    assert audit["selector_cache_hits"] == 1
    assert audit["reviewer_cache_hits"] == 0

    # Changing unrelated module metadata must not call either model stage.
    saved = json.loads(second.read_text())
    saved["identity"]["editor_code_sha256"] = "unrelated-module-edit"
    second.write_text(json.dumps(saved))
    _, audit = editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "third.json",
        reuse_path=second,
        assessor=assess,
        reviewer=review,
    )
    assert calls == {"selector": 1, "reviewer": 2}
    assert audit["selector_cache_hits"] == audit["reviewer_cache_hits"] == 1

    # Changed source invalidates both stages.
    editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="b" * 64,
        cache_path=tmp_path / "changed-source.json",
        reuse_path=second,
        assessor=assess,
        reviewer=review,
    )
    assert calls == {"selector": 2, "reviewer": 3}

    # A known v3 artifact migrates selector decisions, never its faulty reviews.
    legacy = json.loads(first.read_text())
    legacy["identity"].pop("selector_sha256")
    legacy["identity"].pop("reviewer_sha256")
    legacy["identity"]["version"] = "podcast_structured_editor_v3"
    legacy["identity"]["editor_code_sha256"] = (
        "d977781963fc02015cfb58bd70dae33f161e6b3773da3dd6aa6fb228597dd371"
    )
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy))
    _, audit = editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "migrated.json",
        reuse_path=legacy_path,
        assessor=assess,
        reviewer=review,
    )
    assert calls == {"selector": 2, "reviewer": 4}
    assert audit["selector_cache_hits"] == 1 and audit["reviewer_cache_hits"] == 0

    monkeypatch.setattr(editor, "EDITOR_PROMPT", editor.EDITOR_PROMPT + " Changed selector.")
    editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "changed-selector.json",
        reuse_path=second,
        assessor=assess,
        reviewer=review,
    )
    assert calls == {"selector": 3, "reviewer": 5}


def test_legacy_migration_preserves_positive_windows_but_rechecks_negative_headline_gates(tmp_path):
    import json

    from clipper.models import ClipCandidate
    from scripts import tjr_semantic_editor as editor

    segments = [
        TranscriptSegment(0, 22, "Security stopped me but the owner finally let me into the show."),
        TranscriptSegment(
            30, 52, "Security stopped the guest but the owner finally opened the door."
        ),
    ]
    proposals = [ClipCandidate("v", s.start, s.end, s.text, 10) for s in segments]
    calls = []

    def initial(context):
        calls.append(context)
        if context["proposed_start"] == 30:
            return dict(keep=False, reason="No strong literal headline.")
        return dict(
            keep=True,
            start_unit=0,
            end_unit=0,
            opening=4,
            story=4,
            ending=5,
            hook_index=0,
            hook=4,
            reason="The owner resolved the security problem.",
        )

    path = tmp_path / "legacy.json"
    editor.refine_contextual_candidates(
        _brief(),
        proposals,
        segments,
        source_sha256="a" * 64,
        cache_path=path,
        assessor=initial,
        reviewer=_review,
    )
    saved = json.loads(path.read_text())
    saved["identity"].pop("selector_sha256")
    saved["identity"].pop("reviewer_sha256")
    saved["identity"]["editor_code_sha256"] = (
        "d977781963fc02015cfb58bd70dae33f161e6b3773da3dd6aa6fb228597dd371"
    )
    saved["identity"]["version"] = "podcast_structured_editor_v3"
    path.write_text(json.dumps(saved))

    def updated(context):
        calls.append(context)
        assert context["proposed_start"] == 30
        return dict(
            keep=True,
            start_unit=1,
            end_unit=1,
            opening=4,
            story=4,
            ending=5,
            selection_policy="exchange_only_v1",
            reason="The owner opened the door.",
        )

    clips, audit = editor.refine_contextual_candidates(
        _brief(),
        proposals,
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "updated.json",
        reuse_path=path,
        assessor=updated,
        reviewer=_review,
    )
    assert len(clips) == 2 and len(calls) == 3
    assert audit["selector_cache_hits"] == 1 and audit["reviewer_cache_hits"] == 0


def test_stage_ast_identity_ignores_version_optional_fields_but_keeps_semantics():
    import ast
    import copy

    from scripts.tjr_semantic_editor import _canonical_ast

    tree = ast.parse("def select(value):\n    return value + 1\n")
    older = copy.deepcopy(tree)
    older.body[0]._fields = tuple(
        name for name in older.body[0]._fields if name not in {"type_params", "type_comment"}
    )
    newer = copy.deepcopy(tree)
    newer.body[0]._fields = tuple(
        dict.fromkeys((*newer.body[0]._fields, "type_params", "type_comment"))
    )
    newer.body[0].type_params = []
    newer.body[0].type_comment = None
    assert _canonical_ast(older) == _canonical_ast(newer)
    assert _canonical_ast(tree) != _canonical_ast(
        ast.parse("def select(value):\n    return value + 2\n")
    )


def test_discovery_cache_skips_embedding_and_invalidates_source(tmp_path):
    from unittest.mock import patch

    from scripts import tjr_semantic_editor as editor

    segments = [TranscriptSegment(0, 24, "The owner finally let the performer into his own show.")]
    first = tmp_path / "discovery.json"
    editor.build_semantic_editorial_candidates(
        _brief(), "v", segments, embedder=_fake_embedder, source_sha256="a" * 64, cache_path=first
    )
    with patch.object(editor, "_build_semantic_editorial_candidates") as discover:
        _, audit = editor.build_semantic_editorial_candidates(
            _brief(),
            "v",
            segments,
            source_sha256="a" * 64,
            cache_path=tmp_path / "retry.json",
            reuse_path=first,
        )
        assert audit["cache_reused"] is True
        discover.assert_not_called()
        discover.return_value = ([], {})
        editor.build_semantic_editorial_candidates(
            _brief(), "v", segments, source_sha256="b" * 64, reuse_path=first
        )
        discover.assert_called_once()


def test_preflight_rejects_wrong_reason_even_when_acceptance_matches(tmp_path, monkeypatch):
    import json

    import scripts.tjr_semantic_editor as editor

    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            [
                {
                    "start": 2308.64,
                    "end": 2328.88,
                    "text": "The audience is large but earnings remain small.",
                },
                {
                    "start": 2281.2,
                    "end": 2308.0,
                    "text": "The audience is large but earnings remain small.",
                },
                {
                    "start": 8.28,
                    "end": 52.16,
                    "text": "The audience is large but earnings remain small.",
                },
            ]
        )
    )

    class FakeEditor:
        def __init__(self, profile):
            self.index = 0

        def review(self, context):
            self.index += 1
            positive = self.index != 3
            return {
                "headline": "Large audiences do not guarantee earnings",
                "setup_quote": "The audience is large",
                "payoff_quote": "but earnings remain small",
                "reason": "Evidence from the selected exchange",
                "opening_standalone": positive,
                "payoff_complete": positive,
                "ending_complete": positive,
                "headline_supported": positive,
                "headline_self_contained": positive,
                "contains_promotion_or_intro": self.index != 1,
            }

        def close(self):
            pass

    monkeypatch.setattr(editor, "LocalSourceReviewer", FakeEditor)
    monkeypatch.setattr(
        editor,
        "_focused_span_review",
        lambda instance, context, **kwargs: instance.review(context),
    )
    output = tmp_path / "review.json"
    assert editor.reviewer_preflight(transcript, output) == 1
    records = json.loads(output.read_text())
    assert all(r["actual_accept"] == r["expected_accept"] for r in records)
    assert records[1]["flags_match"] is False
    assert records[1]["expected_flags"] == {
        "payoff_complete": False,
        "contains_promotion_or_intro": False,
    }


def test_review_context_marks_neighbors_excluded():
    from scripts.tjr_semantic_editor import SemanticUnit, _review_context

    units = [SemanticUnit(i, i + 1, f"Thought {i}") for i in range(7)]
    assert _review_context(units, 2, 4) == {
        "before": ["Thought 0", "Thought 1"],
        "selected_units": ["Thought 2", "Thought 3", "Thought 4"],
        "after": ["Thought 5", "Thought 6"],
    }


def test_long_source_evidence_is_distinct_from_short_quote_format():
    from scripts.tjr_semantic_editor import _source_quote_span

    source = (
        "Well, bro, if you had a podcast getting 60 million views, "
        "you'd be making millions of dollars."
    )
    assert _source_quote_span(source, [source]) is None
    span = _source_quote_span(source, [source], max_words=64)
    assert span is not None
    assert span["text"] == source.rstrip(".")
    assert _source_quote_span(source.replace("60", "600"), [source], max_words=64) is None


def test_bounded_evidence_preserves_full_source_and_shared_contract():
    from scripts.tjr_semantic_editor import (
        _evidence_excerpt,
        _review_evidence_valid,
        _source_quote_span,
    )

    source = (
        "Well, bro, if you had a podcast getting 60 million views, "
        "you'd be making millions of dollars."
    )
    span = _source_quote_span(source, [source], max_words=64)
    assert span is not None
    excerpt = _evidence_excerpt(span["text"])
    assert excerpt in source and len(excerpt.split()) <= 12
    assert span["text"] == source.rstrip(".")
    review = {
        "headline": "A large audience can support a podcast",
        "setup_quote": excerpt,
        "payoff_quote": "making millions of dollars",
    }
    assert _review_evidence_valid(review, [source])
    assert not _review_evidence_valid({**review, "setup_quote": source}, [source])
    assert not _review_evidence_valid(
        {**review, "payoff_quote": "making billions of dollars"}, [source]
    )


def test_intro_rejection_does_not_run_boundary_or_headline_inference():
    from scripts.tjr_semantic_editor import _focused_span_review

    context = {
        "selected_units": ["This show is presented by the sponsor.", "I was telling him."],
        "before": [],
        "after": ["Here is what I said."],
    }
    replies = [
        {
            "reason": "Actual show introduction",
            "ad_read_quote": "presented by the sponsor",
            "show_intro_quote": "",
        },
        {
            "reason": "Unfinished quoted speech",
            "setup_quote": "I was telling him",
            "resolution_quote": "",
            "opening_independent": 0,
            "last_thought_finished": 0,
        },
    ]

    class Editor:
        def _review_completion(self, *args):
            assert replies, "Rejected intro must not trigger extra model calls"
            return replies.pop(0)

    review = _focused_span_review(Editor(), context)
    assert review["contains_promotion_or_intro"] is True
    assert review["ending_complete"] is False
    assert review["continuation_review"] is None
    assert review["headline"] == ""
    assert not replies


def test_opening_dependency_requires_source_evidence_without_prior_verdict():
    from scripts.tjr_semantic_editor import _focused_span_review

    context = {
        "selected_units": [
            "At my first performance the door staff stopped me.",
            "The owner came out and let me inside.",
        ],
        "before": [],
        "after": [],
    }
    replies = [
        {"reason": "Conversation", "ad_read_quote": "", "show_intro_quote": ""},
        {
            "reason": "A complete incident",
            "setup_quote": "the door staff stopped me",
            "resolution_quote": "let me inside",
            "opening_independent": 0,
            "last_thought_finished": 1,
        },
        {
            "subject_quote": "At my first performance",
            "missing_context_quote": "",
            "reason": "The speaker states the event and situation.",
            "opening_standalone": 1,
        },
        {"headline": "Door staff stopped the performer at his own show"},
    ]
    replies.append(_audit_reply("supported", "The door staff stopped the performer."))
    payloads = []

    class Editor:
        def _review_completion(self, prompt, payload, *args):
            if "opening_text" in payload:
                assert args[0]["reason"]["minLength"] == 1
                assert "for either verdict" in prompt
            payloads.append(payload)
            return replies.pop(0)

    review = _focused_span_review(Editor(), context)
    assert review["opening_standalone"] is True
    assert set(payloads[2]) == {"delivered_transcript", "opening_text"}
    assert review["opening_review"]["subject_quote"] in context["selected_units"][0]


def test_headline_audit_rejects_invented_roles_and_requires_bound_source_units():
    import pytest

    from scripts.tjr_semantic_editor import _audit_headline

    units = ["A visitor waited while the owner unlocked the shop."]
    response = _audit_reply("unsupported", "The visitor is not identified as the owner.")

    class Editor:
        def _review_completion(self, prompt, payload, *args):
            assert set(payload) == {"source_units", "headline"}
            return response

    assert (
        _audit_headline(Editor(), "The owner waited outside his shop", units)["verdict"]
        == "unsupported"
    )
    response["evidence_unit_ids"] = [99]
    with pytest.raises(RuntimeError, match="source unit evidence"):
        _audit_headline(Editor(), "The owner waited outside his shop", units)


def test_headline_critic_prose_never_becomes_generation_input():
    from scripts.tjr_semantic_editor import _source_grounded_headline

    units = ["A visitor waited outside the shop.", "The owner unlocked the shop and let him in."]
    replies = [
        {
            "central_quote": "A visitor waited outside the shop",
            "context_quote": "The owner unlocked the shop",
            "payoff_quote": "and let him in",
        },
        {"headline": "The owner waited outside his shop"},
        _audit_reply("unsupported", "The visitor broke into the shop instead"),
    ]
    calls = []

    class Editor:
        def _review_completion(self, prompt, payload, *args):
            calls.append(payload)
            assert replies
            return replies.pop(0)

    result = _source_grounded_headline(Editor(), units)
    assert result["headline_supported"] is False
    assert len(result["headline_audits"]) == 1
    assert len(calls) == 3 and not replies
    assert set(calls[1]) == {"source_passages", "source_context"}
    assert [item["text"] for item in calls[1]["source_context"]] == units
    assert "broke into" not in str(calls[1])
    assert "fact_check" not in str(calls)
    assert result["headline_source_spans"]["central_quote"]["first_unit"] == 0


def test_headline_generation_requires_literal_evidence_and_preserves_conditions():
    import pytest

    from scripts.tjr_semantic_editor import _source_grounded_headline

    units = [
        "The analyst said the trial failed.",
        "If the control worked, the researcher would repeat the trial.",
        "The sponsor thanked the analyst for reporting the failure.",
    ]
    evidence = {
        "central_quote": "The analyst said the trial failed",
        "context_quote": "If the control worked, the researcher would repeat the trial",
        "payoff_quote": "The sponsor thanked the analyst for reporting the failure",
    }
    replies = [
        evidence,
        {"headline": "Analyst reports failed trial and sponsor thanks them"},
        _audit_reply("supported", "Reporting is explicit"),
    ]
    calls = []

    class Editor:
        def _review_completion(self, prompt, payload, *args):
            calls.append(payload)
            return replies.pop(0)

    result = _source_grounded_headline(Editor(), units)
    assert result["headline_supported"] is True
    assert calls[1]["source_passages"]["context_quote"].startswith("If")
    assert [item["text"] for item in calls[1]["source_context"]] == units
    assert "reason" not in calls[1]["source_passages"]
    # A copied model interpretation cannot be laundered into source evidence.
    for bad in (
        "The researcher said the trial failed",
        "The control worked and the researcher repeated the trial",
    ):
        replies[:] = [{**evidence, "central_quote": bad}]
        calls.clear()
        with pytest.raises(RuntimeError, match="exact delivered source passage"):
            _source_grounded_headline(Editor(), units)
        assert len(calls) == 1


def test_review_parser_enforces_declared_fields_without_inventing_reason_requirement():
    import json

    import pytest

    from scripts.tjr_semantic_editor import LocalContextualEditor

    value = {"central_quote": "A visitor waited outside"}

    class Model:
        def create_chat_completion(self, **request):
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]
            }

    editor = object.__new__(LocalContextualEditor)
    editor.model = Model()
    quote_schema = {"central_quote": {"type": "string"}}
    assert editor._review_completion("Extract", {}, quote_schema, 32) == value
    value["reason"] = "Invented explanation"
    with pytest.raises(RuntimeError, match="declared schema"):
        editor._review_completion("Extract", {}, quote_schema, 32)
    value.clear()
    value["reason"] = ""
    with pytest.raises(RuntimeError, match="evidence reason"):
        editor._review_completion("Assess", {}, {"reason": {"type": "string"}}, 32)
    value.clear()
    with pytest.raises(RuntimeError, match="declared schema"):
        editor._review_completion("Extract", {}, quote_schema, 32)


def test_factual_probe_uses_separate_model_with_bounded_memory_and_no_exchange_rerun(
    tmp_path, monkeypatch
):
    import json
    import sys
    from types import SimpleNamespace

    from scripts import tjr_semantic_editor as editor

    monkeypatch.setattr(editor.Path, "home", lambda: tmp_path)
    path = tmp_path / ".cache/clipper/editor/Qwen_Qwen3.5-9B-Q4_K_S.gguf"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(
        editor.hashlib,
        "file_digest",
        lambda *args: SimpleNamespace(
            hexdigest=lambda: "25bacefaea1654a359bab316793f217a646da8610ee7973a26548fb2d624c7a9"
        ),
    )
    loads = []

    class Model:
        def __init__(self, **kwargs):
            loads.append(kwargs)
            self.metadata = {"tokenizer.chat_template": "template"}

        def token_eos(self):
            return 1

        def token_bos(self):
            return 2

        def detokenize(self, *args, **kwargs):
            return b"token"

        def create_chat_completion(self, **kwargs):
            raise AssertionError("only the stubbed factual generation may run")

        def close(self):
            pass

    class Formatter:
        def __init__(self, **kwargs):
            self.template = kwargs["template"]

        def __call__(self, **kwargs):
            return SimpleNamespace(prompt="assistant\n<think>\n\n</think>\n\n")

        def to_chat_handler(self):
            return "hard-non-thinking"

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(
            Llama=Model, llama_chat_format=SimpleNamespace(Jinja2ChatFormatter=Formatter)
        ),
    )
    monkeypatch.setattr(
        editor,
        "_focused_span_review",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("factual comparison must not repeat exchange assessment")
        ),
    )
    generated = []

    def generate(model, units):
        generated.append(units)
        return {
            "headline": "A visitor waited outside the shop",
            "headline_supported": True,
            "headline_self_contained": True,
            "headline_source_spans": {"central_quote": {}, "context_quote": {}, "payoff_quote": {}},
        }

    monkeypatch.setattr(editor, "_source_grounded_headline", generate)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "fixture": str(i),
                    "expected_accept": i == 0,
                    "review_context": {"selected_units": ["A visitor waited outside the shop."]},
                }
                for i in range(3)
            ]
        )
    )
    output = tmp_path / "comparison.json"
    assert editor.reviewer_model_probe(baseline, output, factual_probe=True) == 0
    report = json.loads(output.read_text())
    assert report["production_approved"] is False
    assert report["probe_scope"] == "headline_facts_only"
    assert report["model_repo"] == "bartowski/Qwen_Qwen3.5-9B-GGUF"
    assert loads[0]["n_ctx"] == 2048 and loads[0]["n_batch"] == 128
    assert len(loads) == len(generated) == len(report["cases"]) == 1


def test_workflow_routes_named_probe_modes_to_the_actual_cli(tmp_path, monkeypatch):
    import os
    import subprocess
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/tjr-weekly-hd.yml").read_text())
    script = next(
        step["run"]
        for step in workflow["jobs"]["editorial_preflight"]["steps"]
        if step.get("name", "").startswith("Assess reviewer evidence")
    )
    (tmp_path / "reviewer-baseline").mkdir()
    (tmp_path / "reviewer-baseline/reviewer-preflight.json").write_text("{}")
    source_bound_cache = tmp_path / "reviewer-baseline/review-request-cache.json"
    source_bound_cache.write_text("{}")
    structured_proof = (
        tmp_path / "reviewer-baseline/reviewer-gpu-evidence/baseline_4b/cold/proof.json"
    )
    structured_proof.parent.mkdir(parents=True)
    structured_proof.write_text("{}")
    (tmp_path / "reviewer-input").mkdir()
    (tmp_path / "reviewer-input/transcript.json").write_text("[]")
    executable = tmp_path / "bin/python"
    executable.parent.mkdir()
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$PROBE_ARGUMENT_CAPTURE"\n')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{executable.parent}:{os.environ['PATH']}")
    capture = tmp_path / "args.txt"
    monkeypatch.setenv("PROBE_ARGUMENT_CAPTURE", str(capture))
    for mode in (
        "disabled",
        "focused_4b",
        "factual_9b",
        "factual_consensus",
        "factual_ablation_4b",
        "factual_nli",
        "evidence_qa",
        "structured_claim",
        "source_bound_headline",
        "evidence_gpu",
    ):
        rendered = (
            script.replace("${{ inputs.reviewer_model_probe }}", mode)
            .replace(
                "${{ inputs.reviewer_model_probe == 'factual_9b' "
                "|| inputs.reviewer_model_probe == 'factual_consensus' }}",
                "true" if mode in {"factual_9b", "factual_consensus"} else "false",
            )
            .replace(
                "${{ inputs.reviewer_model_probe == 'factual_consensus' }}",
                "true" if mode == "factual_consensus" else "false",
            )
        )
        rendered = rendered.replace(
            "${{ inputs.reviewer_model_probe == 'factual_ablation_4b' }}",
            "true" if mode == "factual_ablation_4b" else "false",
        )
        rendered = rendered.replace(
            "${{ inputs.reviewer_model_probe == 'factual_nli' }}",
            "true" if mode == "factual_nli" else "false",
        )
        rendered = rendered.replace(
            "${{ inputs.reviewer_model_probe == 'evidence_qa' }}",
            "true" if mode == "evidence_qa" else "false",
        )
        rendered = rendered.replace(
            "${{ inputs.reviewer_model_probe == 'evidence_gpu' }}",
            "true" if mode == "evidence_gpu" else "false",
        )
        rendered = rendered.replace(
            "${{ inputs.reviewer_model_probe == 'structured_claim' }}",
            "true" if mode == "structured_claim" else "false",
        )
        assert "${{" not in rendered
        subprocess.run(["bash", "-e", "-c", rendered], cwd=tmp_path, check=True)
        args = capture.read_text().splitlines()
        assert ("--reviewer-model-probe-baseline" in args) == (mode != "disabled")
        if mode == "source_bound_headline":
            assert source_bound_cache.relative_to(tmp_path).as_posix() in args
        if mode == "structured_claim":
            assert structured_proof.relative_to(tmp_path).as_posix() in args
        assert ("--reviewer-diagnostics-baseline" in args) == (mode == "disabled")
        assert ("--headline-factual-probe" in args) == (mode in {"factual_9b", "factual_consensus"})
        assert ("--headline-consensus-probe" in args) == (mode == "factual_consensus")
        assert ("--headline-ablation-probe" in args) == (mode == "factual_ablation_4b")
        assert ("--headline-nli-probe" in args) == (mode == "factual_nli")
        assert ("--evidence-qa-probe" in args) == (mode == "evidence_qa")
        assert ("--structured-claim-probe" in args) == (mode == "structured_claim")
        assert ("--source-bound-headline-probe" in args) == (mode == "source_bound_headline")
        assert ("--evidence-gpu-probe" in args) == (mode == "evidence_gpu")
        assert "scripts.tjr_semantic_editor" in args


def test_workflow_validation_reuse_requires_complete_recent_exact_head_proof():
    import copy
    import datetime
    import json
    import subprocess
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/tjr-weekly-hd.yml").read_text())
    steps = workflow["jobs"]["tests"]["steps"]
    script = next(step["with"]["script"] for step in steps if step.get("id") == "validation_cache")
    required_names = [
        'Run python -m pip install -e ".[dev]"',
        "Run ruff check .",
        "Run ruff format --check .",
        "Run mypy",
        "Install FFmpeg for media regression tests and HD canary",
        "Run pytest",
        "Validate campaign rules",
        "Verify pinned PO-token provider readiness without downloading video",
        "Render and inspect a synthetic HD FFmpeg canary",
    ]
    run = dict(
        id=42,
        head_sha="a" * 40,
        status="completed",
        conclusion="success",
        event="push",
        path=".github/workflows/tjr-weekly-hd.yml",
        html_url="https://github.com/example/run/42",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    job = dict(
        name="Validate source, policy and code",
        status="completed",
        conclusion="success",
        labels=["ubuntu-24.04"],
        steps=[
            dict(name=name, status="completed", conclusion="success") for name in required_names
        ],
    )
    scenarios = [(run, job, False, True)]
    for changes in (
        {"head_sha": "b" * 40},
        {"conclusion": "failure"},
        {"event": "workflow_dispatch"},
        {"created_at": "2020-01-01T00:00:00Z"},
        {"path": ".github/workflows/other.yml"},
    ):
        scenarios.append(({**run, **changes}, job, False, False))
    for changes in (
        {"status": "in_progress"},
        {"labels": ["ubuntu-latest"]},
        {"conclusion": "cancelled"},
        {"steps": job["steps"][:-1]},
    ):
        scenarios.append((run, {**job, **changes}, False, False))
    failed = copy.deepcopy(job)
    next(s for s in failed["steps"] if s["name"] == "Run pytest")["conclusion"] = "failure"
    scenarios.extend([(run, failed, False, False), (run, job, True, False)])
    harness = """
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const outputs = {};
const core = {setOutput: (k,v) => outputs[k]=v, info: () => {}, warning: () => {}};
const context = {repo: {owner: 'owner', repo: 'repo'}, sha: 'a'.repeat(40)};
const github = {rest: {actions: {
 listWorkflowRuns: async () => {if(input.error)throw Error('API unavailable');
   return {data: {workflow_runs: [input.run]}};},
 listJobsForWorkflowRun: async () => ({data: {jobs: [input.job]}}),
}}};
(async () => { SCRIPT })().then(() => process.stdout.write(JSON.stringify(outputs)));
""".replace("SCRIPT", script)
    for candidate, validation, error, expected in scenarios:
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(dict(run=candidate, job=validation, error=error)),
            text=True,
            capture_output=True,
            check=True,
        )
        assert (json.loads(result.stdout)["reused"] == "true") is expected
    inputs = next(s for s in steps if s.get("name", "").startswith("Validate production inputs"))
    assert inputs["if"] == "github.event_name == 'workflow_dispatch'"


def test_headline_consensus_vetoes_complementary_failures_and_requires_all_components():
    import pytest

    from scripts.tjr_semantic_editor import _headline_consensus

    supported = {
        key: "supported"
        for key in ("actor_action", "relationship_role", "setting_time", "quantities_outcomes")
    }
    assert _headline_consensus(supported, supported)["supported"] is True
    for component in supported:
        rejected = {**supported, component: "unsupported"}
        for first, second in ((supported, rejected), (rejected, supported)):
            combined = _headline_consensus(first, second)
            assert combined["supported"] is False and combined[component] == "unsupported"
        assert not _headline_consensus(supported, {**supported, component: "uncertain"})[
            "supported"
        ]
    with pytest.raises(RuntimeError, match="complete component evidence"):
        _headline_consensus(supported, {"actor_action": "supported"})


def test_ablation_does_not_count_wrong_reason_or_invalid_citation_as_success():
    import json

    import pytest

    from scripts import tjr_semantic_editor as editor

    fixture = dict(
        expected_supported=False,
        expected_unsupported_component="setting_time",
        request={
            "messages": [
                {},
                {
                    "content": json.dumps(
                        {"source_units": [{"id": 0, "text": "A quoted instruction."}]}
                    )
                },
            ]
        },
    )
    result = dict(
        evidence_unit_ids=[0], verdict="unsupported", unsupported_component="actor_action"
    )
    assert not editor._ablation_result(result, fixture, True)["passed"]
    result["unsupported_component"] = "setting_time"
    assert editor._ablation_result(result, fixture, True)["passed"]
    result["evidence_unit_ids"] = [True]
    with pytest.raises(RuntimeError, match="source references"):
        editor._ablation_result(result, fixture, True)
    result["evidence_unit_ids"] = [0]
    result["verdict"] = "supported"
    with pytest.raises(RuntimeError, match="inconsistent"):
        editor._ablation_result(result, fixture, True)


def test_ablation_reuses_every_unchanged_request_and_invalidates_other_model(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from scripts import tjr_semantic_editor as editor

    units = [{"id": 0, "text": "A speaker reports a quoted instruction."}]
    fixtures = []
    for i in range(8):
        fixtures.append(
            dict(
                headline=f"Test claim {i}",
                expected_supported=i < 2,
                expected_unsupported_component=None if i < 2 else "setting_time",
                request=dict(
                    messages=[
                        {"role": "system", "content": "Audit literal source."},
                        {
                            "role": "user",
                            "content": json.dumps(
                                dict(
                                    source_units=units, headline=f"Test claim {i}", output_schema={}
                                )
                            ),
                        },
                    ],
                    max_tokens=144,
                    response_format={
                        "schema": {
                            "properties": {
                                "evidence_unit_ids": {
                                    "type": "array",
                                    "items": {"type": "integer", "enum": [0]},
                                },
                                "reason": {"type": "string"},
                                **{
                                    k: {
                                        "type": "string",
                                        "enum": ["supported", "unsupported", "uncertain"],
                                    }
                                    for k in editor._HEADLINE_COMPONENTS
                                },
                            }
                        }
                    },
                ),
            )
        )
    source = tmp_path / "baseline.json"
    source.write_text(
        json.dumps(
            dict(experiment="source_presentation_x_verdict_schema", annotated_fixtures=fixtures)
        )
    )
    profile = {"model_sha256": "first-model", "context_tokens": 4096}
    calls = []

    def completion(**request):
        calls.append(request)
        payload = json.loads(request["messages"][1]["content"])
        expected = int(payload["headline"].rsplit(" ", 1)[-1]) < 2
        result = dict(evidence_unit_ids=[0], reason="Literal source check.")
        if "verdict" in request["response_format"]["schema"]["properties"]:
            result.update(
                verdict="supported" if expected else "unsupported",
                unsupported_component="none" if expected else "setting_time",
            )
        else:
            result.update(
                {
                    k: "unsupported" if not expected and k == "setting_time" else "supported"
                    for k in editor._HEADLINE_COMPONENTS
                }
            )
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}

    def load(runtime):
        instance = editor.LocalContextualEditor.__new__(editor.LocalContextualEditor)
        instance.model = SimpleNamespace(create_chat_completion=completion, close=lambda: None)
        return instance, dict(profile)

    monkeypatch.setattr(editor, "_load_probe_peer", load)
    monkeypatch.setattr("importlib.metadata.version", lambda _: "test-runtime")
    first, second, third = [tmp_path / f"{i}.json" for i in range(3)]
    assert editor.reviewer_model_ablation(source, first) == 0
    assert len(calls) == 32
    report = json.loads(first.read_text())
    assert report["production_approved"] is False
    assert report["experiment_complete"] is True
    assert len(report["qualified_variants"]) == 4
    for call in calls:
        payload = json.loads(call["messages"][1]["content"])
        assert payload["source_units"] == units
        if "continuous_source_text" in payload:
            assert payload["continuous_source_text"] == units[0]["text"]
    assert editor.reviewer_model_ablation(first, second) == 0
    assert len(calls) == 32
    assert json.loads(second.read_text())["cache_hits"] == 32
    profile["model_sha256"] = "changed-model"
    assert editor.reviewer_model_ablation(second, third) == 0
    assert len(calls) == 64


def test_headline_preserves_reviewed_resolution_without_reselecting_quotes():
    from scripts.tjr_semantic_editor import _source_grounded_headline, _source_quote_span

    units = [
        "I have sixty million social views and earn nothing.",
        "If a podcast had sixty million views, it could earn millions.",
        "Those social views only drive listeners to the podcast.",
    ]
    spans = {
        "setup_quote": _source_quote_span(units[0], units, max_words=64),
        "resolution_quote": _source_quote_span(units[2], units, max_words=64),
    }
    replies = [
        {"headline": "Huge social views earn nothing but drive podcast listeners"},
        _audit_reply("supported", "Views bring listeners, not proven earnings"),
    ]
    calls = []

    class Editor:
        def _review_completion(self, prompt, payload, *args):
            calls.append(payload)
            return replies.pop(0)

    result = _source_grounded_headline(Editor(), units, exchange_spans=spans)
    assert len(calls) == 2 and not replies
    assert result["headline_source_spans"] == spans
    assert calls[0]["source_passages"] == {key: span["text"] for key, span in spans.items()}
    assert calls[0]["source_passages"]["resolution_quote"] == units[2][:-1]
    assert [unit["text"] for unit in calls[0]["source_context"]] == units
    assert "reason" not in str(calls[0])
    assert "could earn" not in str(calls[0]["source_passages"])


def test_headline_rejects_forged_reviewed_spans_before_inference():
    import pytest

    from scripts.tjr_semantic_editor import _source_grounded_headline, _source_quote_span

    units = ["The visitor waited outside.", "The owner finally unlocked the door."]
    spans = {
        "setup_quote": _source_quote_span(units[0], units, max_words=64),
        "resolution_quote": _source_quote_span(units[1], units, max_words=64),
    }

    class Editor:
        def _review_completion(self, *args):
            raise AssertionError("Invalid source handoff must not call any model")

    for changed in (
        {"text": "The owner locked the door", "first_unit": 1, "last_unit": 1},
        {**spans["resolution_quote"], "first_unit": 0},
        {**spans["resolution_quote"], "last_unit": 0},
        {**spans["resolution_quote"], "first_unit": True},
        {**spans["resolution_quote"], "reason": "Invented interpretation"},
        None,
    ):
        with pytest.raises(RuntimeError, match="canonical source evidence"):
            _source_grounded_headline(
                Editor(), units, exchange_spans={**spans, "resolution_quote": changed}
            )
    with pytest.raises(RuntimeError, match="setup and resolution"):
        _source_grounded_headline(
            Editor(), units, exchange_spans={"setup_quote": spans["setup_quote"]}
        )


def test_production_calls_position_review_with_exact_source_context(tmp_path, monkeypatch):
    from clipper.models import ClipCandidate
    from scripts import tjr_semantic_editor as editor

    calls = []
    segments = [
        TranscriptSegment(0, 7, "How did security stop you at your own show?"),
        TranscriptSegment(7, 14, "I had left my own access pass in the car."),
        TranscriptSegment(14, 22, "The owner came outside and finally let me inside."),
        TranscriptSegment(22, 29, "That is a different story for another day."),
    ]

    def initialize(self):
        pass

    def selection(self, prompt, payload, properties, tokens):
        assert "headlines" not in payload
        return dict(
            keep=True,
            window_index=0,
            opening=4,
            story=4,
            ending=5,
            reason="Missing pass; owner let him inside.",
        )

    def focused(instance, context, *, factual_audit):
        calls.append(context)
        assert isinstance(instance, editor.ReviewRequestCache)
        assert factual_audit is editor._position_headline_audit
        return _review(context)

    def legacy(self, context):
        raise AssertionError("Production must not return to the legacy reviewer")

    monkeypatch.setattr(editor.LocalContextualEditor, "__init__", initialize)
    monkeypatch.setattr(editor.LocalContextualEditor, "_review_completion", selection)
    monkeypatch.setattr(editor.LocalContextualEditor, "review", legacy)
    monkeypatch.setattr(editor.LocalContextualEditor, "close", lambda self: None)
    monkeypatch.setattr(editor, "_source_position_review", focused)
    monkeypatch.setattr(editor, "source_headline_candidates", lambda _: [])
    proposal = ClipCandidate("_kDrxucOx9g", 0, 22, " ".join(item.text for item in segments[:3]), 10)
    clips, audit = editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "editorial-cache.json",
    )
    assert len(clips) == 1
    assert calls == [
        {
            "selected_units": [item.text for item in segments[:3]],
            "before": [],
            "after": [segments[3].text],
        }
    ]
    packet = audit["assessments"][0]["claim_review_packet"]
    assert packet["source_video_id"] == "_kDrxucOx9g"
    assert packet["production_approved"] is False
    assert packet["reviewed_spans"]["setup_quote"]["text"] in segments[0].text
    assert packet["reviewed_spans"]["resolution_quote"]["text"] in segments[2].text
    replayed, replay_audit = editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "replayed-editorial-cache.json",
        reuse_path=tmp_path / "editorial-cache.json",
    )
    assert replayed == clips
    assert replay_audit["cache_reused"] is True
    assert replay_audit["assessments"][0]["claim_review_packet"] == packet
    assert len(calls) == 1


def test_invalid_claim_packet_rejects_one_window_without_aborting_editor(tmp_path, monkeypatch):
    from clipper.models import ClipCandidate
    from scripts import tjr_semantic_editor as editor

    segments = [
        TranscriptSegment(0, 7, "How did security stop you at your own show?"),
        TranscriptSegment(7, 14, "I left my access pass in the car."),
        TranscriptSegment(14, 22, "The owner finally let me inside."),
    ]
    proposal = ClipCandidate("_kDrxucOx9g", 0, 22, " ".join(s.text for s in segments), 10)
    monkeypatch.setattr(editor, "source_headline_candidates", lambda _: [])

    def select(_context):
        return dict(
            keep=True,
            start_unit=0,
            end_unit=2,
            opening=4,
            story=4,
            ending=5,
            reason="The owner let him inside after the lost pass.",
        )

    def review(context):
        return {
            **_review(context),
            "headline_source_spans": {
                "setup_quote": {
                    "text": "fabricated setup",
                    "first_unit": 0,
                    "last_unit": 0,
                },
                "resolution_quote": {
                    "text": "finally let me inside",
                    "first_unit": 2,
                    "last_unit": 2,
                },
            },
        }

    clips, audit = editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "editorial-cache.json",
        assessor=select,
        reviewer=review,
    )
    assert clips == []
    assert audit["assessments"][0]["rejection"] == "INVALID_CLAIM_REVIEW_PACKET"
    assert "claim_review_packet" not in audit["assessments"][0]


def test_exchange_selection_does_not_require_literal_headlines():
    from scripts.tjr_semantic_editor import _select_exchange

    calls = []

    class Editor:
        def _review_completion(self, prompt, payload, properties, tokens):
            calls.append(payload)
            assert "headlines" not in payload and "headline_quality" not in properties
            return dict(
                keep=True,
                window_index=0,
                opening=4,
                story=4,
                ending=5,
                reason="The lost key is found at the end.",
            )

    result = _select_exchange(
        Editor(),
        {
            "units": [
                dict(
                    id=0,
                    start=0,
                    end=22,
                    text="The owner found the lost key and unlocked the door.",
                )
            ],
            "headlines": [],
            "min_seconds": 20,
            "max_seconds": 60,
        },
    )
    assert result["keep"] is True and result["selection_policy"] == "exchange_only_v1"
    assert result["start_unit"] == result["end_unit"] == 0
    assert len(calls) == 1


def test_review_request_cache_skips_weights_and_invalidates_changed_or_corrupt_request(tmp_path):
    import json

    from scripts.tjr_semantic_editor import LocalContextualEditor, ReviewRequestCache

    created = []

    class Model:
        def create_chat_completion(self, **request):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"reason": "No source support"})},
                    }
                ]
            }

    def factory():
        created.append(True)
        instance = LocalContextualEditor.__new__(LocalContextualEditor)
        instance.model = Model()
        return instance

    path = tmp_path / "review-request-cache.json"
    identity = {"source": "original", "model": "pinned"}
    first = ReviewRequestCache(path, factory, identity)
    properties = {"reason": {"type": "string"}}
    first._review_completion("Check source", {"source": "unchanged"}, properties, 64)
    assert len(created) == 1 and first.metrics["model_calls"] == 1
    assert first.calls[0]["raw_calls"][0]["request"]["temperature"] == 0
    warm = ReviewRequestCache(path, factory, identity)
    assert warm._review_completion("Check source", {"source": "unchanged"}, properties, 64) == {
        "reason": "No source support"
    }
    assert len(created) == 1 and warm.metrics["cache_hits"] == 1
    warm._review_completion("Check source", {"source": "changed"}, properties, 64)
    assert len(created) == 2
    saved = json.loads(path.read_text())
    first_key = next(iter(saved["records"]))
    saved["records"][first_key]["response"]["reason"] = "Corrupted approval"
    path.write_text(json.dumps(saved))
    corrupt = ReviewRequestCache(path, factory, identity)
    assert corrupt._review_completion("Check source", {"source": "unchanged"}, properties, 64) == {
        "reason": "No source support"
    }
    assert len(created) == 2
    assert corrupt.metrics["recovered_responses"] == 1
    changed = ReviewRequestCache(path, factory, {**identity, "model": "different"})
    changed._review_completion("Check source", {"source": "unchanged"}, properties, 64)
    assert len(created) == 3


def test_source_qa_is_blind_to_headline_and_requires_canonical_evidence(monkeypatch):
    import pytest

    from scripts import tjr_semantic_editor as editor

    units = ["Sixty million views earned no money.", "Those views only drive podcast listeners."]
    payloads = []
    facts = {
        key: dict(answer="Views bring listeners, not proven earnings.", quote=units[1])
        for key in editor._HEADLINE_COMPONENTS
    }
    monkeypatch.setattr(
        editor,
        "_semantic_draft",
        lambda instance, prompt, payload, tokens: (
            payloads.append(payload) or "Source says views bring listeners."
        ),
    )

    class Editor:
        def _review_completion(self, prompt, payload, properties, tokens):
            payloads.append(payload)
            return facts

    source = editor._source_fact_record(Editor(), units)
    assert all("headline" not in payload for payload in payloads)
    assert source["facts"]["quantities_outcomes"]["source_span"]["first_unit"] == 1
    facts["actor_action"]["quote"] = "They already earned millions of dollars"
    with pytest.raises(RuntimeError, match="canonical delivered"):
        editor._source_fact_record(Editor(), units)
    facts["actor_action"] = dict(answer="Actual earnings", quote="")
    with pytest.raises(RuntimeError, match="must remain unknown"):
        editor._source_fact_record(Editor(), units)


def test_hook_revision_preserves_exchange_and_separates_factual_and_text_quality():
    from scripts import tjr_semantic_editor as editor

    units = [
        "I have sixty million views and earn nothing.",
        "Those views only drive listeners to the podcast.",
    ]
    spans = {
        "setup_quote": editor._source_quote_span(units[0], units, max_words=64),
        "resolution_quote": editor._source_quote_span(units[1], units, max_words=64),
    }
    calls = []
    replies = iter(
        [
            "The podcast already earned millions from social views",
            "Sixty million views bring listeners but no income",
        ]
    )

    class Editor:
        def _review_completion(self, prompt, payload, properties, tokens):
            calls.append(payload)
            return {"headline": next(replies)}

    def audit(instance, headline, selected):
        assert selected == units
        valid = "already earned" not in headline
        result = dict(
            verdict="supported" if valid else "unsupported",
            **{key: "supported" for key in editor._HEADLINE_COMPONENTS},
            headline_self_contained=True,
            central_highlight=True,
        )
        result["quantities_outcomes"] = "supported" if valid else "unsupported"
        return result

    result = editor._source_grounded_headline(
        Editor(), units, exchange_spans=spans, factual_audit=audit
    )
    assert result["hook_status"] == "accepted" and len(result["headline_audits"]) == 2
    assert calls[0]["source_passages"] == calls[1]["source_passages"]
    assert result["headline_source_spans"] == spans
    assert calls[1]["failed_dimensions"] == ["quantities_outcomes"]
    replies = iter(["A perfectly factual but incomplete fragment"] * 2)
    result = editor._source_grounded_headline(
        Editor(),
        units,
        exchange_spans=spans,
        factual_audit=lambda *args: dict(
            verdict="supported", headline_self_contained=False, central_highlight=True
        ),
    )
    assert result["headline_supported"] is True
    assert result["headline_self_contained"] is False
    assert result["hook_status"] == "blocked_after_bounded_revision"


def test_nli_fixed_labels_and_invalid_probabilities():
    import pytest

    from scripts.tjr_semantic_editor import _nli_verdict

    assert _nli_verdict([0.9, 0.05, 0.05]) == "contradiction"
    assert _nli_verdict([0.05, 0.9, 0.05]) == "entailment"
    assert _nli_verdict([0.05, 0.05, 0.9]) == "neutral"
    for scores in ([], [1.0], [0.1, 0.1, 0.1], [-0.1, 1.1, 0], [float("nan"), 0, 1]):
        with pytest.raises(RuntimeError, match="invalid NLI probabilities"):
            _nli_verdict(scores)


def test_nli_probe_reuses_exact_source_claim_proofs_without_loading_model(tmp_path, monkeypatch):
    import builtins
    import hashlib
    import importlib.metadata
    import json

    from scripts.tjr_semantic_editor import reviewer_nli_probe

    fixtures = [
        {
            "headline": f"Claim number {index} is stated here",
            "request": {
                "messages": [
                    {},
                    {
                        "content": json.dumps(
                            {
                                "headline": f"Claim number {index} is stated here",
                                "source_units": [
                                    {"id": 0, "text": "Some unchanged exact source words."}
                                ],
                            }
                        )
                    },
                ]
            },
            "expected_supported": index < 2,
        }
        for index in range(8)
    ]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "experiment": "source_presentation_x_verdict_schema",
                "annotated_fixtures": fixtures,
            }
        )
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "pinned-runtime")
    imports = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            imports.append(name)
            raise ImportError("Diagnostic must not load models in this unit test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    output = tmp_path / "nli.json"
    # Missing proofs produce recorded failures; no fabricated model verdicts.
    assert reviewer_nli_probe(baseline, output) == 1
    report = json.loads(output.read_text())
    assert len(imports) == 8 and report["cache_hits"] == 0
    profile = report["model_profile"]
    for row in report["comparisons"]:
        row.pop("error")
        row["probabilities"] = (
            [0.05, 0.9, 0.05] if row["fixture"]["expected_supported"] else [0.05, 0.05, 0.9]
        )
    output.write_text(json.dumps(report))
    imports.clear()
    assert reviewer_nli_probe(baseline, output) == 0
    report = json.loads(output.read_text())
    assert not imports and report["cache_hits"] == 8
    assert report["production_approved"] is False
    assert all(row["request"]["truncation"] is False for row in report["comparisons"])
    for row in report["comparisons"]:
        assert (
            row["request_key"]
            == hashlib.sha256(
                json.dumps(
                    {
                        "profile": profile,
                        "request": row["request"],
                        "code": "nli-full-premise-v1",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
    changed_payload = json.loads(fixtures[0]["request"]["messages"][1]["content"])
    changed_payload["source_units"][0]["text"] += " Changed."
    fixtures[0]["request"]["messages"][1]["content"] = json.dumps(changed_payload)
    baseline.write_text(
        json.dumps(
            {
                "experiment": "source_presentation_x_verdict_schema",
                "annotated_fixtures": fixtures,
            }
        )
    )
    assert reviewer_nli_probe(baseline, output) == 1
    assert len(imports) == 1
    assert json.loads(output.read_text())["cache_hits"] == 7


def test_nli_adapter_reads_persisted_audit_request_and_preserves_full_source():
    import copy
    import json

    import pytest

    from scripts.tjr_semantic_editor import _nli_request

    units = [
        {"id": 0, "text": "She reported no earnings from her social audience."},
        {"id": 1, "text": "Those views only brought listeners to her podcast."},
    ]
    fixture = {
        "headline": "Social views brought listeners without paying her",
        "request": {
            "messages": [
                {"role": "system", "content": "An old audit instruction"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_units": units,
                            "headline": "Social views brought listeners without paying her",
                        }
                    ),
                },
            ]
        },
    }
    request = _nli_request(fixture)
    assert request == {
        "premise": " ".join(unit["text"] for unit in units),
        "hypothesis": fixture["headline"],
        "truncation": False,
    }
    for changed in (
        [{"id": True, "text": units[0]["text"]}],
        [{"id": 9, "text": units[0]["text"]}],
        [{"id": 0, "text": ""}],
        [],
    ):
        bad = copy.deepcopy(fixture)
        payload = json.loads(bad["request"]["messages"][1]["content"])
        payload["source_units"] = changed
        bad["request"]["messages"][1]["content"] = json.dumps(payload)
        with pytest.raises(ValueError, match="invalid NLI"):
            _nli_request(bad)
    with pytest.raises(ValueError, match="invalid NLI"):
        _nli_request({**fixture, "headline": "A different unsupported claim"})


def test_unstructured_source_reasoning_truncation_fails_and_retains_request(tmp_path):
    import pytest

    from scripts.tjr_semantic_editor import LocalContextualEditor, ReviewRequestCache

    requests = []

    class Model:
        def create_chat_completion(self, **request):
            requests.append(request)
            return {"choices": [{"finish_reason": "length", "message": {"content": "Incomplete"}}]}

    instance = LocalContextualEditor.__new__(LocalContextualEditor)
    instance.model = Model()
    cache = ReviewRequestCache(tmp_path / "requests.json", lambda: instance, {"source": "exact"})
    with pytest.raises(RuntimeError, match="truncated evidence"):
        cache.semantic_draft(
            "Answer from source", {"source_units": ["An exact source passage."]}, 128
        )
    assert "response_format" not in requests[0]
    assert requests[0]["seed"] == 0 and requests[0]["temperature"] == 0
    assert cache.metrics["model_calls"] == 1 and cache.metrics["cache_hits"] == 0
    assert len(cache.calls) == 1 and "error" in cache.calls[0]
    assert cache.records == {}
    assert cache.calls[0]["raw_calls"][0]["response"]["choices"][0]["finish_reason"] == "length"


def test_qa_qualification_adapters_preserve_sources_before_model_loading(tmp_path, monkeypatch):
    import hashlib
    import json

    from clipper.models import WordTiming
    from scripts import tjr_semantic_editor as editor

    text = "A preserved original source passage with a complete thought."
    transcript = tmp_path / "transcript.json"
    segment = TranscriptSegment(0, 22, text, (WordTiming(0, 1, "A"),))
    transcript.write_text(json.dumps([segment.to_dict()]))
    transcript_hash = hashlib.sha256(
        json.dumps([segment.to_dict()], sort_keys=True).encode()
    ).hexdigest()
    provenance = tmp_path / "editorial-cache.json"
    provenance.write_text(
        json.dumps({"identity": dict(source_sha256="a" * 64, transcript_sha256=transcript_hash)})
    )
    fixture = dict(
        headline="An original source highlight",
        expected_supported=False,
        expected_unsupported_component="actor_action",
        request={
            "messages": [
                {"role": "system", "content": "saved control"},
                {
                    "role": "user",
                    "content": json.dumps({"source_units": [{"id": 0, "text": text}]}),
                },
            ]
        },
    )
    fixtures = [{**fixture, "expected_supported": index < 2} for index in range(8)]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            dict(experiment="source_presentation_x_verdict_schema", annotated_fixtures=fixtures)
        )
    )
    calls = []
    monkeypatch.setattr(editor, "LocalSourceReviewer", lambda profile: calls.append(True))
    import pytest

    with pytest.raises(ValueError, match="missing qualification thought units"):
        editor.reviewer_evidence_qualification(baseline, transcript, tmp_path / "proof.json")
    assert calls == []
    provenance.write_text(
        json.dumps({"identity": dict(source_sha256="a" * 64, transcript_sha256="b" * 64)})
    )
    import pytest

    with pytest.raises(ValueError, match="identity does not match"):
        editor.reviewer_evidence_qualification(baseline, transcript, tmp_path / "proof.json")
    assert calls == []
    fixtures[0]["request"]["messages"][1]["content"] = json.dumps(
        {"source_units": [{"id": 0, "text": "Substituted invented source words."}]}
    )
    baseline.write_text(
        json.dumps(
            dict(experiment="source_presentation_x_verdict_schema", annotated_fixtures=fixtures)
        )
    )
    with pytest.raises(ValueError, match="do not match"):
        editor.reviewer_evidence_qualification(baseline, transcript, tmp_path / "proof.json")
    assert calls == []


def test_structured_claim_probe_reuses_verified_baseline_without_rerunning_old_gates(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    from clipper.editorial_benchmark import HeldoutClaim
    from scripts import tjr_semantic_editor as editor

    transcript_data = [{"start": 0.0, "end": 1.0, "text": "Exact source speech."}]
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps(transcript_data))
    transcript_sha = hashlib.sha256(
        json.dumps(transcript_data, sort_keys=True).encode()
    ).hexdigest()
    source_sha = "a" * 64
    (tmp_path / "editorial-cache.json").write_text(
        json.dumps({"identity": {"source_sha256": source_sha, "transcript_sha256": transcript_sha}})
    )
    fixtures = [
        {
            "headline": f"Case {index}",
            "expected_supported": index % 2 == 0,
            "expected_unsupported_component": None,
            "request": {
                "messages": [
                    {"role": "system", "content": "saved"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"source_units": [{"id": 0, "text": "Exact source speech."}]}
                        ),
                    },
                ]
            },
        }
        for index in range(12)
    ]
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps({"source_video_id": "_kDrxucOx9g"}))
    heldout_sha = hashlib.sha256(
        json.dumps(json.loads(heldout_path.read_text()), sort_keys=True).encode()
    ).hexdigest()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "experiment": "evidence_preserving_source_qa",
                "experiment_complete": True,
                "transcript_sha256": transcript_sha,
                "model_profile": {
                    "source_sha256": source_sha,
                    "model_sha256": editor._review_model_profile()["sha256"],
                },
                "heldout_fixture_sha256": heldout_sha,
                "annotated_fixtures": fixtures,
                "comparisons": [{} for _ in range(12)],
                "heldout_comparisons": [{} for _ in range(12)],
            }
        )
    )
    heldout = [
        HeldoutClaim(
            case_id=f"heldout_{index}",
            headline=f"Heldout {index}",
            expected_supported=index % 2 == 0,
            annotation_reason="Synthetic routing test only.",
            annotation_status="transcript_only_pending_audio_review",
            source_start=0.0,
            source_end=1.0,
            source_clip="00-double-coverage-_kDrxucOx9g.mp4",
            source_units=("Exact source speech.",),
        )
        for index in range(12)
    ]
    monkeypatch.setattr(editor, "load_heldout_claims", lambda *args: heldout)
    monkeypatch.setattr(
        editor,
        "LocalSourceReviewer",
        lambda *args: (_ for _ in ()).throw(AssertionError("old gate ran")),
    )
    monkeypatch.setattr(
        editor,
        "audit_structured_claims",
        lambda _reviewer, headline, _units, **_kwargs: {
            "validated": {"all_claims_supported": int(headline.split()[-1]) % 2 == 0}
        },
    )
    output = tmp_path / "proof.json"
    assert (
        editor.reviewer_evidence_qualification(
            baseline,
            transcript,
            output,
            structured_claim_probe=True,
            heldout_path=heldout_path,
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["structured_claim_scores"]["frozen_correct"] == 12
    assert report["structured_claim_scores"]["heldout_correct"] == 12
    assert report["baseline_reference"]["replayed_model_calls"] == 0
    assert report["request_cache_metrics"]["model_calls"] == 0
    assert report["production_approved"] is False
    changed = json.loads(baseline.read_text())
    changed["heldout_fixture_sha256"] = "0" * 64
    baseline.write_text(json.dumps(changed))
    import pytest

    with pytest.raises(ValueError, match="complete exact-source proof"):
        editor.reviewer_evidence_qualification(
            baseline,
            transcript,
            tmp_path / "rejected.json",
            structured_claim_probe=True,
            heldout_path=heldout_path,
        )


def test_source_unit_evidence_rejects_invalid_contracts_and_preserves_repeated_positions():
    import pytest

    from scripts.tjr_semantic_editor import _resolve_source_units, _unit_span_valid

    units = [
        "This exact sentence repeats.",
        "Some other complete thought.",
        "This exact sentence repeats.",
    ]
    span = _resolve_source_units(dict(first_unit=2, last_unit=2), units)
    assert span == dict(text=units[2], first_unit=2, last_unit=2)
    assert _unit_span_valid(span, units)
    assert _resolve_source_units(dict(first_unit=-1, last_unit=-1), units) is None
    assert _resolve_source_units(dict(first_unit=0, last_unit=2), units)["text"] == " ".join(units)
    for pointer in (
        dict(first_unit=True, last_unit=1),
        dict(first_unit=-1, last_unit=0),
        dict(first_unit=2, last_unit=1),
        dict(first_unit=0, last_unit=3),
        dict(first_unit=0, last_unit=0, text="A rewritten quotation"),
        dict(first_unit=0),
    ):
        with pytest.raises(RuntimeError):
            _resolve_source_units(pointer, units)
    assert not _unit_span_valid(dict(text="A stitched quote", first_unit=0, last_unit=2), units)


def test_position_review_preserves_evidence_and_excluded_namespace(monkeypatch):
    from scripts import tjr_semantic_editor as editor

    units = ["I was stuck outside the building.", "The owner let me inside."]
    captured = []
    relation = {"value": "new_topic"}
    promotion = {"value": False}

    class Model:
        def _review_completion(self, prompt, payload, properties, tokens):
            if "ad_read_span" in properties:
                return dict(
                    ad_read_present=int(promotion["value"]),
                    ad_read_span=dict(
                        first_unit=0 if promotion["value"] else -1,
                        last_unit=0 if promotion["value"] else -1,
                    ),
                    show_intro_present=0,
                    show_intro_span=dict(first_unit=-1, last_unit=-1),
                    reason="An ordinary account.",
                )
            if "setup_span" in properties:
                assert "excluded_after" not in payload
                return dict(
                    setup_span=dict(first_unit=0, last_unit=0),
                    resolution_span=dict(first_unit=1, last_unit=1),
                    opening_independent=1,
                    last_thought_finished=1,
                    reason="Owner resolves exclusion.",
                )
            assert payload["excluded_after"] == [
                dict(id=0, text="Then we discussed a different topic.")
            ]
            return dict(
                final_span=dict(first_unit=1, last_unit=1),
                continuation_span=dict(first_unit=0, last_unit=0),
                relation=relation["value"],
                reason="A separate topic follows.",
            )

    def generate(instance, delivered, *, exchange_spans, factual_audit):
        captured.append(exchange_spans)
        assert delivered == units
        assert factual_audit is editor._position_headline_audit
        return dict(
            headline="The owner rescued his guest from security",
            headline_supported=True,
            headline_self_contained=True,
        )

    monkeypatch.setattr(editor, "_source_grounded_headline", generate)
    context = dict(selected_units=units, before=[], after=["Then we discussed a different topic."])
    review = editor._source_position_review(Model(), context)
    assert review["exchange_accepted"] is True
    assert captured == [review["source_quote_spans"]]
    assert captured[0]["resolution_quote"]["text"] == units[1]
    assert review["continuation_review"]["continuation_span_source"]["text"] == context["after"][0]
    relation["value"] = "missing_answer"
    rejected = editor._source_position_review(Model(), context)
    assert rejected["payoff_complete"] is False
    assert rejected["ending_complete"] is False
    assert "exchange_accepted" not in rejected
    assert len(captured) == 1

    # A purpose veto must not conceal a separate missing-payoff failure.
    promotion["value"] = True
    rejected = editor._source_position_review(Model(), context)
    assert rejected["contains_promotion_or_intro"] is True
    assert rejected["continuation_review"]["relation"] == "missing_answer"
    assert rejected["payoff_complete"] is False
    assert rejected["ending_complete"] is False
    assert "exchange_accepted" not in rejected
    assert len(captured) == 1

    # Optional continuation leaves the ending complete even when purpose rejects it.
    relation["value"] = "optional_elaboration"
    rejected = editor._source_position_review(Model(), context)
    assert rejected["payoff_complete"] is True
    assert rejected["ending_complete"] is True
    assert "exchange_accepted" not in rejected
    assert len(captured) == 1


def test_source_bound_diagnostic_cannot_approve_its_own_model_audit():
    from scripts.tjr_semantic_editor import _source_bound_headline_diagnostic

    units = [
        "Because right now I got like 60 million views on my Instagram.",
        "No, it just drives the podcast.",
    ]
    spans = {
        "setup_quote": {"text": units[0], "first_unit": 0, "last_unit": 0},
        "resolution_quote": {"text": units[1], "first_unit": 1, "last_unit": 1},
    }

    class Editor:
        def _review_completion(self, prompt, payload, properties, tokens):
            return {
                "setup_quote": {"unit_id": 0, "text": "60 million views on my Instagram"},
                "resolution_quote": {"unit_id": 1, "text": "No, it just drives the podcast."},
            }

    result = _source_bound_headline_diagnostic(
        Editor(),
        units,
        exchange_spans=spans,
        factual_audit=lambda *_: {
            "verdict": "supported",
            "headline_self_contained": True,
            "central_highlight": True,
        },
    )
    assert result["source_bound"] is True
    assert result["candidate_model_audit"]["verdict"] == "supported"
    assert result["headline_supported"] is False
    assert result["production_approved"] is False


def test_position_reviewer_fingerprint_serializes_injected_headline_type():
    from scripts.tjr_semantic_editor import _source_position_review, _stage_fingerprint

    assert len(_stage_fingerprint(_source_position_review)) == 64


def test_continuation_is_bound_to_last_substantive_business_point():
    import pytest

    from scripts.tjr_semantic_editor import _final_substantive_unit_id, _source_position_review

    selected = [
        "Because right now I got like 60 million views on my Instagram.",
        "And like, my thing is going crazy.",
        "I'm like not making a dime off of it.",
        "Well, bro, if you had a podcast getting 60 million views, "
        "you'd be making millions of dollars.",
        "Yeah.",
        "So that's the thing.",
        "You got to transition to somewhere that's monetizable.",
        "Yeah, for sure.",
        "Yeah.",
        "You don't make anything off Instagram.",
        "Not really.",
        "No, it just drives the podcast.",
        "Yeah.",
    ]
    assert _final_substantive_unit_id(selected) == 11

    class Model:
        def __init__(self, wrong_final=False, wrong_resolution=False):
            self.wrong_final = wrong_final
            self.wrong_resolution = wrong_resolution

        def _review_completion(self, prompt, payload, properties, tokens):
            if "ad_read_span" in properties:
                return dict(
                    ad_read_present=0,
                    ad_read_span=dict(first_unit=-1, last_unit=-1),
                    show_intro_present=0,
                    show_intro_span=dict(first_unit=-1, last_unit=-1),
                    reason="Ordinary business discussion.",
                )
            if "setup_span" in properties:
                assert payload["final_substantive_unit_id"] == 11
                assert properties["resolution_span"]["properties"]["last_unit"]["enum"] == [
                    -1,
                    11,
                ]
                return dict(
                    setup_span=dict(first_unit=0, last_unit=2),
                    resolution_span=dict(
                        first_unit=6, last_unit=6 if self.wrong_resolution else 11
                    ),
                    opening_independent=1,
                    last_thought_finished=1,
                    reason="The conversation has a delivered contrast.",
                )
            assert payload["final_substantive_unit_id"] == 11
            assert payload["final_substantive_unit_text"] == selected[11]
            assert properties["final_span"]["properties"]["first_unit"]["enum"] == [11]
            return dict(
                final_span=dict(
                    first_unit=0 if self.wrong_final else 11,
                    last_unit=0 if self.wrong_final else 11,
                ),
                continuation_span=dict(first_unit=0, last_unit=1),
                relation="optional_elaboration",
                reason="The tools list elaborates an already delivered contrast.",
            )

    review = _source_position_review(
        Model(),
        dict(
            selected_units=selected,
            before=[],
            after=["It's like tools.", "Instagram, Facebook, TikTok."],
        ),
        headline_generator=lambda *args, **kwargs: dict(
            headline="",
            headline_supported=False,
            headline_self_contained=False,
        ),
    )
    assert review["payoff_complete"] is True
    assert review["continuation_review"]["final_span_source"]["text"] == selected[11]
    with pytest.raises(RuntimeError, match="resolution omitted the final substantive"):
        _source_position_review(
            Model(wrong_resolution=True),
            dict(selected_units=selected, before=[], after=["It's like tools."]),
            headline_generator=lambda *args, **kwargs: {},
        )
    with pytest.raises(RuntimeError, match="ignored the final substantive unit"):
        _source_position_review(
            Model(wrong_final=True),
            dict(selected_units=selected, before=[], after=["It's like tools."]),
            headline_generator=lambda *args, **kwargs: {},
        )


def test_source_bound_probe_rejects_changed_transcript_before_model_loading(tmp_path, monkeypatch):
    import pytest

    from scripts import tjr_semantic_editor as editor

    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        editor,
        "LocalSourceReviewer",
        lambda *_: (_ for _ in ()).throw(AssertionError("model weights loaded")),
    )
    with pytest.raises(ValueError, match="pinned exact-source transcript"):
        editor.source_bound_headline_probe(transcript, tmp_path / "probe.json")


def test_position_purpose_rejects_label_without_matching_evidence():
    import pytest

    from scripts.tjr_semantic_editor import _source_position_review

    class Model:
        def _review_completion(self, prompt, payload, properties, tokens):
            return dict(
                ad_read_present=1,
                ad_read_span=dict(first_unit=-1, last_unit=-1),
                show_intro_present=0,
                show_intro_span=dict(first_unit=-1, last_unit=-1),
                reason="Claims an ad but supplies no delivered evidence.",
            )

    with pytest.raises(RuntimeError, match="label and evidence positions disagree"):
        _source_position_review(Model(), dict(selected_units=["A complete exchange."], after=[]))


def test_position_headline_unclaimed_dimensions_do_not_veto_supported_claim():
    from scripts.tjr_semantic_editor import _qa_headline_audit

    class Model:
        def _review_completion(self, prompt, payload, properties, tokens):
            assert "not_claimed" in properties["setting_time"]["enum"]
            return dict(
                actor_action="supported",
                relationship_role="not_claimed",
                setting_time="not_claimed",
                quantities_outcomes="not_claimed",
                headline_self_contained=1,
                central_highlight=1,
                reason="Actor and action are explicit.",
            )

    result = _qa_headline_audit(
        Model(),
        "Bobby Green paced during the meeting",
        ["Bobby Green paced during the meeting."],
        fact_backend=lambda editor, units: {"facts": {}},
    )
    assert result["verdict"] == "supported"


def test_position_headline_cannot_pass_without_any_supported_claim():
    from scripts.tjr_semantic_editor import _qa_headline_audit

    class Model:
        def _review_completion(self, prompt, payload, properties, tokens):
            return {
                **dict.fromkeys(
                    ("actor_action", "relationship_role", "setting_time", "quantities_outcomes"),
                    "not_claimed",
                ),
                "headline_self_contained": 1,
                "central_highlight": 1,
                "reason": "No factual claim identified.",
            }

    result = _qa_headline_audit(
        Model(),
        "An empty tease without a fact",
        ["Bobby Green paced during the meeting."],
        fact_backend=lambda editor, units: {"facts": {}},
    )
    assert result["verdict"] == "uncertain"


def test_position_facts_are_blind_and_extract_source_text():
    import pytest

    from scripts import tjr_semantic_editor as editor

    units = ["The guest waited outside.", "The owner opened the door."]
    malformed = {"value": False}

    class Model:
        def _review_completion(self, prompt, payload, properties, tokens):
            assert set(payload) == {"source_units"}
            assert "headline" not in payload
            answer = "An unstated relationship" if malformed["value"] else "unknown"
            return {
                key: dict(
                    answer="The owner opened the door." if key == "actor_action" else answer,
                    evidence=dict(
                        first_unit=1 if key == "actor_action" else -1,
                        last_unit=1 if key == "actor_action" else -1,
                    ),
                )
                for key in editor._HEADLINE_COMPONENTS
            }

    facts = editor._source_position_facts(Model(), units)
    assert facts["facts"]["actor_action"]["source_span"]["text"] == units[1]
    assert facts["facts"]["relationship_role"]["source_span"] is None
    malformed["value"] = True
    with pytest.raises(RuntimeError, match="unknown"):
        editor._source_position_facts(Model(), units)


def test_reviewer_profile_change_preserves_selection_and_invalidates_completion(
    tmp_path, monkeypatch
):
    from clipper.models import ClipCandidate
    from scripts import tjr_modal_runner as runner
    from scripts import tjr_semantic_editor as editor

    segments = [
        TranscriptSegment(0, 7, "How did security stop you at your own show?"),
        TranscriptSegment(7, 14, "I had left my own access pass in the car."),
        TranscriptSegment(14, 22, "The owner came outside and finally let me inside."),
    ]
    proposal = ClipCandidate("v", 0, 22, " ".join(s.text for s in segments), 10)
    calls = {"selector": 0, "reviewer": 0}

    def assess(context):
        calls["selector"] += 1
        return dict(
            keep=True, start_unit=0, end_unit=2, opening=4, story=4, ending=5, reason="Pass found."
        )

    def review(context):
        calls["reviewer"] += 1
        return _review(context)

    first = tmp_path / "first.json"
    editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=first,
        assessor=assess,
        reviewer=review,
    )
    completion = runner._pipeline_identity("a" * 64)
    profile = editor._review_model_profile()
    monkeypatch.setattr(editor, "_review_model_profile", lambda: dict(profile, context_tokens=2048))
    _, audit = editor.refine_contextual_candidates(
        _brief(),
        [proposal],
        segments,
        source_sha256="a" * 64,
        cache_path=tmp_path / "second.json",
        reuse_path=first,
        assessor=assess,
        reviewer=review,
    )
    assert calls == {"selector": 1, "reviewer": 2}
    assert audit["selector_cache_hits"] == 1 and audit["reviewer_cache_hits"] == 0
    assert runner._pipeline_identity("a" * 64) != completion


def test_review_cache_owns_raw_evidence_and_replay_never_loads_weights(tmp_path):
    import hashlib
    import json

    import pytest

    from scripts.tjr_semantic_editor import LocalContextualEditor, ReviewRequestCache

    class Model:
        def create_chat_completion(self, **request):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"reason":"Original source judgment"}'},
                    }
                ]
            }

    instance = LocalContextualEditor.__new__(LocalContextualEditor)
    instance.model = Model()
    path = tmp_path / "cache.json"
    identity = {"source_sha256": "a" * 64, "model_sha256": "b" * 64}
    cache = ReviewRequestCache(path, lambda: instance, identity)
    properties = {"reason": {"type": "string"}}
    received = cache._review_completion("Evidence", {"units": ["Exact source"]}, properties, 64)
    received["source_span"] = {"first_unit": 0, "last_unit": 0}
    record = next(iter(cache.records.values()))
    assert "source_span" not in record["response"]
    assert "source_span" not in cache.calls[0]["response"]
    assert (
        hashlib.sha256(json.dumps(record["response"], sort_keys=True).encode()).hexdigest()
        == record["response_sha256"]
    )

    def forbidden():
        raise AssertionError("Replay must not load weights")

    warm = ReviewRequestCache(
        path, forbidden, identity, replay_only=True, recorded_runtime=cache.identity["runtime"]
    )
    assert warm._review_completion("Evidence", {"units": ["Exact source"]}, properties, 64) == {
        "reason": "Original source judgment"
    }
    assert warm.metrics["model_calls"] == 0 and warm.metrics["cache_hits"] == 1
    with pytest.raises(RuntimeError, match="inference is forbidden"):
        warm._review_completion("Changed request", {"units": ["Exact source"]}, properties, 64)
    with pytest.raises(ValueError, match="permitted only"):
        ReviewRequestCache(path, forbidden, identity, recorded_runtime="0.3.35")


def test_review_cache_recovery_requires_original_api_checksum(tmp_path):
    import json

    import pytest

    from scripts.tjr_semantic_editor import LocalContextualEditor, ReviewRequestCache

    class Model:
        def create_chat_completion(self, **request):
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"reason":"Original"}'}}
                ]
            }

    instance = LocalContextualEditor.__new__(LocalContextualEditor)
    instance.model = Model()
    path = tmp_path / "cache.json"
    identity = {"source_sha256": "a" * 64}
    original = ReviewRequestCache(path, lambda: instance, identity)
    properties = {"reason": {"type": "string"}}
    original._review_completion("Evidence", {}, properties, 64)
    saved = json.loads(path.read_text())
    record = next(iter(saved["records"].values()))
    record["response"]["reason"] = "Tampered"
    record["raw_calls"][0]["response"]["choices"][0]["message"]["content"] = '{"reason":"Tampered"}'
    path.write_text(json.dumps(saved))
    replay = ReviewRequestCache(
        path,
        lambda: pytest.fail("No inference"),
        identity,
        replay_only=True,
        recorded_runtime=original.identity["runtime"],
    )
    with pytest.raises(RuntimeError, match="inference is forbidden"):
        replay._review_completion("Evidence", {}, properties, 64)
    assert replay.metrics["model_calls"] == 0 and replay.metrics.get("recovered_responses", 0) == 0


def test_gpu_qualification_cache_identity_ignores_packing_time(tmp_path, monkeypatch):
    import gzip
    import json
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from scripts import tjr_semantic_editor as editor

    calls = []

    def remote(packed, profile, key, code_hash):
        calls.append((packed, profile, key, code_hash))
        return {"files": [], "passed": False}

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(enable_output=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "scripts.tjr_modal_probe",
        SimpleNamespace(
            app=SimpleNamespace(run=nullcontext),
            qualify_source_reviewer_gpu=SimpleNamespace(remote=remote),
            volume=None,
        ),
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]")
    source_hash = "2a7e07b37074f3073d71b65e10a3efb4019b3cdd4277bc2d3770a99dcbc55e0a"
    transcript.with_name("editorial-cache.json").write_text(
        json.dumps({"identity": {"source_sha256": source_hash}})
    )
    monkeypatch.setattr(gzip.time, "time", lambda: 1)
    editor.reviewer_gpu_qualification(baseline, transcript, tmp_path / "first.json")
    first = list(calls)
    calls.clear()
    monkeypatch.setattr(gzip.time, "time", lambda: 10000)
    editor.reviewer_gpu_qualification(baseline, transcript, tmp_path / "second.json")
    assert calls == first
    assert len(calls) == 2
    assert json.loads(gzip.decompress(calls[0][0]))["transcript"] == []
