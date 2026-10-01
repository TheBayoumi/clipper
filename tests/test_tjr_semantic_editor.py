from clipper.models import CampaignBrief, TranscriptSegment
from scripts.tjr_semantic_editor import build_semantic_editorial_candidates


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
            opening=4,
            story=4,
            ending=4,
            hook=17,
            reason="Invalid rating.",
        )

    path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="out-of-range"):
        refine_contextual_candidates(
            _brief(), proposals, segments, source_sha256="a" * 64, cache_path=path, assessor=assess
        )
    saved = json.loads(path.read_text())
    assert saved["complete"] is False and saved["processed_count"] == 1
    assert saved["decisions"][1]["decision"]["hook"] == 17

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

    context = dict(
        selected_units=["The owner let me into my own show."],
        before=[],
        after=["What happened next?"],
    )

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
                                    **{
                                        key: int(value) if type(value) is bool else value
                                        for key, value in _review(context).items()
                                    },
                                    "reason": "The selected speech resolves the stated setup.",
                                }
                            )
                        },
                    }
                ]
            }

    editor = object.__new__(LocalContextualEditor)
    editor.model = Model()
    assert editor.review(context)["payoff_complete"] is True
    request = editor.model.request
    assert json.loads(request["messages"][1]["content"]) == context
    assert "ratings" not in request["messages"][1]["content"]
    assert request["response_format"]["schema"]["properties"]["contains_promotion_or_intro"] == {
        "type": "integer",
        "enum": [0, 1],
    }
    assert request["seed"] == 0 and request["temperature"] == 0


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
