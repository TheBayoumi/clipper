from dataclasses import replace

import pytest

from clipper.editorial import (
    _find_hook_sentence,
    _hook_start_anchor,
    _moment_type,
    _overlay_duplicates_opening,
    _source_excerpt,
    _topic,
    aggregate_editorial_score,
    build_edit_plan,
    cluster_concepts,
    discover_story_moments,
    end_boundary_score,
    generate_hook_variants,
    mine_clip_concepts,
    score_editorial_text,
    select_distinct_concepts,
    select_render_plan_queue,
    select_render_plans,
    select_submission_shortlist,
    semantic_similarity,
    start_boundary_score,
    transcript_fingerprint,
)
from clipper.models import (
    CampaignBrief,
    ClipConcept,
    EditorialScores,
    EditPlan,
    HookVariant,
    SourceSpan,
    StoryMoment,
    TranscriptSegment,
)


def brief() -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "c",
            "title": "Podcast clips",
            "objective": "Find strong self-contained creator stories",
            "keywords": ["creator", "money", "business", "Fortnite"],
            "allowed_video_ids": ["v"],
            "rights_confirmed": True,
            "min_clip_seconds": 12,
            "max_clip_seconds": 32,
            "production": {
                "candidate_pool_size": 30,
                "concept_count": 8,
                "variants_per_concept": 3,
                "final_render_budget": 6,
            },
            "diversity": {"semantic_similarity_threshold": 0.7, "max_concepts_per_topic": 2},
            "hooks": {
                "enabled": [
                    "direct",
                    "curiosity_text",
                    "question",
                    "number",
                    "conflict",
                    "strong_opinion",
                    "payoff_first",
                ]
            },
            "editorial": {
                "platform": "tiktok",
                "max_punch_ins_per_clip": 2,
                "post_speech_tail_seconds": 0.2,
            },
        }
    )


def transcript() -> list[TranscriptSegment]:
    rows = [
        (0, 4, "So I used to think streaming was easy, but I was completely wrong."),
        (4, 8, "The first month I made only 200 dollars from streaming."),
        (8, 12, "Then I learned why consistency matters for every creator."),
        (12, 16, "I posted every day and the business finally started growing."),
        (16, 20, "That changed everything for me."),
        (21, 25, "How did Fortnite become the thing that paid for my first car?"),
        (25, 29, "I was 16 when I won my first real tournament money."),
        (29, 33, "But the pressure was insane because everyone expected me to keep winning."),
        (33, 37, "I think confidence was the biggest difference."),
        (37, 41, "That is why I still compete now."),
        (42, 46, "My worst business mistake was hiring too fast."),
        (46, 50, "We spent 5000 dollars before the product was ready."),
        (50, 54, "The problem was I never asked what customers actually wanted."),
        (54, 58, "Eventually we fixed it by talking to users every week."),
        (58, 62, "That lesson saved the business."),
        (63, 67, "What is the best advice for a new creator?"),
        (67, 71, "Do not chase every trend just because it is popular."),
        (71, 75, "Build something people remember and keep showing up."),
        (75, 79, "That is the truth."),
    ]
    return [TranscriptSegment(start, end, text) for start, end, text in rows]


def scores(value: float = 8.0) -> EditorialScores:
    return EditorialScores(*(value for _ in range(12)))


def concept(
    concept_id: str,
    text: str,
    *,
    start: float = 0,
    end: float = 24,
    topic: str = "creator",
    cluster: str = "unassigned",
    score: float = 8.0,
) -> ClipConcept:
    return ClipConcept(
        concept_id,
        "v",
        start,
        end,
        text,
        topic,
        "setup",
        "payoff",
        "story",
        end - start,
        scores(),
        score,
        cluster,
        transcript_fingerprint(text),
    )


def test_start_and_end_boundary_scoring_penalizes_weak_incomplete_edges() -> None:
    assert start_boundary_score("Why did I risk $10,000 on this business?") > start_boundary_score(
        "So like I mean it was this thing"
    )
    complete = end_boundary_score("That lesson saved the business.", next_gap=0.8)
    assert complete > end_boundary_score("and then because", next_gap=0)
    assert complete > end_boundary_score("get motivated because of it but you're", next_gap=0.01)
    assert start_boundary_score("") == 0


def test_semantic_similarity_and_fingerprint_are_deterministic() -> None:
    first = "I made money building a creator business"
    similar = "My creator business made money"
    different = "The weather outside is cold and rainy"
    assert semantic_similarity(first, similar) > semantic_similarity(first, different)
    assert semantic_similarity("a an the", "the a") == 0
    assert transcript_fingerprint(first) == transcript_fingerprint(first.upper())


def test_editorial_scores_cover_required_dimensions() -> None:
    result = score_editorial_text(
        brief(),
        "Why did I lose 5000 dollars? I failed, learned the truth, and finally fixed the business.",
        start_text="Why did I lose 5000 dollars?",
        end_text="I finally fixed the business.",
        next_gap=1.0,
    )
    assert len(result.to_dict()) == 12
    assert result.hook_strength >= 6
    assert result.payoff_strength >= 6
    assert result.specificity >= 4
    assert 0 <= aggregate_editorial_score(result) <= 10


def test_story_moment_discovery_uses_natural_boundaries() -> None:
    moments = discover_story_moments(brief(), "v", transcript(), min_seconds=12, max_seconds=24)
    assert len(moments) >= 3
    assert all(moment.end > moment.start for moment in moments)
    assert all(moment.transcript_fingerprint for moment in moments)
    assert {moment.moment_type for moment in moments}
    assert moments[0].setup and moments[0].payoff
    assert discover_story_moments(brief(), "v", []) == []


def test_moment_type_topic_and_excerpt_are_source_derived() -> None:
    assert _moment_type("Why did this happen?") == "question_answer"
    assert _moment_type("I made 2 million dollars in the business.") == "money_story"
    assert _moment_type("My biggest failure taught me this.") == "failure_lesson"
    assert _moment_type("This is the best strategy.") == "strong_opinion"
    assert _moment_type("When I started, I learned because it was hard.") == "story"
    assert _moment_type("There were 4 people there.") == "specific_fact"
    assert _moment_type("Simple creator insight.") == "insight"
    assert _topic("Fortnite changed my career", brief().keywords) == "career"
    assert _topic("alpha alpha beta gamma", []) == "alpha-beta-gamma"
    assert _source_excerpt("one two three", max_words=8) == "ONE TWO THREE"
    assert _source_excerpt("one two three four five six seven eight nine", max_words=4).endswith(
        "…"
    )


def test_full_mining_produces_complete_bounded_concepts() -> None:
    b = brief()
    moments = discover_story_moments(b, "v", transcript(), min_seconds=12, max_seconds=24)
    concepts = mine_clip_concepts(b, "v", transcript(), moments)
    assert concepts
    assert len(concepts) <= b.production.candidate_pool_size
    assert all(b.min_clip_seconds <= item.duration <= b.max_clip_seconds for item in concepts)
    assert all(item.scores.story_completeness >= 4 for item in concepts)
    assert all(not item.text.lower().endswith(" because") for item in concepts)


def test_cluster_and_distinct_selection_reject_semantic_near_duplicates() -> None:
    b = brief()
    items = [
        concept("a", "I made money building a creator business", score=9),
        concept("b", "My creator business made money", start=40, end=64, score=8.8),
        concept(
            "c",
            "Fortnite pressure changed how I compete",
            start=80,
            end=104,
            topic="Fortnite",
            score=8.5,
        ),
        concept(
            "d",
            "Hiring too fast was my worst business mistake",
            start=120,
            end=144,
            topic="business",
            score=8.4,
        ),
    ]
    clustered = cluster_concepts(items, similarity_threshold=0.55)
    assert clustered[0].semantic_cluster == clustered[1].semantic_cluster
    selected = select_distinct_concepts(b, items)
    assert len({item.semantic_cluster for item in selected}) == len(selected)
    assert len(selected) >= 2


def test_housekeeping_is_not_mined_and_number_hook_requires_meaningful_number() -> None:
    b = brief()
    noisy = [
        TranscriptSegment(0, 6, "What's up guys and this is the podcast intro."),
        TranscriptSegment(6, 14, "We have 100000 dollars on the line for this game."),
        TranscriptSegment(14, 22, "That is why this match matters."),
    ]
    moments = discover_story_moments(b, "v", noisy, min_seconds=8, max_seconds=24)
    concepts = mine_clip_concepts(b, "v", noisy, moments)
    assert all("what's up guys" not in item.text.lower() for item in concepts)
    assert _find_hook_sentence("Before we head out, what's one message for fans?", "number") is None
    assert _find_hook_sentence("I won 100000 dollars in the final.", "number") is not None


def test_hook_variants_are_legitimate_and_not_filename_duplicates() -> None:
    b = brief()
    c = concept(
        "c1",
        (
            "How did I make 5000 dollars? "
            "I think consistency is the best advantage, but pressure is real."
        ),
        end=28,
    )
    segs = [
        TranscriptSegment(0, 8, "How did I make 5000 dollars?"),
        TranscriptSegment(8, 18, "I think consistency is the best advantage."),
        TranscriptSegment(18, 28, "But pressure is real."),
    ]
    variants = generate_hook_variants(b, c, segs)
    assert len(variants) == b.production.variants_per_concept
    assert len({item.fingerprint for item in variants}) == len(variants)
    assert all(item.source_spans for item in variants)
    assert all(item.mode != "payoff_first" for item in variants)
    assert _find_hook_sentence(c.text, "question") is not None
    assert _find_hook_sentence(c.text, "number") is not None
    assert _find_hook_sentence(c.text, "conflict") is not None
    assert _find_hook_sentence(c.text, "strong_opinion") is not None
    assert _find_hook_sentence("plain text.", "number") is None
    trimmed = _find_hook_sentence(
        "Dude, before we head out, what's one message for esports fans?", "question"
    )
    assert trimmed is not None
    assert trimmed.lower().startswith("what's one message")
    for source in (
        "Uh, before we head out, what's one message for esports fans?",
        "Um, so anyway, what is the biggest lesson you learned?",
        "You know, dude, how did you actually make it work?",
    ):
        question = _find_hook_sentence(source, "question")
        assert question is not None
        assert not question.lower().startswith(("uh", "um", "you know", "dude", "before"))
    assert (
        _find_hook_sentence("When I was young is I couldn't really buy stuff.", "question") is None
    )
    assert _find_hook_sentence("I did it because buying the car motivated me.", "question") is None


def test_edit_plan_expands_unanswered_setup_to_first_real_payoff_and_caps_tail() -> None:
    b = brief()
    c = replace(
        concept(
            "origin",
            "How did you start? I needed a computer. My brother had one to play.",
            start=0,
            end=20,
        ),
        setup="How did you start?",
        payoff="My brother had one to play.",
        scores=replace(scores(6.5), payoff_strength=6.5),
    )
    variant = HookVariant("h", "origin", "direct", (SourceSpan(5, 20),), None, 8, "direct", "fp")
    segs = [
        TranscriptSegment(18, 20, "My brother had one to play."),
        TranscriptSegment(20.01, 22, "And he got me a computer, but he said,"),
        TranscriptSegment(22.01, 24, "you have to pay me back this summer"),
        TranscriptSegment(24.01, 26, "or work for me for the whole year."),
        TranscriptSegment(26.01, 28, "I paid him back in two weeks."),
        TranscriptSegment(28.01, 30, "Then I qualified for another event."),
    ]
    plan = build_edit_plan(b, c, variant, segs)
    assert plan.source_spans[0].end == pytest.approx(28.01)

    complete = replace(c, setup="First car at 16.", payoff="You know what I'm saying?")
    direct = HookVariant("h2", "origin", "direct", (SourceSpan(5, 20),), None, 8, "direct", "fp2")
    adjacent = [
        TranscriptSegment(5, 20, "You know what I'm saying?"),
        TranscriptSegment(20.01, 22, "What's next?"),
    ]
    plan2 = build_edit_plan(b, complete, direct, adjacent)
    assert plan2.source_spans[0].end == pytest.approx(20.01)


def test_edit_plan_has_limited_meaningful_punch_ins_and_tail() -> None:
    b = brief()
    c = concept(
        "c1",
        "Creator context. I made 5000 dollars. This was the biggest problem. That lesson saved me.",
        end=24,
    )
    variant = HookVariant("v1", "c1", "direct", (SourceSpan(0, 24),), None, 8.5, "direct", "fp")
    segs = [
        TranscriptSegment(0, 4, "Creator context."),
        TranscriptSegment(6, 9, "I made 5000 dollars."),
        TranscriptSegment(12, 15, "This was the biggest problem."),
        TranscriptSegment(20, 24, "That lesson saved me."),
    ]
    plan = build_edit_plan(b, c, variant, segs)
    punch = [beat for beat in plan.beats if beat.beat_type == "punch_in"]
    assert b.editorial.punch_ins_enabled is False
    assert punch == []
    assert plan.source_spans[0].end == pytest.approx(24.2)
    assert any(beat.beat_type == "payoff_hold" for beat in plan.beats)
    assert plan.to_clip_candidate(c.text).end == pytest.approx(24.2)


def test_source_span_and_multispan_render_guard() -> None:
    with pytest.raises(ValueError, match="source span"):
        SourceSpan(2, 1)
    plan = EditPlan(
        "p",
        "v",
        "c",
        "h",
        "direct",
        (SourceSpan(0, 2), SourceSpan(3, 5)),
        None,
        (),
        "tiktok",
        8,
        "fp",
    )
    with pytest.raises(ValueError, match="contiguous"):
        plan.to_clip_candidate("text")


def test_render_budget_prioritizes_concept_diversity_then_variants() -> None:
    def plan(name: str, concept_id: str, score: float) -> EditPlan:
        return EditPlan(
            name,
            "v",
            concept_id,
            name,
            "direct",
            (SourceSpan(0, 20),),
            None,
            (),
            "tiktok",
            score,
            concept_id,
        )

    plans = [plan("a1", "a", 9), plan("a2", "a", 8.9), plan("b1", "b", 8.8), plan("c1", "c", 8.7)]
    assert [item.concept_id for item in select_render_plans(plans, budget=3)] == ["a", "b", "c"]
    assert len(select_render_plans(plans, budget=4)) == 4
    shortlist = select_submission_shortlist(plans, clip_count=2, max_per_source=2)
    assert len(shortlist) == 2 and len({item.concept_id for item in shortlist}) == 2


def test_render_queue_treats_model_plan_ids_as_concept_scoped() -> None:
    def plan(concept_id: str, variant_id: str, score: float, start: float) -> EditPlan:
        return EditPlan(
            "p1",
            "video",
            concept_id,
            variant_id,
            "direct",
            (SourceSpan(start, start + 20),),
            None,
            (),
            "tiktok",
            score,
            f"fp-{concept_id}",
        )

    plans = [
        plan("c1", "v1", 9.5, 0),
        plan("c2", "v1", 9.4, 60),
        plan("c3", "v1", 9.3, 120),
        plan("c4", "v1", 9.2, 180),
    ]
    primary, reserve = select_render_plan_queue(plans, budget=3)
    assert len(primary) == 3
    assert len({item.concept_id for item in primary}) == 3
    assert len(reserve) == 1
    assert reserve[0].concept_id not in {item.concept_id for item in primary}


def test_distinct_selection_respects_topic_cap() -> None:
    b = replace(brief(), diversity=replace(brief().diversity, max_concepts_per_topic=1))
    items = [
        concept("a", "creator alpha unique insight one", topic="creator", score=9),
        concept(
            "b", "creator beta different lesson two", start=30, end=54, topic="creator", score=8.9
        ),
        concept(
            "c", "Fortnite gamma competitive story", start=60, end=84, topic="Fortnite", score=8.8
        ),
    ]
    selected = select_distinct_concepts(b, items)
    assert sum(item.topic == "creator" for item in selected) == 1


def test_v10_diversity_rejection_ledger_explains_topic_and_semantic_losses() -> None:
    base = brief()
    strict = replace(
        base,
        diversity=replace(base.diversity, max_concepts_per_topic=1),
        production=replace(base.production, concept_count=3),
    )
    first = concept("a", "creator money story with a strong payoff", topic="creator", score=9.0)
    topic_duplicate = concept(
        "b",
        "creator tournament strategy with a different lesson",
        topic="creator",
        start=40,
        end=64,
        score=8.0,
    )
    semantic_duplicate = concept(
        "c",
        "creator money story with a strong payoff",
        topic="money",
        start=80,
        end=104,
        score=7.0,
    )
    rejections: list[dict[str, object]] = []
    selected = select_distinct_concepts(
        strict, [first, topic_duplicate, semantic_duplicate], rejections=rejections
    )
    reasons = {reason for item in rejections for reason in item["reasons"]}
    assert [item.concept_id for item in selected] == [first.concept_id]
    assert "topic_quota" in reasons
    assert "semantic_cluster_duplicate" in reasons
    assert all(item["scores"] for item in rejections)


def test_v10_mining_rejection_ledger_records_boundary_attrition() -> None:
    b = replace(brief(), min_clip_seconds=12, max_clip_seconds=30)
    segments = [
        TranscriptSegment(0, 6, "Why did I risk 5000 dollars on this business?"),
        TranscriptSegment(6, 12, "and then because"),
        TranscriptSegment(12, 18, "So like maybe this was nothing"),
        TranscriptSegment(18, 24, "Before we head out thanks for watching."),
        TranscriptSegment(24, 30, "The final lesson saved the business."),
    ]
    moment = StoryMoment(
        "m",
        "v",
        0,
        30,
        " ".join(item.text for item in segments),
        "story",
        "business",
        "setup",
        "payoff",
        scores(),
        8.0,
        "fp",
    )
    rejections: list[dict[str, object]] = []
    mine_clip_concepts(b, "v", segments, [moment], rejections=rejections)
    reasons = {reason for item in rejections for reason in item["reasons"]}
    assert "no_semantic_closure" in reasons


def test_v10_hook_anchor_overlay_and_render_reserve_helpers() -> None:
    c = concept("hook", "What's one message for esports fans? Keep practicing.", end=20)
    segments = [TranscriptSegment(0, 20, "What's one message for esports fans? Keep practicing.")]
    anchor = _hook_start_anchor(segments, c, "What's one message for esports fans?")
    assert anchor is not None and anchor[1].lower() == "what's"
    assert _hook_start_anchor(segments, c, "sentence that is absent") is None
    span = SourceSpan(0, 20)
    assert _overlay_duplicates_opening("What's one message for esports fans", span, segments)
    assert not _overlay_duplicates_opening(None, span, segments)

    def plan(index: int, score: float) -> EditPlan:
        return EditPlan(
            f"p{index}",
            "v",
            f"c{index}",
            f"v{index}",
            "direct",
            (SourceSpan(index * 30, index * 30 + 20),),
            None,
            (),
            "tiktok",
            score,
            f"fp{index}",
        )

    plans = [plan(1, 9.0), plan(2, 8.0), plan(3, 7.0)]
    primary, reserve = select_render_plan_queue(plans, budget=2)
    assert len(primary) == 2 and len(reserve) == 1
    assert {item.plan_id for item in primary}.isdisjoint({item.plan_id for item in reserve})


def test_v10_topic_signals_choose_dominant_editorial_theme() -> None:
    keywords = brief().keywords
    assert _topic("My dad saw the skin and asked what I was doing", keywords) == "family"
    assert (
        _topic("I made $1.5 million from my creator code and bought nothing", keywords) == "money"
    )
    assert (
        _topic("I won five World Cup tournament weeks and qualified again", keywords)
        == "competition"
    )
    assert (
        _topic("My reaction training coach made me practice aim every day", keywords) == "training"
    )
    assert _topic("We give away gaming computers to kids", keywords) == "giveaway"
    assert _topic("I dropped out of school to build my career", keywords) == "career"


def test_v10_story_moment_can_extend_forward_to_capture_answer_payoff() -> None:
    b = replace(brief(), min_clip_seconds=18, max_clip_seconds=40)
    segments = [
        TranscriptSegment(0, 6, "What skill matters most in this game?"),
        TranscriptSegment(6, 12, "I think it is reaction time because every fight happens fast."),
        TranscriptSegment(12, 18, "The match can change instantly."),
        TranscriptSegment(18, 24, "My coach made me train reaction drills every single day."),
        TranscriptSegment(24, 30, "That training is why I can react before most players."),
    ]
    moment = StoryMoment(
        "split",
        "v",
        0,
        13,
        "question and setup",
        "question_answer",
        "training",
        "What skill matters most?",
        "reaction time",
        scores(),
        8.0,
        "fp-split",
    )
    stats: dict[str, int] = {}
    concepts = mine_clip_concepts(b, "v", segments, [moment], stats=stats)
    assert concepts
    assert max(item.source_end for item in concepts) > moment.end
    assert stats["candidate_starts"] > 0
    assert stats["eligible_endpoints"] > 0
    assert stats["concepts_after_quality"] >= len(concepts)
    assert stats["concepts_after_moment_dedup"] >= len(concepts)
    assert stats["semantic_representatives"] >= len(concepts)
    assert stats["raw_pool"] == len(concepts)


def test_v10_empty_mining_populates_zero_funnel_stats() -> None:
    stats: dict[str, int] = {}
    concepts = mine_clip_concepts(brief(), "v", [], [], stats=stats)
    assert concepts == []
    assert stats == {
        "candidate_starts": 0,
        "eligible_endpoints": 0,
        "concepts_after_quality": 0,
        "concepts_after_moment_dedup": 0,
        "semantic_representatives": 0,
        "raw_pool": 0,
    }


def test_render_reserves_prioritize_unrepresented_concepts_before_extra_variants() -> None:
    def make(plan_id: str, concept_id: str, variant: str, score: float, start: float) -> EditPlan:
        return EditPlan(
            plan_id,
            "video",
            concept_id,
            variant,
            "direct",
            (SourceSpan(start, start + 20),),
            None,
            (),
            "tiktok",
            score,
            f"fp-{plan_id}",
        )

    plans = [
        make("a1", "A", "v1", 10.0, 0),
        make("a2", "A", "v2", 9.9, 0),
        make("a3", "A", "v3", 9.8, 0),
        make("b1", "B", "v1", 9.7, 60),
        make("b2", "B", "v2", 9.6, 60),
        make("c1", "C", "v1", 9.0, 120),
        make("c2", "C", "v2", 8.9, 120),
        make("d1", "D", "v1", 8.0, 180),
        make("d2", "D", "v2", 7.9, 180),
    ]
    primary, reserve = select_render_plan_queue(plans, budget=2)
    primary_concepts = {plan.concept_id for plan in primary}
    unseen = [plan.concept_id for plan in reserve if plan.concept_id not in primary_concepts]
    assert unseen[:2] == ["C", "D"]
    first_repeat = next(
        index for index, plan in enumerate(reserve) if plan.concept_id in primary_concepts
    )
    assert first_repeat >= 2
