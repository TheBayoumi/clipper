from dataclasses import replace

import pytest

from clipper.editorial import (
    _find_hook_sentence,
    _moment_type,
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
    assert _topic("Fortnite changed my career", brief().keywords) == "fortnite"
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
