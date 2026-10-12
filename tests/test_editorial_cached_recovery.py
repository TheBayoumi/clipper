"""Cross-stage recovery cannot waive earlier structural/factual rejections."""

from copy import deepcopy

import pytest

from clipper.editorial import _verified_verbatim_cached_quote, candidate_gate_failures
from clipper.editorial_cached_recovery import _literal_payoff_title, recover_source_quote_drafts
from clipper.models import TranscriptSegment


@pytest.fixture
def source_case():
    source = [
        "I was drafted and made it to Washington.",
        "My family is taken care of forever because I was drafted by Washington.",
    ]
    start, end = 2347.69, 2372.67
    rows = [
        TranscriptSegment(start, 2359, source[0]),
        TranscriptSegment(2359, end, source[1]),
    ]
    review = {
        "opening_standalone": True,
        "payoff_complete": True,
        "ending_complete": True,
        "contains_promotion_or_intro": False,
        "delivered_units": source,
        "headline_audits": [
            {
                "headline": "Family secured by Washington draft",
                "verdict": "uncertain",
                "actor_action": "uncertain",
                "relationship_role": "supported",
                "setting_time": "not_claimed",
                "quantities_outcomes": "not_claimed",
            }
        ],
        "source_quote_spans": {
            "setup_quote": {"text": source[0], "first_unit": 0, "last_unit": 0},
            "resolution_quote": {"text": source[1], "first_unit": 1, "last_unit": 1},
        },
    }
    item = {
        "proposal_start": 2354.1,
        "proposal_end": 2380.2,
        "reviewed_start": start,
        "reviewed_end": end,
        "rejection": "HEADLINE_REVIEW_REJECTED",
        "decision": {"keep": True, "opening": 5, "story": 5, "ending": 5},
        "exchange_review": review,
    }
    return item, rows


def _recover(source_case):
    item, rows = source_case
    return recover_source_quote_drafts(
        [item], rows, video_id="3S-fL9rNPp0", min_duration=20, max_duration=45
    )


def test_recovered_draft_uses_exact_delivered_words_not_failed_model_headline(source_case):
    clips, audit = _recover(source_case)
    assert len(clips) == 1
    clip = clips[0]
    assert "MY FAMILY IS TAKEN CARE OF FOREVER" in clip.reasons[0]
    assert "Family secured by Washington draft" not in clip.reasons[0]
    assert clip.start == source_case[0]["reviewed_start"]
    assert audit["selector_reassessment_count"] == 0
    assert audit["boundary_reassessment_count"] == 0
    assert audit["headline_model_reassessment_count"] == 0
    assert audit["automatic_publication_allowed"] is False
    hook = next(
        x.removeprefix("semantic_hook=") for x in clip.reasons if x.startswith("semantic_hook=")
    )
    assert _verified_verbatim_cached_quote(clip, hook)
    assert candidate_gate_failures(clip, hook_override=hook) == []
    assert not _verified_verbatim_cached_quote(clip, "INVENTED OTHER HEADLINE")


def test_missing_contrast_remains_rejected_and_audited(source_case):
    item, rows = source_case
    wrong = deepcopy(item)
    wrong["rejection"] = "SPAN_REVIEW_REJECTED"
    wrong["exchange_review"]["continuation_review"] = {
        "relation": "missing_contrast",
        "reason": "The source continuation reverses the claim.",
    }
    clips, report = recover_source_quote_drafts(
        [wrong], rows, video_id="3S-fL9rNPp0", min_duration=20, max_duration=45
    )
    assert clips == []
    assert report["breakdown"]["boundary_requires_independent_review:missing_contrast"] == 1
    assert report["records"][0]["draft_approved"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(rejection="INVALID_SOURCE_EVIDENCE"),
        lambda x: x["exchange_review"].update(
            headline_audits=[{"verdict": "unsupported", "actor_action": "unsupported"}]
        ),
        lambda x: x["exchange_review"].update(opening_standalone=False),
        lambda x: x["exchange_review"].update(payoff_complete=False),
        lambda x: x["exchange_review"].update(ending_complete=False),
        lambda x: x["exchange_review"].update(contains_promotion_or_intro=True),
        lambda x: x["exchange_review"].update(delivered_units=[]),
        lambda x: x["exchange_review"].update(source_quote_spans={}),
        lambda x: x.update(reviewed_end=5000.0),
        lambda x: x["decision"].update(story=2),
        lambda x: x["exchange_review"].update(headline_audits=[]),
    ],
)
def test_invalid_or_uncertain_claims_never_become_auto_approved(source_case, change):
    item, rows = source_case
    modified = deepcopy(item)
    change(modified)
    clips, report = recover_source_quote_drafts(
        [modified], rows, video_id="3S-fL9rNPp0", min_duration=20, max_duration=45
    )
    assert clips == []
    assert report["automatic_publication_allowed"] is False


def test_recovery_requires_actual_timed_source_and_deduplicates(source_case):
    item, rows = source_case
    clips, report = recover_source_quote_drafts(
        [item, deepcopy(item)],
        rows,
        video_id="3S-fL9rNPp0",
        min_duration=20,
        max_duration=45,
    )
    assert len(clips) == 1
    assert report["breakdown"]["duplicate_reviewed_window"] == 1
    altered = [TranscriptSegment(x.start, x.end, "He mentioned unrelated events.") for x in rows]
    clips, report = recover_source_quote_drafts(
        [item], altered, video_id="3S-fL9rNPp0", min_duration=20, max_duration=45
    )
    assert clips == []
    assert report["breakdown"]["not_in_timed_source"] == 1


@pytest.mark.parametrize(
    "words",
    [
        "",
        "so that made me a fan of the Broncos.",
        "My growth.",
        (
            "The most unbelievably unnecessarily verbose descriptive content about "
            "completely unrelated topics that goes on forever and ever."
        ),
    ],
)
def test_literal_hook_rejects_incomplete_overlong_or_dependent_quotes(words):
    assert _literal_payoff_title(words) is None


def test_literal_hook_never_invents_actor_or_ignores_negation():
    assert _literal_payoff_title("My family is taken care of forever because I was drafted.") == (
        "My family is taken care of forever"
    )
    assert _literal_payoff_title("If you are not healthy, none of this matters.") == (
        "If you are not healthy, none of this matters."
    )


def test_missing_model_reviews_and_unusable_last_speech_are_never_recovered(source_case):
    item, rows = source_case
    missing = deepcopy(item)
    missing.pop("exchange_review")
    dependent = deepcopy(item)
    dependent["exchange_review"]["delivered_units"][-1] = "So that made me a fan of the Broncos."
    clips, report = recover_source_quote_drafts(
        [missing, dependent],
        rows,
        video_id="3S-fL9rNPp0",
        min_duration=20,
        max_duration=45,
    )
    assert not clips
    assert report["breakdown"]["missing_review"] == 1
    assert report["breakdown"]["no_standalone_literal_payoff"] == 1


def test_invalid_cached_window_and_tampered_quote_do_not_become_review_drafts(source_case):
    item, rows = source_case
    missing_time = deepcopy(item)
    missing_time.pop("reviewed_start")
    corrupted = deepcopy(item)
    corrupted["exchange_review"]["source_quote_spans"]["setup_quote"]["text"] = (
        "Invented dialogue never in the source."
    )
    clips, report = recover_source_quote_drafts(
        [missing_time, corrupted],
        rows,
        video_id="3S-fL9rNPp0",
        min_duration=20,
        max_duration=45,
    )
    assert not clips
    assert report["breakdown"]["invalid_reviewed_window"] == 1
    assert report["breakdown"]["unmatched_setup_or_resolution"] == 1


def test_unusable_layout_produces_no_quote_recovery(monkeypatch):
    from clipper import editorial_cached_recovery as recovery

    def reject_all_layouts(text):
        raise ValueError("no portrait safe hook")

    monkeypatch.setattr(recovery, "persistent_hook_layout", reject_all_layouts)
    assert recovery._literal_payoff_title("My family is taken care of forever") is None


def test_verbatim_evidence_gate_fails_closed_on_tampered_quote(source_case):
    clip = _recover(source_case)[0][0]
    hook = next(
        x.removeprefix("semantic_hook=") for x in clip.reasons if x.startswith("semantic_hook=")
    )
    corrupted = deepcopy(clip.to_dict())
    corrupted["reasons"] = [
        "setup_quote=unsupported text" if reason.startswith("setup_quote=") else reason
        for reason in corrupted["reasons"]
    ]
    from clipper.models import ClipCandidate

    wrong = ClipCandidate(
        clip.video_id, clip.start, clip.end, clip.text, clip.score, tuple(corrupted["reasons"])
    )
    assert not _verified_verbatim_cached_quote(wrong, hook)
    assert "UNSUPPORTED_SOURCE_HOOK" in candidate_gate_failures(wrong, hook_override=hook)
    other = ClipCandidate(
        clip.video_id,
        clip.start,
        clip.end,
        clip.text,
        clip.score,
        tuple(x for x in clip.reasons if not x.startswith("semantic_hook=")),
    )
    assert not _verified_verbatim_cached_quote(other, hook)


def test_hypothetical_touchdown_is_not_misrepresented_as_real_game_result(source_case):
    item, rows = source_case
    generated = deepcopy(item)
    generated["exchange_review"]["delivered_units"] = [
        "Let's do a spin move, go this way.",
        "Oh, touchdown, 40 -yard gain.",
    ]
    review = generated["exchange_review"]
    review["source_quote_spans"] = {
        "setup_quote": {"text": review["delivered_units"][0], "first_unit": 0, "last_unit": 0},
        "resolution_quote": {"text": review["delivered_units"][1], "first_unit": 1, "last_unit": 1},
    }
    clips, report = recover_source_quote_drafts(
        [generated], rows, video_id="3S-fL9rNPp0", min_duration=20, max_duration=45
    )
    assert clips == []
    assert report["breakdown"]["hypothetical_outcome_not_established"] == 1
