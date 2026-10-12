"""Package ownership and explicit headline handoff for production editorial review."""

from clipper import editorial_engine as engine
from scripts import tjr_semantic_editor as adapter


def test_script_uses_package_source_evidence_helpers():
    for name in (
        "_unit_span_schema",
        "_resolve_source_units",
        "_unit_span_valid",
        "_numbered_source",
        "_final_substantive_unit_id",
        "_evidence_excerpt",
    ):
        assert getattr(adapter, name) is getattr(engine, name)


def test_adapter_passes_explicit_headline_dependencies(monkeypatch):
    context = {"selected_units": ["A complete exchange."]}
    reviewer = object()
    audit = object()
    generator = object()
    expected = {"ending_complete": False}

    def review(instance, supplied_context, *, factual_audit, headline_generator):
        assert instance is reviewer
        assert supplied_context is context
        assert factual_audit is audit
        assert headline_generator is generator
        return expected

    monkeypatch.setattr(adapter, "review_source_positions", review)
    assert (
        adapter._source_position_review(
            reviewer, context, factual_audit=audit, headline_generator=generator
        )
        is expected
    )


def test_invalid_mixed_source_range_gets_one_grounded_regeneration():
    from clipper.editorial_engine import review_source_positions

    units = [
        "The reporter asked about the result.",
        "The coach confirmed that the match ended in a draw.",
    ]
    purpose = {
        "ad_read_present": 0,
        "ad_read_span": {"first_unit": -1, "last_unit": -1},
        "show_intro_present": 0,
        "show_intro_span": {"first_unit": -1, "last_unit": -1},
        "reason": "Conversation, not promotion.",
    }
    story = {
        "setup_span": {"first_unit": 0, "last_unit": 0},
        "resolution_span": {"first_unit": 1, "last_unit": -1},
        "opening_independent": 1,
        "last_thought_finished": 1,
        "reason": "The question is answered.",
    }
    replies = [
        purpose,
        story,
        {
            **story,
            "resolution_span": {"first_unit": 1, "last_unit": 1},
        },
    ]
    seen = []

    class Editor:
        def _review_completion(self, prompt, payload, *_):
            seen.append((prompt, payload))
            return replies.pop(0)

    def headline_generator(editor, delivered, *, exchange_spans, factual_audit):
        assert delivered == units
        assert exchange_spans["resolution_quote"]["text"] == units[1]
        return {
            "headline": "Coach confirms that the match ended in a draw",
            "headline_supported": False,
            "headline_self_contained": False,
        }

    result = review_source_positions(
        Editor(),
        {"selected_units": units, "before": [], "after": []},
        factual_audit=lambda *_: {},
        headline_generator=headline_generator,
    )
    assert len(seen) == 3 and not replies
    assert seen[2][1]["invalid_model_positions"]["resolution_span"] == {
        "first_unit": 1,
        "last_unit": -1,
    }
    assert "not source evidence" in seen[2][0].lower()
    assert result["source_quote_spans"]["resolution_quote"]["last_unit"] == 1
    assert result["headline_supported"] is False


def test_repeated_invalid_source_positions_fail_closed():
    import pytest

    from clipper.editorial_engine import SourceEvidenceError, review_source_positions

    purpose = {
        "ad_read_present": 0,
        "ad_read_span": {"first_unit": -1, "last_unit": -1},
        "show_intro_present": 0,
        "show_intro_span": {"first_unit": -1, "last_unit": -1},
        "reason": "Conversation.",
    }
    broken = {
        "setup_span": {"first_unit": 0, "last_unit": 0},
        "resolution_span": {"first_unit": 9, "last_unit": -1},
        "opening_independent": 1,
        "last_thought_finished": 1,
        "reason": "Malformed source span.",
    }
    replies = [purpose, broken, broken]

    class Editor:
        def _review_completion(self, prompt, payload, *_):
            return replies.pop(0)

    def forbidden_headline(*args, **kwargs):
        raise AssertionError("invalid model evidence must not reach headline generation")

    with pytest.raises(SourceEvidenceError, match="evidence range"):
        review_source_positions(
            Editor(),
            {"selected_units": ["The guest explained the decision."], "before": [], "after": []},
            factual_audit=lambda *_: {},
            headline_generator=forbidden_headline,
        )
    assert not replies


def test_completed_source_boundaries_can_be_reused_for_new_headline_only():
    from clipper.editorial_engine import complete_source_review, review_source_boundaries

    selected = [
        "The coach asked who would take the field.",
        "The quarterback confirmed he would start on Sunday.",
    ]
    absent = {"first_unit": -1, "last_unit": -1}
    model_calls = []

    class Model:
        def _review_completion(self, prompt, payload, schema, tokens):
            model_calls.append(prompt)
            if "speech act performed" in prompt:
                return {
                    "ad_read_present": 0,
                    "ad_read_span": absent,
                    "show_intro_present": 0,
                    "show_intro_span": absent,
                    "reason": "Ordinary sporting discussion.",
                }
            if "Assess ONLY delivered_units as the finished clip" in prompt:
                return {
                    "setup_span": {"first_unit": 0, "last_unit": 0},
                    "resolution_span": {"first_unit": 1, "last_unit": 1},
                    "opening_independent": 1,
                    "last_thought_finished": 1,
                    "reason": "The quarterback gives a specific answer.",
                }
            raise AssertionError("Only new hook production is authorized")

    context = {"selected_units": selected, "before": [], "after": []}
    editor = Model()
    boundary = review_source_boundaries(editor, context)
    assert len(model_calls) == 2
    headlines = []

    def generate(editor, units, *, exchange_spans, factual_audit):
        assert units == selected
        assert exchange_spans["resolution_quote"]["text"] == selected[1]
        headlines.append(1)
        return {
            "headline": "Quarterback confirms Sunday start",
            "headline_supported": False,
            "headline_self_contained": False,
        }

    first = complete_source_review(
        editor, context, boundary, factual_audit=lambda *_: {}, headline_generator=generate
    )
    second = complete_source_review(
        editor, context, first, factual_audit=lambda *_: {}, headline_generator=generate
    )
    assert len(model_calls) == 2 and len(headlines) == 2
    assert second["headline"] == "Quarterback confirms Sunday start"
    assert second["headline_supported"] is False
    assert second["source_quote_spans"] == boundary["source_quote_spans"]


def test_cached_story_cannot_mutate_completion_flags_or_payoff_locations():
    import pytest

    from clipper.editorial_engine import (
        SourceEvidenceError,
        complete_source_review,
        review_source_boundaries,
    )

    selected = ["The owner answered the question."]
    absent = {"first_unit": -1, "last_unit": -1}

    class Model:
        def _review_completion(self, prompt, payload, schema, tokens):
            if "speech act performed" in prompt:
                return {
                    "ad_read_present": 0,
                    "ad_read_span": absent,
                    "show_intro_present": 0,
                    "show_intro_span": absent,
                    "reason": "No promotion",
                }
            return {
                "setup_span": {"first_unit": 0, "last_unit": 0},
                "resolution_span": {"first_unit": 0, "last_unit": 0},
                "opening_independent": 1,
                "last_thought_finished": 1,
                "reason": "Complete exchange",
            }

    context = {"selected_units": selected, "after": []}
    boundary = review_source_boundaries(Model(), context)

    def should_not_generate(*args, **kwargs):
        raise AssertionError("tampered boundary should never generate headlines")

    for key, value in (("payoff_complete", False), ("contains_promotion_or_intro", True)):
        corrupted = {**boundary, key: value}
        with pytest.raises(SourceEvidenceError, match="differs from source review"):
            complete_source_review(
                Model(),
                context,
                corrupted,
                factual_audit=lambda *_: {},
                headline_generator=should_not_generate,
            )
    corrupted = {
        **boundary,
        "source_quote_spans": {
            "setup_quote": boundary["source_quote_spans"]["setup_quote"],
            "resolution_quote": None,
        },
    }
    with pytest.raises(SourceEvidenceError, match="positions changed"):
        complete_source_review(
            Model(),
            context,
            corrupted,
            factual_audit=lambda *_: {},
            headline_generator=should_not_generate,
        )


def test_cached_source_boundary_does_not_trust_altered_factual_flags():
    import copy

    import pytest

    from clipper.editorial_engine import (
        SourceEvidenceError,
        complete_source_review,
        review_source_boundaries,
    )

    speech = ["The owner answered the final question."]
    absent = {"first_unit": -1, "last_unit": -1}

    class Model:
        def _review_completion(self, prompt, payload, schema, tokens):
            if "speech act performed" in prompt:
                return {
                    "ad_read_present": 0,
                    "ad_read_span": absent,
                    "show_intro_present": 0,
                    "show_intro_span": absent,
                    "reason": "No advertisement",
                }
            return {
                "setup_span": {"first_unit": 0, "last_unit": 0},
                "resolution_span": {"first_unit": 0, "last_unit": 0},
                "opening_independent": 1,
                "last_thought_finished": 1,
                "reason": "Resolved final answer",
            }

    context = {"selected_units": speech, "before": [], "after": []}
    original = review_source_boundaries(Model(), context)
    variants = []
    mismatch = copy.deepcopy(original)
    mismatch["evidence_contract"] = "wrong contract"
    variants.append(mismatch)
    mismatch = copy.deepcopy(original)
    mismatch["speech_purpose_review"]["ad_read_present"] = 1
    variants.append(mismatch)
    mismatch = copy.deepcopy(original)
    mismatch["speech_purpose_quote_spans"]["ad_read_span"] = {
        "text": speech[0],
        "first_unit": 0,
        "last_unit": 0,
    }
    variants.append(mismatch)
    mismatch = copy.deepcopy(original)
    mismatch["thought_completion_review"]["resolution_span"] = absent
    variants.append(mismatch)
    mismatch = copy.deepcopy(original)
    mismatch["thought_completion_review"]["last_thought_finished"] = 0
    variants.append(mismatch)
    mismatch = copy.deepcopy(original)
    mismatch["boundary_audit"]["payoff_location"] = "absent"
    variants.append(mismatch)
    mismatch = copy.deepcopy(original)
    mismatch["continuation_review"] = {"relation": "new_topic"}
    variants.append(mismatch)
    for altered in variants:
        with pytest.raises(SourceEvidenceError):
            complete_source_review(
                Model(),
                context,
                altered,
                factual_audit=lambda *_: {},
                headline_generator=lambda *args, **kwargs: pytest.fail("false approval"),
            )


def test_verified_excluded_continuation_is_rechecked_before_headline_reuse():
    from clipper.editorial_engine import complete_source_review, review_source_boundaries

    source = ["The quarterback confirmed that his team won its last game."]
    later = ["Next, we should talk about the upcoming draft."]
    absent = {"first_unit": -1, "last_unit": -1}
    calls = []

    class Model:
        def _review_completion(self, prompt, payload, schema, tokens):
            calls.append(prompt)
            if "speech act performed" in prompt:
                return {
                    "ad_read_present": 0,
                    "ad_read_span": absent,
                    "show_intro_present": 0,
                    "show_intro_span": absent,
                    "reason": "Normal interview",
                }
            if "finished clip" in prompt:
                return {
                    "setup_span": {"first_unit": 0, "last_unit": 0},
                    "resolution_span": {"first_unit": 0, "last_unit": 0},
                    "opening_independent": 1,
                    "last_thought_finished": 1,
                    "reason": "The speaker answered.",
                }
            return {
                "final_span": {"first_unit": 0, "last_unit": 0},
                "continuation_span": {"first_unit": 0, "last_unit": 0},
                "relation": "new_topic",
                "reason": "The next question starts a new topic.",
            }

    context = {"selected_units": source, "before": [], "after": later}
    model = Model()
    boundary = review_source_boundaries(model, context)
    result = complete_source_review(
        model,
        context,
        boundary,
        factual_audit=lambda *_: {},
        headline_generator=lambda *_args, **_kw: {
            "headline": "Team confirms recent victory",
            "headline_supported": False,
            "headline_self_contained": False,
        },
    )
    assert len(calls) == 3
    assert result["boundary_audit"]["ending"] == "closed"
    assert result["source_quote_spans"] == boundary["source_quote_spans"]
    assert result["headline_supported"] is False
    import copy

    import pytest

    from clipper.editorial_engine import SourceEvidenceError

    bad = copy.deepcopy(boundary)
    bad["continuation_review"]["continuation_span"] = {"first_unit": -1, "last_unit": -1}
    with pytest.raises(SourceEvidenceError, match="cached continuation"):
        complete_source_review(
            model,
            context,
            bad,
            factual_audit=lambda *_: {},
            headline_generator=lambda *_args, **_kw: pytest.fail("false evidence"),
        )
    bad = copy.deepcopy(boundary)
    bad["continuation_review"] = None
    with pytest.raises(SourceEvidenceError, match="continuation decision"):
        complete_source_review(
            model,
            context,
            bad,
            factual_audit=lambda *_: {},
            headline_generator=lambda *_args, **_kw: pytest.fail("missing source proof"),
        )


def test_missing_contrast_veto_requires_two_independent_grounded_optional_proofs():
    import copy

    import pytest

    from clipper.editorial_engine import (
        SourceEvidenceError,
        _optional_contrast_proof_valid,
        complete_source_review,
        review_source_boundaries,
    )

    delivered = ["Brock Purdy became a quarterback you win because of, not just with."]
    excluded = ["He used to be someone they could win with, but now they win because of."]
    absent = {"first_unit": -1, "last_unit": -1}
    grounded = {"first_unit": 0, "last_unit": 0}

    def obligation(state="delivered", *, payoff_span=None):
        return {
            "setup_span": grounded.copy(),
            "payoff_span": (grounded.copy() if state == "delivered" else absent.copy())
            if payoff_span is None
            else payoff_span,
            "payoff_state": state,
            "reason": "The comparison concludes within delivered speech.",
        }

    def dependency(effect="restates_or_illustrates_delivered_payoff", *, span=None):
        if effect == "supplies_missing_payoff":
            requirement, open_span = "unrevealed_outcome", grounded.copy()
        elif effect == "changes_or_qualifies_delivered_payoff":
            requirement, open_span = "necessary_correction", grounded.copy()
        elif effect == "uncertain":
            requirement, open_span = "uncertain", absent.copy()
        else:
            requirement, open_span = "none", absent.copy()
        return {
            "excluded_span": grounded.copy() if span is None else span,
            "open_obligation_span": open_span,
            "missing_requirement": requirement,
            "effect": effect,
            "reason": "The omitted source is checked against the same delivered obligation.",
        }

    def event_guard(status="non_event"):
        absent_event = {"first_unit": -1, "last_unit": -1}
        return {
            "excluded_event_span": absent_event,
            "prior_event_setup_span": absent_event,
            "delivered_event_outcome_span": absent_event,
            "status": status,
            "reason": "The excluded speech adds no separately teased consequential event.",
        }

    def evaluate(first, second):
        responses = [
            first["obligation"],
            first["dependency"],
            second["obligation"],
            second["dependency"],
        ]
        prompts = []

        class Model:
            def _review_completion(self, prompt, payload, _schema, _tokens):
                prompts.append(prompt)
                if "speech act performed" in prompt:
                    return {
                        "ad_read_present": 0,
                        "ad_read_span": absent,
                        "show_intro_present": 0,
                        "show_intro_span": absent,
                        "reason": "Ordinary football conversation.",
                    }
                if "Assess ONLY delivered_units as the finished clip" in prompt:
                    return {
                        "setup_span": grounded,
                        "resolution_span": grounded,
                        "opening_independent": 1,
                        "last_thought_finished": 1,
                        "reason": "The speaker delivers the complete comparison.",
                    }
                if "Judge the CUT" in prompt:
                    return {
                        "final_span": grounded,
                        "continuation_span": grounded,
                        "relation": "missing_contrast",
                        "reason": "First reviewer suspects a missing comparison.",
                    }
                assert "missing_contrast" not in str(payload)
                if "Perform a BLIND excluded-event inventory" in prompt:
                    assert "frozen_delivered_obligation" not in payload
                    return event_guard()
                if "excluded_after" not in payload:
                    assert set(payload) == {"delivered_units", "final_substantive_unit_id"}
                else:
                    assert "frozen_delivered_obligation" in payload
                return responses.pop(0)

        model = Model()
        context = {"selected_units": delivered, "before": [], "after": excluded}
        boundary = review_source_boundaries(model, context)
        assert len(prompts) in (7, 8) and not responses
        assert boundary["continuation_review"]["relation"] == "missing_contrast"
        assert len(boundary["continuation_review"]["optional_contrast_proof"]["verdicts"]) == 2
        return model, context, boundary

    optional = {"obligation": obligation(), "dependency": dependency()}
    model, context, boundary = evaluate(optional, optional)
    assert boundary["ending_complete"] is True
    assert boundary["payoff_complete"] is True
    assert boundary["continuation_review"]["optional_contrast_proof"]["optional_verified"]
    result = complete_source_review(
        model,
        context,
        boundary,
        factual_audit=lambda *_: {},
        headline_generator=lambda *_args, **_kw: {
            "headline": "Purdy became the reason they can win",
            "headline_supported": False,
            "headline_self_contained": False,
        },
    )
    assert result["headline_supported"] is False

    missing = {
        "obligation": obligation("pending"),
        "dependency": dependency("supplies_missing_payoff"),
    }
    _model, _context, veto = evaluate(optional, missing)
    assert veto["ending_complete"] is False
    assert veto["payoff_complete"] is False
    assert not veto["continuation_review"]["optional_contrast_proof"]["optional_verified"]

    _model, _context, unknown = evaluate(
        optional,
        {"obligation": obligation(), "dependency": dependency("uncertain")},
    )
    assert unknown["ending_complete"] is False

    with pytest.raises(SourceEvidenceError, match="canonical proof"):
        evaluate(
            {"obligation": obligation(), "dependency": dependency(span=absent)},
            optional,
        )

    old_contract = {
        "verdicts": [
            {
                "delivered_span": grounded.copy(),
                "excluded_span": grounded.copy(),
                "materiality": "not_needed_for_delivered_point",
                "reason": "Old approval contract is no longer valid.",
            }
        ]
        * 2,
        "optional_verified": True,
    }
    with pytest.raises(SourceEvidenceError, match="narrative obligation"):
        _optional_contrast_proof_valid(old_contract, delivered, excluded, 0)

    corrupted = copy.deepcopy(boundary)
    corrupted["continuation_review"]["optional_contrast_proof"]["verdicts"][1]["dependency"][
        "effect"
    ] = "supplies_missing_payoff"
    corrupted["continuation_review"]["optional_contrast_proof"]["verdicts"][1]["dependency"][
        "missing_requirement"
    ] = "unrevealed_outcome"
    corrupted["continuation_review"]["optional_contrast_proof"]["verdicts"][1]["dependency"][
        "open_obligation_span"
    ] = grounded.copy()
    with pytest.raises(SourceEvidenceError, match="approval was altered"):
        complete_source_review(
            model,
            context,
            corrupted,
            factual_audit=lambda *_: {},
            headline_generator=lambda *_args, **_kw: pytest.fail(
                "tampered proof must not reach headline generation"
            ),
        )


def test_optional_proof_rejects_schema_valid_but_fake_delivered_payoff():
    import pytest

    from clipper.editorial_engine import SourceEvidenceError, _optional_contrast_proof_valid

    pointer = {"first_unit": 0, "last_unit": 0}
    absent = {"first_unit": -1, "last_unit": -1}
    delivered = ["The runner crossed the field, then I looked behind."]
    after = ["The defender hit me and drove me into the ground."]

    def build(state, span):
        return {
            "obligation": {
                "setup_span": pointer,
                "payoff_span": span,
                "payoff_state": state,
                "reason": "Determine whether the promised impact is spoken.",
            },
            "dependency": {
                "excluded_span": pointer,
                "open_obligation_span": absent,
                "missing_requirement": "none",
                "effect": "restates_or_illustrates_delivered_payoff",
                "reason": "Candidate continuation comparison.",
            },
        }

    with pytest.raises(SourceEvidenceError, match="canonical proof"):
        _optional_contrast_proof_valid(
            {"verdicts": [build("pending", pointer)] * 2, "event_gap": None, "optional_verified": False},
            delivered,
            after,
            0,
        )
    with pytest.raises(SourceEvidenceError, match="canonical proof"):
        _optional_contrast_proof_valid(
            {"verdicts": [build("delivered", absent)] * 2, "event_gap": None, "optional_verified": False},
            delivered,
            after,
            0,
        )
    proof = {
        "verdicts": [build("pending", absent)] * 2,
        "event_gap": None,
        "optional_verified": False,
    }
    assert _optional_contrast_proof_valid(proof, delivered, after, 0) is False


def test_obligation_review_rejects_missing_payoff_span_before_dependency_inference():
    import pytest

    from clipper.editorial_engine import SourceEvidenceError, review_optional_contrast

    calls = []

    class Editor:
        def _review_completion(self, prompt, payload, schema, tokens):
            calls.append((prompt, payload))
            assert "excluded_after" not in payload
            return {
                "setup_span": {"first_unit": 0, "last_unit": 0},
                "payoff_span": {"first_unit": -1, "last_unit": -1},
                "payoff_state": "delivered",
                "reason": "The selected story allegedly finishes its obligation.",
            }

    with pytest.raises(SourceEvidenceError, match="payoff-state disagrees"):
        review_optional_contrast(Editor(), ["The defender was closing in."], ["He tackled me."], 0)
    assert len(calls) == 1


def test_narrative_proof_validation_fails_closed_on_invalid_combinations():
    import pytest

    from clipper.editorial_engine import SourceEvidenceError, _optional_contrast_proof_valid

    pointer = {"first_unit": 0, "last_unit": 0}
    absent = {"first_unit": -1, "last_unit": -1}
    selected = ["The speaker finished the comparison."]
    excluded = ["Another speaker asks a different question."]
    valid = {
        "obligation": {
            "setup_span": pointer,
            "payoff_span": pointer,
            "payoff_state": "delivered",
            "reason": "The comparison concludes before the cut.",
        },
        "dependency": {
            "excluded_span": pointer,
            "open_obligation_span": absent,
            "missing_requirement": "none",
            "effect": "new_subject",
            "reason": "The next speaker asks a separate question.",
        },
    }
    safe_event_guard = {
        "excluded_event_span": absent,
        "prior_event_setup_span": absent,
        "delivered_event_outcome_span": absent,
        "status": "non_event",
        "reason": "No additional consequential event follows this completed conclusion.",
    }
    assert _optional_contrast_proof_valid(
        {"verdicts": [valid, valid], "event_gap": safe_event_guard, "optional_verified": True},
        selected,
        excluded,
        0,
    )
    assert _optional_contrast_proof_valid(None, selected, excluded, 0) is False
    for invalid in (
        [],
        {"verdicts": [valid], "event_gap": None, "optional_verified": True},
        {"verdicts": "untrusted", "event_gap": None, "optional_verified": False},
        {"verdicts": [valid, valid], "event_gap": safe_event_guard, "optional_verified": False},
        {"verdicts": [valid, valid], "event_gap": safe_event_guard, "optional_verified": True, "other": "tamper"},
        {"verdicts": [valid, valid], "event_gap": safe_event_guard, "optional_verified": 1},
        {"verdicts": [valid, {"obligation": valid["obligation"]}], "event_gap": None, "optional_verified": True},
        {
            "verdicts": [
                valid,
                {
                    **valid,
                    "dependency": {**valid["dependency"], "effect": "not_a_source_relation"},
                },
            ],
            "event_gap": None,
            "optional_verified": True,
        },
        {
            "verdicts": [
                valid,
                {
                    **valid,
                    "obligation": {**valid["obligation"], "reason": ""},
                },
            ],
            "optional_verified": True,
        },
    ):
        with pytest.raises(SourceEvidenceError):
            _optional_contrast_proof_valid(invalid, selected, excluded, 0)
    # A model may not allege an essential continuation without identifying
    # which delivered question, outcome or correction remains unresolved.
    for omitted_claim in (
        {"effect": "supplies_missing_payoff", "missing_requirement": "none"},
        {
            "effect": "supplies_missing_payoff",
            "missing_requirement": "unrevealed_outcome",
        },
        {
            "effect": "new_subject",
            "missing_requirement": "unrevealed_outcome",
            "open_obligation_span": pointer,
        },
        {
            "effect": "changes_or_qualifies_delivered_payoff",
            "missing_requirement": "necessary_correction",
        },
    ):
        dep = {**valid["dependency"], **omitted_claim}
        with pytest.raises(SourceEvidenceError, match="canonical proof"):
            _optional_contrast_proof_valid(
                {
                    "verdicts": [valid, {**valid, "dependency": dep}],
                    "optional_verified": False,
                },
                selected,
                excluded,
                0,
            )
    linked_missing = {
        "excluded_span": pointer,
        "open_obligation_span": pointer,
        "missing_requirement": "unrevealed_outcome",
        "effect": "supplies_missing_payoff",
        "reason": "The omitted source reveals the awaited event outcome.",
    }
    assert (
        _optional_contrast_proof_valid(
            {
                "verdicts": [valid, {**valid, "dependency": linked_missing}],
                "optional_verified": False,
            },
            selected,
            excluded,
            0,
        )
        is False
    )

    invalid_pending = {
        **valid,
        "obligation": {
            **valid["obligation"],
            "payoff_state": "pending",
            "payoff_span": absent,
        },
    }
    assert (
        _optional_contrast_proof_valid(
            {"verdicts": [valid, invalid_pending], "optional_verified": False},
            selected,
            excluded,
            0,
        )
        is False
    )


def test_missing_contrast_without_resolution_remains_vetoed():
    from clipper.editorial_engine import _optional_contrast_proof_valid

    assert (
        _optional_contrast_proof_valid(
            None,
            ["The speaker starts a claim but gives no answer."],
            ["The missing answer occurs later."],
            0,
        )
        is False
    )
