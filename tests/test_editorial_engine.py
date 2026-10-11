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
