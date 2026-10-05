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
