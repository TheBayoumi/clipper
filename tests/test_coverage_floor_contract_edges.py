from pathlib import Path

import pytest

from clipper.autonomous_quality_planner import AutonomousQualityPlanner
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.providers.base import ModelIdentity, ProviderResult
from clipper.stage_contracts import StageIdentity
from clipper.story_graph import NarrativeEnvelope, SemanticCore


class _UnusedEditorial:
    identity = ModelIdentity("unused", "rev", "none", "test", "editor", "schema")

    def complete_json(
        self, *, task: str, payload: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        raise AssertionError((task, payload))


def _timeline(count: int = 220) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:x",
                f"word-{index}",
                float(index),
                float(index + 1),
                "speaker",
                1.0,
                "word_exact",
                "test",
            )
            for index in range(count)
        ),
    )


def test_stage_identity_rejects_each_missing_required_identity_component() -> None:
    with pytest.raises(ValueError, match="stage, source, and contract hashes"):
        StageIdentity("", "source", "contract")
    with pytest.raises(ValueError, match="stage, source, and contract hashes"):
        StageIdentity("stage", "", "contract")
    with pytest.raises(ValueError, match="stage, source, and contract hashes"):
        StageIdentity("stage", "source", "")


def test_story_graph_factories_reject_empty_word_provenance() -> None:
    timeline = _timeline()
    with pytest.raises(ValueError, match="semantic core requires source words"):
        SemanticCore.from_word_ids(
            timeline,
            core_id="core",
            source_word_ids=(),
            semantic_summary="summary",
            editorial_reason="reason",
            confidence=0.9,
        )

    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=(timeline.words[100].word_id,),
        semantic_summary="summary",
        editorial_reason="reason",
        confidence=0.9,
    )
    with pytest.raises(ValueError, match="narrative envelope requires source words"):
        NarrativeEnvelope.from_word_ids(
            timeline,
            core,
            envelope_id="envelope",
            source_word_ids=(),
            setup_resolved=True,
            payoff_resolved=True,
            confidence=0.9,
        )


def test_planner_empty_chunks_and_wide_core_context_paths(tmp_path: Path) -> None:
    planner = AutonomousQualityPlanner(
        _UnusedEditorial(),
        DagStore(tmp_path / "dag"),
        max_words_per_chunk=200,
        chunk_overlap_words=20,
        envelope_context_words=100,
    )
    empty = CanonicalTimeline("empty-video", "empty-source", ())
    assert planner._chunks(empty) == ()

    timeline = _timeline()
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="wide-core",
        source_word_ids=tuple(word.word_id for word in timeline.words[50:171]),
        semantic_summary="wide complete event",
        editorial_reason="exercise wide context path",
        confidence=0.9,
    )
    assert planner._context_range(timeline, core) == (50, 171)
