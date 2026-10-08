from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clipper_engine import speech, spoken, spoken_semantics
from clipper_engine.gameplay import allocation, contract
from clipper_engine.profiles import load_profile
from clipper_engine.rendering import overlays
from clipper_engine.sources import catalog, mediasilo
from clipper_engine.sources import qa as source_qa

CAMPAIGN = Path("campaigns/mw4-dmz-from-the-ward.json")


def _word(start: float, end: float, text: str) -> speech.TranscriptWord:
    return speech.TranscriptWord(start=start, end=end, text=text, probability=0.95)


def _segment(start: float, end: float, text: str) -> speech.TranscriptSegment:
    return speech.TranscriptSegment(
        start=start,
        end=end,
        text=text,
        words=(_word(start, end, text),),
    )


def test_campaign_override_deep_merges_canonical_mw4_profile() -> None:
    profile = load_profile("mw4", CAMPAIGN)
    assert profile.expected_source_count == 3
    assert profile.source_url.endswith("aab332d7-ae82-4825-a88b-dec831ecc327")
    assert profile.config["content_strategy"]["mode"] == "spoken_content"
    assert profile.config["semantic_editor"]["minimum_output_seconds"] == 10
    assert profile.config["semantic_editor"]["maximum_output_seconds"] == 12
    assert profile.config["duplicate_policy"]["finishing_move_exclusive"] is True
    assert profile.config["output"]["width"] == 1080
    assert profile.config["output"]["height"] == 1920
    assert profile.config["output"]["full_source_frame"] is True
    assert profile.config["output"]["fps"] == "source"
    assert profile.config["spoken_content"]["asr"]["model"] == "distil-large-v3"
    assert profile.config["spoken_content"]["semantic"]["model"] == "BAAI/bge-small-en-v1.5"
    assert profile.config["spoken_content"]["semantic"]["minimum_topic_similarity"] == 0.52
    assert profile.config["spoken_content"]["semantic"]["variants_per_topic"] == 5
    assert profile.config["spoken_content"]["semantic"]["minimum_topic_margin"] == 0.05
    assert profile.config["output"]["portrait_layout"]["enabled"] is True
    assert profile.config["output"]["portrait_layout"]["background_mode"] == "blurred_source"
    assert profile.config["output"]["portrait_layout"]["background_blur_sigma"] == 18
    assert profile.config["output"]["portrait_layout"]["title_bar_enabled"] is False


def test_spotlight_template_parser_extracts_provider_context() -> None:
    payload = {
        "elements": [
            {
                "playlistId": "playlist-1",
                "providedData": {
                    "playlistId": "playlist-1",
                    "assets": [
                        {
                            "id": "asset-1",
                            "title": "video.mp4",
                            "fileName": "video.mp4",
                            "derivatives": [
                                {"type": "proxy", "url": "https://example.test/video.mp4"}
                            ],
                        }
                    ],
                },
            }
        ]
    }
    html = f"<script>window.template = {json.dumps(payload)}; window.pageIndex = 0;</script>"
    assets = mediasilo._spotlight_template_assets(html, "presentation-1")
    assert len(assets) == 1
    assert assets[0]["_presentation_id"] == "presentation-1"
    assert assets[0]["_playlist_id"] == "playlist-1"


def test_catalog_can_select_declared_proxy_for_analysis(tmp_path: Path) -> None:
    asset = {
        "id": "asset-1",
        "title": "DMZ.mp4",
        "fileName": "DMZ.mp4",
        "duration": 12_000,
        "derivatives": [
            {"type": "source", "fileSize": 500_000},
            {
                "type": "proxy",
                "url": "https://example.test/proxy.mp4",
                "fileSize": 10_000,
                "width": 1280,
                "height": 720,
                "duration": 12_000,
            },
        ],
        "_presentation_id": "presentation-1",
        "_playlist_id": "playlist-1",
    }
    source_profile = {
        "analysis_derivative": "proxy",
        "assets": [
            {
                "title": "DMZ.mp4",
                "source_key": "dmz",
                "content_type": "developer",
            }
        ],
    }
    output = tmp_path / "catalog.json"
    with patch("clipper_engine.sources.catalog._capture_assets", return_value=[asset]):
        result = catalog.discover(
            "https://app.mediasilo.com/spotlight/test",
            output,
            None,
            1,
            source_profile,
        )
    selected = result["sources"][0]
    assert selected["source_key"] == "dmz"
    assert selected["derivative_type"] == "proxy"
    assert selected["content_type"] == "developer"
    assert selected["original_source_declared"] is True


def test_production_verification_rejects_certified_proxy(tmp_path: Path) -> None:
    source = tmp_path / "proxy.mp4"
    source.write_bytes(b"proxy")
    manifest = tmp_path / "proxy.source_qa.json"
    manifest.write_text(
        json.dumps(
            {
                "derivative_type": "proxy",
                "downloaded_bytes": source.stat().st_size,
                "sha256": "unused-before-production-policy-check",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="production render requires"):
        source_qa.verify(source, manifest, require_original=True)


def _window(
    start: float,
    end: float,
    text: str,
    anchor: float,
    event: str,
    score: float,
) -> spoken_semantics.SemanticWindow:
    return spoken_semantics.SemanticWindow(
        start=start,
        end=end,
        text=text,
        anchor_time=anchor,
        event_label=event,
        event_similarity=score,
        opening_quality=0.8,
        story_quality=0.85,
        ending_quality=0.9,
        coherence=0.88,
        score=score,
        vector=(score, 1.0 - score),
    )


def _match(
    topic_id: str,
    window_index: int,
    similarity: float,
    runner_up: float,
    score: float,
) -> spoken_semantics.TopicMatch:
    return spoken_semantics.TopicMatch(
        topic_id=topic_id,
        window_index=window_index,
        similarity=similarity,
        runner_up_similarity=runner_up,
        margin=similarity - runner_up,
        score=score,
    )


def test_spoken_semantic_discovery_runs_independent_topic_searches() -> None:
    transcript = speech.Transcript(
        language="en",
        duration=40.0,
        model="synthetic",
        settings={},
        segments=(
            _segment(0.0, 12.0, "Freedom changes how you play."),
            _segment(14.0, 26.0, "Persistent extraction changes the next run."),
            _segment(28.0, 40.0, "The stakes change every fight."),
        ),
    )
    windows = [
        _window(0.0, 12.0, "Freedom changes how you play.", 6.0, "lesson_explanation", 0.82),
        _window(
            14.0,
            26.0,
            "Persistent extraction changes the next run.",
            20.0,
            "reveal_payoff",
            0.84,
        ),
        _window(28.0, 40.0, "The stakes change every fight.", 34.0, "conflict_stakes", 0.86),
    ]
    matches = {
        "player_freedom": [_match("player_freedom", 0, 0.81, 0.55, 0.83)],
        "persistent_progression": [_match("persistent_progression", 1, 0.83, 0.54, 0.85)],
        "higher_stakes": [_match("higher_stakes", 2, 0.85, 0.52, 0.87)],
    }
    topic_summary = {
        topic_id: {
            "best_similarity": items[0].similarity,
            "best_runner_up_similarity": items[0].runner_up_similarity,
            "best_margin": items[0].margin,
            "minimum_similarity": 0.52,
            "minimum_topic_margin": 0.05,
            "candidate_count": 1,
            "independent_search": True,
        }
        for topic_id, items in matches.items()
    }
    profile = load_profile("mw4", CAMPAIGN)
    config = dict(profile.config)
    config["spoken_content"] = dict(config["spoken_content"])
    config["spoken_content"]["source_roles"] = {"synthetic": "developer"}

    with (
        patch(
            "clipper_engine.spoken_semantics.discover_windows",
            return_value=(windows, {"candidate_count": 3}),
        ),
        patch(
            "clipper_engine.spoken_semantics.topic_matches",
            return_value=(matches, topic_summary),
        ),
    ):
        plans, required, diagnostics = spoken.build_candidate_plans(
            transcript,
            "synthetic",
            config,
            embedder=lambda texts: [[1.0, 0.0] for _ in texts],
        )

    assert {item["id"] for item in required} == {
        "developer:player_freedom",
        "developer:persistent_progression",
        "developer:higher_stakes",
    }
    assert len(plans) == 3
    assert all(10.0 <= plan.output_duration <= 12.0 for plan in plans)
    assert all(len(plan.covered_semantic_anchor_ids) == 1 for plan in plans)
    assert all(item["disposition"] == "search_required" for item in required)
    assert diagnostics["independent_topic_search"] is True
    assert diagnostics["maximum_topics_per_clip"] == 1


def test_spoken_semantic_discovery_records_no_admissible_topic() -> None:
    transcript = speech.Transcript(
        language="en",
        duration=12.0,
        model="synthetic",
        settings={},
        segments=(_segment(0.0, 12.0, "A complete but unrelated source moment."),),
    )
    profile = load_profile("mw4", CAMPAIGN)
    config = dict(profile.config)
    config["spoken_content"] = dict(config["spoken_content"])
    config["spoken_content"]["source_roles"] = {"synthetic": "worldbuilder"}
    empty_matches = {
        "hajin_fallout": [],
        "rogue_operators": [],
        "off_books_operation": [],
    }
    summary = {
        topic_id: {
            "best_similarity": 0.2,
            "best_runner_up_similarity": 0.19,
            "best_margin": 0.01,
            "minimum_similarity": 0.52,
            "minimum_topic_margin": 0.05,
            "candidate_count": 0,
            "independent_search": True,
        }
        for topic_id in empty_matches
    }

    with (
        patch(
            "clipper_engine.spoken_semantics.discover_windows",
            return_value=(
                [
                    _window(
                        0.0,
                        12.0,
                        "A complete but unrelated source moment.",
                        6.0,
                        "story_experience",
                        0.8,
                    )
                ],
                {"candidate_count": 1},
            ),
        ),
        patch(
            "clipper_engine.spoken_semantics.topic_matches",
            return_value=(empty_matches, summary),
        ),
    ):
        plans, required, _ = spoken.build_candidate_plans(
            transcript,
            "synthetic",
            config,
            embedder=lambda texts: [[1.0, 0.0] for _ in texts],
        )

    assert plans == []
    assert {item["disposition"] for item in required} == {"no_admissible_candidate"}


def test_window_search_uses_soft_regions_and_silence_padding() -> None:
    units = [
        spoken_semantics.SemanticUnit(
            40.32,
            43.26,
            "The nuclear meltdown spread radiation throughout the region.",
        ),
        spoken_semantics.SemanticUnit(
            45.02,
            50.06,
            "Dangerous winds made the surrounding region uninhabitable.",
        ),
    ]

    bounds = spoken_semantics._window_bounds(
        0,
        units,
        10.0,
        12.0,
        60.0,
        4.5,
    )

    assert bounds
    assert any(
        10.0 <= fitted_end - fitted_start <= 12.0 for _, _, fitted_start, fitted_end in bounds
    )


def test_topic_matches_rejects_broad_sibling_similarity() -> None:
    windows = [
        _window(
            0.0,
            11.0,
            "Taking down the lieutenant is only half the job. Secure the dog tags.",
            5.5,
            "conflict_stakes",
            0.8,
        ),
        _window(
            20.0,
            31.0,
            "Extract them intact. They advance our intel and build the next lead.",
            25.5,
            "reveal_payoff",
            0.8,
        ),
    ]
    topics = [
        {
            "id": "lieutenant",
            "coverage_descriptions": ["lieutenant target"],
        },
        {
            "id": "intel",
            "coverage_descriptions": ["extract intel"],
        },
    ]

    def embed(texts: list[str]) -> list[list[float]]:
        values: list[list[float]] = []
        for text in texts:
            lowered = text.casefold()
            if "advance our intel" in lowered or "extract intel" in lowered:
                values.append([0.1, 1.0])
            elif "lieutenant" in lowered:
                values.append([1.0, 0.45])
            else:
                values.append([0.5, 0.5])
        return values

    matches, _ = spoken_semantics.topic_matches(
        windows,
        topics,
        embed,
        {
            "spoken_content": {
                "semantic": {
                    "minimum_topic_similarity": 0.52,
                    "minimum_topic_margin": 0.05,
                    "variants_per_topic": 5,
                }
            }
        },
    )

    assert [item.window_index for item in matches["lieutenant"]] == [0]
    assert [item.window_index for item in matches["intel"]] == [1]


def test_allocator_covers_string_semantic_anchors_without_redundant_clips() -> None:
    config = load_profile("mw4", CAMPAIGN).config

    def candidate(key: str, anchor: str, start: float, score: float) -> dict[str, object]:
        return {
            "plan_key": key,
            "score": score,
            "start": start,
            "end": start + 11.0,
            "segments": [{"start": start, "end": start + 11.0, "speed": 1.0, "reason": anchor}],
            "covered_semantic_anchor_ids": [anchor],
            "finishing_move": None,
        }

    candidates = [
        candidate("a-low", "developer:a", 0.0, 0.5),
        candidate("a-high", "developer:a", 20.0, 0.9),
        candidate("b", "developer:b", 40.0, 0.8),
        candidate("c", "developer:c", 60.0, 0.8),
    ]
    selected = allocation._solve_source(
        "developer",
        candidates,
        {"developer:a", "developer:b", "developer:c"},
        False,
        config,
    )
    assert [item["plan_key"] for item in selected] == ["a-high", "b", "c"]


def test_spoken_contract_requires_headline_and_transcript_evidence() -> None:
    config = load_profile("mw4", CAMPAIGN).config
    plan = {
        "start": 1.0,
        "end": 12.0,
        "output_duration": 11.0,
        "opening_quality": 0.5,
        "ending_quality": 0.5,
        "retention_quality": 0.5,
        "payoff_quality": 0.5,
        "story_coherence": 0.5,
        "weakest_quarter_interest": 0.5,
        "low_interest_fraction": 0.1,
        "max_unexplained_low_interest_run_seconds": 0.0,
        "content_type": "developer",
        "topic_id": "topic",
        "headline": "",
        "transcript_text": "",
        "covered_semantic_anchor_ids": ["developer:topic"],
        "segments": [{"start": 1.0, "end": 12.0, "speed": 1.0, "reason": "topic"}],
        "finishing_move": None,
    }
    failures = contract.validate_plan("developer", 1, plan, config)
    assert any("no on-screen headline" in item for item in failures)
    assert any("no transcript evidence" in item for item in failures)


def test_plan_key_binds_editorial_headline() -> None:
    base = {
        "story_type": "developer_statement",
        "effect_profile": "spoken_content_text_overlay",
        "content_type": "developer",
        "topic_id": "freedom",
        "headline": "Headline A",
        "segments": [{"start": 1.0, "end": 12.0, "speed": 1.0, "reason": "topic"}],
        "finishing_move": None,
    }
    changed = dict(base)
    changed["headline"] = "Headline B"
    assert contract.plan_key(base) != contract.plan_key(changed)


def test_catalog_resolves_spotlight_original_through_provider_download_api(
    tmp_path: Path,
) -> None:
    asset = {
        "id": "asset-1",
        "title": "DMZ.mp4",
        "fileName": "DMZ.mp4",
        "duration": 12_000,
        "derivatives": [
            {
                "type": "source",
                "fileSize": 500_000,
                "width": 3840,
                "height": 2160,
                "duration": 12_000,
            },
            {
                "type": "proxy",
                "url": "https://example.test/proxy.mp4",
                "fileSize": 10_000,
                "width": 1280,
                "height": 720,
                "duration": 12_000,
            },
        ],
        "_presentation_id": "presentation-1",
        "_playlist_id": "playlist-1",
    }
    source_profile = {
        "analysis_derivative": "source",
        "assets": [{"title": "DMZ.mp4", "source_key": "dmz"}],
    }
    output = tmp_path / "catalog.json"
    with patch("clipper_engine.sources.catalog._capture_assets", return_value=[asset]):
        result = catalog.discover(
            "https://app.mediasilo.com/spotlight/presentation-1",
            output,
            None,
            1,
            source_profile,
        )

    selected = result["sources"][0]
    assert selected["derivative_type"] == "source"
    assert selected["url_mode"] == "provider_download_api"
    assert selected["url"] == (
        "https://api.mediasilo.com/v3/presentations/presentation-1/"
        "playlists/playlist-1/assets/asset-1/download"
    )


def test_proxy_original_alignment_requires_same_provider_asset_and_timing(
    tmp_path: Path,
) -> None:
    common = {
        "source_key": "dmz",
        "title": "DMZ.mp4",
        "file_name": "DMZ.mp4",
        "provider_asset_id": "asset-1",
        "provider_presentation_id": "presentation-1",
        "provider_playlist_id": "playlist-1",
        "declared_duration_ms": 12_000,
    }
    analysis = {
        **common,
        "derivative_type": "proxy",
        "duration_seconds": 12.01,
        "sha256": "proxy-sha",
        "media_contract": {
            "video": {"timing": {"nominal_rate": "60/1"}},
        },
    }
    original = {
        **common,
        "derivative_type": "source",
        "duration_seconds": 12.0,
        "sha256": "source-sha",
        "media_contract": {
            "video": {"timing": {"nominal_rate": "60/1"}},
        },
    }
    analysis_path = tmp_path / "analysis.json"
    original_path = tmp_path / "original.json"
    alignment_path = tmp_path / "alignment.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    original_path.write_text(json.dumps(original), encoding="utf-8")

    result = source_qa.certify_analysis_alignment(
        analysis_path,
        original_path,
        alignment_path,
    )

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    updated = json.loads(original_path.read_text(encoding="utf-8"))
    assert updated["analysis_alignment"]["production_sha256"] == "source-sha"


def test_proxy_original_alignment_rejects_different_provider_asset(tmp_path: Path) -> None:
    base = {
        "source_key": "dmz",
        "title": "DMZ.mp4",
        "file_name": "DMZ.mp4",
        "provider_presentation_id": "presentation-1",
        "provider_playlist_id": "playlist-1",
        "declared_duration_ms": 12_000,
        "duration_seconds": 12.0,
        "media_contract": {
            "video": {"timing": {"nominal_rate": "60/1"}},
        },
    }
    analysis_path = tmp_path / "analysis.json"
    original_path = tmp_path / "original.json"
    analysis_path.write_text(
        json.dumps(
            {
                **base,
                "provider_asset_id": "asset-proxy",
                "derivative_type": "proxy",
                "sha256": "proxy-sha",
            }
        ),
        encoding="utf-8",
    )
    original_path.write_text(
        json.dumps(
            {
                **base,
                "provider_asset_id": "asset-source",
                "derivative_type": "source",
                "sha256": "source-sha",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="temporal alignment failed"):
        source_qa.certify_analysis_alignment(
            analysis_path,
            original_path,
            tmp_path / "alignment.json",
        )


def test_topic_specificity_dominates_editorial_window_quality() -> None:
    transcript = speech.Transcript(
        language="en",
        duration=30.0,
        model="synthetic",
        settings={},
        segments=(
            _segment(0.0, 11.0, "Generic polished statement."),
            _segment(15.0, 26.0, "Everything extracted is persistent and can be lost."),
        ),
    )
    generic = _window(
        0.0,
        11.0,
        "Generic polished statement.",
        5.5,
        "memorable_quote_opinion",
        0.98,
    )
    specific = _window(
        15.0,
        26.0,
        "Everything extracted is persistent and can be lost.",
        20.5,
        "lesson_explanation",
        0.60,
    )
    matches = {
        "player_freedom": [],
        "persistent_progression": [
            _match("persistent_progression", 0, 0.60, 0.54, 0.55),
            _match("persistent_progression", 1, 0.72, 0.60, 0.78),
        ],
        "higher_stakes": [],
    }
    summary = {
        topic_id: {
            "best_similarity": (items[0].similarity if items else None),
            "best_runner_up_similarity": (items[0].runner_up_similarity if items else None),
            "best_margin": (items[0].margin if items else None),
            "minimum_similarity": 0.52,
            "minimum_topic_margin": 0.05,
            "candidate_count": len(items),
            "independent_search": True,
        }
        for topic_id, items in matches.items()
    }
    profile = load_profile("mw4", CAMPAIGN)
    config = dict(profile.config)
    config["spoken_content"] = dict(config["spoken_content"])
    config["spoken_content"]["source_roles"] = {"synthetic": "developer"}

    with (
        patch(
            "clipper_engine.spoken_semantics.discover_windows",
            return_value=([generic, specific], {"candidate_count": 2}),
        ),
        patch(
            "clipper_engine.spoken_semantics.topic_matches",
            return_value=(matches, summary),
        ),
    ):
        plans, _, _ = spoken.build_candidate_plans(
            transcript,
            "synthetic",
            config,
            embedder=lambda texts: [[1.0, 0.0] for _ in texts],
        )

    persistent = [plan for plan in plans if plan.topic_id == "persistent_progression"]
    persistent.sort(key=lambda plan: -plan.score)
    assert persistent[0].transcript_text == specific.text
    assert persistent[0].score > persistent[1].score

def test_mw4_title_has_no_accent_line_but_keeps_headline(tmp_path: Path) -> None:
    config = load_profile("mw4", CAMPAIGN).config
    target = tmp_path / "editorial_master.nut"
    with patch(
        "clipper_engine.rendering.overlays.portrait_layout.title_lines",
        return_value=[{"text": "TEST HOOK", "font_size": 66, "y": 330, "width": 330.0}],
    ):
        filter_graph, info = overlays._portrait_filter(
            "TEST HOOK",
            target,
            config,
            {"width": 1080, "height": 1920},
            Path("/unused-font-path.ttf"),
        )

    assert "drawbox=" not in filter_graph
    assert "drawtext=" in filter_graph
    assert "fontcolor=0xFAFAFA" in filter_graph
    assert info["title_lines"][0]["text"] == "TEST HOOK"

