from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clipper_engine import speech, spoken
from clipper_engine.gameplay import allocation, contract
from clipper_engine.profiles import load_profile
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
    assert profile.config["output"]["width"] == "source"
    assert profile.config["output"]["fps"] == "source"


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


def test_spoken_topic_search_emits_legal_windows_and_explicit_coverage() -> None:
    transcript = speech.Transcript(
        language="en",
        duration=40.0,
        model="synthetic",
        settings={},
        segments=tuple(
            _segment(float(index * 2), float(index * 2 + 2), text)
            for index, text in enumerate(
                [
                    "setup one",
                    "setup two",
                    "DMZ gives a different experience",
                    "you can play with a whole bunch of toys",
                    "the way you want to play",
                    "complete thought",
                    "spacing one",
                    "spacing two",
                    "everything is persistent",
                    "everything can be taken out and lost",
                    "more persistence",
                    "complete thought two",
                    "spacing three",
                    "spacing four",
                    "there are stakes",
                    "should I engage this person",
                    "dying has a different impact",
                    "complete thought three",
                    "tail one",
                    "tail two",
                ]
            )
        ),
    )
    profile = load_profile("mw4", CAMPAIGN)
    config = dict(profile.config)
    config["spoken_content"] = dict(config["spoken_content"])
    config["spoken_content"]["source_roles"] = {"synthetic": "developer"}
    plans, required, diagnostics = spoken.build_candidate_plans(transcript, "synthetic", config)
    required_ids = {item["id"] for item in required}
    assert required_ids == {
        "developer:player_freedom",
        "developer:persistent_progression",
        "developer:higher_stakes",
    }
    assert {plan.topic_id for plan in plans} == {
        "player_freedom",
        "persistent_progression",
        "higher_stakes",
    }
    assert all(10.0 <= plan.output_duration <= 12.0 for plan in plans)
    assert all(len(plan.covered_semantic_anchor_ids) == 1 for plan in plans)
    assert all(plan.headline for plan in plans)
    assert diagnostics["content_type"] == "developer"


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
