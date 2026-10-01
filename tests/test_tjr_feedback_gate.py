"""Deterministic offline checks of post-render source and editorial feedback."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tjr_feedback_gate import review_run

CHANNEL = "UCf1q6dhccWr6eQEcFFnJSbA"


def _probe(_path: Path) -> dict[str, object]:
    return {
        "width": 1080,
        "height": 1920,
        "fps": 60.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "duration": 29.2,
        "video_duration": 29.2,
        "audio_duration": 29.2,
    }


def _fixture(root: Path, *, generic: bool = False) -> Path:
    artifact = root / "tjr-real-original-youtube-hd-test"
    base = artifact / "tjr-modal-artifacts" / "attempt-1" / "render"
    clips = base / "clips"
    clips.mkdir(parents=True)
    stem = "01-tjr-test"
    for suffix in (".mp4", ".srt", ".ssim.txt", "-contact.png", "-preview.png"):
        (clips / (stem + suffix)).write_bytes(b"fixture")
    (clips / (stem + ".ass")).write_text(
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n"
        "Style: Caption,DejaVu Sans,64,white,white,black,black,-1,0,0,0,"
        "100,100,0,0,3,18,0,2,120,120,375,1\n"
        "[Events]\n"
        r"Dialogue: 5,0:00:00.00,0:00:29.20,Hook,,0,0,0,,{\an8\pos(540,185)}"
        "WHY DID THE REFEREE CANCEL THAT GOAL?\n"
        r"Dialogue: 2,0:00:00.10,0:00:00.55,Caption,,0,0,0,,{\c&H0059DEFF&}"
        r"WORD{\rCaption}"
        "\n",
        encoding="utf-8",
    )
    quality = {
        "status": "MEASURED_SOURCE_MATCHED_ENCODING",
        "source_to_delivery_mean_ssim": 0.997,
        "compared_frames": 1750,
        "output_bytes": len(b"fixture"),
        "campaign_watermark_applied": True,
        "edit_plan": {
            "style": "semantic_micro_punch",
            "punch_scale": 1.025,
            "attention_beats": [
                {"start": 0.0, "end": 0.68},
                {"start": 8.0, "end": 8.52},
            ],
            "random_effects": False,
        },
    }
    (clips / (stem + ".quality.json")).write_text(json.dumps(quality))
    report = {
        "source_channel_id": CHANNEL,
        "source_sha256": "a" * 64,
        "source_published_at": "2026-09-27T15:00:00Z",
        "source_url": "https://www.youtube.com/watch?v=AbCdEfGhI12",
        "source_profile": {"fps": "60/1"},
        "clips": [
            {
                "file": f"clips/{stem}.mp4",
                "srt": f"clips/{stem}.srt",
                "ass_sidecar": f"clips/{stem}.ass",
                "source_matched_quality": f"clips/{stem}.quality.json",
                "contact_sheet": f"clips/{stem}-contact.png",
                "preview": f"clips/{stem}-preview.png",
                "duration_seconds": 29.2,
                "overlay_acceptance": {
                    "style": "B2",
                    "persistent_hook_seconds": 29.2,
                    "spoken_word_highlight_events": 1,
                },
                "hook_candidate": (
                    'THE MOMENT: "PARTIAL QUOTE"'
                    if generic
                    else "WHY DID THE REFEREE CANCEL THAT GOAL?"
                ),
                "source_start_seconds": 2861.2,
                "source_end_seconds": 2890.4,
                "review_required": True,
            }
        ],
    }
    report["clips"][0]["editorial_integrity_gate"] = {
        "status": "unverified",
        "evidence": ["manual semantic review still required"],
    }
    (base / "tjr-youtube-qa-report.json").write_text(json.dumps(report))
    (base / "transcript.json").write_text(
        json.dumps(
            [
                {
                    "start": 2860.0,
                    "end": 2891.0,
                    "text": (
                        "Why did the referee cancel that goal? The replay showed a handball before "
                        "the shot went in, so we accepted the decision and started again."
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    (base / "source-analysis-coverage.json").write_text(
        json.dumps(
            {
                "reported_original_seconds": 3218,
                "analyzed_source_seconds": 3217.6,
                "full_source_analyzed": True,
            }
        )
    )
    return artifact


def _noop_fixture(root: Path) -> Path:
    artifact = root / "tjr-real-original-youtube-hd-noop"
    base = artifact / "tjr-modal-artifacts" / "attempt-1" / "render"
    base.mkdir(parents=True)
    (base / "tjr-youtube-qa-report.json").write_text(
        json.dumps(
            {
                "status": "NO_CREATOR_GRADE_MOMENTS",
                "source_channel_id": CHANNEL,
                "source_sha256": "b" * 64,
                "source_published_at": "2026-09-29T15:00:00Z",
                "source_url": "https://www.youtube.com/watch?v=NoopVideo12",
                "selection_policy": "quality_driven_zero_to_n",
                "selected_clip_count": 0,
                "clips": [],
            }
        ),
        encoding="utf-8",
    )
    (base / "source-analysis-coverage.json").write_text(
        json.dumps(
            {
                "reported_original_seconds": 1800,
                "analyzed_source_seconds": 1800,
                "full_source_analyzed": True,
            }
        ),
        encoding="utf-8",
    )
    (base / "editorial-candidate-audit.json").write_text(
        json.dumps(
            {
                "selection_policy": "quality_driven_zero_to_n",
                "selected": [],
                "selected_count": 0,
                "rejected": [{"reason": "BELOW_CREATOR_QUALITY_FLOOR"}],
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_zero_creator_grade_clips_is_a_valid_audited_noop(tmp_path: Path) -> None:
    _noop_fixture(tmp_path)
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert result["status"] == "NO_CREATOR_GRADE_MOMENTS__NO_RENDER_REQUIRED"
    assert result["issues"] == []
    assert result["technically_verified_mp4_count"] == 0
    assert result["editorial_noop_count"] == 1
    assert result["next_actions"] == ["SKIP_SOURCE_NO_CREATOR_GRADE_MOMENTS"]
    assert result["automatic_publication_allowed"] is False


def test_zero_clip_noop_requires_explicit_consistent_counts(tmp_path: Path) -> None:
    artifact = _noop_fixture(tmp_path)
    report_path = next(artifact.rglob("tjr-youtube-qa-report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("selected_clip_count")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "INVALID_EDITORIAL_NOOP" in result["issues"]

    artifact = _noop_fixture(tmp_path / "second")
    audit_path = next(artifact.rglob("editorial-candidate-audit.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("selected_count")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    result = review_run(tmp_path / "second", expected_channels=1, probe=_probe)
    assert "INVALID_EDITORIAL_NOOP" in result["issues"]

    artifact = _noop_fixture(tmp_path / "third")
    report_path = next(artifact.rglob("tjr-youtube-qa-report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selected_clip_count"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = review_run(tmp_path / "third", expected_channels=1, probe=_probe)
    assert "INVALID_EDITORIAL_NOOP" in result["issues"]


def test_success_is_review_only_and_checks_sidecars(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert result["status"] == "TECHNICAL_QA_PASSED__HUMAN_REVIEW_REQUIRED"
    assert result["technically_verified_mp4_count"] == 1
    assert result["automatic_publication_allowed"] is False
    assert result["channels"][0]["minimum_ssim"] == 0.997


def test_ass_hook_must_match_audited_hook(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    ass = next(artifact.rglob("*.ass"))
    ass.write_text(
        ass.read_text(encoding="utf-8").replace(
            "WHY DID THE REFEREE CANCEL THAT GOAL?", "MISLEADING STALE HOOK"
        ),
        encoding="utf-8",
    )
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "CAPTION_OR_HOOK_TIMING_FAILED" in result["issues"]


def test_fidelity_report_must_match_actual_mp4_size(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    mp4 = next(artifact.rglob("*.mp4"))
    mp4.write_bytes(b"replaced-render-bytes")
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "SOURCE_FIDELITY_FAILED" in result["issues"]


def test_generic_hooks_cannot_be_mislabeled_as_good_production(tmp_path: Path) -> None:
    _fixture(tmp_path, generic=True)
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "GENERIC_HOOK_OVERUSE" in result["issues"]
    assert "IMPROVE_GROUNDED_CREATIVE_HOOKS" in result["next_actions"]


def test_missing_sidecar_and_partial_source_are_reported(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    next(artifact.rglob("01-tjr-test.ass")).unlink()
    coverage = next(artifact.rglob("source-analysis-coverage.json"))
    coverage.write_text(
        json.dumps(
            {
                "reported_original_seconds": 3218,
                "analyzed_source_seconds": 840,
                "full_source_analyzed": False,
            }
        )
    )
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "MISSING_OR_INVALID_CLIP_EVIDENCE" in result["issues"]
    assert "INCOMPLETE_SOURCE_COVERAGE" in result["issues"]


def test_failed_egress_is_reported_without_claiming_missing_campaign_channel(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "tjr-real-original-youtube-hd-blocked"
    failed.mkdir()
    (failed / "verified-original-egress.json").write_text(
        json.dumps({"status": "YOUTUBE_EGRESS_BOT_CHALLENGE"})
    )
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "MISSING_CHANNEL_ARTIFACT" in result["issues"]
    assert "YOUTUBE_EGRESS_BLOCKED" in result["issues"]
    assert result["status"] == "YOUTUBE_EGRESS_BLOCKED"
    assert "REPAIR_YOUTUBE_SOURCE_TRANSPORT" in result["next_actions"]


def test_workflow_stores_independent_two_channel_evidence() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "feedback_audit:" in workflow
    assert "youtube_preview, youtube_modal_egress, render" in workflow
    assert "python -m scripts.tjr_feedback_gate" in workflow
    assert "actions/download-artifact@v4" in workflow
    assert "gh issue comment 7" in workflow


def test_replay_mode_never_requests_source_budget_or_new_media() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "feedback_replay" in workflow
    assert "feedback_source_run_id:" in workflow
    assert "run-id: ${{ inputs.feedback_source_run_id }}" in workflow
    assert "Replay existing Modal production artifacts at no new media cost" in workflow
    assert "source_run_id = os.getenv('TJR_FEEDBACK_SOURCE_RUN_ID', '')" in workflow


def test_replay_channel_count_and_artifact_family_are_derived() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "EXPECTED_CHANNELS=1" in workflow
    assert '"$TJR_SOURCE_MODE" == "feedback_replay"' in workflow
    assert "EXPECTED_CHANNELS=0" in workflow
    assert "TJR_MODAL_CHANNEL_ID: ${{ inputs.target_channel_id }}" in workflow
    for pattern in (
        "tjr-real-original-youtube-hd-*",
        "tjr-real-youtube-hd-*",
        "tjr-weekly-hd-*",
    ):
        assert pattern in workflow
    assert "AUDITOR_CRASH" in workflow
    assert "TJR_EXPECTED_CHANNELS: true" not in workflow


def test_failing_fidelity_is_not_counted_as_technically_verified(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    quality = next(artifact.rglob("01-tjr-test.quality.json"))
    evidence = json.loads(quality.read_text(encoding="utf-8"))
    evidence["source_to_delivery_mean_ssim"] = 0.91
    quality.write_text(json.dumps(evidence), encoding="utf-8")
    report = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert report["status"] == "BLOCKED"
    assert report["technically_verified_mp4_count"] == 0
    assert report["channels"][0]["minimum_ssim"] == 0.91
    assert "SOURCE_FIDELITY_FAILED" in report["issues"]


def test_nonfinite_fidelity_is_never_accepted_or_reported_as_perfect(
    tmp_path: Path,
) -> None:
    artifact = _fixture(tmp_path)
    quality = next(artifact.rglob("01-tjr-test.quality.json"))
    evidence = json.loads(quality.read_text(encoding="utf-8"))
    evidence["source_to_delivery_mean_ssim"] = float("nan")
    quality.write_text(json.dumps(evidence), encoding="utf-8")
    report = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert report["technically_verified_mp4_count"] == 0
    assert report["channels"][0]["minimum_ssim"] is None
    assert "SOURCE_FIDELITY_FAILED" in report["issues"]


def test_independent_ass_audit_rejects_false_persistent_hook(
    tmp_path: Path,
) -> None:
    artifact = _fixture(tmp_path)
    ass = next(artifact.rglob("01-tjr-test.ass"))
    ass.write_text(
        ass.read_text(encoding="utf-8").replace("0:00:29.20,Hook", "0:00:04.00,Hook"),
        encoding="utf-8",
    )
    report = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "CAPTION_OR_HOOK_TIMING_FAILED" in report["issues"]
    assert report["technically_verified_mp4_count"] == 0


def test_independent_ass_audit_rejects_missing_word_highlight(
    tmp_path: Path,
) -> None:
    artifact = _fixture(tmp_path)
    ass = next(artifact.rglob("01-tjr-test.ass"))
    ass.write_text(
        ass.read_text(encoding="utf-8").replace(r"\c&H0059DEFF&", ""),
        encoding="utf-8",
    )
    report = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "CAPTION_OR_HOOK_TIMING_FAILED" in report["issues"]
    assert report["technically_verified_mp4_count"] == 0


def test_caption_event_outside_clip_is_rejected(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    ass = next(artifact.rglob("01-tjr-test.ass"))
    ass.write_text(
        ass.read_text(encoding="utf-8").replace(
            "0:00:00.10,0:00:00.55,Caption",
            "0:00:40.00,0:00:41.00,Caption",
        ),
        encoding="utf-8",
    )
    report = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "CAPTION_OR_HOOK_TIMING_FAILED" in report["issues"]
    assert report["technically_verified_mp4_count"] == 0


def test_random_or_aggressive_edit_plan_is_rejected(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    quality = next(artifact.rglob("01-tjr-test.quality.json"))
    evidence = json.loads(quality.read_text(encoding="utf-8"))
    evidence["edit_plan"]["punch_scale"] = 1.08
    quality.write_text(json.dumps(evidence), encoding="utf-8")
    report = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "EDITORIAL_EDIT_PLAN_FAILED" in report["issues"]
    assert "IMPROVE_EDITORIAL_EDITING" in report["next_actions"]
    assert report["technically_verified_mp4_count"] == 0


def test_unaudited_split_screen_layout_is_rejected_for_double_coverage(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    quality = next(artifact.rglob("01-tjr-test.quality.json"))
    evidence = json.loads(quality.read_text(encoding="utf-8"))
    evidence["edit_plan"] = {
        "style": "split_screen_montage",
        "punch_scale": 1.0,
        "attention_beats": [],
        "random_effects": False,
        "editorial_layout": "tjr-memecoin-logo-safe",
    }
    quality.write_text(json.dumps(evidence), encoding="utf-8")
    report = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "EDITORIAL_EDIT_PLAN_FAILED" in report["issues"]


def test_independent_audit_rejects_overlapping_source_windows(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    report_path = next(artifact.rglob("tjr-youtube-qa-report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    second = dict(report["clips"][0])
    stem = "02-tjr-test"
    clips_dir = report_path.parent / "clips"
    for suffix in (".mp4", ".srt", ".ssim.txt", "-contact.png", "-preview.png"):
        (clips_dir / (stem + suffix)).write_bytes(b"fixture")
    (clips_dir / (stem + ".ass")).write_text(
        (clips_dir / "01-tjr-test.ass").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (clips_dir / (stem + ".quality.json")).write_text(
        (clips_dir / "01-tjr-test.quality.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    second.update(
        {
            "file": f"clips/{stem}.mp4",
            "srt": f"clips/{stem}.srt",
            "ass_sidecar": f"clips/{stem}.ass",
            "source_matched_quality": f"clips/{stem}.quality.json",
            "contact_sheet": f"clips/{stem}-contact.png",
            "preview": f"clips/{stem}-preview.png",
            "source_start_seconds": 2880.0,
            "source_end_seconds": 2909.2,
        }
    )
    report["clips"].append(second)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "DUPLICATE_OR_INVALID_CLIP_WINDOW" in result["issues"]


def test_every_real_production_mode_routes_through_independent_feedback_audit() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "youtube_preview, youtube_modal_egress, render" in workflow
    assert '["modal_direct","youtube_direct","verified_mirror"]' in workflow
    assert "pattern: tjr-real-original-youtube-hd-*" in workflow
    assert "pattern: tjr-real-youtube-hd-*" in workflow
    assert "pattern: tjr-weekly-hd-*" in workflow


def test_production_has_no_automatic_transport_fallback() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "auto_direct" not in workflow
    assert "youtube_alternate_egress:" not in workflow
    assert "Fallback YouTube original" not in workflow
    assert "needs.youtube_preview.result == 'failure'" not in workflow
    assert "acquisition_route" not in workflow
    assert "inputs.source_mode == 'modal_direct'" in workflow
    assert "inputs.source_mode == 'youtube_direct'" in workflow
    assert "bgutil-ytdlp-pot-provider:2.0.0" in workflow
    assert "-p 127.0.0.1:4416:4416" in workflow
    assert "TJR_BGUTIL_POT_PROVIDER_URL: http://127.0.0.1:4416" in workflow
    assert "TJR_ASR_MODEL: distil-large-v3" in workflow


def test_double_coverage_has_no_tjr_source_layout_exception(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    report_path = next(artifact.rglob("tjr-youtube-qa-report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_url"] = "https://www.youtube.com/watch?v=LvnemCfJpQU"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "KNOWN_SOURCE_LAYOUT_REGRESSION" not in result["issues"]


def test_duplicate_semantic_headlines_are_rejected_even_for_distinct_windows(
    tmp_path: Path,
) -> None:
    artifact = _fixture(tmp_path)
    report_path = next(artifact.rglob("tjr-youtube-qa-report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    second = dict(report["clips"][0])
    stem = "02-tjr-test"
    clips_dir = report_path.parent / "clips"
    for suffix in (".mp4", ".srt", ".ssim.txt", "-contact.png", "-preview.png"):
        (clips_dir / (stem + suffix)).write_bytes(b"fixture")
    (clips_dir / (stem + ".ass")).write_text(
        (clips_dir / "01-tjr-test.ass").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (clips_dir / (stem + ".quality.json")).write_text(
        (clips_dir / "01-tjr-test.quality.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    second.update(
        {
            "file": f"clips/{stem}.mp4",
            "srt": f"clips/{stem}.srt",
            "ass_sidecar": f"clips/{stem}.ass",
            "source_matched_quality": f"clips/{stem}.quality.json",
            "contact_sheet": f"clips/{stem}-contact.png",
            "preview": f"clips/{stem}-preview.png",
            "source_start_seconds": 3000.0,
            "source_end_seconds": 3029.2,
        }
    )
    report["clips"].append(second)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "WEAK_OR_DUPLICATE_HOOK" in result["issues"]
    assert "IMPROVE_GROUNDED_CREATIVE_HOOKS" in result["next_actions"]


def test_independent_hook_audit_rejects_negated_millions_claim(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    report_path = next(artifact.rglob("tjr-youtube-qa-report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["clips"][0]["hook_candidate"] = "A TRADER CLAIMS MILLIONS: HOW?"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    transcript = next(artifact.rglob("transcript.json"))
    transcript.write_text(
        json.dumps(
            [
                {
                    "start": 2860.0,
                    "end": 2891.0,
                    "text": "I never made a million dollars trading.",
                }
            ]
        ),
        encoding="utf-8",
    )
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "HOOK_SOURCE_MISMATCH" in result["issues"]
    assert result["technically_verified_mp4_count"] == 0


def test_failed_direct_attempt_does_not_poison_a_successful_fallback(tmp_path: Path) -> None:
    _fixture(tmp_path)
    failed = tmp_path / "tjr-real-youtube-hd-failed"
    failed.mkdir()
    (failed / "source-acquisition-errors.json").write_text(
        json.dumps({"error": "BOT_CHALLENGE"}),
        encoding="utf-8",
    )
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "YOUTUBE_EGRESS_BLOCKED" not in result["issues"]
    assert result["technically_verified_mp4_count"] == 1
    assert result["attempt_failures"]


def test_replay_auto_channel_count_uses_single_double_coverage_channel(
    tmp_path: Path,
) -> None:
    first = _fixture(tmp_path)
    first.rename(tmp_path / f"tjr-real-original-youtube-hd-{CHANNEL}-123")
    result = review_run(tmp_path, expected_channels=0, probe=_probe)
    assert result["expected_channels"] == 1
    assert "MISSING_CHANNEL_ARTIFACT" not in result["issues"]


def test_verified_mirror_upload_keeps_transcript_and_source_coverage() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "tjr-mirror-artifacts/**/transcript.json" in workflow
    assert "tjr-mirror-artifacts/**/source-analysis-coverage.json" in workflow


def test_decoded_mp4_duration_must_match_report_and_source_window(tmp_path: Path) -> None:
    _fixture(tmp_path)

    def short_probe(_path: Path) -> dict[str, object]:
        result = _probe(_path)
        result["duration"] = 10.0
        result["video_duration"] = 10.0
        result["audio_duration"] = 29.2
        return result

    report = review_run(tmp_path, expected_channels=1, probe=short_probe)
    assert "INVALID_REAL_MEDIA_DURATION" in report["issues"]
    assert report["technically_verified_mp4_count"] == 0


def test_multiple_successful_fallbacks_are_all_audited_and_blocked_until_canonicalized(
    tmp_path: Path,
) -> None:
    first = _fixture(tmp_path)
    second = tmp_path / "tjr-youtube-alt-macos-success"
    import shutil

    shutil.copytree(first, second)
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert "MULTIPLE_PRODUCTION_ARTIFACTS" in result["issues"]
    assert "DUPLICATE_CHANNEL_ARTIFACT" in result["issues"]
    assert len(result["channels"]) == 2


def test_summary_audit_requires_matching_span_source_and_payoff(tmp_path):
    import json

    from scripts.tjr_feedback_gate import reviewed_summary_evidence

    hook = "Security stopped the performer at his own show"
    text = "Security stopped me outside my own show. The owner finally let me inside."
    review = dict(
        headline=hook,
        opening_standalone=True,
        payoff_complete=True,
        ending_complete=True,
        headline_supported=True,
        headline_self_contained=True,
        contains_promotion_or_intro=False,
        setup_quote="Security stopped me outside",
        payoff_quote="The owner finally let me inside.",
    )
    evidence = dict(reviewed_start=0, reviewed_end=24, exchange_review=review)
    saved = dict(
        complete=True,
        identity=dict(version="podcast_structured_editor_v3", source_sha256="a" * 64),
        audit=dict(assessments=[evidence]),
    )
    path = tmp_path / "editorial-cache.json"
    assert not reviewed_summary_evidence(tmp_path, hook, text, 0, 24, "a" * 64)
    path.write_text(json.dumps(saved))
    assert reviewed_summary_evidence(tmp_path, hook, text, 0, 24, "a" * 64)
    assert not reviewed_summary_evidence(tmp_path, hook, text, 0, 23, "a" * 64)
    assert not reviewed_summary_evidence(tmp_path, hook, text, 0, 24, "b" * 64)
    assert not reviewed_summary_evidence(tmp_path, hook, text, 0, 24, "")
    for field, value in [
        ("payoff_complete", False),
        ("contains_promotion_or_intro", True),
        ("payoff_quote", "He paid me millions of dollars"),
    ]:
        saved["audit"]["assessments"][0]["exchange_review"] = {**review, field: value}
        path.write_text(json.dumps(saved))
        assert not reviewed_summary_evidence(tmp_path, hook, text, 0, 24, "a" * 64)


def test_independent_v5_review_rejects_false_boundary_or_quote_provenance(tmp_path):
    import json

    from scripts.tjr_feedback_gate import reviewed_summary_evidence

    units = ["Security stopped me outside.", "The owner finally let me inside."]
    text = " ".join(units)
    hook = "The owner rescued his guest from security"
    boundary = dict(
        promotion_unit_ids=[],
        opening="standalone",
        ending="closed",
        payoff_location="selected",
        setup_unit_id=0,
        payoff_unit_id=1,
    )
    review = dict(
        headline=hook,
        opening_standalone=True,
        payoff_complete=True,
        ending_complete=True,
        headline_supported=True,
        headline_self_contained=True,
        contains_promotion_or_intro=False,
        setup_quote=units[0],
        payoff_quote=units[1],
        delivered_units=units,
        boundary_audit=boundary,
    )
    saved = dict(
        complete=True,
        identity=dict(version="podcast_structured_editor_v5", source_sha256="a" * 64),
        audit=dict(assessments=[dict(reviewed_start=0, reviewed_end=24, exchange_review=review)]),
    )
    path = tmp_path / "editorial-cache.json"
    path.write_text(json.dumps(saved))
    assert reviewed_summary_evidence(tmp_path, hook, text, 0, 24, "a" * 64)
    for changes in (
        {"payoff_location": "after"},
        {"payoff_unit_id": 0},
        {"payoff_unit_id": 99},
        {"promotion_unit_ids": [0]},
        {"ending": "unresolved"},
    ):
        review["boundary_audit"] = {**boundary, **changes}
        path.write_text(json.dumps(saved))
        assert not reviewed_summary_evidence(tmp_path, hook, text, 0, 24, "a" * 64)


def test_centered_black_ass_plate_audit_rejects_offset_or_tinted_plate(tmp_path):
    import pytest

    from clipper.models import ClipCandidate, TranscriptSegment, WordTiming
    from clipper.tiktok import audit_tiktok_ass, create_tiktok_ass
    from scripts.tjr_feedback_gate import verify_ass_sidecar

    hook = "THIS IS THE COMPLETE EXCHANGE"
    ass = create_tiktok_ass(
        ClipCandidate("v", 0, 2, "Yeah, exactly.", 1),
        [
            TranscriptSegment(
                0.1,
                1.2,
                "Yeah, exactly.",
                (WordTiming(0.1, 0.5, "yeah,"), WordTiming(0.5, 1.2, "exactly.")),
            )
        ],
        tmp_path / "centered.ass",
        hook_text=hook,
    )
    proof = audit_tiktok_ass(ass, clip_duration=2)
    verify_ass_sidecar(
        ass,
        duration_seconds=2,
        reported_word_events=proof["spoken_word_highlight_events"],
        expected_hook=hook,
    )
    original = ass.read_text()
    for mutated in (
        original.replace("&H00000000", "&H00201613"),
        original.replace(r"\alpha&H00&", r"\alpha&H58&"),
        original.replace("m 0 0 l ", "m 12 0 l "),
    ):
        ass.write_text(mutated)
        with pytest.raises(ValueError, match=r"black|geometry"):
            verify_ass_sidecar(
                ass,
                duration_seconds=2,
                reported_word_events=proof["spoken_word_highlight_events"],
                expected_hook=hook,
            )


def test_independent_decode_cache_reuses_only_identical_media(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import tjr_feedback_gate as gate

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original delivery")
    calls = []

    def probe(path, *, full_decode=False):
        calls.append(full_decode)
        return {"duration": 20.0, "width": 1080, "height": 1920}

    monkeypatch.setattr(gate, "probe_media", probe)
    monkeypatch.setattr(
        gate.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="ffmpeg pinned")
    )
    prior = tmp_path / "prior"
    gate.cached_probe_media(clip, full_decode=True, cache_root=None, output=prior)
    gate.cached_probe_media(clip, full_decode=True, cache_root=prior, output=tmp_path / "retry")
    assert calls == [True]
    clip.write_bytes(b"changed delivery")
    gate.cached_probe_media(clip, full_decode=True, cache_root=prior, output=tmp_path / "changed")
    assert calls == [True, True]
