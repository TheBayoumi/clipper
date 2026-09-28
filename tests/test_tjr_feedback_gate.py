"""Deterministic offline checks of post-render source and editorial feedback."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tjr_feedback_gate import review_run

CHANNEL = "UCGHBUXjDCeiIXNdKR0HUZnA"


def _probe(_path: Path) -> dict[str, object]:
    return {
        "width": 1080,
        "height": 1920,
        "fps": 60.0,
        "video_codec": "h264",
        "audio_codec": "aac",
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
        "TRUTHFUL HOOK\n"
        r"Dialogue: 2,0:00:00.10,0:00:00.55,Caption,,0,0,0,,{\c&H0059DEFF&}"
        r"WORD{\rCaption}"
        "\n",
        encoding="utf-8",
    )
    quality = {
        "status": "MEASURED_SOURCE_MATCHED_ENCODING",
        "source_to_delivery_mean_ssim": 0.997,
        "compared_frames": 1750,
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
        "source_url": "https://www.youtube.com/watch?v=LvnemCfJpQU",
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
                    else "WHEN IS THIS MARKET-CAP ENTRY TOO LATE?"
                ),
                "source_start_seconds": 2861.2,
                "source_end_seconds": 2890.4,
                "review_required": True,
            }
        ],
    }
    (base / "tjr-youtube-qa-report.json").write_text(json.dumps(report))
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


def test_success_is_review_only_and_checks_sidecars(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = review_run(tmp_path, expected_channels=1, probe=_probe)
    assert result["status"] == "TECHNICAL_QA_PASSED__HUMAN_REVIEW_REQUIRED"
    assert result["technically_verified_mp4_count"] == 1
    assert result["automatic_publication_allowed"] is False
    assert result["channels"][0]["minimum_ssim"] == 0.997


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


def test_failed_egress_is_distinct_from_missing_channel(tmp_path: Path) -> None:
    _fixture(tmp_path)
    first = review_run(tmp_path, expected_channels=2, probe=_probe)
    assert "MISSING_CHANNEL_ARTIFACT" in first["issues"]
    failed = tmp_path / "tjr-real-original-youtube-hd-blocked"
    failed.mkdir()
    (failed / "verified-original-egress.json").write_text(
        json.dumps({"status": "YOUTUBE_EGRESS_BOT_CHALLENGE"})
    )
    result = review_run(tmp_path, expected_channels=2, probe=_probe)
    assert "MISSING_CHANNEL_ARTIFACT" not in result["issues"]
    assert "YOUTUBE_EGRESS_BLOCKED" in result["issues"]
    assert result["technically_verified_mp4_count"] == 1


def test_workflow_stores_independent_two_channel_evidence() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "feedback_audit:" in workflow
    assert "needs: [tests, youtube_modal_egress]" in workflow
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
    assert "Replay existing production artifacts at no Modal cost" in workflow
    assert "source_run_id = os.getenv('TJR_FEEDBACK_SOURCE_RUN_ID', '')" in workflow


def test_replay_channel_count_is_a_deterministic_shell_decision() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tjr-weekly-hd.yml"
    ).read_text(encoding="utf-8")
    assert "EXPECTED_CHANNELS=2" in workflow
    assert '"$TJR_SOURCE_MODE" != "feedback_replay"' in workflow
    assert "EXPECTED_CHANNELS=1" in workflow
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

