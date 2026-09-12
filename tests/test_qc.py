import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from clipper.qc import (
    _caption_margin,
    _float,
    _fps,
    _render_evidence,
    _tracking_evidence,
    _transition_issues,
    run_technical_qc,
    tracking_plan_issues,
)


def completed(returncode: int = 0, *, stdout: str = "", stderr: str = "") -> CompletedProcess[str]:
    return CompletedProcess(["tool"], returncode, stdout, stderr)


def fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    ass = tmp_path / "clip.ass"
    ass.write_text(
        "; TimingMode: word_exact\n"
        "Style: Default,DejaVu Sans,58,&H00FFFFFF,&HFFFFFFFF,&H00000000,&H80000000,"
        "1,0,0,0,100,100,0,0,1,4,0,2,80,80,461,1\n",
        encoding="utf-8",
    )
    audit = tmp_path / "clip.caption-audit.json"
    audit.write_text(
        json.dumps(
            {
                "first_audio_word": "hello",
                "first_audio_word_time": 0.0,
                "first_audio_words": "hello world",
                "first_caption_text": "hello world",
                "first_caption_time": 0.0,
                "first_caption_timing_delta_seconds": 0.0,
                "alignment": "PASS",
                "partial_words_dropped": 0,
                "simultaneous_narrative_layers_max": 1,
            }
        ),
        encoding="utf-8",
    )
    tracking = tmp_path / "clip.tracking.json"
    tracking.write_text(
        json.dumps(
            {
                "framing_mode": "speaker_locked_portrait",
                "background_fill": "none",
                "crop_width": 1214,
                "crop_height": 2158,
                "speaker_tracks": 2,
                "speaker_switches": 1,
                "reframe_events": 2,
                "zoom_factor": 1.0,
                "source_width": 3840,
                "source_height": 2160,
                "transitions": [],
                "source_cuts": [],
                "image_quality": {
                    "source_width": 3840,
                    "source_height": 2160,
                    "crop_width": 1214,
                    "crop_height": 2158,
                    "max_portrait_crop_width": 1214,
                    "max_portrait_crop_height": 2158,
                    "effective_upscale_factor": 0.89,
                    "digital_zoom_used": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return video, ass, tracking


def probe_payload(*, audio: bool = True, width: int = 1080, height: int = 1920) -> str:
    streams = [
        {
            "index": 0,
            "codec_name": "h264",
            "codec_type": "video",
            "width": width,
            "height": height,
            "r_frame_rate": "30/1",
            "duration": "20.0",
            "bit_rate": "8000000",
        }
    ]
    if audio:
        streams.append(
            {
                "index": 1,
                "codec_name": "aac",
                "codec_type": "audio",
                "r_frame_rate": "0/0",
                "duration": "20.0",
            }
        )
    return json.dumps(
        {"streams": streams, "format": {"duration": "20.0", "size": "123", "bit_rate": "8200000"}}
    )


def test_qc_passes_complete_vertical_render(tmp_path: Path) -> None:
    video, ass, tracking = fixtures(tmp_path)
    loud = '{"input_i":"-14.4","input_tp":"-1.2","input_lra":"2.3"}'
    video.with_suffix(".render.json").write_text(
        json.dumps(
            {
                "profile": "production",
                "preset": "slow",
                "crf": 17,
                "resampling_stages": 1,
                "digital_zoom_used": False,
                "post_upscale_punch_in": False,
            }
        )
    )
    calls = iter(
        [completed(stdout=probe_payload()), completed(), completed(stderr=loud), completed()]
    )
    with (
        patch("clipper.qc.shutil.which", return_value="/usr/bin/tool"),
        patch("clipper.qc._run", side_effect=lambda *_a, **_k: next(calls)),
    ):
        report = run_technical_qc(
            video,
            expected_duration=20,
            caption_path=ass,
            tracking_path=tracking,
            caption_platform="tiktok",
            watermark_required=True,
            watermark_present=True,
        )
    assert report["status"] == "PASS" and report["issues"] == []
    assert report["video"]["decode_pass"] is True
    assert report["audio"]["integrated_lufs"] == -14.4
    assert report["captions"]["bottom_margin_px"] == 461
    assert report["captions"]["timing_mode"] == "word_exact"
    assert report["captions"]["word_exact"] is True
    assert report["framing"]["no_filler_pass"] is True


def test_qc_rejects_missing_media_and_missing_tools(tmp_path: Path) -> None:
    missing = run_technical_qc(
        tmp_path / "none.mp4",
        expected_duration=20,
        caption_path=tmp_path / "none.ass",
        tracking_path=tmp_path / "none.json",
    )
    assert missing["status"] == "FAIL"
    video, ass, tracking = fixtures(tmp_path)
    with patch("clipper.qc.shutil.which", return_value=None):
        report = run_technical_qc(
            video, expected_duration=20, caption_path=ass, tracking_path=tracking
        )
    assert "ffprobe/ffmpeg" in report["issues"][0]


def test_qc_rejects_unreadable_probe(tmp_path: Path) -> None:
    video, ass, tracking = fixtures(tmp_path)
    with (
        patch("clipper.qc.shutil.which", return_value="tool"),
        patch("clipper.qc._run", return_value=completed(1, stderr="bad")),
    ):
        report = run_technical_qc(
            video, expected_duration=20, caption_path=ass, tracking_path=tracking
        )
    assert report == {"status": "FAIL", "issues": ["ffprobe could not read rendered video"]}


def test_qc_reports_geometry_audio_caption_tracking_and_watermark_failures(tmp_path: Path) -> None:
    video, ass, tracking = fixtures(tmp_path)
    ass.write_text("Style: Default,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,80,100,1\n")
    tracking.write_text(
        json.dumps({"background_fill": "blur", "crop_width": 100, "crop_height": 100})
    )
    bad_probe = json.dumps(
        {
            "streams": [
                {
                    "codec_name": "vp9",
                    "codec_type": "video",
                    "width": 720,
                    "height": 1280,
                    "r_frame_rate": "24/1",
                },
                {"codec_name": "mp3", "codec_type": "audio"},
            ],
            "format": {"duration": "17.5", "size": "1"},
        }
    )
    loud = '{"input_i":"-25.0","input_tp":"0.1","input_lra":"9.0"}'
    calls = iter(
        [
            completed(stdout=bad_probe),
            completed(1, stderr="decode error"),
            completed(stderr=loud),
            completed(stderr="silence_duration: 2.50\nsilence_duration: 1.20"),
        ]
    )
    with (
        patch("clipper.qc.shutil.which", return_value="tool"),
        patch("clipper.qc._run", side_effect=lambda *_a, **_k: next(calls)),
    ):
        report = run_technical_qc(
            video,
            expected_duration=20,
            caption_path=ass,
            tracking_path=tracking,
            watermark_required=True,
            watermark_present=False,
        )
    issues = " | ".join(report["issues"])
    assert report["status"] == "FAIL"
    for expected in (
        "width",
        "height",
        "fps",
        "codec",
        "duration",
        "decode",
        "loudness",
        "true peak",
        "silence",
        "caption",
        "no-filler",
        "portrait crop",
        "watermark",
    ):
        assert expected in issues


def test_qc_handles_no_audio_and_failed_loudness_parse(tmp_path: Path) -> None:
    video, ass, tracking = fixtures(tmp_path)
    no_audio_calls = iter([completed(stdout=probe_payload(audio=False)), completed()])
    with (
        patch("clipper.qc.shutil.which", return_value="tool"),
        patch("clipper.qc._run", side_effect=lambda *_a, **_k: next(no_audio_calls)),
    ):
        no_audio = run_technical_qc(
            video, expected_duration=20, caption_path=ass, tracking_path=tracking
        )
    assert "audio stream is missing" in no_audio["issues"]
    bad_loud_calls = iter(
        [completed(stdout=probe_payload()), completed(), completed(stderr="not json"), completed()]
    )
    with (
        patch("clipper.qc.shutil.which", return_value="tool"),
        patch("clipper.qc._run", side_effect=lambda *_a, **_k: next(bad_loud_calls)),
    ):
        bad_loud = run_technical_qc(
            video, expected_duration=20, caption_path=ass, tracking_path=tracking
        )
    assert "objective loudness analysis failed" in bad_loud["issues"]


def test_qc_helpers_handle_invalid_values_and_missing_evidence(tmp_path: Path) -> None:
    assert _float("2.5") == 2.5 and _float(None, 3.0) == 3.0
    assert _fps(None) == 0 and _fps("0/0") == 0 and _fps("25") == 25 and _fps("30000/1001") > 29
    assert _caption_margin(tmp_path / "missing.ass") is None
    malformed = tmp_path / "bad.ass"
    malformed.write_text("Style: Default,short\n")
    assert _caption_margin(malformed) is None
    bad_number = tmp_path / "bad-number.ass"
    bad_number.write_text("Style: Default," + ",".join(["x"] * 21) + ",oops,1\n")
    assert _caption_margin(bad_number) is None
    assert _tracking_evidence(tmp_path / "missing.json") == {}


def test_transition_and_render_quality_gates_reject_v8_failure_modes(tmp_path: Path) -> None:
    bad = [
        {
            "reason": "source_cut",
            "mode": "eased_reframe",
            "start": 2.0,
            "end": 2.2,
            "normalized_distance": 0.8,
            "target_visible_at": 2.0,
            "from_x": 0.0,
            "to_x": 300.0,
        },
        {
            "reason": "speaker_change",
            "mode": "eased_reframe",
            "start": 2.4,
            "end": 2.6,
            "normalized_distance": 0.8,
            "target_visible_at": 2.5,
            "from_x": 300.0,
            "to_x": 0.0,
        },
    ]
    issues = " | ".join(_transition_issues(bad))
    assert "source camera cut" in issues
    assert "velocity" in issues
    assert "before the target face" in issues
    assert "back-and-forth" not in issues
    render = tmp_path / "x.render.json"
    render.write_text(json.dumps({"resampling_stages": 2, "digital_zoom_used": True}))
    assert _render_evidence(render)["digital_zoom_used"] is True


def test_transition_gate_rejects_same_speaker_camera_motion() -> None:
    issues = _transition_issues(
        [
            {
                "reason": "subject_motion",
                "mode": "eased_reframe",
                "start": 4.0,
                "end": 4.5,
                "normalized_distance": 0.2,
                "target_visible_at": 4.0,
                "from_x": 100.0,
                "to_x": 300.0,
            }
        ]
    )
    assert "same-speaker crop motion is not allowed inside a source shot" in issues


def test_transition_gate_rejects_one_and_a_half_second_camera_reversal() -> None:
    issues = _transition_issues(
        [
            {
                "reason": "speaker_change",
                "mode": "hard_cut",
                "start": 41.79175,
                "end": 41.79175,
                "normalized_distance": 0.6,
                "target_visible_at": 41.79175,
                "from_x": 1129.0,
                "to_x": 1856.0,
            },
            {
                "reason": "speaker_change",
                "mode": "hard_cut",
                "start": 43.29325,
                "end": 43.29325,
                "normalized_distance": 0.6,
                "target_visible_at": 43.29325,
                "from_x": 1856.0,
                "to_x": 1129.0,
            },
        ]
    )
    assert "back-and-forth crop oscillation detected" in issues


def test_qc_rejects_first_caption_misalignment(tmp_path: Path) -> None:
    video, ass, tracking = fixtures(tmp_path)
    video.with_suffix(".caption-audit.json").write_text(
        json.dumps(
            {
                "first_audio_word": "what's",
                "first_audio_word_time": 0.0,
                "first_audio_words": "what's one message",
                "first_caption_text": "before we head",
                "first_caption_time": 0.21,
                "first_caption_timing_delta_seconds": 0.21,
                "alignment": "FAIL",
            }
        )
    )
    calls = iter(
        [
            completed(stdout=probe_payload()),
            completed(),
            completed(stderr='{"input_i":"-14","input_tp":"-1.5","input_lra":"2"}'),
            completed(),
        ]
    )
    with (
        patch("clipper.qc.shutil.which", return_value="tool"),
        patch("clipper.qc._run", side_effect=lambda *_a, **_k: next(calls)),
    ):
        report = run_technical_qc(
            video, expected_duration=20, caption_path=ass, tracking_path=tracking
        )
    assert report["status"] == "FAIL"
    assert "first caption" in " | ".join(report["issues"])


def test_qc_rejects_overlapping_narrative_text_layers(tmp_path: Path) -> None:
    video, ass, tracking = fixtures(tmp_path)
    audit_path = video.with_suffix(".caption-audit.json")
    audit = json.loads(audit_path.read_text())
    audit["simultaneous_narrative_layers_max"] = 2
    audit_path.write_text(json.dumps(audit))
    calls = iter(
        [
            completed(stdout=probe_payload()),
            completed(),
            completed(stderr='{"input_i":"-14","input_tp":"-1.5","input_lra":"2"}'),
            completed(),
        ]
    )
    with (
        patch("clipper.qc.shutil.which", return_value="tool"),
        patch("clipper.qc._run", side_effect=lambda *_a, **_k: next(calls)),
    ):
        report = run_technical_qc(
            video, expected_duration=20, caption_path=ass, tracking_path=tracking
        )
    assert "multiple narrative text layers overlap" in report["issues"]


def test_qc_rejects_missing_caption_layer_concurrency_evidence(tmp_path: Path) -> None:
    video, ass, tracking = fixtures(tmp_path)
    audit_path = video.with_suffix(".caption-audit.json")
    audit = json.loads(audit_path.read_text())
    audit.pop("simultaneous_narrative_layers_max")
    audit_path.write_text(json.dumps(audit))
    calls = iter(
        [
            completed(stdout=probe_payload()),
            completed(),
            completed(stderr='{"input_i":"-14","input_tp":"-1.5","input_lra":"2"}'),
            completed(),
        ]
    )
    with (
        patch("clipper.qc.shutil.which", return_value="tool"),
        patch("clipper.qc._run", side_effect=lambda *_a, **_k: next(calls)),
    ):
        report = run_technical_qc(
            video, expected_duration=20, caption_path=ass, tracking_path=tracking
        )
    assert "caption layer concurrency evidence is missing" in report["issues"]


def test_tracking_preflight_rejects_bad_geometry_filler_and_zoom() -> None:
    issues = tracking_plan_issues(
        {
            "background_fill": "blur",
            "crop_width": 100,
            "crop_height": 100,
            "transitions": [],
            "image_quality": {"max_portrait_crop_height": 640, "digital_zoom_used": True},
        }
    )
    assert "tracking evidence does not confirm no-filler portrait composition" in issues
    assert "tracking crop is not a valid portrait crop" in issues
    assert "crop resolution is materially below the maximum portrait crop" in issues
    assert "tracking plan uses digital zoom that discards source pixels" in issues


def test_tracking_preflight_rejects_off_center_or_cropped_selected_faces() -> None:
    issues = tracking_plan_issues(
        {
            "background_fill": "none",
            "crop_width": 1214,
            "crop_height": 2158,
            "transitions": [],
            "image_quality": {
                "max_portrait_crop_height": 2158,
                "digital_zoom_used": False,
            },
            "composition": {
                "sample_count": 20,
                "centered_sample_ratio": 0.7,
                "fully_visible_sample_ratio": 0.9,
            },
        }
    )
    assert "selected speaker is outside the horizontal framing safe region" in issues
    assert "selected speaker face is cropped in tracking evidence" in issues


def test_tracking_preflight_requires_composition_evidence_when_face_detected() -> None:
    issues = tracking_plan_issues(
        {
            "background_fill": "none",
            "crop_width": 1214,
            "crop_height": 2158,
            "face_detected": True,
            "transitions": [],
            "image_quality": {
                "max_portrait_crop_height": 2158,
                "digital_zoom_used": False,
            },
        }
    )
    assert "selected face composition evidence is missing" in issues


def test_tracking_preflight_rejects_stale_crop_in_uncovered_source_shot() -> None:
    issues = tracking_plan_issues(
        {
            "background_fill": "none",
            "crop_width": 1214,
            "crop_height": 2158,
            "transitions": [],
            "image_quality": {
                "max_portrait_crop_height": 2158,
                "digital_zoom_used": False,
            },
            "shot_coverage": [
                {
                    "start": 15.766,
                    "end": 17.267,
                    "duration": 1.501,
                    "status": "uncovered",
                    "safe_centered": False,
                }
            ],
        }
    )
    assert "source shot carries a stale crop without face coverage" in issues


def test_tracking_preflight_accepts_centered_fallback_for_unobserved_shot() -> None:
    issues = tracking_plan_issues(
        {
            "background_fill": "none",
            "crop_width": 1214,
            "crop_height": 2158,
            "transitions": [],
            "image_quality": {
                "max_portrait_crop_height": 2158,
                "digital_zoom_used": False,
            },
            "shot_coverage": [
                {
                    "start": 15.766,
                    "end": 17.267,
                    "duration": 1.501,
                    "status": "safe_center_fallback",
                    "safe_centered": True,
                }
            ],
        }
    )
    assert "source shot carries a stale crop without face coverage" not in issues
    assert "unobserved source shot is not using the safe centered crop" not in issues


def test_tracking_preflight_rejects_malformed_or_off_center_fallback_evidence() -> None:
    issues = tracking_plan_issues(
        {
            "background_fill": "none",
            "crop_width": 1214,
            "crop_height": 2158,
            "transitions": [],
            "image_quality": {
                "max_portrait_crop_height": 2158,
                "digital_zoom_used": False,
            },
            "shot_coverage": [
                "bad",
                {
                    "duration": 1.0,
                    "status": "safe_center_fallback",
                    "safe_centered": False,
                },
            ],
        }
    )
    assert "source shot coverage evidence is malformed" in issues
    assert "unobserved source shot is not using the safe centered crop" in issues
