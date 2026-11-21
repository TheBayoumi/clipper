"""Synthetic assembly trust-boundary regressions; never run real ffprobe/ffmpeg."""

from __future__ import annotations

import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import book_audio_timeline as timeline  # noqa: E402
import book_shot_review as creative  # noqa: E402
import book_timeline_assemble as assembly  # noqa: E402
import wan_full_book_episode as episode  # noqa: E402
from wan_shared_shot_cache import WAN_MODEL_ID, WAN_REVISION  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seal_legacy(plan: dict[str, Any]) -> None:
    """Bind a synthetic historical plan with the actual legacy hash algorithm."""
    source = {key: value for key, value in plan.items() if key != "fingerprint"}
    plan["fingerprint"] = _sha(json.dumps(source, sort_keys=True).encode())


def _source(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, list[Path]]:
    """Build a fully hash-consistent four-shot historical source, using synthetic bytes."""
    frames = tmp_path / "artifacts/full-book/frames"
    source = tmp_path / "artifacts/full-book/source"
    frames.mkdir(parents=True)
    source.mkdir(parents=True)
    audio = source / "full-kokoro.wav"
    with wave.open(str(audio), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(timeline.RATE)
        recording.writeframes(b"\0\0" * (36 * timeline.RATE))

    shots: list[dict[str, Any]] = []
    folders: list[Path] = []
    for index, keyframe in enumerate(("hook", "lesson_example", "book_intro", "hook")):
        frame = frames / f"{keyframe}.png"
        frame.write_bytes(f"synthetic-frame-{keyframe}".encode())
        speech = source / f"scene-{index:02d}.wav"
        speech.write_bytes(f"synthetic-locked-speech-{index}".encode())
        prompt = f"Approved synthetic book-shot prompt {index}"
        folder = tmp_path / f"rendered/scene-{index:02d}"
        folder.mkdir(parents=True)
        video = folder / "wan-s2v-candidate.mp4"
        video.write_bytes(f"synthetic-video-{index}".encode() * 22_000)
        evidence = {
            "schema": "clipper-wan-s2v-modal-v1",
            "model_id": WAN_MODEL_ID,
            "model_revision": WAN_REVISION,
            "image_sha256": timeline.file_sha(frame),
            "audio_sha256": timeline.file_sha(speech),
            "prompt_sha256": _sha(prompt.encode()),
            "seed": 735 + index,
            "status": "CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED",
            "publish_eligible": False,
            "creative_review": "PENDING",
            "video_sha256": timeline.file_sha(video),
        }
        (folder / "model-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        shot: dict[str, Any] = {
            "id": f"scene-{index:02d}",
            "keyframe_name": keyframe,
            "audio": speech.name,
            "audio_sha256": timeline.file_sha(speech),
            "prompt": prompt,
            "seed": 735 + index,
            "duration_seconds": 9.0,
            "end_seconds": 9.0 * (index + 1),
        }
        if keyframe == "book_intro":
            shot["required_on_screen_text"] = ["ATOMIC HABITS", "JAMES CLEAR"]
        shots.append(shot)
        folders.append(folder)
    plan: dict[str, Any] = {
        "schema": episode.SCHEMA,
        "seconds": 36.0,
        "full_audio_sha256": timeline.file_sha(audio),
        "voice": {"voice_id": "am_adam"},
        "shots": shots,
        "publish_eligible": False,
    }
    _seal_legacy(plan)
    return plan, audio, frames, folders


def _write_approvals(
    shots: list[dict[str, Any]], plan_fingerprint: str, folders: list[Path], frames: Path
) -> None:
    """Create synthetic, self-consistent content-bound review records, not real approvals."""
    previous: str | None = None
    for shot, folder in zip(shots, folders, strict=True):
        video = folder / "wan-s2v-candidate.mp4"
        base = creative.contract(
            shot,
            plan_fingerprint,
            timeline.file_sha(video),
            timeline.file_sha(frames / f"{shot['keyframe_name']}.png"),
            previous,
        )
        record: dict[str, Any] = {
            **base,
            "sampled_frame_sha256": ["a" * 64, "b" * 64, "c" * 64],
            "previous_sampled_frame_sha256": ["d" * 64] if previous else [],
            "provider": {
                "name": "SyntheticVision",
                "model_id": "synthetic-vlm",
                "model_revision": "synthetic-v1",
            },
            "automated_criteria": {
                name: {
                    "status": "PENDING_HUMAN" if name == "speech_synchronization" else "PASS",
                    "confidence": 0.0 if name == "speech_synchronization" else 0.99,
                    "evidence": "synthetic criterion",
                }
                for name in creative.CRITERION_VERSIONS
            },
            "status": "PENDING_HUMAN",
            "human_approval": None,
        }
        record["human_approval"] = {
            "status": "APPROVED",
            "reviewer": "synthetic-operator",
            "approved_at": "2026-11-21T00:00:00Z",
            "video_sha256": base["video_sha256"],
            "shot_fingerprint": base["shot_fingerprint"],
            "criterion_versions": base["criterion_versions"],
            "observation_sha256": creative.digest(
                {key: value for key, value in record.items() if key != "human_approval"}
            ),
            "criteria": {
                name: {"status": "PASS", "evidence": "synthetic human criterion"}
                for name in creative.CRITERION_VERSIONS
            },
        }
        (folder / "creative-review.json").write_text(json.dumps(record), encoding="utf-8")
        previous = timeline.file_sha(video)


def _generalized(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], list[Path]]:
    plan, audio, frames, folders = _source(tmp_path)
    edl, assets = assembly.make_legacy_replay(plan, audio, frames, folders)
    _write_approvals(edl["shots"], edl["fingerprint"], folders, frames)
    for asset in assets["assets"]:
        asset["review_sha256"] = timeline.file_sha(Path(asset["review_path"]))
    return edl, audio, assets, folders


def test_legacy_intro_preserves_identity_and_declared_cover_text(tmp_path: Path) -> None:
    plan, audio, frames, folders = _source(tmp_path)
    edl, _assets = assembly.make_legacy_replay(plan, audio, frames, folders)
    intro = edl["shots"][2]
    assert intro["id"] == "scene-02"
    assert intro["keyframe_name"] == "book_intro"
    assert intro["required_on_screen_text"] == ["ATOMIC HABITS", "JAMES CLEAR"]
    assert creative.expected_text(intro) == ["ATOMIC HABITS", "JAMES CLEAR"]
    assert intro["director_role"] == "LEGACY_APPROVED_SHOT"
    assert (intro["start_frame"], intro["end_frame"]) == (432, 648)


def test_legacy_intro_without_declared_text_fails_closed(tmp_path: Path) -> None:
    plan, audio, frames, folders = _source(tmp_path)
    plan["shots"][2].pop("required_on_screen_text")
    _seal_legacy(plan)
    with pytest.raises(assembly.AssemblyError, match=r"intro.*required_on_screen_text"):
        assembly.make_legacy_replay(plan, audio, frames, folders)


def test_legacy_intro_explicit_empty_text_survives_replay(tmp_path: Path) -> None:
    plan, audio, frames, folders = _source(tmp_path)
    plan["shots"][2]["required_on_screen_text"] = []
    _seal_legacy(plan)
    edl, _assets = assembly.make_legacy_replay(plan, audio, frames, folders)
    intro = edl["shots"][2]
    assert "required_on_screen_text" in intro
    assert intro["required_on_screen_text"] == []
    assert creative.expected_text(intro) == []


def test_legacy_non_intro_absence_stays_absent_and_hash_binding_remains(tmp_path: Path) -> None:
    plan, audio, frames, folders = _source(tmp_path)
    edl, _assets = assembly.make_legacy_replay(plan, audio, frames, folders)
    non_intro = edl["shots"][0]
    assert non_intro["keyframe_name"] == "hook"
    assert "required_on_screen_text" not in non_intro
    assert creative.expected_text(non_intro) == []
    assembly._fingerprinted(edl)
    edl["shots"][2]["required_on_screen_text"] = ["WRONG TITLE"]
    with pytest.raises(assembly.AssemblyError, match="fingerprint mismatch"):
        assembly._fingerprinted(edl)


def test_generalized_last_invalid_review_blocks_every_media_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edl, audio, assets, folders = _generalized(tmp_path)
    (folders[-1] / "creative-review.json").unlink()
    calls: list[list[str]] = []

    def fake_probe(argv: list[str]) -> str:
        calls.append(argv)
        return json.dumps({"format": {"duration": "9.0"}, "streams": [{"codec_type": "video"}]})

    monkeypatch.setattr(assembly, "run", fake_probe)
    with pytest.raises(assembly.AssemblyError, match="creative review rejected"):
        assembly.assemble(edl, audio, assets, tmp_path / "rejected")
    assert calls == [], "no earlier shot may reach ffprobe before all approvals pass"


def test_generalized_valid_reviews_preserve_probes_and_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edl, audio, assets, _folders = _generalized(tmp_path)
    calls: list[list[str]] = []

    def fake_media(argv: list[str]) -> str:
        calls.append(argv)
        if argv[0] == "ffprobe" and ".clean-master-incomplete" in argv[-1]:
            data = {
                "format": {"duration": "36.0"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1080,
                        "height": 1920,
                        "r_frame_rate": "24/1",
                        "nb_frames": str(edl["output"]["frames"]),
                    },
                    {"codec_type": "audio"},
                ],
            }
            return json.dumps(data)
        if argv[0] == "ffprobe":
            return json.dumps(
                {"format": {"duration": "9.25"}, "streams": [{"codec_type": "video"}]}
            )
        if "-filter_complex" in argv:
            Path(argv[-1]).write_bytes(b"synthetic-assembled-video")
        return ""

    monkeypatch.setattr(assembly, "run", fake_media)
    result = assembly.assemble(edl, audio, assets, tmp_path / "accepted")
    assert result["shot_count"] == 4
    assert sum(argv[0] == "ffprobe" for argv in calls) == 5
    assert any(argv[0] == "ffmpeg" and "-filter_complex" in argv for argv in calls)


def test_episode_last_invalid_review_blocks_every_media_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _audio, frames, folders = _source(tmp_path)
    _write_approvals(plan["shots"], plan["fingerprint"], folders, frames)
    (folders[-1] / "creative-review.json").unlink()
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def fake_probe(argv: list[str]) -> str:
        calls.append(argv)
        return json.dumps({"format": {"duration": "9.0"}})

    monkeypatch.setattr(episode, "command_run", fake_probe)
    with pytest.raises(episode.EpisodeError, match="creative review rejected"):
        episode.prepare_concat(plan, folders, tmp_path / "rejected", captions=None, clean=True)
    assert calls == [], "no earlier shot may reach ffprobe before all approvals pass"


def test_episode_valid_reviews_preserve_probes_and_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _audio, frames, folders = _source(tmp_path)
    _write_approvals(plan["shots"], plan["fingerprint"], folders, frames)
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def fake_media(argv: list[str]) -> str:
        calls.append(argv)
        if argv[0] == "ffprobe" and str(argv[-1]).endswith("master.mp4"):
            data = {
                "format": {"duration": "36.0"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1080,
                        "height": 1920,
                        "r_frame_rate": "24/1",
                    },
                    {"codec_type": "audio"},
                ],
            }
            return json.dumps(data)
        if argv[0] == "ffprobe":
            return json.dumps({"format": {"duration": "9.0"}})
        if "-filter_complex" in argv:
            Path(argv[-1]).write_bytes(b"synthetic-full-book-output")
        return ""

    monkeypatch.setattr(episode, "command_run", fake_media)
    episode.prepare_concat(plan, folders, tmp_path / "accepted", captions=None, clean=True)
    assert sum(argv[0] == "ffprobe" for argv in calls) == 5
    assert any(argv[0] == "ffmpeg" and "-filter_complex" in argv for argv in calls)
    assert (tmp_path / "accepted/atomic-habits-clean-master.mp4").is_file()
