"""P3/P6 deterministic regression tests: source audio and cached video only."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import book_ai_director as director  # noqa: E402
import book_audio_timeline as timeline  # noqa: E402
import book_episode_selection as selection  # noqa: E402
import book_timeline_assemble as assembly  # noqa: E402


def _wav(path: Path, duration: float = 37.27) -> None:
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(timeline.RATE)
        recording.writeframes(b"\0\0" * round(duration * timeline.RATE))


def _source(tmp_path: Path) -> tuple[dict, dict, Path, dict, dict]:
    catalog = json.loads((ROOT / "niches/book-editorial-catalog.json").read_text())
    choice = selection.choose_lesson(
        catalog, book_id="atomic-habits", lesson_id="make-it-easy"
    )
    episode = {
        "title": "Choose the small step",
        "scenes": [
            {
                "headline": "Hook",
                "narration": "You do not need a complicated new habit to begin making progress today "
                "because the first step can be simple and clear",
            },
            {
                "headline": "Example",
                "narration": "James Clear describes making a habit easy. Put your notebook on the desk "
                "before breakfast and move your phone away from the work surface",
            },
            {
                "headline": "Payoff",
                "narration": "The next time you want to start something useful choose an action that "
                "requires almost no setup and finish it before changing your plan",
            },
        ],
    }
    request = director.build_request(choice, episode)
    words = [
        word
        for section in request["narrative_sections"]
        for word in section["narration"].split()
    ]
    shots = []
    for index, section in enumerate(request["narrative_sections"]):
        start, end = section["start_word"], section["end_word"]
        shots.append(
            {
                "id": f"director-{index}",
                "role": ("hook", "demonstration", "payoff")[index],
                "scene_index": index,
                "start_word": start,
                "end_word": end,
                "narration_text": " ".join(words[start:end]),
                "renderer": "s2v" if index != 1 else "i2v",
                "character_action": "Host handles a notebook",
                "camera": "Steady medium shot",
                "setting": "Warm study",
                "emotion": "Engaged",
                "required_objects": ["notebook"],
                "source_evidence": [request["approved_claims"][0]] if index == 1 else [],
                "qa_criteria": ["No duplicate hands", "Original identity"],
            }
        )
    plan = director.validate_plan(
        request,
        {
            "schema": director.PLAN_SCHEMA,
            "request_fingerprint": request["fingerprint"],
            "shots": shots,
        },
    )
    audio = tmp_path / "narration.wav"
    _wav(audio)
    voice = {
        "model": timeline.VOICE_MODEL,
        "revision": timeline.VOICE_REVISION,
        "voice_id": timeline.VOICE_ID,
        "style": "original_warm_low_cinematic_baritone",
        "speed": 0.98,
        "text_sha256": timeline.digest(" ".join(words)),
        "wav_sha256": timeline.file_sha(audio),
    }
    count = len(words)
    sample_frames = round(37.27 * timeline.RATE)
    alignment = {
        "audio_sha256": timeline.file_sha(audio),
        "script_sha256": request["script_sha256"],
        "aligner_model": "fixture-forced-aligner",
        "aligner_revision": "v1-test-only",
        "words": [
            {
                "text": word,
                "start_sample": i * sample_frames // count + 10,
                "end_sample": (i + 1) * sample_frames // count - 10,
            }
            for i, word in enumerate(words)
        ],
    }
    return request, plan, audio, voice, alignment


def _assets(edl: dict, tmp_path: Path) -> dict:
    sources = []
    for shot in edl["shots"]:
        index = len(sources)
        video = tmp_path / f"scene-{index}.mp4"
        video.write_bytes(b"cached-real-hash-only-clip-" + bytes([index]) * 300)
        record = tmp_path / f"scene-{index}.json"
        record.write_text(
            json.dumps(
                {
                    "video_sha256": timeline.file_sha(video),
                    "model_id": "Wan-AI/Wan2.2-S2V-14B",
                    "model_revision": "pinned-fixture",
                    "publish_eligible": False,
                }
            ),
            encoding="utf-8",
        )
        sources.append(
            {
                "shot_id": shot["id"],
                "path": str(video),
                "sha256": timeline.file_sha(video),
                "source_start_frame": 0,
                "evidence_path": str(record),
                "evidence_sha256": timeline.file_sha(record),
            }
        )
    return {
        "schema": assembly.ASSETS_SCHEMA,
        "edl_fingerprint": edl["fingerprint"],
        "assets": sources,
    }


def _lock(tmp_path: Path) -> tuple[dict, Path, dict]:
    request, plan, audio, voice, alignment = _source(tmp_path)
    edl = timeline.build_timeline(request, plan, audio, voice, alignment)
    return edl, audio, _assets(edl, tmp_path)


def test_audio_lock_uses_real_wav_samples_and_exact_word_boundaries(tmp_path: Path) -> None:
    request, plan, audio, voice, alignment = _source(tmp_path)
    locked = timeline.build_timeline(request, plan, audio, voice, alignment)
    assert locked["audio"]["sha256"] == timeline.file_sha(audio)
    assert locked["voice_cache_key"] == timeline.build_timeline(
        request, plan, audio, voice, alignment
    )["voice_cache_key"]
    assert locked["output"]["frames"] == 895
    assert locked["shots"][0]["start_frame"] == 0
    assert locked["shots"][-1]["end_frame"] == 895
    assert all(
        left["end_frame"] == right["start_frame"]
        for left, right in zip(locked["shots"], locked["shots"][1:], strict=False)
    )
    assert locked["alignment_review"] == "PENDING_ACOUSTIC_VERIFICATION"
    assert locked["publish_eligible"] is False


@pytest.mark.parametrize("defect", [
    "audio", "voice", "word", "reorder", "alignment_hash", "model", "audio_format"
])
def test_audio_lock_rejects_stale_or_false_inputs(tmp_path: Path, defect: str) -> None:
    request, plan, audio, voice, alignment = _source(tmp_path)
    if defect == "audio":
        _wav(audio, 38.0)
    elif defect == "voice":
        voice["voice_id"] = "af_heart"
    elif defect == "word":
        alignment["words"][0]["text"] = "invented"
    elif defect == "reorder":
        alignment["words"][1]["start_sample"] = 0
    elif defect == "alignment_hash":
        alignment["audio_sha256"] = "0" * 64
    elif defect == "model":
        voice["revision"] = "changed"
    else:
        with wave.open(str(audio), "wb") as reader:
            reader.setnchannels(2)
            reader.setsampwidth(2)
            reader.setframerate(24000)
            reader.writeframes(b"\0\0\0\0" * 100_000)
    with pytest.raises(timeline.AudioTimelineError):
        timeline.build_timeline(request, plan, audio, voice, alignment)


def test_clean_assembler_maps_only_locked_audio_and_no_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edl, audio, assets = _lock(tmp_path)
    used: list[list[str]] = []

    def command(argv: list[str]) -> str:
        used.append(argv)
        if argv[0] == "ffprobe" and ".clean-master-incomplete" in argv[-1]:
            return json.dumps(
                {
                    "format": {"duration": "37.27"},
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
            )
        if argv[0] == "ffprobe":
            return json.dumps({
                "format": {"duration": "37.30"},
                "streams": [{"codec_type": "video"}],
            })
        if argv[0] == "ffmpeg" and "-filter_complex" in argv:
            Path(argv[-1]).write_bytes(b"technically-valid-fixture-output")
        return ""

    monkeypatch.setattr(assembly, "run", command)
    result = assembly.assemble(edl, audio, assets, tmp_path / "final")
    cmd = next(argv for argv in used if "-filter_complex" in argv)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=3:v=1:a=0" in graph
    assert "subtitles" not in graph and "drawtext" not in graph
    assert "-shortest" not in cmd
    assert cmd[cmd.index("-map") + 1] == "[vout]"
    assert cmd[cmd.index("-map", cmd.index("-map") + 1) + 1] == "3:a:0"
    assert result["single_authoritative_narration"] is True
    assert result["overlay_mode"] == "NONE"
    assert (tmp_path / "final/clean-master.mp4").is_file()


@pytest.mark.parametrize("defect", [
    "video_hash", "evidence_hash", "missing", "short", "gap", "wrong_audio", "asset_order"
])
def test_assembler_rejects_invalid_cache_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    edl, audio, assets = _lock(tmp_path)
    encoded = False

    def command(argv: list[str]) -> str:
        nonlocal encoded
        if argv[0] == "ffprobe":
            return json.dumps({
                "format": {"duration": "0.1" if defect == "short" else "38.0"},
                "streams": [{"codec_type": "video"}],
            })
        encoded = True
        return ""

    monkeypatch.setattr(assembly, "run", command)
    if defect == "video_hash":
        Path(assets["assets"][0]["path"]).write_bytes(b"stale")
    elif defect == "evidence_hash":
        Path(assets["assets"][0]["evidence_path"]).write_bytes(b"stale")
    elif defect == "missing":
        Path(assets["assets"][0]["path"]).unlink()
    elif defect == "gap":
        old = copy.deepcopy(edl)
        edl["shots"][1]["start_frame"] += 1
        edl["fingerprint"] = timeline.digest({
            k: v for k, v in edl.items() if k != "fingerprint"
        })
        assets["edl_fingerprint"] = edl["fingerprint"]
        assert old["fingerprint"] != edl["fingerprint"]
    elif defect == "wrong_audio":
        _wav(audio, 38)
    elif defect == "asset_order":
        assets["assets"].reverse()
    with pytest.raises((assembly.AssemblyError, timeline.AudioTimelineError)):
        assembly.assemble(edl, audio, assets, tmp_path / "out")
    assert not encoded


def test_legacy_adapter_refuses_fabricated_source_plan(tmp_path: Path) -> None:
    audio = tmp_path / "full-kokoro.wav"
    _wav(audio)
    with pytest.raises(assembly.AssemblyError, match="historical full-book"):
        assembly.make_legacy_replay({}, audio, tmp_path, [])


def test_alignment_cannot_be_replaced_with_proportional_word_timings(
    tmp_path: Path,
) -> None:
    request, plan, audio, voice, alignment = _source(tmp_path)
    alignment["words"] = []
    with pytest.raises(timeline.AudioTimelineError, match="every spoken word"):
        timeline.build_timeline(request, plan, audio, voice, alignment)
