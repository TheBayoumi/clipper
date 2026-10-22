import json
from pathlib import Path
from typing import Any

import pytest

from clipper.hilp import HilpSimulationError, simulate_hilp_cycle, validate_hilp_evidence


class FakeRenderer:
    def render(
        self,
        source_path: Path,
        output_path: Path,
        clip: Any,
        segments: Any,
        watermark_path: Path | None = None,
        edit_plan: Any = None,
    ) -> Path:
        del source_path, clip, segments, watermark_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            f"render:{edit_plan.plan_id}:{edit_plan.caption_start_source_time}".encode()
        )
        output_path.with_suffix(".ass").write_text("captions", encoding="utf-8")
        output_path.with_suffix(".tracking.json").write_text("{}", encoding="utf-8")
        output_path.with_suffix(".caption-audit.json").write_text(
            json.dumps({"alignment": "PASS"}), encoding="utf-8"
        )
        return output_path


def _fake_qc(output: Path, **kwargs: object) -> dict[str, Any]:
    del output, kwargs
    return {"status": "PASS"}


def _plan(index: int) -> dict[str, Any]:
    return {
        "plan_id": f"plan-{index}",
        "video_id": "video-1",
        "concept_id": f"concept-{index}",
        "variant_id": f"variant-{index}",
        "hook_mode": "direct",
        "source_spans": [{"start": 0.0, "end": 10.0}],
        "hook_text": None,
        "beats": [],
        "caption_platform": "tiktok",
        "score": 0.9 - index * 0.01,
        "transcript_fingerprint": "fingerprint",
        "caption_start_source_time": 0.0,
        "caption_start_word": "one",
    }


def _run(root: Path, *, finalists: int = 6) -> None:
    rendered = []
    for index in range(finalists):
        path = root / "clips" / f"plan-{index}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"original-{index}".encode())
        rendered.append(
            {
                "video_id": "video-1",
                "output_path": str(path),
                "start": 0.0,
                "end": 10.0,
                "score": 0.9 - index * 0.01,
                "source_url": "https://www.youtube.com/watch?v=video-1",
                "concept_id": f"concept-{index}",
                "plan_id": f"plan-{index}",
                "hook_mode": "direct",
                "render_sha256": f"sha-{index}",
            }
        )
    manifest = {
        "status": "SUCCESS",
        "rendered_clips": rendered,
        "submission_shortlist": rendered[:3],
        "edit_plans": [_plan(index) for index in range(finalists)],
    }
    transcript = {
        "video-1": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "one two three four five",
                "words": [
                    {"start": 0.0, "end": 0.3, "text": "one"},
                    {"start": 0.8, "end": 1.1, "text": "two"},
                    {"start": 1.5, "end": 1.8, "text": "three"},
                    {"start": 2.2, "end": 2.5, "text": "four"},
                    {"start": 3.0, "end": 3.3, "text": "five"},
                ],
            }
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")


def test_simulation_exercises_approve_reject_and_real_revision(tmp_path: Path) -> None:
    _run(tmp_path)
    source = tmp_path / "source.mkv"
    source.write_bytes(b"master")
    watermark = tmp_path / "watermark.png"
    watermark.write_bytes(b"watermark")

    result = simulate_hilp_cycle(
        tmp_path,
        source_path=source,
        renderer=FakeRenderer(),
        watermark_path=watermark,
        qc_runner=_fake_qc,
    )

    assert result["status"] == "PASS"
    assert set(result["branches_exercised"]) == {"APPROVE", "REJECT", "REVISE"}
    assert [item["concept_id"] for item in result["final_shortlist"]] == [
        "concept-0",
        "concept-3",
        "concept-2",
    ]
    proof = result["revision_proof"]
    assert proof["before_sha256"] != proof["after_sha256"]
    assert proof["before_review"]["decision"] == "REVISE"
    assert proof["after_review"]["decision"] == "APPROVE"
    assert proof["after_qc"] == "PASS"
    review = json.loads((tmp_path / "editorial-review.json").read_text())
    assert review["status"] == "SIMULATED_HILP_COMPLETE"
    validate_hilp_evidence(json.loads((tmp_path / "hilp-simulation.json").read_text()))


def test_simulation_refuses_to_fake_reject_replacement_without_reserve(tmp_path: Path) -> None:
    _run(tmp_path, finalists=3)
    source = tmp_path / "source.mkv"
    source.write_bytes(b"master")
    with pytest.raises(HilpSimulationError, match="at least four"):
        simulate_hilp_cycle(
            tmp_path,
            source_path=source,
            renderer=FakeRenderer(),
            qc_runner=_fake_qc,
        )


def test_validator_rejects_missing_branch_and_no_changed_render() -> None:
    payload = {
        "status": "PASS",
        "events": [{"decision": "APPROVE"}],
        "final_shortlist": [
            {"concept_id": "a"},
            {"concept_id": "b"},
            {"concept_id": "c"},
        ],
        "revision_proof": {
            "before_sha256": "same",
            "after_sha256": "same",
            "after_qc": "PASS",
        },
    }
    with pytest.raises(HilpSimulationError, match="decision branches"):
        validate_hilp_evidence(payload)

    payload["events"] = [
        {"decision": "APPROVE"},
        {"decision": "REJECT"},
        {"decision": "REVISE"},
    ]
    with pytest.raises(HilpSimulationError, match="changed render"):
        validate_hilp_evidence(payload)
