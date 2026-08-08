import json
from pathlib import Path

import pytest

from clipper.acceptance import validate_live_run


def _write_live_run(root: Path, *, finalists: int = 2, shortlist: int = 1) -> None:
    for directory in ("clips", "captions", "tracking"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    rendered = []
    qc = []
    for index in range(finalists):
        plan_id = f"plan-{index}"
        clip = root / "clips" / f"{index:02d}.mp4"
        clip.write_bytes(b"mp4")
        (root / "captions" / f"{index:02d}.ass").write_text("captions")
        (root / "captions" / f"{index:02d}.caption-audit.json").write_text(
            json.dumps({"alignment": "PASS"})
        )
        (root / "tracking" / f"{index:02d}.tracking.json").write_text("{}")
        rendered.append(
            {"plan_id": plan_id, "concept_id": f"concept-{index}", "output_path": str(clip)}
        )
        qc.append(
            {
                "plan_id": plan_id,
                "status": "PASS",
                "captions": {"alignment": "PASS"},
                "framing": {"transition_qc_pass": True},
                "watermark": {"required": True, "renderer_asset_present": True},
            }
        )
    manifest = {
        "status": "SUCCESS",
        "status_reason": None,
        "errors": [],
        "targets": {"rendered_finalists": finalists, "submission_shortlist": shortlist},
        "actual": {"rendered_finalists": finalists, "submission_shortlist": shortlist},
        "rendered_clips": rendered,
        "submission_shortlist": rendered[:shortlist],
        "technical_qc": qc,
        "funnel": {
            "transcript_segments": 100,
            "story_moments": 20,
            "raw_concepts": 10,
            "selected_concepts": finalists,
            "hook_variants": finalists * 2,
            "edit_plans": finalists * 2,
            "render_plans": finalists,
            "render_attempts": finalists,
            "technical_qc_pass": finalists,
            "render_success": finalists,
            "submission_shortlist": shortlist,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for required in (
        "funnel.json",
        "rejections.json",
        "coverage.json",
        "transcript.json",
        "editorial-review.json",
    ):
        (root / required).write_text("{}")


def test_live_validator_accepts_complete_qc_passed_batch(tmp_path: Path) -> None:
    _write_live_run(tmp_path, finalists=3, shortlist=1)
    manifest = validate_live_run(tmp_path, expected_finalists=3, expected_shortlist=1)
    assert manifest["status"] == "SUCCESS"


def test_live_validator_rejects_underproduction_and_caption_mismatch(tmp_path: Path) -> None:
    _write_live_run(tmp_path, finalists=3, shortlist=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["actual"]["rendered_finalists"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="yield"):
        validate_live_run(tmp_path, expected_finalists=3, expected_shortlist=1)

    manifest["actual"]["rendered_finalists"] = 3
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "captions" / "00.caption-audit.json").write_text('{"alignment":"FAIL"}')
    with pytest.raises(ValueError, match="first-caption"):
        validate_live_run(tmp_path, expected_finalists=3, expected_shortlist=1)


def test_live_workflow_targets_current_branch_campaign_and_every_mp4() -> None:
    workflow = Path(".github/workflows/live-campaign.yml").read_text()
    assert "feat/word-reveal-face-tracking" in workflow
    assert "reach-double-coverage-dedicated.yaml" in workflow
    assert "head -n 1" not in workflow
    assert 'for clip in "$run_dir"/clips/*.mp4' in workflow
    assert "reach-live-${{ github.sha }}" in workflow


def test_live_validator_rejects_release_gate_failures(tmp_path: Path) -> None:
    cases = []

    def case(name: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        _write_live_run(root, finalists=3, shortlist=1)
        return root

    root = case("status")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["status"] = "FAILED"
    (root / "manifest.json").write_text(json.dumps(manifest))
    cases.append((root, "production status"))

    root = case("errors")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["errors"] = [{"error": "render failed"}]
    (root / "manifest.json").write_text(json.dumps(manifest))
    cases.append((root, "pipeline errors"))

    root = case("shortlist")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["submission_shortlist"][0]["plan_id"] = "missing-plan"
    (root / "manifest.json").write_text(json.dumps(manifest))
    cases.append((root, "shortlist references"))

    root = case("inventory")
    (root / "tracking" / "00.tracking.json").unlink()
    cases.append((root, "artifact inventory"))

    root = case("transition")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["technical_qc"][0]["framing"]["transition_qc_pass"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    cases.append((root, "transition QC"))

    root = case("watermark")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["technical_qc"][0]["watermark"]["renderer_asset_present"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    cases.append((root, "watermark"))

    root = case("funnel")
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["funnel"]["raw_concepts"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    cases.append((root, "funnel ledger"))

    root = case("required-artifact")
    (root / "coverage.json").unlink()
    cases.append((root, "required live acceptance artifact"))

    for run_dir, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_live_run(run_dir, expected_finalists=3, expected_shortlist=1)


def test_live_validator_rejects_qc_count_and_concept_diversity(tmp_path: Path) -> None:
    root = tmp_path / "qc-count"
    root.mkdir()
    _write_live_run(root, finalists=3, shortlist=1)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["technical_qc"] = manifest["technical_qc"][:2]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="technical-QC"):
        validate_live_run(root, expected_finalists=3, expected_shortlist=1)

    root = tmp_path / "diversity"
    root.mkdir()
    _write_live_run(root, finalists=3, shortlist=1)
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["rendered_clips"]:
        item["concept_id"] = "same"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="concept diversity"):
        validate_live_run(root, expected_finalists=3, expected_shortlist=1)


def test_live_validator_rejects_non_object_manifest_and_wrong_targets(tmp_path: Path) -> None:
    non_object = tmp_path / "non-object"
    non_object.mkdir()
    (non_object / "manifest.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        validate_live_run(non_object, expected_finalists=3, expected_shortlist=1)

    wrong = tmp_path / "wrong-targets"
    wrong.mkdir()
    _write_live_run(wrong, finalists=3, shortlist=1)
    manifest = json.loads((wrong / "manifest.json").read_text())
    manifest["targets"] = {"rendered_finalists": 2, "submission_shortlist": 1}
    (wrong / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unexpected production targets"):
        validate_live_run(wrong, expected_finalists=3, expected_shortlist=1)


def test_live_validator_rejects_count_qc_and_funnel_mismatches(tmp_path: Path) -> None:
    def make(name: str) -> tuple[Path, dict]:
        root = tmp_path / name
        root.mkdir()
        _write_live_run(root, finalists=3, shortlist=1)
        return root, json.loads((root / "manifest.json").read_text())

    root, manifest = make("render-count")
    manifest["rendered_clips"] = manifest["rendered_clips"][:2]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rendered finalists"):
        validate_live_run(root, expected_finalists=3, expected_shortlist=1)

    root, manifest = make("shortlist-count")
    manifest["submission_shortlist"] = []
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shortlist clips"):
        validate_live_run(root, expected_finalists=3, expected_shortlist=1)

    root, manifest = make("caption-qc")
    manifest["technical_qc"][0]["captions"]["alignment"] = "FAIL"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="caption alignment QC"):
        validate_live_run(root, expected_finalists=3, expected_shortlist=1)

    for name, field, message in (
        ("funnel-render", "render_success", "render_success"),
        ("funnel-qc", "technical_qc_pass", "technical_qc_pass"),
        ("funnel-shortlist", "submission_shortlist", "submission_shortlist"),
    ):
        root, manifest = make(name)
        manifest["funnel"][field] = 0
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=message):
            validate_live_run(root, expected_finalists=3, expected_shortlist=1)
