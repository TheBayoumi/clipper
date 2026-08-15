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
            json.dumps(
                {
                    "alignment": "PASS",
                    "hook_overlay_rendered": False,
                    "potential_hook_caption_overlap_seconds": 1.2,
                    "simultaneous_narrative_layers_max": 1,
                }
            )
        )
        (root / "tracking" / f"{index:02d}.tracking.json").write_text("{}")
        rendered.append(
            {"plan_id": plan_id, "concept_id": f"concept-{index}", "output_path": str(clip)}
        )
        qc.append(
            {
                "plan_id": plan_id,
                "status": "PASS",
                "captions": {
                    "alignment": "PASS",
                    "hook_overlay_rendered": False,
                    "simultaneous_narrative_layers_max": 1,
                },
                "framing": {"transition_qc_pass": True},
                "watermark": {"required": True, "renderer_asset_present": True},
            }
        )
    manifest = {
        "status": "SUCCESS",
        "status_reason": None,
        "errors": [],
        "targets": {
            "rendered_finalists": finalists,
            "submission_shortlist": shortlist,
            "distinct_finalist_concepts": min(shortlist, finalists),
            "distinct_shortlist_concepts": shortlist,
        },
        "actual": {
            "rendered_finalists": finalists,
            "submission_shortlist": shortlist,
            "distinct_finalist_concepts": min(shortlist, finalists),
            "distinct_shortlist_concepts": shortlist,
        },
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


def test_live_validator_rejects_missing_or_overlapping_caption_concurrency(
    tmp_path: Path,
) -> None:
    _write_live_run(tmp_path, finalists=2, shortlist=1)
    audit_path = tmp_path / "captions" / "00.caption-audit.json"
    audit = json.loads(audit_path.read_text())
    audit.pop("simultaneous_narrative_layers_max")
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="concurrency evidence is missing"):
        validate_live_run(tmp_path, expected_finalists=2, expected_shortlist=1)

    audit["simultaneous_narrative_layers_max"] = 2
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="overlapping narrative captions"):
        validate_live_run(tmp_path, expected_finalists=2, expected_shortlist=1)

    audit["simultaneous_narrative_layers_max"] = 1
    audit["hook_overlay_rendered"] = True
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="hook and spoken captions overlap"):
        validate_live_run(tmp_path, expected_finalists=2, expected_shortlist=1)


def test_v10_workflow_targets_current_branch_campaign_and_every_mp4() -> None:
    workflow = Path(".github/workflows/v10-production-cycle.yml").read_text()
    assert "feat/word-reveal-face-tracking" in workflow
    assert "reach-double-coverage-dedicated.yaml" in workflow
    assert 'assert int(result["rendered_finalists"]) >= 6' in workflow
    assert 'assert int(result["initial_shortlist"]) >= 3' in workflow
    assert "for item in rendered:" in workflow
    assert '"ffmpeg"' in workflow
    assert "manual-review-queue.json" in workflow
    assert "PENDING_ACTUAL_REVIEW" in workflow
    assert '"automated_hilp_allowed": False' in workflow
    assert "v10-production-review-${{ github.sha }}" in workflow


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
    manifest["targets"]["distinct_finalist_concepts"] = 3
    manifest["actual"]["distinct_finalist_concepts"] = 3
    for item in manifest["rendered_clips"]:
        item["concept_id"] = "same"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="concept diversity"):
        validate_live_run(
            root, expected_finalists=3, expected_shortlist=1, expected_distinct_finalists=3
        )


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


def test_live_validator_rejects_six_files_representing_only_two_concepts(tmp_path: Path) -> None:
    _write_live_run(tmp_path, finalists=6, shortlist=3)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for index, item in enumerate(manifest["rendered_clips"]):
        item["concept_id"] = f"concept-{index % 2}"
    manifest["submission_shortlist"] = manifest["rendered_clips"][:3]
    manifest["actual"]["distinct_finalist_concepts"] = 2
    manifest["actual"]["distinct_shortlist_concepts"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"yield|concept diversity"):
        validate_live_run(
            tmp_path, expected_finalists=6, expected_shortlist=3, expected_distinct_finalists=3
        )


def test_package_root_lazily_exposes_pipeline_without_eager_dependency_import() -> None:
    import clipper

    assert clipper.PipelineSettings.__name__ == "PipelineSettings"
    assert callable(clipper.run_pipeline)
    missing_name = "missing_export"
    with pytest.raises(AttributeError):
        getattr(clipper, missing_name)


def test_live_validator_accepts_more_distinct_finalists_than_minimum(tmp_path: Path) -> None:
    _write_live_run(tmp_path, finalists=6, shortlist=3)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["targets"]["distinct_finalist_concepts"] = 3
    manifest["actual"]["distinct_finalist_concepts"] = 6
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = validate_live_run(
        tmp_path,
        expected_finalists=6,
        expected_shortlist=3,
        expected_distinct_finalists=3,
    )
    assert report["actual"]["distinct_finalist_concepts"] == 6


def test_balanced_editor_defaults_to_free_dual_l4_and_keeps_managed_opt_in() -> None:
    worker = Path("scripts/modal_open_models.py").read_text()
    factory = Path("src/clipper/providers/factory.py").read_text()
    production_workflow = Path(".github/workflows/v10-production-cycle.yml").read_text()
    endpoint_bootstrap = Path("scripts/modal_endpoint_bootstrap.py").read_text()
    assert "def editorial(" in worker
    assert "AutoModelForCausalLM" in worker
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in worker
    assert 'gpu="L4:2"' in worker
    assert "ModalEditorialProvider" in factory
    assert "ModalEndpointEditorialProvider" in factory
    assert 'CLIPPER_MODAL_EDITORIAL_BACKEND", "function"' in factory
    assert "Qwen/Qwen3.5-4B" in factory
    assert "Qwen/Qwen3.6-27B-FP8" in factory
    assert "modal-managed-endpoint" in factory
    assert '"endpoint", "create"' in endpoint_bootstrap
    assert "modal_endpoint_bootstrap.py" not in production_workflow
    assert "modal deploy scripts/modal_open_models.py" in production_workflow


def test_v10_production_workflow_uses_modal_hf_and_original_quality_master() -> None:
    workflow = Path(".github/workflows/v10-production-cycle.yml").read_text()
    deployment_workflow = Path(".github/workflows/modal-workers-deploy.yml").read_text()
    source_worker = Path("scripts/modal_v10_cycle.py").read_text()
    model_worker = Path("scripts/modal_open_models.py").read_text()
    assert "MODAL_TOKEN_ID" in workflow
    assert "MODAL_TOKEN_SECRET" in workflow
    assert "HF_TOKEN" not in workflow
    assert "HF_TOKEN" not in deployment_workflow
    assert 'HF_SECRET_NAME = "custom-secret"' in model_worker
    assert "modal.Secret.from_name(HF_SECRET_NAME)" in model_worker
    assert "modal.Secret.from_dict" not in model_worker
    assert model_worker.count("secrets=[hf_secret]") == 8
    assert "hf_access_smoke" in workflow
    assert "hf_access_smoke" in deployment_workflow
    assert "modal deploy scripts/modal_open_models.py" in workflow
    assert "modal deploy scripts/modal_v10_cycle.py" in workflow
    assert 'Function.from_name("clipper-v10-cycle", "acquire_source")' in workflow
    assert 'result["quality_policy"] == "highest_available_no_transcode"' in workflow
    assert "source-master.json" in workflow
    assert "with_options(cloud=value, timeout=1800)" in workflow
    assert "with_options(region=value, timeout=1800)" in workflow
    assert '"cloud:gcp"' in workflow
    assert '"cloud:aws"' in workflow
    assert '"cloud:oci"' in workflow
    assert 'Function.from_name("clipper-v10-cycle", "run_full_cycle")' in workflow
    assert 'assert result["pipeline_status"] == "SUCCESS"' in workflow
    assert 'assert int(result["rendered_finalists"]) >= 6' in workflow
    assert 'assert int(result["initial_shortlist"]) >= 3' in workflow
    assert "modal volume get --force clipper-v10-artifacts" in workflow
    assert "manual-review-queue.json" in workflow
    assert "PENDING_ACTUAL_MP4_REVIEW" in workflow
    assert "PENDING_ACTUAL_REVIEW" in workflow
    assert '"automated_hilp_allowed": False' in workflow
    assert "v10-production-review-${{ github.sha }}" in workflow
    assert "simulate_hilp_cycle" not in workflow
    assert 'modal app stop "$CLIPPER_V10_MODAL_APP" --yes' in workflow
    assert 'modal app stop "$CLIPPER_MODAL_APP" --yes' in workflow
    assert "def acquire_source(" in source_worker
    assert '"yt-dlp[default]>=2026.7.4,<2027"' in source_worker
    assert '"quality_policy": "highest_available_no_transcode"' in source_worker
    assert "scaledown_window=2" in source_worker
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in model_worker
    assert '"torchvision==0.23.0"' in model_worker
