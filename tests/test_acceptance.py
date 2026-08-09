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


def test_live_workflow_targets_current_branch_campaign_and_every_mp4() -> None:
    workflow = Path(".github/workflows/live-campaign.yml").read_text()
    assert "feat/word-reveal-face-tracking" in workflow
    assert "reach-double-coverage-dedicated.yaml" in workflow
    assert "head -n 1" not in workflow
    assert "for clip in /review/clips/*.mp4" in workflow
    assert "clipper-live:${GITHUB_SHA}" in workflow
    assert "--entrypoint /bin/sh" in workflow
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


def test_balanced_editor_uses_pinned_qwen3_30b_on_two_l4s() -> None:
    worker = Path("scripts/modal_open_models.py").read_text()
    factory = Path("src/clipper/providers/factory.py").read_text()
    local = Path("src/clipper/providers/local.py").read_text()
    editorial_block = worker[worker.index("def editorial") : worker.index("def _vision_infer")]
    decorator = worker[
        worker.rfind("@app.function", 0, worker.index("def editorial")) : worker.index(
            "def editorial"
        )
    ]
    assert 'gpu="L4:2"' in decorator
    assert 'EDITORIAL_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"' in worker
    assert '"transformers>=4.57,<5"' in worker
    assert '"transformers>=5.14,<6"' not in worker
    assert '"kernels>=0.15.2,<0.16"' not in worker
    assert '"sentencepiece>=0.2,<1"' in worker
    assert '"tiktoken>=0.11,<1"' in worker
    assert 'EDITORIAL_MODEL_REVISION = "110954009be4a882781a90356c7d2b8a9e3428dc"' in worker
    assert "BitsAndBytesConfig(" in editorial_block
    assert "load_in_4bit=True" in editorial_block
    assert 'bnb_4bit_quant_type="nf4"' in editorial_block
    assert 'device_map="auto"' in editorial_block
    assert 'max_memory={0: "22GiB", 1: "22GiB"}' in editorial_block
    assert '"L4:2"' in editorial_block
    assert "_editorial_output_budget(payload)" in editorial_block
    assert "_editorial_contract(task)" in editorial_block
    assert '"start_word_id"' in worker
    assert '"source_start_word_id"' in worker
    assert "short word_ref values" in worker
    assert "Do not copy full word-ID lists" in worker
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in factory
    assert 'CLIPPER_EDITORIAL_QUANTIZATION", "bnb-4bit-nf4"' in factory
    assert "110954009be4a882781a90356c7d2b8a9e3428dc" in factory
    # local-lite deliberately keeps the smaller model; balanced is the high-quality Modal path.
    assert "Qwen/Qwen3-4B-Instruct-2507" in local


def test_open_model_workflow_uses_modal_hf_and_full_episode_fixture() -> None:
    workflow = Path(".github/workflows/open-model-acceptance.yml").read_text()
    assert "MODAL_TOKEN_ID" in workflow
    assert "MODAL_TOKEN_SECRET" in workflow
    assert "HF_TOKEN" in workflow
    assert "modal deploy scripts/modal_open_models.py" in workflow
    assert 'Function.from_name(app, "editorial")' in workflow
    assert '"Qwen/Qwen3-30B-A3B-Instruct-2507"' in workflow
    assert "editorial_usage" in workflow
    assert "hf_access_smoke" in workflow
    assert workflow.index("Modal Qwen embedding and 30B editorial execution") < workflow.index(
        "gated Hugging Face diarization access"
    )
    assert workflow.index("gated Hugging Face diarization access") < workflow.index(
        "Run full-episode open editorial analysis through Modal"
    )
    assert 'speech_providers("balanced")' in workflow
    assert "CLIPPER_EDITORIAL_ENGINE: open" in workflow
    assert "CLIPPER_GROUNDING_ENGINE: open" in workflow
    assert "CLIPPER_COMPUTE_PROFILE: balanced" in workflow
    assert "CLIPPER_VISUAL_SCOUT" in workflow
    assert "CLIPPER_OPEN_PROXY_URL" in workflow
    assert "CLIPPER_OPEN_PROXY_SHA256" in workflow
    assert "reach-open-proxy-v1" in workflow
    assert "Restore prior compatible open-model cache" in workflow
    assert 'artifact="open-model-acceptance-$head_sha"' in workflow
    assert "prior-open-evidence/_cache" in workflow
    assert "open-evidence/_cache" in workflow
    assert 'manifest["full_media"]' in workflow
    assert "--no-render" in workflow
    assert "progress.json" in workflow
    assert 'kill -0 "$pid"' in workflow
    assert 'wait "$pid"' in workflow
    assert "reach-double-coverage-dedicated.yaml" in workflow
