import json
from pathlib import Path

import yaml


def _workflow() -> str:
    return Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")


def _watchdog() -> str:
    return Path("scripts/modal_hilp_watchdog.py").read_text(encoding="utf-8")


def test_production_workflow_is_single_pass_resumable_and_exact_head() -> None:
    workflow = _workflow()
    watchdog = _watchdog()

    deployment_gate = workflow.index("Wait for successful exact-head Modal deployment")
    execution = workflow.index("Run current-model pipeline with cancellable Modal spy")
    validation = workflow.index(
        "Validate dynamic yield, resumable inference, cost bounds, and actual media"
    )

    assert deployment_gate < execution < validation
    assert "python scripts/modal_hilp_watchdog.py" in workflow
    assert "editorial_acceptance_only:" in workflow
    assert "CLIPPER_RENDER" in workflow
    assert "fresh_inference:" in workflow
    assert '"fresh_inference": True' not in workflow
    assert '"fresh_inference": os.environ["CLIPPER_FRESH_INFERENCE"] == "true"' in watchdog
    assert '"resume_from_run_id": os.environ.get("CLIPPER_RESUME_FROM_RUN_ID") or None' in watchdog
    assert '"sources": [source_payload]' in watchdog
    assert "scoped_brief_yaml = _scoped_brief_yaml()" in watchdog
    assert '"brief_yaml": scoped_brief_yaml' in watchdog
    assert '"videos"] = [dict(matches[0])]' in watchdog
    assert '"git_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"]' in watchdog
    assert "function.spawn(request)" in watchdog
    assert "call_started = time.monotonic()" in watchdog
    assert "call.get(timeout=min(poll_seconds, remaining_wall_seconds))" in watchdog
    assert "production_call_cancel_retry" in watchdog
    assert "cancelled.set()" in watchdog
    assert "call.cancel(terminate_containers=False)" in watchdog
    assert "modal-function-call.json" in watchdog
    assert "content-addressed-resume" in workflow
    assert "content-addressed-stage-resume" in workflow
    assert "PENDING_ACTUAL_MP4_REVIEW" in watchdog
    assert "READY_FOR_HUMAN_REVIEW" in workflow
    assert "READY_TO_PUBLISH" in workflow
    assert "cycle-evidence" in workflow
    assert "hilp-review" in workflow


def test_production_workflow_resolves_campaign_and_target_from_request_data() -> None:
    workflow = _workflow()

    assert "campaign_brief:" in workflow
    assert "target_video_id:" in workflow
    assert '"acceptance/production-run-request.json"' in workflow
    assert 'marker.get("campaign_brief")' in workflow
    assert 'marker.get("target_video_id")' in workflow
    assert 'targets.get("videos")' in workflow
    assert 'rights.get("authorized_channels")' in workflow
    watchdog = _watchdog()
    assert 'os.environ["CLIPPER_TARGET_VIDEO_ID"]' in watchdog
    assert 'os.environ["CLIPPER_TARGET_VIDEO_URL"]' in watchdog
    assert 'os.environ["CLIPPER_TARGET_CHANNEL_ID"]' in watchdog
    assert 'brief_path = Path(os.environ["CLIPPER_CAMPAIGN_BRIEF"])' in watchdog
    assert 'text = brief_path.read_text(encoding="utf-8")' in watchdog
    assert "scoped_brief_yaml = _scoped_brief_yaml()" in watchdog


def test_production_acceptance_contains_no_campaign_specific_identity() -> None:
    workflow = _workflow()

    forbidden = (
        "reach-double-coverage",
        "Double Coverage",
        "2Y4LP85PTak",
        "UCf1q6dhccWr6eQEcFFnJSbA",
        "#DoubleCoverage",
    )
    assert all(value not in workflow for value in forbidden)


def test_production_workflow_is_dynamic_yield_and_human_review_gated() -> None:
    workflow = _workflow()

    assert 'int(result["rendered_finalists"]) >= 6' not in workflow
    assert 'int(result["initial_shortlist"]) >= 3' not in workflow
    assert "eligible_quality_moments" in workflow
    watchdog = _watchdog()
    assert 'rendered = int(normalized.get("rendered") or 0)' in watchdog
    assert 'reviewable = int(normalized.get("reviewable") or 0)' in watchdog
    assert "if reviewable != rendered:" in watchdog
    assert "contract permits zero quality yield" in workflow
    assert "PENDING_ACTUAL_MP4_REVIEW" in watchdog
    assert '"human_review": "PENDING_ACTUAL_REVIEW"' in workflow
    assert "Wait for successful exact-head Modal deployment" in workflow
    assert "modal-workers-deploy.yml" in workflow
    assert '"head_sha": sha' in workflow
    assert 'item.get("status") == "completed"' in workflow
    assert 'item.get("conclusion") == "success"' in workflow
    assert "time.sleep(15)" in workflow
    assert "modal-deployment-prerequisite.json" in workflow
    assert "modal app stop" not in workflow
    assert "modal deploy scripts/modal_open_models.py" not in workflow
    assert "modal deploy scripts/modal_pipeline.py" not in workflow


def test_production_workflow_enforces_current_model_and_budget_evidence() -> None:
    workflow = _workflow()

    assert 'int(schema.get("task_families", 0)) != 4' in workflow
    assert 'os.environ["CLIPPER_EXECUTION_MODE"] == "fresh-inference" and hits != 0' in workflow
    assert (
        'os.environ["CLIPPER_EXECUTION_MODE"] == "fresh-inference"'
        " and stage_cache_hits != 0" in workflow
    )
    assert 'editorial.get("model_invocations")' in workflow
    assert '"semantic_cores", "narrative_envelope", "quality_windows"' in workflow
    assert "gpu_seconds > gpu_limit" in workflow
    assert "estimated_usd > cost_limit" in workflow


def test_production_workflow_requires_exact_head_modal_deployment_without_mutation() -> None:
    workflow = _workflow()
    parsed = yaml.safe_load(workflow)
    assert isinstance(parsed, dict)
    assert "jobs" in parsed
    assert "actions: read" in workflow
    assert "Wait for successful exact-head Modal deployment" in workflow
    assert 'workflow = "modal-workers-deploy.yml"' in workflow
    assert '"head_sha": sha' in workflow
    assert 'item.get("head_sha") == sha' in workflow
    assert 'item.get("conclusion") == "success"' in workflow
    assert "exact-head Deploy Modal workers completed unsuccessfully" in workflow
    assert "timed out waiting for successful exact-head Deploy Modal workers run" in workflow
    assert "modal app stop" not in workflow
    assert "modal deploy " not in workflow


def test_production_workflow_has_cancellable_modal_spy_and_editorial_acceptance() -> None:
    workflow = _workflow()
    watchdog = _watchdog()
    spy = Path("scripts/modal_execution_spy.py").read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    assert isinstance(parsed, dict)

    assert "issues: write" in workflow
    assert "pull-requests: read" in workflow
    assert "editorial_acceptance_only:" in workflow
    assert "REQUEST_EDITORIAL_ACCEPTANCE_ONLY" in workflow
    assert 'os.environ.get("REQUEST_EDITORIAL_ACCEPTANCE_ONLY", "").lower() == "true"' in workflow
    assert "Run current-model pipeline with cancellable Modal spy" in workflow
    assert "python scripts/modal_hilp_watchdog.py" in workflow
    assert "Validate live editorial projection and token-aware repartition" in workflow
    assert "editorial_evidence_projection" in workflow
    assert "editorial_capacity_probe" in workflow
    assert "editorial_acceptance_probe_result" in workflow
    assert "maximum_partition_count" in workflow
    assert "legacy raw negative-control payload unexpectedly fit model context" in workflow
    assert "token-aware repartitioner did not produce measured multi-way recovery" in workflow
    assert "if: env.CLIPPER_RENDER == 'false'" in workflow
    assert "if: env.CLIPPER_RENDER == 'true'" in workflow

    assert "function.spawn(request)" in watchdog
    assert '"editorial_acceptance_probe": not render' in watchdog
    assert "call.cancel(terminate_containers=False)" in watchdog
    assert "production_call_spawned" in watchdog
    assert "production_call_cancel" in watchdog
    assert "SIGTERM" in watchdog
    assert "SIGINT" in watchdog
    assert "modal-spy-summary.json" in watchdog

    assert '"editorial_context_repartition"' in spy
    assert "under-partitioned measured input" in spy
    assert "repeated without forward progress" in spy
    assert "projection expanded serialized evidence" in spy
    assert "--show-function-call-id" in spy
    assert '"--follow"' in spy
    assert '"--since"' not in spy


def test_modal_editorial_capacity_probe_is_non_generating() -> None:
    worker = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    assert "def _editorial_capacity_probe(" in worker
    assert "def capacity_probe(" in worker
    probe_start = worker.index("def _editorial_capacity_probe(")
    probe_end = worker.index("@app.cls(", probe_start)
    probe = worker[probe_start:probe_end]
    assert "_editorial_generation_plan(" in probe
    assert "structured_model(" not in probe
    assert "model.generate(" not in probe
    assert '"event": "editorial_capacity_probe"' in probe

    assert "def _editorial_acceptance_probe(" in pipeline
    assert "invoke_editorial_capacity_probe(" in pipeline
    assert "worker.capacity_probe.remote" in pipeline
    assert "token_aware_repartition(" in pipeline
    assert '"event": "editorial_acceptance_probe_result"' in pipeline
    assert "editorial-acceptance-probe.json" in pipeline


def test_editorial_deadline_acceptance_forces_and_correlates_real_generation() -> None:
    worker = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    provider = Path("src/clipper/providers/modal.py").read_text(encoding="utf-8")
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    spy = Path("scripts/modal_execution_spy.py").read_text(encoding="utf-8")
    workflow = _workflow()

    assert "EDITORIAL_DEADLINE_PROBE_MIN_NEW_TOKENS = 65_536" in worker
    assert "def _editorial_deadline_probe(" in worker
    assert "def deadline_probe(self, payload:" in worker
    deadline_start = worker.index("def _editorial_deadline_probe(")
    deadline_end = worker.index("def _editorial_capacity_probe(", deadline_start)
    deadline_probe = worker[deadline_start:deadline_end]
    assert "structured_model(" in deadline_probe
    assert "rendered,\n            None," in deadline_probe
    assert "min_new_tokens=output_budget" in deadline_probe
    assert "max_time=EDITORIAL_GENERATION_DEADLINE_SECONDS" in deadline_probe
    assert "elapsed_seconds < EDITORIAL_GENERATION_DEADLINE_SECONDS" in deadline_probe
    assert "elapsed_seconds > maximum_deadline_elapsed" in deadline_probe
    assert "output_units is None or output_units >= output_budget" in deadline_probe
    assert "_editorial_generation_deadline_error(" in deadline_probe

    production_infer_start = worker.index("def _editorial_infer(")
    production_infer_end = worker.index("def _editorial_deadline_probe(", production_infer_start)
    production_infer = worker[production_infer_start:production_infer_end]
    assert "late_candidate_parseable = True" in production_infer
    deadline_rejection = production_infer.index(
        'message="editorial generation reached the runtime latency boundary"'
    )
    candidate_acceptance = production_infer.index("generated_text = candidate")
    assert deadline_rejection < candidate_acceptance

    assert "def invoke_editorial_deadline_probe(" in provider
    assert 'application_status != "CAPACITY_REJECTED"' in provider
    assert 'error_type != "EditorialCapacityError"' in provider
    assert 'reason != "generation_runtime_deadline"' in provider

    assert "invoke_editorial_deadline_probe(" in pipeline
    assert "worker.deadline_probe.remote" in pipeline
    assert "deadline_probe_min_new_tokens = 65_536" in pipeline
    assert "deadline_seconds != 300.0" in pipeline
    assert "deadline_target_tokens >= deadline_input_tokens" in pipeline
    assert '"generation_deadline_probe": deadline_evidence' in pipeline
    assert '"invocation_id": str(deadline_result.get("invocation_id") or "")' in pipeline

    assert '"editorial_generation_deadline"' in spy
    assert '"elapsed_seconds"' in spy
    assert '"forced_min_new_tokens"' in spy
    assert 'counts.get("editorial_generation_deadline")' in workflow
    assert 'event.get("reason") == "generation_runtime_deadline"' in workflow
    assert "len(deadline_terminals) != 1" in workflow
    assert "len(matching_deadlines) != 1 or len(matching_repartitions) != 1" in workflow
    assert "elapsed_seconds > deadline_seconds + deadline_tolerance_seconds" in workflow
    assert "forced_min_new_tokens < 65_536" in workflow
    assert "output_tokens <= 0 or output_tokens >= forced_min_new_tokens" in workflow
    assert "target_repartition_tokens >= observed_repartition_tokens" in workflow
    assert 'probe.get("generation_deadline_probe")' in workflow


def test_production_workflow_waits_for_exact_head_modal_deploy() -> None:
    workflow = Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")
    assert "Wait for successful exact-head Modal deployment" in workflow
    assert "for _attempt in range(180):" in workflow
    assert "time.sleep(15)" in workflow
    assert '"status") in {"queued", "in_progress", "waiting", "pending"}' in workflow
    assert "exact-head Deploy Modal workers completed unsuccessfully" in workflow
    assert "timed out waiting for successful exact-head Deploy Modal workers run" in workflow


def test_modal_deployment_embeds_and_verifies_exact_source_sha() -> None:
    deploy = Path(".github/workflows/modal-workers-deploy.yml").read_text(encoding="utf-8")
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    models = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    parsed_deploy = yaml.safe_load(deploy)

    assert isinstance(parsed_deploy, dict)
    permissions = parsed_deploy.get("permissions")
    assert isinstance(permissions, dict)
    assert permissions.get("actions") == "read"
    assert "CLIPPER_DEPLOYED_GIT_SHA" in deploy
    assert "Verify exact deployment checkout" in deploy
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in deploy
    assert "Verify immutable deployed SHA identities" in deploy
    assert '"deployment_identity"' in deploy
    assert "production pipeline worker SHA mismatch" in pipeline
    assert "open-model worker SHA mismatch" in models
    assert "def deployment_identity()" in pipeline
    assert "def deployment_identity()" in models


def test_editorial_runtime_safety_is_preflighted_and_fail_closed() -> None:
    worker = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    provider = Path("src/clipper/providers/modal.py").read_text(encoding="utf-8")
    deploy = Path(".github/workflows/modal-workers-deploy.yml").read_text(encoding="utf-8")
    workflow = _workflow()
    watchdog = _watchdog()

    assert 'EDITORIAL_OUTLINES_VERSION = "1.3.0"' in worker
    assert 'EDITORIAL_TRANSFORMERS_VERSION = "4.57.3"' in worker
    assert 'f"outlines=={EDITORIAL_OUTLINES_VERSION}"' in worker
    assert 'f"transformers=={EDITORIAL_TRANSFORMERS_VERSION}"' in worker
    assert "def _verify_editorial_generation_runtime_contract()" in worker
    assert "GenerationConfig(max_time=EDITORIAL_GENERATION_DEADLINE_SECONDS)" in worker
    assert "Transformers._generate_output_seq(" in worker
    assert 'observed_generate_kwargs.get("max_time")' in worker
    load_start = worker.index("def load_model(self) -> None:")
    load_end = worker.index("@modal.method()", load_start)
    load_model = worker[load_start:load_end]
    assert load_model.index("_verify_editorial_generation_runtime_contract()") < load_model.index(
        "_load_editorial_model()"
    )

    runtime_contract_gate = deploy.index("Verify editorial generation runtime contract")
    gpu_placement_gate = deploy.index("Verify editorial model spans allocated GPUs")
    assert runtime_contract_gate < gpu_placement_gate
    assert 'contract.get("outlines_version") != "1.3.0"' in deploy
    assert 'contract.get("transformers_version") != "4.57.3"' in deploy
    assert 'contract.get("outlines_max_time_forwarded") is not True' in deploy
    assert 'contract.get("transformers_max_time_supported") is not True' in deploy
    assert "modal-editorial-runtime-contract.json" in deploy

    assert "CLIPPER_EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS" in worker
    assert 'actual_payload.get("capacity_repartitionable") is True' in worker
    assert '"reason": "runtime_input_guard"' in worker
    assert "startup_timeout=EDITORIAL_STARTUP_TIMEOUT_SECONDS" in worker
    assert "timeout=EDITORIAL_EXECUTION_TIMEOUT_SECONDS" in worker
    assert '"editorial_execution_timeout"' in provider
    assert '"reason": "execution_timeout"' in provider
    assert 'getattr(exception_namespace, "FunctionTimeoutError", None)' in provider
    assert "isinstance(exc, timeout_type)" in provider

    assert "CLIPPER_EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS: 32768" in workflow
    assert "CLIPPER_EDITORIAL_GENERATION_DEADLINE_SECONDS: 300" in workflow
    assert "CLIPPER_EDITORIAL_GENERATION_DEADLINE_TOLERANCE_SECONDS: 30" in workflow
    assert "CLIPPER_MODAL_GENERATION_STALL_SECONDS: 720" in workflow
    assert 'authoritative_counts.get("editorial_remote_call_start")' in workflow
    assert 'authoritative_counts.get("editorial_remote_call_terminal")' in workflow
    assert 'counts.get("editorial_execution_timeout")' in workflow
    assert "producer terminal carried unsafe runtime capacity evidence" in workflow
    assert "safe_completes" in workflow
    assert "if not projections or not probe_results:" in workflow
    assert "if not projections or not capacity_probes or not probe_results:" not in workflow
    assert '"max_gpu_seconds": max_gpu_seconds' in watchdog
    assert '"max_estimated_usd": max_estimated_usd' in watchdog
    assert "conservative in-flight production budget reached before completion" in watchdog
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    assert "invoke_editorial_capacity_probe(" in pipeline
    assert "_run_editorial_capacity_probe(" not in pipeline
    assert "worker.capacity_probe.remote" in pipeline


def test_modal_spy_is_bound_to_spawned_call_and_execution_id() -> None:
    watchdog = _watchdog()
    spy = Path("scripts/modal_execution_spy.py").read_text(encoding="utf-8")

    assert "execution_id = uuid.uuid4().hex" in watchdog
    assert "spy.root_function_call_id = call_id" in watchdog
    assert "execution_id=execution_id" in watchdog
    assert "spy_thread.start()" in watchdog
    assert watchdog.index("spy_thread.start()") < watchdog.index("function.spawn(request)")
    assert watchdog.index("call.hydrate()") < watchdog.index("spy.root_function_call_id = call_id")

    assert "root_function_call_id" in spy
    assert "_belongs_to_execution" in spy
    assert "app == self.pipeline_app" in spy
    assert 'payload.get("execution_id")' in spy
    assert "editorial remote call made no terminal progress before watchdog deadline" in spy
    assert "wait_for_producer_barrier" in spy
    assert "active_editorial_calls" in spy
    assert "authoritative_event_counts" in spy
    assert "wait_for_terminal_and_quiet" not in spy
    assert '"outlines_version"' in spy
    assert '"transformers_version"' in spy
    assert '"generation_deadline_seconds"' in spy
    assert '"editorial_generation_deadline"' in spy


def test_production_runtime_rechecks_deployed_identity_before_inference() -> None:
    workflow = _workflow()

    identity = workflow.index("Verify immutable deployed Modal SHA identities")
    schemas = workflow.index("Verify four production editorial schemas and gated-model access")
    execution = workflow.index("Run current-model pipeline with cancellable Modal spy")
    assert identity < schemas < execution
    assert '"deployment_identity"' in workflow
    assert "deployed SHA mismatch before inference" in workflow


def test_all_paid_modal_calls_carry_expected_deployed_sha() -> None:
    workflow = _workflow()
    speech = Path("src/clipper/providers/modal_speech.py").read_text(encoding="utf-8")
    models = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")

    watchdog = _watchdog()
    assert 'expected_git_sha=os.environ["CLIPPER_ACCEPTANCE_SHA"]' in watchdog
    assert '"expected_git_sha"' in speech
    assert "_assert_expected_git_sha(payload)" in models
    assert "_assert_expected_git_sha(payload)" in pipeline


def test_paid_workflows_are_not_triggered_by_pull_request_synchronization() -> None:
    production = Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")
    deploy = Path(".github/workflows/modal-workers-deploy.yml").read_text(encoding="utf-8")
    production_marker = json.loads(
        Path("acceptance/production-run-request.json").read_text(encoding="utf-8")
    )
    deploy_marker = json.loads(
        Path("acceptance/modal-deploy-request.json").read_text(encoding="utf-8")
    )

    assert "pull_request:" not in production
    assert "pull_request:" not in deploy
    assert 'paths:\n      - "acceptance/production-run-request.json"' in production
    assert 'paths:\n      - "acceptance/modal-deploy-request.json"' in deploy
    assert "Require enabled production acceptance request" in production
    assert "Require enabled Modal deployment request" in deploy
    assert isinstance(production_marker.get("enabled"), bool)
    assert isinstance(production_marker.get("confirm_production"), bool)
    assert isinstance(deploy_marker.get("enabled"), bool)


def test_pull_request_smoke_is_read_only_and_package_publish_requires_exact_head_ci() -> None:
    smoke_path = Path(".github/workflows/deploy-and-smoke.yml")
    publish_path = Path(".github/workflows/publish-tested-image.yml")
    smoke = smoke_path.read_text(encoding="utf-8")
    publish = publish_path.read_text(encoding="utf-8")
    parsed_smoke = yaml.safe_load(smoke)
    parsed_publish = yaml.safe_load(publish)

    assert isinstance(parsed_smoke, dict)
    assert parsed_smoke.get("permissions") == {"contents": "read"}
    assert "pull_request:" in smoke
    assert "packages: write" not in smoke
    assert "docker login ghcr.io" not in smoke
    assert "docker push" not in smoke

    assert isinstance(parsed_publish, dict)
    assert parsed_publish.get("permissions") == {
        "contents": "read",
        "packages": "write",
    }
    assert "workflow_run:" in publish
    assert "workflows:\n      - CI" in publish
    assert "types:\n      - completed" in publish
    assert "branches:\n      - main" in publish
    assert "github.event.workflow_run.conclusion == 'success'" in publish
    assert "github.event.workflow_run.event == 'push'" in publish
    assert "github.event.workflow_run.head_branch == 'main'" in publish
    assert "ref: ${{ github.event.workflow_run.head_sha }}" in publish
    assert 'expected_sha="${{ github.event.workflow_run.head_sha }}"' in publish
    assert 'test "${source_sha}" = "${expected_sha}"' in publish
    assert "docker login ghcr.io" in publish
    assert "docker push" in publish
    assert ":latest" not in publish
    assert 'tag="sha-${source_sha::12}"' in publish


def test_production_pipeline_rejects_prohibited_watermark_before_paid_work() -> None:
    pipeline = Path("src/clipper/pipeline.py").read_text(encoding="utf-8")
    brief_load = pipeline.index("brief = load_brief(brief_path)")
    watermark_gate = pipeline.index("campaign watermark_url is prohibited by")
    provider_resolution = pipeline.index("editor = editorial_provider or build_editorial_provider")
    assert brief_load < watermark_gate < provider_resolution


def test_published_image_installs_and_preflights_real_cli_runtimes() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    smoke = Path(".github/workflows/deploy-and-smoke.yml").read_text(encoding="utf-8")
    publish = Path(".github/workflows/publish-tested-image.yml").read_text(encoding="utf-8")

    assert 'pip install ".[open-models]"' in dockerfile
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert pyproject.count('"transformers==4.57.3"') >= 3
    assert '"transformers>=5.14,<6"' not in pyproject
    assert 'pip install ".[asr]"' not in dockerfile
    for workflow in (smoke, publish):
        assert "Preflight published CLI runtimes inside image" in workflow
        assert "preflight --profile balanced" in workflow
        assert "preflight --profile local-lite --allow-local-lite" in workflow
        assert '--build-arg "CLIPPER_SOURCE_SHA=${source_sha}"' in workflow
        assert "_runtime_source_sha" in workflow
    assert "ARG CLIPPER_SOURCE_SHA" in dockerfile
    assert ".clipper-source-sha" in dockerfile


def test_disabled_acceptance_guards_precede_any_paid_modal_work() -> None:
    production = Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")
    deploy = Path(".github/workflows/modal-workers-deploy.yml").read_text(encoding="utf-8")

    production_guard = production.index("Require enabled production acceptance request")
    static_gate = production.index("Wait for successful exact-head CI and smoke")
    production_compute = production.index("Verify Modal credentials before compute")
    production_spawn = production.index("Run current-model pipeline with cancellable Modal spy")
    assert production_guard < static_gate < production_compute < production_spawn
    assert '"ci.yml"' in production
    assert '"deploy-and-smoke.yml"' in production
    assert "exact-head static gate failed before production compute" in production

    deploy_guard = deploy.index("Require enabled Modal deployment request")
    deploy_static_gate = deploy.index("Wait for successful exact-head CI and smoke")
    deploy_install = deploy.index("Install Modal orchestration dependencies")
    deploy_open_models = deploy.index("Deploy exact-HEAD open-model workers")
    assert deploy_guard < deploy_static_gate < deploy_install < deploy_open_models
    assert '"ci.yml"' in deploy
    assert '"deploy-and-smoke.yml"' in deploy
    assert "exact-head static gate failed before Modal deployment" in deploy


def test_production_budget_limits_are_finite_and_cli_is_cancellable() -> None:
    workflow = _workflow()
    watchdog = _watchdog()
    cli_modal = Path("src/clipper/modal_execution.py").read_text(encoding="utf-8")
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")

    assert "math.isfinite(gpu_limit)" in workflow
    assert "math.isfinite(cost_limit)" in workflow
    assert "def _finite_positive_env(" in watchdog
    assert '_finite_positive_env("CLIPPER_MAX_GPU_SECONDS")' in watchdog
    assert '_finite_positive_env("CLIPPER_MAX_ESTIMATED_USD")' in watchdog
    assert "math.isfinite(max_gpu_seconds)" in pipeline
    assert "math.isfinite(max_estimated_usd)" in pipeline
    assert "def _invoke_remote_with_budget(" in cli_modal
    assert "function.spawn(request)" in cli_modal
    assert "call.get(timeout=min(poll_seconds, remaining_seconds))" in cli_modal
    assert "remaining_budget_wall_seconds" in cli_modal
    assert "call.cancel(terminate_containers=False)" in cli_modal
    assert "math.isfinite(resolved)" in cli_modal


def test_failed_runner_response_keeps_verified_execution_identity() -> None:
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    failed_start = pipeline.index("if failed:")
    failed_end = pipeline.index("editorial_probe_result:", failed_start)
    failed = pipeline[failed_start:failed_end]

    assert '"execution_id": execution_id.lower()' in failed
    assert '"deployed_git_sha": DEPLOYED_GIT_SHA' in failed
    assert '"execution_mode": execution_mode' in failed


def test_file_cache_uses_per_writer_atomic_temporary_paths() -> None:
    cache = Path("src/clipper/cache.py").read_text(encoding="utf-8")

    assert "uuid.uuid4().hex" in cache
    assert 'f".{path.name}.{uuid.uuid4().hex}.tmp"' in cache
    assert "temporary.replace(path)" in cache
    assert "temporary.unlink(missing_ok=True)" in cache


def test_editorial_capacity_state_has_one_serialized_remote_writer() -> None:
    worker = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    state = Path("src/clipper/editorial_capacity_state.py").read_text(encoding="utf-8")

    assert "def editorial_capacity_state_writer(" in worker
    assert "max_containers=1" in worker
    assert "model_cache.reload()" in worker
    assert "merge_editorial_capacity_state(current, incoming)" in worker
    assert "write_editorial_capacity_state(path, merged)" in worker
    assert "model_cache.commit()" in worker
    assert "editorial_capacity_state_writer.remote(" in worker
    assert "editorial_capacity_state_persist_failed" in worker
    assert 'f".{path.name}.{uuid.uuid4().hex}.tmp"' in state
    assert "temporary.replace(path)" in state


def test_watchdog_fails_closed_if_spy_thread_exits() -> None:
    watchdog = _watchdog()
    spy = Path("scripts/modal_execution_spy.py").read_text(encoding="utf-8")

    assert "active_calls = list(self._active_editorial_calls.items())" in spy
    assert "with self.lock:" in spy
    assert "if not self.stop.is_set():" in spy
    assert "Modal log follower exited before explicit stop" in spy
    assert "returncode={process.returncode}" in spy
    assert "returncode not in {0, None}" not in spy
    assert "if not spy_thread.is_alive():" in watchdog
    assert "Modal spy thread exited unexpectedly before production completion" in watchdog
    assert "cancel_call(reason)" in watchdog


def test_watchdog_validates_all_timing_before_spawn_and_cleans_partial_setup() -> None:
    watchdog = _watchdog()
    run_start = watchdog.index("def run(*, render: bool)")
    run_body = watchdog[run_start:]

    poll_validation = run_body.index('_finite_positive_env("CLIPPER_MODAL_SPY_POLL_SECONDS"')
    barrier_validation = run_body.index('"CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS"')
    spy_start = run_body.index("spy_thread.start()")
    spawn = run_body.index("function.spawn(request)")
    assert poll_validation < spy_start < spawn
    assert barrier_validation < spy_start < spawn
    assert "try:\n        spy_thread.start()" in run_body
    assert "if call is not None and not remote_completed and not cancelled.is_set():" in run_body
    assert 'cancel_call("watchdog exited before production call completed")' in run_body


def test_source_acquisition_is_inside_watchdog_budget_envelope() -> None:
    workflow = _workflow()
    watchdog = _watchdog()
    prepare = workflow.index("Prepare budget-accounted source acquisition")
    execution = workflow.index("Run current-model pipeline with cancellable Modal spy")
    assert prepare < execution
    between_steps = workflow[prepare:execution]
    assert ".remote(" not in between_steps
    assert ".spawn(" not in between_steps
    assert "_acquire_remote_source(" in watchdog
    assert "budget=budget" in watchdog
    assert "source-budget.json" in watchdog
    assert "remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()" in watchdog
    assert '"max_gpu_seconds": remaining_gpu_seconds' in watchdog
    assert '"max_estimated_usd": remaining_estimated_usd' in watchdog
