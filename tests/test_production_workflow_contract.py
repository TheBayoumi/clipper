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
    assert '"sources": [_source_payload()]' in watchdog
    assert '"git_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"]' in watchdog
    assert "function.spawn(request)" in watchdog
    assert "call.get(timeout=poll_seconds)" in watchdog
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
    assert 'Path(os.environ["CLIPPER_CAMPAIGN_BRIEF"]).read_text(' in watchdog


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
    assert "pull-requests: write" in workflow
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
    assert 'f"outlines=={EDITORIAL_OUTLINES_VERSION}"' in worker
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
    assert "conservative in-flight GPU budget reached" in watchdog
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

    assert '"expected_git_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"]' in workflow
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
    assert "math.isfinite(max_gpu_seconds)" in watchdog
    assert "math.isfinite(max_estimated_usd)" in watchdog
    assert "math.isfinite(max_gpu_seconds)" in pipeline
    assert "math.isfinite(max_estimated_usd)" in pipeline
    assert "def _invoke_remote_with_budget(" in cli_modal
    assert "function.spawn(request)" in cli_modal
    assert "call.get(timeout=poll_seconds)" in cli_modal
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
    assert "if not spy_thread.is_alive():" in watchdog
    assert "Modal spy thread exited unexpectedly before production completion" in watchdog
    assert "cancel_call(reason)" in watchdog
