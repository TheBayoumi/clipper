from pathlib import Path

import yaml


def _workflow() -> str:
    return Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")


def _watchdog() -> str:
    return Path("scripts/modal_hilp_watchdog.py").read_text(encoding="utf-8")


def _bootstrap_workflow() -> str:
    return Path(".github/workflows/lovable-production-bootstrap.yml").read_text(encoding="utf-8")


def _modal_deploy_workflow() -> str:
    return Path(".github/workflows/modal-workers-deploy.yml").read_text(encoding="utf-8")


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
    assert "_spawn_recoverable_modal_call(" in watchdog
    assert "function.spawn(request)" not in watchdog
    assert "call, call_started, submission_error = _spawn_recoverable_modal_call(" in watchdog
    assert "call.get(timeout=min(poll_seconds, remaining_wall_seconds))" in watchdog
    assert "production_call_cancel_retry" in watchdog
    assert "production_call_cancel_requested" in watchdog
    assert "production_call_cancel_unconfirmed" in watchdog
    assert "cancelled.set()" in watchdog
    assert "call.cancel(terminate_containers=terminate_containers)" in watchdog
    assert "terminate_containers=False" in watchdog
    assert "terminate_containers=True" in watchdog
    assert "call.get(timeout=cancel_confirmation_seconds)" in watchdog
    assert '"root_call_terminal_confirmed": (' in watchdog
    assert '"root_call_hard_termination_succeeded": root_hard_termination_succeeded' in watchdog
    assert '"root_call_stopped": (' in watchdog
    assert "modal-function-call.json" in watchdog
    assert "content-addressed-resume" in workflow
    assert "content-addressed-stage-resume" in workflow
    assert "PENDING_ACTUAL_MP4_REVIEW" in watchdog
    assert "READY_FOR_HUMAN_REVIEW" in workflow
    assert "READY_TO_PUBLISH" in workflow
    assert "cycle-evidence" in workflow
    assert "hilp-review" in workflow


def test_lovable_bootstrap_pins_modal_deployment_to_triggering_sha() -> None:
    bootstrap = _bootstrap_workflow()
    deploy = _modal_deploy_workflow()

    assert 'tag="clipper-hilp-${EXPECTED_SHA}"' in bootstrap
    assert '{"ref": sys.argv[1], "inputs": {"deployment_sha": sys.argv[2]}}' in bootstrap
    assert "modal-workers-deploy.yml/dispatches" in bootstrap
    assert "production-pipeline.yml/dispatches" in bootstrap
    assert "pinned to ${EXPECTED_SHA}" in bootstrap
    assert "deployment_sha:" in deploy
    assert "run-name: Deploy Modal workers ${{ inputs.deployment_sha || github.sha }}" in deploy
    assert (
        "CLIPPER_DEPLOYED_GIT_SHA: ${{ inputs.deployment_sha || "
        "github.event.pull_request.head.sha || github.sha }}" in deploy
    )
    assert (
        "ref: ${{ inputs.deployment_sha || github.event.pull_request.head.sha || github.sha }}"
        in deploy
    )
    assert '[[ "$CLIPPER_DEPLOYED_GIT_SHA" =~ ^[0-9a-f]{40}$ ]]' in deploy
    assert 'test "$(git rev-parse HEAD)" = "$CLIPPER_DEPLOYED_GIT_SHA"' in deploy


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

    assert "_spawn_recoverable_modal_call(" in watchdog
    assert "function.spawn(request)" not in watchdog
    assert '"editorial_acceptance_probe": not render' in watchdog
    assert "call.cancel(terminate_containers=terminate_containers)" in watchdog
    assert "terminate_containers=False" in watchdog
    assert "terminate_containers=True" in watchdog
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
