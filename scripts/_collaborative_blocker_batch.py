from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement target, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# P1: default Modal call cancellation must confirm exact terminal state.
replace_once(
    "src/clipper/modal_execution.py",
    '_RETRY_DELAYS_SECONDS = (2.0, 5.0)\n',
    '''_RETRY_DELAYS_SECONDS = (2.0, 5.0)
_CANCEL_CONFIRM_TERMINAL_MODAL_ERRORS = frozenset(
    {
        "DeserializationError",
        "ExecutionError",
        "FunctionTimeoutError",
        "InputCancellation",
        "InternalFailure",
        "OutputExpiredError",
        "RemoteError",
    }
)
_CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS = frozenset(
    {
        "AuthError",
        "ClientClosed",
        "ConflictError",
        "ConnectionError",
        "DataLossError",
        "InternalError",
        "InvalidError",
        "NotFoundError",
        "PermissionDeniedError",
        "RequestSizeError",
        "ResourceExhaustedError",
        "ServiceError",
        "UnimplementedError",
        "VersionError",
    }
)
''',
)
replace_once(
    "src/clipper/modal_execution.py",
    '''def _retry_delay(attempt: int) -> float:
    index = max(0, min(attempt - 1, len(_RETRY_DELAYS_SECONDS) - 1))
    return _RETRY_DELAYS_SECONDS[index]
''',
    '''def _retry_delay(attempt: int) -> float:
    index = max(0, min(attempt - 1, len(_RETRY_DELAYS_SECONDS) - 1))
    return _RETRY_DELAYS_SECONDS[index]


def _cancel_confirmation_seconds() -> float:
    timeout = float(os.getenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "30"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Modal cancellation confirmation timeout must be finite and positive")
    return timeout


def _cancellation_confirmation_is_terminal(exc: BaseException) -> bool:
    names = _exception_class_names(exc)
    if names & _CANCEL_CONFIRM_TERMINAL_MODAL_ERRORS:
        return True
    if names & _CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS:
        return False
    return False
''',
)
replace_once(
    "src/clipper/modal_execution.py",
    '''def _cancel_remote_call(call: Any) -> None:
    """Cancel one exact Modal call and fail closed unless the API confirms cancellation."""
    call_id = str(getattr(call, "object_id", "") or getattr(call, "id", "") or "")
    errors: list[str] = []
    for attempt in range(1, _CONTROL_PLANE_ATTEMPTS + 1):
        try:
            call.cancel(terminate_containers=False)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < _CONTROL_PLANE_ATTEMPTS:
                time.sleep(_retry_delay(attempt))
                continue
            raise ProductionCallNotTerminated(call_id, errors) from exc
        return
    raise ProductionCallNotTerminated(call_id, errors)
''',
    '''def _cancel_remote_call(call: Any) -> None:
    """Cancel one exact Modal call and return only after terminal state is confirmed."""
    call_id = str(getattr(call, "object_id", "") or getattr(call, "id", "") or "")
    errors: list[str] = []
    requested = False
    for attempt in range(1, _CONTROL_PLANE_ATTEMPTS + 1):
        try:
            call.cancel(terminate_containers=False)
        except Exception as exc:
            errors.append(f"cancel {type(exc).__name__}: {exc}")
            if attempt < _CONTROL_PLANE_ATTEMPTS:
                time.sleep(_retry_delay(attempt))
                continue
            raise ProductionCallNotTerminated(call_id, errors) from exc
        requested = True
        break
    if not requested:
        raise ProductionCallNotTerminated(call_id, errors)

    confirmation_seconds = _cancel_confirmation_seconds()
    try:
        call.get(timeout=confirmation_seconds)
    except TimeoutError as exc:
        errors.append(f"terminal confirmation timed out after {confirmation_seconds:.3f}s")
        raise ProductionCallNotTerminated(call_id, errors) from exc
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        if not _cancellation_confirmation_is_terminal(exc):
            errors.append(f"confirmation {type(exc).__name__}: {exc}")
            raise ProductionCallNotTerminated(call_id, errors) from exc
''',
)

# P1: vision producer lifecycle is keyed by Modal FunctionCall, never warm container.
replace_once(
    "scripts/modal_execution_spy.py",
    '''        compact = self._compact_event(payload)
        with self.lock:
''',
    '''        compact = self._compact_event(payload)
        if app != self.pipeline_app and str(compact.get("event") or "").startswith("vision_"):
            function_call_id = self._function_call_id(line)
            if function_call_id:
                compact["invocation_id"] = function_call_id
        with self.lock:
''',
)
replace_once(
    "scripts/modal_execution_spy.py",
    '''            else:
                self._diagnostic_event_counts[name] = self._diagnostic_event_counts.get(name, 0) + 1
                lifecycle_id = str(compact.get("worker_lifecycle_id") or "")
                if self._terminal_event is None and name == "vision_generation_start":
                    if lifecycle_id:
                        self._active_vision_generations[lifecycle_id] = (
                            task,
                            self._positive_int(compact.get("attempt")),
                            self._positive_int(compact.get("frames")),
                            observed_at,
                        )
                elif (
                    name in {"vision_generation_complete", "vision_inference_error"}
                    and lifecycle_id
                ):
                    self._active_vision_generations.pop(lifecycle_id, None)
''',
    '''            else:
                self._diagnostic_event_counts[name] = self._diagnostic_event_counts.get(name, 0) + 1
                invocation_id = str(compact.get("invocation_id") or "")
                if self._terminal_event is None and name == "vision_generation_start":
                    if not invocation_id:
                        violation = f"vision producer start omitted function-call identity: {compact}"
                    elif invocation_id in self._active_vision_generations:
                        violation = f"vision producer repeated active invocation ID: {compact}"
                    else:
                        self._active_vision_generations[invocation_id] = (
                            task,
                            self._positive_int(compact.get("attempt")),
                            self._positive_int(compact.get("frames")),
                            observed_at,
                        )
                elif name in {"vision_generation_complete", "vision_inference_error"}:
                    if not invocation_id:
                        violation = f"vision producer terminal omitted function-call identity: {compact}"
                    else:
                        active = self._active_vision_generations.get(invocation_id)
                        if active is None:
                            violation = f"vision producer terminal has no matching start: {compact}"
                        elif active[0] != task:
                            violation = (
                                "vision producer terminal task does not match start: "
                                f"started={active[0]!r} event={compact}"
                            )
                        else:
                            self._active_vision_generations.pop(invocation_id, None)
''',
)
replace_once(
    "scripts/modal_execution_spy.py",
    '''        stalled = [
            (lifecycle_id, task, attempt, frames, now - started)
            for lifecycle_id, (task, attempt, frames, started) in active
            if now - started >= self.vision_stall_seconds
        ]
        if stalled:
            lifecycle_id, task, attempt, frames, elapsed = max(stalled, key=lambda item: item[4])
            self._set_abort(
                "vision generation made no terminal progress before watchdog deadline",
                {
                    "event": "vision_generation_stall",
                    "worker_lifecycle_id": lifecycle_id,
''',
    '''        stalled = [
            (invocation_id, task, attempt, frames, now - started)
            for invocation_id, (task, attempt, frames, started) in active
            if now - started >= self.vision_stall_seconds
        ]
        if stalled:
            invocation_id, task, attempt, frames, elapsed = max(
                stalled, key=lambda item: item[4]
            )
            self._set_abort(
                "vision generation made no terminal progress before watchdog deadline",
                {
                    "event": "vision_generation_stall",
                    "invocation_id": invocation_id,
''',
)
replace_once(
    "scripts/modal_execution_spy.py",
    '''            active_vision = [
                {
                    "worker_lifecycle_id": lifecycle_id,
                    "task": task,
                    "attempt": attempt,
                    "frames": frames,
                }
                for lifecycle_id, (task, attempt, frames, _started) in sorted(
                    self._active_vision_generations.items()
                )
            ]
''',
    '''            active_vision = [
                {
                    "invocation_id": invocation_id,
                    "task": task,
                    "attempt": attempt,
                    "frames": frames,
                }
                for invocation_id, (task, attempt, frames, _started) in sorted(
                    self._active_vision_generations.items()
                )
            ]
''',
)

# P2: explicit unknown=allow means unlisted/unknown source segments are passable.
replace_once(
    "src/clipper/quality_pipeline.py",
    '''        if classification in segment_policy.allow:
            continue
        buffer = (
''',
    '''        if classification in segment_policy.allow or (
            classification not in segment_policy.forbid and segment_policy.unknown == "allow"
        ):
            continue
        buffer = (
''',
)

# CI coverage must fail independently if pytest-cov metadata is misleading.
replace_once(
    ".github/workflows/ci.yml",
    "      - run: pytest\n",
    '''      - run: pytest
      - name: Enforce measured package coverage floor
        run: coverage report --fail-under=95
''',
)

# Modal cancellation + resume provenance tests.
replace_once(
    "tests/test_modal_execution.py",
    '''    _acquire_remote_source,
    _BudgetLedger,
    _class,
''',
    '''    _acquire_remote_source,
    _BudgetLedger,
    _cancel_confirmation_seconds,
    _cancel_remote_call,
    _class,
''',
)
replace_once(
    "tests/test_modal_execution.py",
    '''    _local_git_sha,
    _materialize_remote_run,
''',
    '''    _load_reviewed_resume_provenance,
    _local_git_sha,
    _materialize_remote_run,
''',
)
replace_once(
    "tests/test_modal_execution.py",
    '''class ServiceError(RuntimeError):
    pass


@pytest.fixture''',
    '''class ServiceError(RuntimeError):
    pass


class InputCancellation(RuntimeError):
    pass


@pytest.fixture''',
)
replace_once(
    "tests/test_modal_execution.py",
    '''        call = Mock()
        call.get.side_effect = RuntimeError(f"{label} blocked")
''',
    '''        call = Mock()
        call.get.side_effect = [RuntimeError(f"{label} blocked"), InputCancellation("cancelled")]
''',
)
replace_once(
    "tests/test_modal_execution.py",
    '''        def get(self, *, timeout: float) -> object:
            assert 0.0 < timeout < 1.0
            clock["now"] = 1.0
            raise TimeoutError
''',
    '''        def get(self, *, timeout: float) -> object:
            if self.cancelled:
                raise InputCancellation("cancelled")
            assert 0.0 < timeout < 1.0
            clock["now"] = 1.0
            raise TimeoutError
''',
)
modal_tests = r'''


def test_cancel_remote_call_requires_exact_terminal_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "7")
    call = Mock()
    call.object_id = "fc-confirmed"
    call.get.side_effect = InputCancellation("cancelled")
    _cancel_remote_call(call)
    call.cancel.assert_called_once_with(terminate_containers=False)
    call.get.assert_called_once_with(timeout=7.0)


def test_cancel_remote_call_rejects_timeout_and_transport_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "2")
    for error in (TimeoutError(), ServiceError("control plane unavailable")):
        call = Mock()
        call.object_id = "fc-unconfirmed"
        call.get.side_effect = error
        with pytest.raises(ProductionCallNotTerminated, match="terminal state"):
            _cancel_remote_call(call)
        call.cancel.assert_called_once_with(terminate_containers=False)


def test_cancel_remote_call_retries_cancel_request_and_validates_confirmation_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "3")
    call = Mock()
    call.object_id = "fc-retry"
    call.cancel.side_effect = [ServiceError("retry"), None]
    call.get.side_effect = InputCancellation("cancelled")
    with patch("clipper.modal_execution.time.sleep") as sleep:
        _cancel_remote_call(call)
    assert call.cancel.call_count == 2
    sleep.assert_called_once_with(2.0)
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "nan")
    with pytest.raises(ValueError, match="finite and positive"):
        _cancel_confirmation_seconds()


def _valid_resume_registry(tmp_path: Path) -> tuple[Path, dict[str, object], list[VideoCandidate]]:
    brief_path = tmp_path / "brief.yaml"
    brief_path.write_text("campaign: reviewed\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(brief_path.read_bytes()).hexdigest()
    record: dict[str, object] = {
        "schema_version": "clipper-resume-provenance-v1",
        "workflow_run_id": "123",
        "campaign_id": "campaign",
        "campaign_brief_sha256": digest,
        "artifact_run_path": "/run-123",
        "artifact_origin_workflow_run_id": "122",
        "artifact_origin_head_sha": "a" * 40,
        "cache_root": "/artifacts/_cache",
        "source_hashes": {"v1": "B" * 64},
        "target_video_id": "v1",
    }
    registry: dict[str, object] = {
        "schema_version": "clipper-resume-provenance-registry-v1",
        "records": {"123": record},
    }
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    (acceptance / "resume-provenance.json").write_text(json.dumps(registry), encoding="utf-8")
    candidates = [VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")]
    return brief_path, registry, candidates


def _write_resume_registry(tmp_path: Path, registry: dict[str, object]) -> None:
    (tmp_path / "acceptance" / "resume-provenance.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )


def test_reviewed_resume_provenance_normalizes_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path, _registry, candidates = _valid_resume_registry(tmp_path)
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    result = _load_reviewed_resume_provenance(
        requested_run_id="123", brief_path=brief_path, campaign_id="campaign", candidates=candidates
    )
    assert result is not None
    assert result["source_hashes"] == {"v1": "b" * 64}


def test_reviewed_resume_provenance_rejects_missing_or_malformed_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path = tmp_path / "brief.yaml"
    brief_path.write_text("campaign: reviewed\n", encoding="utf-8")
    candidates = [VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")]
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="valid reviewed provenance registry"):
        _load_reviewed_resume_provenance(
            requested_run_id="123", brief_path=brief_path, campaign_id="campaign", candidates=candidates
        )
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    (acceptance / "resume-provenance.json").write_text("{bad-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="valid reviewed provenance registry"):
        _load_reviewed_resume_provenance(
            requested_run_id="123", brief_path=brief_path, campaign_id="campaign", candidates=candidates
        )


def test_reviewed_resume_provenance_rejects_identity_and_compatibility_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path, base, candidates = _valid_resume_registry(tmp_path)
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    cases = [
        (("schema_version",), "bad", "registry schema"),
        (("records",), [], "records must be an object"),
        (("records",), {}, "no reviewed compatible"),
        (("record", "schema_version"), "bad", "record schema"),
        (("record", "workflow_run_id"), "other", "does not match workflow run ID"),
        (("record", "campaign_id"), "other", "campaign does not match"),
        (("record", "campaign_brief_sha256"), "0" * 64, "brief digest"),
        (("record", "artifact_run_path"), "run-123", "artifact path"),
        (("record", "artifact_origin_workflow_run_id"), "", "origin workflow run ID"),
        (("record", "artifact_origin_head_sha"), "bad", "origin SHA"),
        (("record", "cache_root"), "/wrong", "cache root"),
        (("record", "source_hashes"), [], "source hashes must be an object"),
        (("record", "source_hashes"), {"other": "b" * 64}, "source identities"),
        (("record", "source_hashes"), {"v1": "bad"}, "invalid source hash"),
        (("record", "target_video_id"), "other", "target does not match"),
    ]
    for path, value, message in cases:
        registry = __import__("json").loads(__import__("json").dumps(base))
        if path[0] == "record":
            registry["records"]["123"][path[1]] = value
        else:
            registry[path[0]] = value
        _write_resume_registry(tmp_path, registry)
        with pytest.raises(RuntimeError, match=message):
            _load_reviewed_resume_provenance(
                requested_run_id="123",
                brief_path=brief_path,
                campaign_id="campaign",
                candidates=candidates,
            )
'''
Path("tests/test_modal_execution.py").write_text(
    Path("tests/test_modal_execution.py").read_text(encoding="utf-8") + modal_tests,
    encoding="utf-8",
)

# Vision lifecycle tests: same warm container may host concurrent distinct calls.
replace_once(
    "tests/test_modal_execution_spy.py",
    '''        "clipper-open-editor",
        '{"event":"vision_generation_start","execution_id":"exec-123",'
''',
    '''        "clipper-open-editor",
        "2026-09-11T13:00:00Z fc-VISION1 "
        '{"event":"vision_generation_start","execution_id":"exec-123",'
''',
)
replace_once(
    "tests/test_modal_execution_spy.py",
    '''        "clipper-open-editor",
        '{"event":"vision_inference_error","execution_id":"exec-123",'
''',
    '''        "clipper-open-editor",
        "2026-09-11T13:00:01Z fc-VISION1 "
        '{"event":"vision_inference_error","execution_id":"exec-123",'
''',
)
spy_tests = r'''


def test_spy_keeps_parallel_generations_from_one_warm_container_distinct(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "vision-call-identity.ndjson",
        execution_id="exec-123",
    )
    for call_id in ("fc-VISIONA", "fc-VISIONB"):
        spy._record(
            "clipper-open-editor",
            f'2026-09-11T13:00:00Z {call_id} '
            '{"event":"vision_generation_start","execution_id":"exec-123",'
            '"worker_lifecycle_id":"same-container","task":"source_policy_visual_scout",'
            '"attempt":1,"frames":8}',
        )
    active = spy.summary()["active_vision_generations"]
    assert [item["invocation_id"] for item in active] == ["fc-VISIONA", "fc-VISIONB"]
    spy._record(
        "clipper-open-editor",
        '2026-09-11T13:00:01Z fc-VISIONA '
        '{"event":"vision_generation_complete","execution_id":"exec-123",'
        '"worker_lifecycle_id":"same-container","task":"source_policy_visual_scout"}',
    )
    assert [item["invocation_id"] for item in spy.summary()["active_vision_generations"]] == [
        "fc-VISIONB"
    ]


def test_spy_rejects_duplicate_or_unmatched_vision_lifecycle(tmp_path: Path) -> None:
    module = _module()
    duplicate = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "vision-duplicate.ndjson",
        execution_id="exec-123",
    )
    line = (
        '2026-09-11T13:00:00Z fc-VISION1 '
        '{"event":"vision_generation_start","execution_id":"exec-123",'
        '"worker_lifecycle_id":"container","task":"source_policy_visual_scout",'
        '"attempt":1,"frames":8}'
    )
    duplicate._record("clipper-open-editor", line)
    duplicate._record("clipper-open-editor", line)
    assert duplicate.abort_reason is not None
    assert "repeated active invocation" in duplicate.abort_reason
    unmatched = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "vision-unmatched.ndjson",
        execution_id="exec-123",
    )
    unmatched._record(
        "clipper-open-editor",
        '2026-09-11T13:00:01Z fc-VISION2 '
        '{"event":"vision_inference_error","execution_id":"exec-123",'
        '"worker_lifecycle_id":"container","task":"source_policy_visual_scout",'
        '"error_type":"InputCancellation"}',
    )
    assert unmatched.abort_reason is not None
    assert "no matching start" in unmatched.abort_reason
'''
Path("tests/test_modal_execution_spy.py").write_text(
    Path("tests/test_modal_execution_spy.py").read_text(encoding="utf-8") + spy_tests,
    encoding="utf-8",
)

# Unknown policy tests.
replace_once(
    "tests/test_coverage_floor_contract_edges.py",
    "from clipper.dag import DagStore\n",
    "from clipper.dag import DagStore\nfrom clipper.editorial_integrity import HazardClassification, SourceHazardSegment\n",
)
replace_once(
    "tests/test_coverage_floor_contract_edges.py",
    "from clipper.quality_batch import RecordingEditorialProvider, plan_quality_batch\n",
    "from clipper.quality_batch import RecordingEditorialProvider, plan_quality_batch\nfrom clipper.quality_pipeline import forbidden_spans_for_campaign\n",
)
quality_tests = r'''


def test_unknown_hazard_allow_excludes_unlisted_spans_but_preserves_explicit_forbid() -> None:
    brief = CampaignBrief.from_dict(
        {
            "campaign_id": "unknown-allow",
            "title": "Unknown allow",
            "objective": "Honor source segment policy",
            "allowed_video_ids": ["video"],
            "rights_confirmed": True,
            "min_clip_seconds": 8,
            "max_clip_seconds": 20,
            "acceptance_policy": {
                "enabled": True,
                "source_segments": {
                    "allow": ["editorial_content"],
                    "forbid": ["advertisement"],
                    "unknown": "allow",
                    "safety_buffer_seconds": 1,
                },
            },
        }
    )

    def hazard(kind: HazardClassification, start: float) -> SourceHazardSegment:
        return SourceHazardSegment(
            start=start,
            end=start + 2,
            classification=kind,
            confidence=0.9,
            evidence=("grounded",),
            model_identity={"model": "test"},
        )

    spans = forbidden_spans_for_campaign(
        brief,
        (
            hazard(HazardClassification.UNKNOWN, 2),
            hazard(HazardClassification.GRAPHIC_HEAVY, 6),
            hazard(HazardClassification.ADVERTISEMENT, 10),
        ),
        (),
    )
    assert [(item.start, item.end) for item in spans] == [(9.0, 13.0)]


def test_unknown_hazard_default_escalation_remains_non_passable() -> None:
    brief = _acceptance_brief()
    hazard = SourceHazardSegment(
        start=2,
        end=4,
        classification=HazardClassification.UNKNOWN,
        confidence=0.5,
        evidence=("uncertain",),
        model_identity={"model": "test"},
    )
    spans = forbidden_spans_for_campaign(brief, (hazard,), ())
    assert [(item.start, item.end) for item in spans] == [(2.0, 4.0)]
'''
Path("tests/test_coverage_floor_contract_edges.py").write_text(
    Path("tests/test_coverage_floor_contract_edges.py").read_text(encoding="utf-8") + quality_tests,
    encoding="utf-8",
)

# Existing uncovered semantic brief fail-closed branches become explicit contract tests.
brief_tests = r'''


def test_brief_fail_closed_validation_matrix_covers_malformed_production_contracts(
    tmp_path: Path,
) -> None:
    def write_and_reject(name: str, value: object, message: str) -> None:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises((BriefValidationError, FileNotFoundError), match=message):
            load_brief(path)

    write_and_reject("root-list", [], "brief root must be an object")

    def clone() -> dict[str, object]:
        return json.loads(json.dumps(DATA))

    case = clone()
    case["targets"]["videos"][0]["video_id"] = "REPLACE_WITH_VIDEO"
    write_and_reject("placeholder-target", case, "example placeholder")
    case = clone()
    case["rights"]["authorized_channels"] = ["UC_REPLACE_CHANNEL"]
    write_and_reject("placeholder-rights", case, "example placeholder")
    case = clone()
    case["targets"] = None
    write_and_reject("missing-targets", case, "targets.mode=explicit")
    case = clone()
    case["targets"]["extra"] = True
    write_and_reject("unknown-target-rule", case, "unsupported targets rule")
    case = clone()
    case["targets"]["mode"] = "discover"
    write_and_reject("bad-target-mode", case, "targets.mode must be explicit")
    case = clone()
    case["targets"]["videos"] = []
    write_and_reject("empty-videos", case, "at least one explicit video")
    case = clone()
    case["targets"]["videos"] = ["not-an-object"]
    write_and_reject("video-shape", case, "must be an object")
    case = clone()
    case["targets"]["videos"][0]["extra"] = True
    write_and_reject("unknown-video-rule", case, "unsupported targets.videos rule")
    case = clone()
    case["targets"]["videos"][0]["video_id"] = ""
    write_and_reject("empty-video-id", case, "must be a non-empty string")
    case = clone()
    case["targets"]["videos"][0]["url"] = "http://example.test/video"
    write_and_reject("non-https-url", case, "must use https")
    case = clone()
    case["targets"]["videos"].append(dict(case["targets"]["videos"][0]))
    write_and_reject("duplicate-video", case, "duplicate video IDs")
    case = clone()
    case["rights"] = None
    write_and_reject("missing-rights", case, "requires a rights object")
    case = clone()
    case["rights"]["authorized_channels"] = "UC123"
    write_and_reject("bad-authorized-list", case, "must be a list of strings")
    case = clone()
    case["rights"]["authorized_channels"] = ["UC_OTHER"]
    write_and_reject("unauthorized-target", case, "outside rights.authorized_channels")
    case = clone()
    case["rights"]["extra"] = True
    write_and_reject("unknown-rights-rule", case, "unsupported rights rule")
    case = clone()
    case["rights"]["confirmed"] = "yes"
    write_and_reject("non-bool-rights", case, "must be true or false")
    case = clone()
    case["content_constraints"] = "bad"
    write_and_reject("bad-constraints", case, "content_constraints must be an object")
    case = clone()
    case["content_constraints"] = {"extra": 1}
    write_and_reject("unknown-constraint", case, "unsupported content_constraints rule")
    case = clone()
    case["acceptance_policy"] = {
        "generated_media": {"synthetic_visuals": "forbid", "extra": "bad"}
    }
    write_and_reject(
        "generated-media-rule", case, "unsupported acceptance_policy.generated_media"
    )
'''
Path("tests/test_brief.py").write_text(
    Path("tests/test_brief.py").read_text(encoding="utf-8") + brief_tests,
    encoding="utf-8",
)
