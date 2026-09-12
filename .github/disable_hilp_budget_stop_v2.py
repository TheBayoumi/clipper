from pathlib import Path


execution_path = Path("src/clipper/modal_execution.py")
source = execution_path.read_text(encoding="utf-8")

old = '''def _spawn_recoverable_modal_call(
    function: Any,
    request: dict[str, Any],
    *,
    budget: _BudgetLedger,
    gpu_count: float,
    estimated_usd_per_second: float,
) -> tuple[Any, float, RuntimeError | None]:
'''
new = '''def _spawn_recoverable_modal_call(
    function: Any,
    request: dict[str, Any],
    *,
    budget: _BudgetLedger,
    gpu_count: float,
    estimated_usd_per_second: float,
    enforce_budget: bool = True,
) -> tuple[Any, float, RuntimeError | None]:
'''
if old not in source:
    raise SystemExit("spawn signature not found")
source = source.replace(old, new, 1)

old = '''    remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
    started = time.monotonic()
    if remaining_gpu_seconds <= 0 or remaining_estimated_usd <= 0:
        return (
            call,
            started,
            ProductionBudgetExceeded(
                "production budget exhausted before Modal producer input attachment: "
                f"gpu_seconds={budget.gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                f"estimated_usd={budget.estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
            ),
        )
'''
new = '''    started = time.monotonic()
    if enforce_budget:
        remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
        if remaining_gpu_seconds <= 0 or remaining_estimated_usd <= 0:
            return (
                call,
                started,
                ProductionBudgetExceeded(
                    "production budget exhausted before Modal producer input attachment: "
                    f"gpu_seconds={budget.gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                    f"estimated_usd={budget.estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                ),
            )
'''
if old not in source:
    raise SystemExit("spawn pre-attachment budget gate not found")
source = source.replace(old, new, 1)

old = '''        remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
        if remaining_gpu_seconds <= 0 or remaining_estimated_usd <= 0:
            raise ProductionBudgetExceeded(
                "production budget exhausted at Modal producer input attachment boundary: "
                f"gpu_seconds={budget.gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                f"estimated_usd={budget.estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
            )
'''
new = '''        if enforce_budget:
            remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
            if remaining_gpu_seconds <= 0 or remaining_estimated_usd <= 0:
                raise ProductionBudgetExceeded(
                    "production budget exhausted at Modal producer input attachment boundary: "
                    f"gpu_seconds={budget.gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                    f"estimated_usd={budget.estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                )
'''
if old not in source:
    raise SystemExit("spawn attachment-boundary budget gate not found")
source = source.replace(old, new, 1)

old = '''def _invoke_remote_with_budget(
    function: Any,
    request: dict[str, Any],
    *,
    max_gpu_seconds: float | None = None,
    max_estimated_usd: float | None = None,
    budget: _BudgetLedger | None = None,
    gpu_count: float = _MODAL_ROOT_GPU_COUNT,
    estimated_usd_per_second: float = _MODAL_ROOT_ESTIMATED_USD_PER_SECOND,
) -> object:
'''
new = '''def _invoke_remote_with_budget(
    function: Any,
    request: dict[str, Any],
    *,
    max_gpu_seconds: float | None = None,
    max_estimated_usd: float | None = None,
    budget: _BudgetLedger | None = None,
    gpu_count: float = _MODAL_ROOT_GPU_COUNT,
    estimated_usd_per_second: float = _MODAL_ROOT_ESTIMATED_USD_PER_SECOND,
    enforce_budget: bool = True,
) -> object:
'''
if old not in source:
    raise SystemExit("invoke signature not found")
source = source.replace(old, new, 1)

old = '''    remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
    if remaining_gpu_seconds <= 0 or remaining_estimated_usd <= 0:
        raise ProductionBudgetExceeded(
            "production budget exhausted before recoverable Modal call allocation"
        )
    call, started, submission_error = _spawn_recoverable_modal_call(
        function,
        request,
        budget=budget,
        gpu_count=gpu_count,
        estimated_usd_per_second=estimated_usd_per_second,
    )
'''
new = '''    if enforce_budget:
        remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
        if remaining_gpu_seconds <= 0 or remaining_estimated_usd <= 0:
            raise ProductionBudgetExceeded(
                "production budget exhausted before recoverable Modal call allocation"
            )
    call, started, submission_error = _spawn_recoverable_modal_call(
        function,
        request,
        budget=budget,
        gpu_count=gpu_count,
        estimated_usd_per_second=estimated_usd_per_second,
        enforce_budget=enforce_budget,
    )
'''
if old not in source:
    raise SystemExit("invoke allocation budget gate not found")
source = source.replace(old, new, 1)

old = '''        while True:
            gpu_seconds, estimated_usd = budget_usage()
            remaining_seconds = remaining_budget_wall_seconds()
            if remaining_seconds <= 0:
                raise ProductionBudgetExceeded(
                    "CLI production call exceeded its in-flight compute budget: "
                    f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                    f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                )
            try:
                result = call.get(timeout=min(poll_seconds, remaining_seconds))
            except TimeoutError:
                continue
            terminal_result = True
            charge_elapsed()
            gpu_seconds = budget.gpu_seconds
            estimated_usd = budget.estimated_usd
            if gpu_seconds > budget.max_gpu_seconds or estimated_usd > budget.max_estimated_usd:
                raise ProductionBudgetExceeded(
                    "CLI production call exceeded its compute budget on the final poll: "
                    f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                    f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                )
            return result
'''
new = '''        while True:
            timeout = poll_seconds
            if enforce_budget:
                gpu_seconds, estimated_usd = budget_usage()
                remaining_seconds = remaining_budget_wall_seconds()
                if remaining_seconds <= 0:
                    raise ProductionBudgetExceeded(
                        "CLI production call exceeded its in-flight compute budget: "
                        f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                        f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                    )
                timeout = min(poll_seconds, remaining_seconds)
            try:
                result = call.get(timeout=timeout)
            except TimeoutError:
                continue
            terminal_result = True
            charge_elapsed()
            if enforce_budget:
                gpu_seconds = budget.gpu_seconds
                estimated_usd = budget.estimated_usd
                if gpu_seconds > budget.max_gpu_seconds or estimated_usd > budget.max_estimated_usd:
                    raise ProductionBudgetExceeded(
                        "CLI production call exceeded its compute budget on the final poll: "
                        f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                        f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                    )
            return result
'''
if old not in source:
    raise SystemExit("invoke live budget gate not found")
source = source.replace(old, new, 1)

old = '''            gpu_seconds = budget.gpu_seconds
            estimated_usd = budget.estimated_usd
            if gpu_seconds > budget.max_gpu_seconds or estimated_usd > budget.max_estimated_usd:
                raise ProductionBudgetExceeded(
                    "CLI production call exceeded its compute budget through cancellation "
                    "acknowledgement: "
                    f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                    f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                )
'''
new = '''            if enforce_budget:
                gpu_seconds = budget.gpu_seconds
                estimated_usd = budget.estimated_usd
                if gpu_seconds > budget.max_gpu_seconds or estimated_usd > budget.max_estimated_usd:
                    raise ProductionBudgetExceeded(
                        "CLI production call exceeded its compute budget through cancellation "
                        "acknowledgement: "
                        f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                        f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                    )
'''
if old not in source:
    raise SystemExit("invoke cancellation budget gate not found")
source = source.replace(old, new, 1)

old = '''def _acquire_remote_source(
    function: Any,
    candidate: VideoCandidate,
    *,
    expected_git_sha: str,
    budget: _BudgetLedger | None = None,
    attempt_evidence: list[dict[str, object]] | None = None,
    execution_id: str | None = None,
) -> dict[str, Any]:
'''
new = '''def _acquire_remote_source(
    function: Any,
    candidate: VideoCandidate,
    *,
    expected_git_sha: str,
    budget: _BudgetLedger | None = None,
    attempt_evidence: list[dict[str, object]] | None = None,
    execution_id: str | None = None,
    enforce_budget: bool = True,
) -> dict[str, Any]:
'''
if old not in source:
    raise SystemExit("acquire signature not found")
source = source.replace(old, new, 1)

old = '''                estimated_usd_per_second=_MODAL_ACQUISITION_ESTIMATED_USD_PER_SECOND,
            )
'''
new = '''                estimated_usd_per_second=_MODAL_ACQUISITION_ESTIMATED_USD_PER_SECOND,
                enforce_budget=enforce_budget,
            )
'''
if source.count(old) < 1:
    raise SystemExit("acquire invoke call not found")
# This exact call shape occurs first in _acquire_remote_source on the current head.
source = source.replace(old, new, 1)
execution_path.write_text(source, encoding="utf-8")

watchdog_path = Path("scripts/modal_hilp_watchdog.py")
watchdog = watchdog_path.read_text(encoding="utf-8")
old = '''                attempt_evidence=attempts,
                execution_id=execution_id,
            )
'''
new = '''                attempt_evidence=attempts,
                execution_id=execution_id,
                enforce_budget=False,
            )
'''
if old not in watchdog:
    raise SystemExit("watchdog source acquisition call not found")
watchdog = watchdog.replace(old, new, 1)
old = '''            estimated_usd_per_second=0.000444,
        )
'''
new = '''            estimated_usd_per_second=0.000444,
            enforce_budget=False,
        )
'''
if old not in watchdog:
    raise SystemExit("watchdog root spawn call not found")
watchdog = watchdog.replace(old, new, 1)
watchdog_path.write_text(watchdog, encoding="utf-8")

test_path = Path("tests/test_modal_execution.py")
tests = test_path.read_text(encoding="utf-8")
marker = "def test_hilp_advisory_budget_opt_out_allows_exhausted_recoverable_submission"
if marker not in tests:
    tests += r'''


def test_hilp_advisory_budget_opt_out_allows_exhausted_recoverable_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    map_response = SimpleNamespace(
        function_call_id="fc-advisory",
        pipelined_inputs=[],
    )
    put_response = SimpleNamespace(inputs=[SimpleNamespace(input_id="in-1")])
    modal, async_utils, function_utils, api_pb2, stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=put_response,
    )

    async def slow_serialize(*_args: object, **_kwargs: object) -> object:
        clock["now"] = 2.0
        return "serialized-input"

    function_utils._create_input.side_effect = slow_serialize
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )
    budget = _BudgetLedger(1.0, 100.0)

    call, _started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=budget,
        gpu_count=1.0,
        estimated_usd_per_second=0.0,
        enforce_budget=False,
    )

    assert call.object_id == "fc-recoverable"
    assert error is None
    assert events == ["from-id", "put-input"]
    stub.FunctionPutInputs.assert_awaited_once()
    assert budget.gpu_seconds > budget.max_gpu_seconds


def test_hilp_advisory_budget_opt_out_does_not_cap_or_cancel_remote_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5")

    class AdvisoryCall:
        def __init__(self) -> None:
            self.timeouts: list[float] = []
            self.cancel_args: list[bool] = []

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            self.timeouts.append(timeout)
            if len(self.timeouts) == 1:
                clock["now"] = 3.0
                raise TimeoutError
            clock["now"] = 4.0
            return {"ok": True}

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_args.append(terminate_containers)

    call = AdvisoryCall()
    function = SimpleNamespace(spawn=Mock(return_value=call))
    budget = _BudgetLedger(1.0, 1.0)
    budget.gpu_seconds = 1.0

    result = _invoke_remote_with_budget(
        function,
        {"request": True},
        budget=budget,
        gpu_count=1.0,
        estimated_usd_per_second=0.0,
        enforce_budget=False,
    )

    assert result == {"ok": True}
    assert call.timeouts == [pytest.approx(5.0), pytest.approx(5.0)]
    assert call.cancel_args == []
    assert budget.gpu_seconds > budget.max_gpu_seconds


def test_source_acquisition_forwards_advisory_budget_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = VideoCandidate(
        "video",
        "title",
        "channel",
        "channel-title",
        "https://www.youtube.com/watch?v=video",
    )
    function = Mock()
    variant = Mock()
    function.with_options.return_value = variant
    observed: list[bool] = []

    def invoke(_function: object, _payload: object, **kwargs: object) -> object:
        observed.append(bool(kwargs.get("enforce_budget")))
        return {
            "video_id": "video",
            "channel_id": "channel",
            "quality_policy": "highest_available_no_transcode",
            "bytes": 1,
            "sha256": "a" * 64,
            "volume_path": "/inputs/video/master.mp4",
        }

    monkeypatch.setattr("clipper.modal_execution._invoke_remote_with_budget", invoke)
    result = _acquire_remote_source(
        function,
        candidate,
        expected_git_sha="b" * 40,
        budget=_BudgetLedger(1.0, 1.0),
        enforce_budget=False,
    )

    assert result["video_id"] == "video"
    assert observed == [False]
'''
    test_path.write_text(tests, encoding="utf-8")

watchdog_test_path = Path("tests/test_modal_hilp_watchdog.py")
watchdog_tests = watchdog_test_path.read_text(encoding="utf-8")
marker = "def test_watchdog_disables_budget_enforcement_for_source_and_root_calls"
if marker not in watchdog_tests:
    watchdog_tests += r'''


def test_watchdog_disables_budget_enforcement_for_source_and_root_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    source_modes: list[bool] = []
    root_modes: list[bool] = []

    def acquire(*_args: object, **kwargs: object) -> dict[str, object]:
        source_modes.append(bool(kwargs.get("enforce_budget")))
        return {
            "video_id": "video",
            "channel_id": "channel",
            "quality_policy": "highest_available_no_transcode",
            "sha256": "a" * 64,
            "volume_path": "/inputs/video/master.mp4",
        }

    class Call(_Call):
        def get(self, *, timeout: float):
            return {
                "status": "PASS",
                "execution_mode": "resume",
                "execution_id": "e" * 32,
                "deployed_git_sha": "a" * 40,
                "pipeline_status": "SUCCESS",
                "review_status": "NOT_RENDERED",
                "run_volume": "volume",
                "run_path": "/run",
            }

    call = Call({})

    def spawn(*_args: object, **kwargs: object):
        root_modes.append(bool(kwargs.get("enforce_budget")))
        return call, module.time.monotonic(), None

    monkeypatch.setattr(module, "_acquire_remote_source", acquire)
    monkeypatch.setattr(module, "_spawn_recoverable_modal_call", spawn)
    monkeypatch.setattr(module, "_cached_source_evidence", lambda: None)
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    result = module.run(render=False)

    assert result["status"] == "PASS"
    assert source_modes == [False]
    assert root_modes == [False]
'''
    watchdog_test_path.write_text(watchdog_tests, encoding="utf-8")
