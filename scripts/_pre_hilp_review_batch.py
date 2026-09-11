from __future__ import annotations

from pathlib import Path


WATCHDOG = Path("scripts/modal_hilp_watchdog.py")
TESTS = Path("tests/test_modal_hilp_watchdog.py")
CONTRACT_TEST = Path("tests/test_pre_hilp_runtime_trigger_contract.py")


def replace_between(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


def patch_watchdog() -> None:
    text = WATCHDOG.read_text(encoding="utf-8")
    variable_anchor = "    cancellation_failure_reason: str | None = None\n    run_succeeded = False\n"
    if text.count(variable_anchor) != 1:
        raise RuntimeError("watchdog cancellation state anchor drifted")
    text = text.replace(
        variable_anchor,
        "    cancellation_failure_reason: str | None = None\n"
        "    root_cancel_terminal_confirmed = False\n"
        "    root_hard_termination_succeeded = False\n"
        "    run_succeeded = False\n",
        1,
    )

    new_cancel = '''    def cancel_call(reason: str) -> None:
        nonlocal cancellation_failure_reason
        nonlocal root_cancel_terminal_confirmed
        nonlocal root_hard_termination_succeeded
        if call is None or cancelled.is_set():
            return

        errors: list[str] = []

        def request_cancel(*, terminate_containers: bool, phase: str) -> bool:
            for attempt in range(1, 4):
                try:
                    call.cancel(terminate_containers=terminate_containers)
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                        raise
                    errors.append(f"{phase} {type(exc).__name__}: {exc}")
                    print(
                        json.dumps(
                            {
                                "event": "production_call_cancel_retry",
                                "function_call_id": call_id,
                                "reason": reason,
                                "phase": phase,
                                "attempt": attempt,
                                "terminate_containers": terminate_containers,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if attempt < 3:
                        time.sleep(0.25 * attempt)
                    continue
                print(
                    json.dumps(
                        {
                            "event": "production_call_cancel_requested",
                            "function_call_id": call_id,
                            "reason": reason,
                            "phase": phase,
                            "attempt": attempt,
                            "terminate_containers": terminate_containers,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return True
            return False

        def confirm_terminal(*, phase: str) -> str | None:
            nonlocal root_cancel_terminal_confirmed
            try:
                call.get(timeout=cancel_confirmation_seconds)
            except TimeoutError:
                errors.append(
                    f"{phase} terminal confirmation timed out after "
                    f"{cancel_confirmation_seconds:.3f}s"
                )
                return None
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                if not _cancellation_confirmation_is_terminal(modal, exc):
                    errors.append(f"{phase} confirmation {type(exc).__name__}: {exc}")
                    return None
                root_cancel_terminal_confirmed = True
                return type(exc).__name__
            root_cancel_terminal_confirmed = True
            return "result"

        soft_requested = request_cancel(
            terminate_containers=False,
            phase="soft_cancel",
        )
        confirmed_by = confirm_terminal(phase="soft_cancel") if soft_requested else None
        if confirmed_by is not None:
            cancelled.set()
            cancellation_failure_reason = None
            print(
                json.dumps(
                    {
                        "event": "production_call_cancelled",
                        "function_call_id": call_id,
                        "reason": reason,
                        "terminal_confirmation": confirmed_by,
                        "hard_termination": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

        print(
            json.dumps(
                {
                    "event": "production_call_cancel_escalated",
                    "function_call_id": call_id,
                    "reason": reason,
                    "errors": list(errors),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        hard_requested = request_cancel(
            terminate_containers=True,
            phase="hard_cancel",
        )
        if not hard_requested:
            failure = ProductionCallNotTerminated(call_id, errors)
            cancellation_failure_reason = str(failure)
            print(
                json.dumps(
                    {
                        "event": "production_call_cancel_unconfirmed",
                        "function_call_id": call_id,
                        "reason": reason,
                        "error": str(failure),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise failure

        # Modal documents terminate_containers=True as terminating the containers
        # running this FunctionCall's cancelled inputs; concurrently affected inputs
        # are rescheduled. Treat the successful exact-call hard-cancel request as the
        # budget-enforcement backstop, while still trying to obtain terminal evidence.
        root_hard_termination_succeeded = True
        confirmed_by = confirm_terminal(phase="hard_cancel")
        cancelled.set()
        cancellation_failure_reason = None
        print(
            json.dumps(
                {
                    "event": "production_call_cancelled",
                    "function_call_id": call_id,
                    "reason": reason,
                    "terminal_confirmation": confirmed_by or "hard_cancel_api_ack",
                    "hard_termination": True,
                    "terminal_confirmed": root_cancel_terminal_confirmed,
                    "confirmation_errors": list(errors),
                },
                sort_keys=True,
            ),
            flush=True,
        )

'''
    text = replace_between(
        text,
        "    def cancel_call(reason: str) -> None:\n",
        "    previous_handlers: dict[int, Any] = {}\n",
        new_cancel,
    )

    old_cleanup = '''            if (
                call is not None
                and not remote_completed
                and not cancelled.is_set()
                and cancellation_failure_reason is None
            ):
                try:
                    cancel_call("watchdog exited before production call completed")
                except BaseException as cleanup_exc:
                    cleanup_failure_reason = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
'''
    new_cleanup = '''            if call is not None and not remote_completed and not cancelled.is_set():
                for cleanup_attempt in range(1, 4):
                    try:
                        cancel_call("watchdog exited before production call completed")
                    except BaseException as cleanup_exc:
                        cleanup_failure_reason = (
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                        if cleanup_attempt < 3:
                            time.sleep(0.25 * cleanup_attempt)
                        continue
                    cleanup_failure_reason = None
                    break
'''
    if text.count(old_cleanup) != 1:
        raise RuntimeError("watchdog cleanup reconciliation anchor drifted")
    text = text.replace(old_cleanup, new_cleanup, 1)

    summary_anchor = '''                "call_cancelled": cancelled.is_set(),
                "root_call_terminal_confirmed": remote_completed or cancelled.is_set(),
                "root_call_cancellation_failure": (
'''
    summary_replacement = '''                "call_cancelled": cancelled.is_set(),
                "root_call_terminal_confirmed": (
                    remote_completed or root_cancel_terminal_confirmed
                ),
                "root_call_hard_termination_succeeded": root_hard_termination_succeeded,
                "root_call_stopped": (
                    remote_completed
                    or root_cancel_terminal_confirmed
                    or root_hard_termination_succeeded
                ),
                "root_call_cancellation_failure": (
'''
    if text.count(summary_anchor) != 1:
        raise RuntimeError("watchdog summary anchor drifted")
    text = text.replace(summary_anchor, summary_replacement, 1)
    WATCHDOG.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    first = '''def test_watchdog_fails_closed_when_root_cancellation_is_not_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    clock = {"now": 0.1}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    class NonTerminalCall(_Call):
        def get(self, *, timeout: float):
            assert timeout > 0
            self.polls += 1
            if self.cancel_requested:
                clock["now"] = 2.1
                raise TimeoutError("root producer still active")
            clock["now"] = 1.1
            if _Spy.instance is not None:
                _Spy.instance.abort_reason = "synthetic bad telemetry"
            raise TimeoutError

    call = NonTerminalCall({})
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(ProductionCallNotTerminated, match="remained nonterminal"):
        module.run(render=False)

    assert call.cancel_args == [False]
    summary = json.loads(
        (tmp_path / "open-evidence" / "modal-spy-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ABORT"
    assert summary["call_cancelled"] is False
    assert summary["root_call_terminal_confirmed"] is False
    assert "terminal confirmation timed out" in summary["root_call_cancellation_failure"]
    assert summary["budget"]["gpu_seconds"] == pytest.approx(4.0)


'''
    first_new = '''def test_watchdog_escalates_unconfirmed_root_cancellation_to_exact_hard_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    clock = {"now": 0.1}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    class NonTerminalCall(_Call):
        def __init__(self) -> None:
            super().__init__({})
            self.hard_terminated = False

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_args.append(terminate_containers)
            self.cancel_requested = True
            if terminate_containers:
                self.hard_terminated = True

        def get(self, *, timeout: float):
            assert timeout > 0
            self.polls += 1
            if self.hard_terminated:
                clock["now"] = 2.6
                raise InputCancellation("hard cancellation reached terminal state")
            if self.cancel_requested:
                clock["now"] = 2.1
                raise TimeoutError("root producer still active")
            clock["now"] = 1.1
            if _Spy.instance is not None:
                _Spy.instance.abort_reason = "synthetic bad telemetry"
            raise TimeoutError

    call = NonTerminalCall()
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(RuntimeError, match="Modal spy aborted production early"):
        module.run(render=False)

    assert call.cancel_args == [False, True]
    summary = json.loads(
        (tmp_path / "open-evidence" / "modal-spy-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ABORT"
    assert summary["call_cancelled"] is True
    assert summary["root_call_terminal_confirmed"] is True
    assert summary["root_call_hard_termination_succeeded"] is True
    assert summary["root_call_stopped"] is True
    assert summary["root_call_cancellation_failure"] is None
    assert summary["budget"]["gpu_seconds"] == pytest.approx(5.0)


'''
    if text.count(first) != 1:
        raise RuntimeError("nonterminal cancellation regression anchor drifted")
    text = text.replace(first, first_new, 1)

    second_start = "def test_watchdog_treats_cancel_confirmation_transport_error_as_unconfirmed(\n"
    second_end = "def test_watchdog_counts_successful_hydration_against_compute_budget(\n"
    second_new = '''def test_watchdog_uses_hard_cancel_ack_when_terminal_confirmation_transport_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))

    class TransportFailureCall(_Call):
        def get(self, *, timeout: float):
            assert timeout > 0
            self.polls += 1
            if self.cancel_requested:
                raise ServiceError("control plane unavailable")
            if _Spy.instance is not None:
                _Spy.instance.abort_reason = "synthetic bad telemetry"
            raise TimeoutError

    call = TransportFailureCall({})
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(RuntimeError, match="Modal spy aborted production early"):
        module.run(render=False)

    assert call.cancel_args == [False, True]
    summary = json.loads(
        (tmp_path / "open-evidence" / "modal-spy-summary.json").read_text(encoding="utf-8")
    )
    assert summary["call_cancelled"] is True
    assert summary["root_call_terminal_confirmed"] is False
    assert summary["root_call_hard_termination_succeeded"] is True
    assert summary["root_call_stopped"] is True
    assert summary["root_call_cancellation_failure"] is None


'''
    text = replace_between(text, second_start, second_end, second_new)
    TESTS.write_text(text, encoding="utf-8")


def add_runtime_trigger_contract() -> None:
    CONTRACT_TEST.write_text(
        '''from pathlib import Path

import yaml


def test_lovable_bootstrap_uses_immutable_tag_for_deploy_and_owner_only_hilp_trigger() -> None:
    workflow = Path(".github/workflows/lovable-production-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(workflow)
    assert isinstance(parsed, dict)
    assert "pull_request" in parsed[True]
    assert "edited" in parsed[True]["pull_request"]["types"]
    assert "contents: write" in workflow
    assert "actions: write" in workflow
    assert "github.actor == github.repository_owner" in workflow
    assert "CLIPPER_HILP_TRIGGER sha=" in workflow
    assert 'tag="clipper-hilp-${EXPECTED_SHA}"' in workflow
    assert '"ref": sys.argv[1], "sha": sys.argv[2]' in workflow
    assert "modal-workers-deploy.yml/dispatches" in workflow
    assert '"deployment_sha": sys.argv[2]' in workflow
    assert "production-pipeline.yml/dispatches" in workflow
    assert '"ref": sys.argv[1]' in workflow
    assert "test \"$(git rev-parse HEAD)\" = \"$EXPECTED_SHA\"" in workflow


def test_runtime_tag_name_is_content_addressed_to_full_sha() -> None:
    workflow = Path(".github/workflows/lovable-production-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    assert 'tag="clipper-hilp-${EXPECTED_SHA}"' in workflow
    assert "refs/tags/${tag}" in workflow
    assert "immutable HILP tag points at the wrong SHA" in workflow
''',
        encoding="utf-8",
    )


def main() -> None:
    patch_watchdog()
    patch_tests()
    add_runtime_trigger_contract()


if __name__ == "__main__":
    main()
