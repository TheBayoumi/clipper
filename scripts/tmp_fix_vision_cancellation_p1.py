from __future__ import annotations

from pathlib import Path


provider_path = Path("src/clipper/providers/modal.py")
provider = provider_path.read_text(encoding="utf-8")

old_helper_anchor = '''    @staticmethod
    def _cancel_confirmation_seconds() -> float:
        timeout = float(os.getenv("CLIPPER_VISION_CANCEL_CONFIRM_SECONDS", "30"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("vision cancellation confirmation timeout must be finite and positive")
        return timeout

    def inspect(
'''
new_helper_anchor = '''    _CANCEL_CONFIRM_TERMINAL_MODAL_ERRORS: ClassVar[frozenset[str]] = frozenset(
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
    _CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS: ClassVar[frozenset[str]] = frozenset(
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

    @staticmethod
    def _cancel_confirmation_seconds() -> float:
        timeout = float(os.getenv("CLIPPER_VISION_CANCEL_CONFIRM_SECONDS", "30"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("vision cancellation confirmation timeout must be finite and positive")
        return timeout

    def _cancellation_confirmation_is_terminal(self, exc: BaseException) -> bool:
        error_name = type(exc).__name__
        try:
            modal_module = self._modal()
        except ProviderUnavailable:
            return error_name not in self._CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS

        exception_namespace = getattr(modal_module, "exception", None)
        if exception_namespace is None:
            return error_name not in self._CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS

        terminal_types = tuple(
            error_type
            for name in self._CANCEL_CONFIRM_TERMINAL_MODAL_ERRORS
            if isinstance((error_type := getattr(exception_namespace, name, None)), type)
        )
        if terminal_types and isinstance(exc, terminal_types):
            return True

        modal_error_type = getattr(exception_namespace, "Error", None)
        if isinstance(modal_error_type, type) and isinstance(exc, modal_error_type):
            return False
        return error_name not in self._CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS

    def inspect(
'''
if old_helper_anchor not in provider:
    raise RuntimeError("ModalVisionProvider cancellation helper anchor drifted")
provider = provider.replace(old_helper_anchor, new_helper_anchor, 1)

old_confirmation = '''            try:
                call.get(timeout=confirmation_timeout)
            except TimeoutError as confirm_exc:
                self._instance_handle = None
                raise ModalRemoteError(
                    function_name=self.function_name,
                    error_type="VisionCancellationUnconfirmedError",
                    message="vision call did not reach a terminal state after cancellation",
                    details={
                        "reason": "vision_cancellation_unconfirmed",
                        "timeout_seconds": deadline,
                        "confirmation_timeout_seconds": confirmation_timeout,
                        "frames": len(frames),
                    },
                ) from confirm_exc
            except Exception:
                self._instance_handle = None
            else:
                self._instance_handle = None
'''
new_confirmation = '''            try:
                call.get(timeout=confirmation_timeout)
            except TimeoutError as confirm_exc:
                self._instance_handle = None
                raise ModalRemoteError(
                    function_name=self.function_name,
                    error_type="VisionCancellationUnconfirmedError",
                    message="vision call did not reach a terminal state after cancellation",
                    details={
                        "reason": "vision_cancellation_unconfirmed",
                        "timeout_seconds": deadline,
                        "confirmation_timeout_seconds": confirmation_timeout,
                        "frames": len(frames),
                    },
                ) from confirm_exc
            except BaseException as confirm_exc:
                self._instance_handle = None
                if isinstance(confirm_exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                if not self._cancellation_confirmation_is_terminal(confirm_exc):
                    raise ModalRemoteError(
                        function_name=self.function_name,
                        error_type="VisionCancellationUnconfirmedError",
                        message=(
                            "vision cancellation confirmation failed through the Modal "
                            "control plane; terminal producer state is unconfirmed"
                        ),
                        details={
                            "reason": "vision_cancellation_unconfirmed",
                            "timeout_seconds": deadline,
                            "confirmation_timeout_seconds": confirmation_timeout,
                            "frames": len(frames),
                            "confirmation_error_type": type(confirm_exc).__name__,
                        },
                    ) from confirm_exc
            else:
                self._instance_handle = None
'''
if old_confirmation not in provider:
    raise RuntimeError("ModalVisionProvider confirmation block drifted")
provider = provider.replace(old_confirmation, new_confirmation, 1)
provider_path.write_text(provider, encoding="utf-8")


test_path = Path("tests/test_vision_runtime_recovery.py")
tests = test_path.read_text(encoding="utf-8")
regression_name = "test_modal_vision_deadline_fails_closed_on_confirmation_transport_error"
if regression_name not in tests:
    tests += '''


def test_modal_vision_deadline_fails_closed_on_confirmation_transport_error(
    tmp_path: Path,
) -> None:
    class FakeModalError(Exception):
        pass

    class ServiceError(FakeModalError):
        pass

    class RemoteError(FakeModalError):
        pass

    exception_namespace = type(
        "ExceptionNamespace",
        (),
        {
            "Error": FakeModalError,
            "ServiceError": ServiceError,
            "RemoteError": RemoteError,
        },
    )
    fake_modal = type("FakeModal", (), {"exception": exception_namespace})()

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    provider = ModalVisionProvider(
        app_name="app",
        identity=_identity(),
        function_name="vision",
    )
    call = Mock()
    call.get.side_effect = [TimeoutError(), ServiceError("control plane unavailable")]
    function = Mock()
    function.spawn.return_value = call

    with (
        patch.object(provider, "_function", return_value=function),
        patch.object(provider, "_modal", return_value=fake_modal),
        pytest.raises(ModalRemoteError) as raised,
    ):
        provider.inspect(task="source_policy_visual_scout", frames=[frame], context={})

    assert raised.value.error_type == "VisionCancellationUnconfirmedError"
    assert raised.value.details["reason"] == "vision_cancellation_unconfirmed"
    assert raised.value.details["confirmation_error_type"] == "ServiceError"
    assert "recovery_action" not in raised.value.details
    assert not visual_ai._is_vision_capacity_error(raised.value)
    assert call.get.call_count == 2
    call.cancel.assert_called_once_with()
'''
test_path.write_text(tests, encoding="utf-8")
