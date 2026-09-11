from pathlib import Path

runtime_contract_path = Path("tests/test_pre_hilp_runtime_trigger_contract.py")
runtime_contract = runtime_contract_path.read_text(encoding="utf-8")
bad_runtime_assertion = (
    '    assert "test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"" in workflow\n'
)
good_runtime_assertion = (
    '    assert \'test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"\' in workflow\n'
)
if runtime_contract.count(bad_runtime_assertion) != 1:
    raise RuntimeError("generated exact-checkout assertion anchor drifted")
runtime_contract_path.write_text(
    runtime_contract.replace(bad_runtime_assertion, good_runtime_assertion, 1),
    encoding="utf-8",
)

production_contract_path = Path("tests/test_production_workflow_contract.py")
production_contract = production_contract_path.read_text(encoding="utf-8")

old_cancel_assertion = '    assert "call.cancel(terminate_containers=False)" in watchdog\n'
new_cancel_assertions = (
    '    assert "call.cancel(terminate_containers=terminate_containers)" in watchdog\n'
    '    assert "terminate_containers=False" in watchdog\n'
    '    assert "terminate_containers=True" in watchdog\n'
)
if production_contract.count(old_cancel_assertion) != 2:
    raise RuntimeError("production cancellation contract anchors drifted")
production_contract = production_contract.replace(
    old_cancel_assertion,
    new_cancel_assertions,
)

old_terminal_assertion = (
    '    assert \'"root_call_terminal_confirmed": remote_completed or cancelled.is_set()\' '
    'in watchdog\n'
)
new_terminal_assertions = (
    '    assert \'"root_call_terminal_confirmed": (\' in watchdog\n'
    '    assert (\n'
    '        \'"root_call_hard_termination_succeeded": root_hard_termination_succeeded\'\n'
    '        in watchdog\n'
    '    )\n'
    '    assert \'"root_call_stopped": (\' in watchdog\n'
)
if production_contract.count(old_terminal_assertion) != 1:
    raise RuntimeError("root terminal-summary contract anchor drifted")
production_contract = production_contract.replace(
    old_terminal_assertion,
    new_terminal_assertions,
    1,
)

old_deploy_assertion = (
    '    assert \'"inputs": {"deployment_sha": os.environ["EXPECTED_SHA"]}\' '
    'in bootstrap\n'
)
new_deploy_assertions = (
    '    assert \'tag="clipper-hilp-${EXPECTED_SHA}"\' in bootstrap\n'
    '    assert (\n'
    '        \'{"ref": sys.argv[1], "inputs": {"deployment_sha": sys.argv[2]}}\'\n'
    '        in bootstrap\n'
    '    )\n'
    '    assert "modal-workers-deploy.yml/dispatches" in bootstrap\n'
    '    assert "production-pipeline.yml/dispatches" in bootstrap\n'
)
if production_contract.count(old_deploy_assertion) != 1:
    raise RuntimeError("immutable deployment contract anchor drifted")
production_contract = production_contract.replace(
    old_deploy_assertion,
    new_deploy_assertions,
    1,
)

production_contract_path.write_text(production_contract, encoding="utf-8")
