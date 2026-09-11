from pathlib import Path

path = Path("tests/test_pre_hilp_runtime_trigger_contract.py")
text = path.read_text(encoding="utf-8")
bad = '    assert "test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"" in workflow\n'
good = '    assert \'test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"\' in workflow\n'
if text.count(bad) != 1:
    raise RuntimeError("generated exact-checkout assertion anchor drifted")
path.write_text(text.replace(bad, good, 1), encoding="utf-8")
