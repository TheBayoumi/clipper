from pathlib import Path

WORKFLOWS = [
    Path('.github/workflows/mw4-semantic-v3-1-shadow.yml'),
    Path('.github/workflows/mw4-fullframe-retention-v3-1.yml'),
]
OUT = Path('.mw4-workflow-patch-v2')

OLD_SIZE = '''          actual_size = target.stat().st_size
          expected_size = int(selected.get('file_size') or 0)
          if expected_size and actual_size != expected_size:
              raise RuntimeError(
                  f'original master byte-size mismatch: downloaded={actual_size} expected={expected_size}'
              )
'''

NEW_SIZE = '''          actual_size = target.stat().st_size
          declared_size_kib = int(selected.get('file_size') or 0)
          declared_floor_bytes = declared_size_kib * 1024 if declared_size_kib else 0
          if declared_floor_bytes and not (
              declared_floor_bytes <= actual_size < declared_floor_bytes + 1024
          ):
              raise RuntimeError(
                  f'original master size mismatch: downloaded={actual_size} '
                  f'declared_kib={declared_size_kib} '
                  f'allowed_bytes=[{declared_floor_bytes},{declared_floor_bytes + 1024})'
              )
'''

OLD_QA = "              'declared_source_bytes': expected_size or None,\n"
NEW_QA = (
    "              'declared_source_size_kib': declared_size_kib or None,\n"
    "              'declared_source_floor_bytes': declared_floor_bytes or None,\n"
)

OUT.mkdir(parents=True, exist_ok=True)
for workflow in WORKFLOWS:
    text = workflow.read_text(encoding='utf-8')
    if text.count(OLD_SIZE) != 1:
        raise RuntimeError(f'{workflow}: expected exactly one legacy literal-byte size validator')
    if text.count(OLD_QA) != 1:
        raise RuntimeError(f'{workflow}: expected exactly one legacy declared_source_bytes field')

    text = text.replace(OLD_SIZE, NEW_SIZE, 1).replace(OLD_QA, NEW_QA, 1)

    if 'expected_size = int(selected.get' in text:
        raise RuntimeError(f'{workflow}: legacy byte-size interpretation remains')
    if "'declared_source_bytes'" in text:
        raise RuntimeError(f'{workflow}: misleading declared_source_bytes remains')
    if "declared_floor_bytes <= actual_size < declared_floor_bytes + 1024" not in text:
        raise RuntimeError(f'{workflow}: strict KiB floor/range check missing')
    if "'derivative_type': 'source'" not in text or "'proxy_fallback_allowed': False" not in text:
        raise RuntimeError(f'{workflow}: original-source certification was accidentally removed')
    if "derivatives.get('x1080')" in text or "derivatives.get('proxy')" in text:
        raise RuntimeError(f'{workflow}: proxy/x1080 fallback reappeared')

    target = OUT / workflow.name
    target.write_text(text, encoding='utf-8')
    print(f'patched {workflow} -> {target}')

# Prove the three observed source sizes satisfy MediaSilo's integer-KiB floor metadata.
observed = {
    'r1': (1524713, 1561306897),
    'batch2': (1619853, 1658730492),
    'week2': (1541181, 1578170213),
}
for key, (declared_kib, actual_bytes) in observed.items():
    floor = declared_kib * 1024
    assert floor <= actual_bytes < floor + 1024, (key, declared_kib, actual_bytes, floor)
    print(f'{key}: KiB floor check PASS actual={actual_bytes} allowed=[{floor},{floor + 1024})')
