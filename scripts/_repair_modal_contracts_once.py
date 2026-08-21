from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if new in text:
        return text
    raise SystemExit(f"{label} anchor missing")


pipeline = Path("scripts/modal_pipeline.py")
text = pipeline.read_text(encoding="utf-8")
constant = '_TARGETED_RECOVERY_PLANS = (("c14", "p3"), ("c5", "p1"))\n'
if '_LEGACY_RECOVERY_PROTOCOL = "v8-six-finalist-recovery"' not in text:
    if constant not in text:
        raise SystemExit("legacy recovery constant anchor missing")
    text = text.replace(
        constant,
        '_LEGACY_RECOVERY_PROTOCOL = "v8-six-finalist-recovery"\n' + constant,
        1,
    )

text = replace_once(
    text,
    '''def recover_finalists(payload: dict[str, Any]) -> dict[str, Any]:
    """Repair only explicitly selected finalists and revalidate every reused clip."""

    from clipper.acceptance import validate_live_run
''',
    '''def recover_finalists(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the explicit legacy V8 six-finalist repair protocol only."""

    if str(payload.get("legacy_recovery_protocol") or "") != _LEGACY_RECOVERY_PROTOCOL:
        raise ValueError(
            "recover_finalists is restricted to the explicit legacy V8 six-finalist protocol"
        )

    from clipper.acceptance import validate_live_run
''',
    "recover_finalists guard",
)
text = replace_once(
    text,
    '''    if render and manifest.get("status") != "SUCCESS":
        raise RuntimeError(
            f"production pipeline did not reach SUCCESS: {manifest.get('status_reason')}"
        )
''',
    '''    if render and manifest.get("status") not in {"SUCCESS", "DEGRADED"}:
        raise RuntimeError(
            "production pipeline did not reach SUCCESS or DEGRADED: "
            f"{manifest.get('status_reason')}"
        )
''',
    "render terminal status",
)
pipeline.write_text(text, encoding="utf-8")

models = Path("scripts/modal_open_models.py")
text = models.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''        "source_hazards:smoke",
        "boundary_audit:smoke",
''',
    '''        "source_hazards:smoke",
        "boundary_audit:smoke",
        "semantic_cores:smoke",
        "narrative_envelope:smoke",
        "quality_windows:smoke",
''',
    "editorial schema smoke tasks",
)
models.write_text(text, encoding="utf-8")

launcher = Path("scripts/run_modal_finalist_recovery.py")
text = launcher.read_text(encoding="utf-8")
if 'LEGACY_RECOVERY_PROTOCOL = "v8-six-finalist-recovery"' not in text:
    text = replace_once(
        text,
        "PLAN_KEYS = (\n",
        'LEGACY_RECOVERY_PROTOCOL = "v8-six-finalist-recovery"\n\nPLAN_KEYS = (\n',
        "launcher protocol",
    )
text = replace_once(
    text,
    '''    request = {
        "source_run_id": args.source_run_id,
''',
    '''    request = {
        "legacy_recovery_protocol": LEGACY_RECOVERY_PROTOCOL,
        "source_run_id": args.source_run_id,
''',
    "launcher request protocol",
)
text = text.replace(
    '"Repair explicitly selected finalists inside Modal, review only newly rendered "',
    '"Legacy V8 recovery only: repair selected finalists inside Modal, review new "',
    1,
)
launcher.write_text(text, encoding="utf-8")

contract = Path("tests/test_modal_pipeline_source_contract.py")
text = contract.read_text(encoding="utf-8")
marker = "def test_modal_schema_smoke_covers_all_active_editorial_task_families() -> None:"
if marker not in text:
    text += '''


def test_modal_schema_smoke_covers_all_active_editorial_task_families() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    smoke = source.split("def editorial_schema_smoke(", 1)[1].split("def hf_access_smoke(", 1)[0]
    expected = (
        "episode_editorial_profile",
        "story_moments:smoke",
        "clip_concepts",
        "global_concept_comparison",
        "hook_variants:smoke",
        "edit_plans:smoke",
        "source_hazards:smoke",
        "boundary_audit:smoke",
        "semantic_cores:smoke",
        "narrative_envelope:smoke",
        "quality_windows:smoke",
    )
    assert len(expected) == 11
    assert all(f'"{task}"' in smoke for task in expected)


def test_modal_full_cycle_accepts_dynamic_degraded_yield_but_rejects_failed() -> None:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    cycle = source.split("def run_full_cycle(", 1)[1]
    assert 'manifest.get("status") not in {"SUCCESS", "DEGRADED"}' in cycle
    assert 'if not render and manifest.get("status") == "FAILED"' in cycle


def test_targeted_finalist_recovery_is_explicit_legacy_protocol_only() -> None:
    pipeline_source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    launcher_source = Path("scripts/run_modal_finalist_recovery.py").read_text(encoding="utf-8")
    recovery = pipeline_source.split("def recover_finalists(", 1)[1].split(
        "def run_full_cycle(", 1
    )[0]
    assert '_LEGACY_RECOVERY_PROTOCOL = "v8-six-finalist-recovery"' in pipeline_source
    assert 'payload.get("legacy_recovery_protocol")' in recovery
    assert "restricted to the explicit legacy V8 six-finalist protocol" in recovery
    assert 'LEGACY_RECOVERY_PROTOCOL = "v8-six-finalist-recovery"' in launcher_source
    assert '"legacy_recovery_protocol": LEGACY_RECOVERY_PROTOCOL' in launcher_source
'''
contract.write_text(text, encoding="utf-8")
