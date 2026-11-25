"""Offline configuration planning, independent of any model or host."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import pipeline_plan as plan
from clipper.books.cli import configure_books_parser, dispatch_books


def _rows(report):
    return {row["stage"]: row for row in report["ordered_stages"]}


def _input(config, root, name, body, dependencies=()):
    path = root / f"{name}.json"
    path.write_bytes(body)
    entry = config["stages"][name]
    specs = plan.registry()
    lineage = plan.semantic_lineage(config, specs)
    entry["input"] = {
        "schema": plan.DESCRIPTOR_SCHEMA,
        "identity": plan.stage_input_identity(
            name, plan.registry()[name], config["source_sha256"],
            entry["semantic"], dependencies,
            tuple(lineage[dep] for dep in specs[name].dependencies),
        ),
        "output_sha256": hashlib.sha256(body).hexdigest(),
        "path": path.name,
    }
    return entry["input"]["output_sha256"]


def test_disabled_dependencies_require_verified_bytes(tmp_path: Path) -> None:
    source = b'{"fixture":"source"}'
    config = plan.template(hashlib.sha256(source).hexdigest(), "script")
    config["stages"]["source"]["enabled"] = False
    forbidden = Mock(side_effect=AssertionError("provider must not run"))
    initial = _rows(plan.resolve(config, tmp_path, verifiers={"source": forbidden}))
    assert initial["source"]["status"] == "BLOCKED_DEPENDENCY"
    assert initial["script"]["status"] == "BLOCKED_DEPENDENCY"
    forbidden.assert_not_called()
    _input(config, tmp_path, "source", source)
    missing = _rows(plan.resolve(config, tmp_path))
    assert missing["source"]["reason"] == "UNVERIFIED_NO_STAGE_VALIDATOR"
    verifier = Mock(return_value=True)
    report = plan.resolve(config, tmp_path, verifiers={"source": verifier})
    rows = _rows(report)
    assert rows["source"]["status"] == "REUSE"
    assert rows["argument"]["status"] == "RUN"
    assert rows["audits"]["identity"] is None
    assert rows["human_review"]["status"] == "AWAITING_HUMAN_REVIEW"
    assert rows["script"]["status"] == "BLOCKED_DEPENDENCY"
    assert rows["narration"]["status"] == "DISABLED"
    assert report["mode"] == "DRY_RUN_ONLY"
    assert report["execution_authorized"] is False
    verifier.assert_called_once()
    (tmp_path / "source.json").write_bytes(b"changed")
    corrupt = _rows(plan.resolve(config, tmp_path, verifiers={"source": verifier}))
    assert corrupt["source"]["reason"] == "INPUT_BYTES_MISMATCH"


def test_exact_reuse_and_dependency_invalidation(tmp_path: Path) -> None:
    config = plan.template("a" * 64, "script")
    checks = {}
    hashes = {}
    for name in ("source", "argument", "audits", "reconcile", "human_review"):
        deps = plan.registry()[name].dependencies
        hashes[name] = _input(
            config, tmp_path, name, name.encode(), tuple(hashes[d] for d in deps)
        )
        config["stages"][name]["enabled"] = False

        def verify(raw, descriptor, inputs, expected=name.encode(), names=deps):
            return raw == expected and tuple(inputs) == names

        checks[name] = verify
    result = _rows(plan.resolve(config, tmp_path, verifiers=checks))
    assert result["human_review"]["status"] == "REUSE"
    assert result["human_review"]["human_attestation"] == "HUMAN_ATTESTED_NOT_INDEPENDENTLY_PROVEN"
    assert result["adaptation"]["status"] == "RUN"
    assert result["script"]["status"] == "RUN"
    config["targets"] = ["brief"]
    brief = _rows(plan.resolve(config, tmp_path, verifiers=checks))
    assert brief["reconcile"]["identity"] == result["reconcile"]["identity"]
    config["stages"]["narration"]["semantic"]["voice"] = "af_bella"
    unrelated = _rows(plan.resolve(config, tmp_path, verifiers=checks))
    assert unrelated["reconcile"]["semantic_lineage_sha256"] == (
        brief["reconcile"]["semantic_lineage_sha256"]
    )
    config["stages"]["argument"]["semantic"]["max_new_tokens"] = 2048
    changed = _rows(plan.resolve(config, tmp_path, verifiers=checks))
    assert changed["argument"]["reason"] == "INPUT_IDENTITY_MISMATCH"
    assert changed["reconcile"]["status"] == "BLOCKED_DEPENDENCY"
    assert changed["reconcile"]["semantic_lineage_sha256"] != (
        brief["reconcile"]["semantic_lineage_sha256"]
    )
    config["source_sha256"] = "b" * 64
    changed_source = _rows(plan.resolve(config, tmp_path, verifiers=checks))
    assert changed_source["source"]["reason"] == "INPUT_IDENTITY_MISMATCH"


def test_untrusted_provenance_and_disabled_review(tmp_path: Path) -> None:
    config = plan.template("a" * 64)
    _input(config, tmp_path, "source", b"source")
    rejected = _rows(plan.resolve(config, tmp_path, verifiers={"source": lambda *_: False}))
    assert rejected["source"]["status"] == "UNVERIFIED"
    assert rejected["source"]["reason"] == "INPUT_CONTRACT_OR_PROVENANCE_REJECTED"
    config["stages"]["human_review"]["enabled"] = False
    rows = _rows(plan.resolve(config, tmp_path, verifiers={"source": lambda *_: True}))
    assert rows["human_review"]["reason"] == "HUMAN_ATTESTATION_REQUIRED"
    assert rows["script"]["status"] == "BLOCKED_DEPENDENCY"


@pytest.mark.parametrize("target", ["visuals", "video", "render", "publish"])
def test_unported_targets_explicit(tmp_path: Path, target: str) -> None:
    report = plan.resolve(plan.template("a" * 64, target), tmp_path)
    assert _rows(report)[target]["status"] == "UNSUPPORTED"
    assert report["status"] == "PLAN_BLOCKED"


@pytest.mark.parametrize("mutation", [
    lambda c: c["stages"].update(unknown={}),
    lambda c: c["stages"]["argument"].update(enabled=1),
    lambda c: c["stages"]["argument"]["semantic"].update(unknown=True),
    lambda c: c["stages"]["argument"]["semantic"].update(max_attempts=3),
    lambda c: c.update(targets=["script", "script"]),
    lambda c: c.update(targets=["unknown"]),
    lambda c: c["stages"]["source"].update(input={"validated": True}),
    lambda c: c["stages"].pop("source"),
])
def test_strict_config(tmp_path: Path, mutation) -> None:
    config = plan.template("a" * 64)
    assert plan.strict_json(json.dumps(config).encode()) == config
    mutation(config)
    with pytest.raises(plan.PlanError):
        plan.resolve(config, tmp_path)


@pytest.mark.parametrize("specs", [
    {"x": plan.StageSpec(("x",), "x", "none")},
    {"x": plan.StageSpec(("y",), "x", "none")},
    {"x": plan.StageSpec(("y", "y"), "x", "none"), "y": plan.StageSpec((), "y", "none")},
])
def test_invalid_graph(specs) -> None:
    with pytest.raises(plan.PlanError):
        plan.ordered_stages(specs)


def test_cli_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    configure_books_parser(parser.add_subparsers())
    config_path = tmp_path / "plan.json"
    config_path.write_text(json.dumps(plan.template("a" * 64)), encoding="utf-8")
    args = parser.parse_args([
        "books", "pipeline-plan", "--config", str(config_path),
        "--artifact-root", str(tmp_path),
    ])
    assert dispatch_books(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "DRY_RUN_ONLY"
    assert report["status"] == "PLAN_BLOCKED"
    config_path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    assert dispatch_books(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID_CONFIGURATION"
