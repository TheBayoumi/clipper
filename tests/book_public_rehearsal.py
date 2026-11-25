"""Opt-in public-synthetic book chapter rehearsal. Test-only, never production."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import re
import resource
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FIXTURE = "PUBLIC_SYNTHETIC_NOT_REAL_CHAPTER"
REVIEW = "SYNTHETIC_TEST_REVIEW_NOT_HUMAN_APPROVAL"
BAN_IMPORTS = frozenset({"modal", "torch", "transformers", "huggingface_hub", "kokoro", "yt_dlp", "gdown"})
STAGES = ("argument", "audit", "reconcile", "adaptation", "chapter_script", "chapter_direction")


class RehearsalStop(RuntimeError):
    """Fail closed: no provider or production claims."""


@contextmanager
def offline_guard(attempts: list[str]) -> Iterator[None]:
    """In-process guard; run in a dedicated process, outside a Modal control plane."""
    originals = (
        builtins.__import__, socket.socket.connect, socket.socket.connect_ex,
        socket.create_connection, urllib.request.urlopen, subprocess.Popen, os.system,
    )

    def deny(name: str) -> Any:
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            attempts.append(name)
            raise RehearsalStop("prohibited operation: " + name)
        return blocked

    def safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] in BAN_IMPORTS:
            return deny("import:" + name)()
        return originals[0](name, *args, **kwargs)

    try:
        builtins.__import__ = safe_import
        socket.socket.connect = deny("socket.connect")
        socket.socket.connect_ex = deny("socket.connect_ex")
        socket.create_connection = deny("socket.create_connection")
        urllib.request.urlopen = deny("urlopen")
        subprocess.Popen = deny("subprocess")
        os.system = deny("system")
        yield
    finally:
        (
            builtins.__import__, socket.socket.connect, socket.socket.connect_ex,
            socket.create_connection, urllib.request.urlopen, subprocess.Popen, os.system,
        ) = originals


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RehearsalStop(reason)


def artifact_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    paths = sorted(root.rglob("*"))
    require(len(paths) <= 1000, "unbounded artifact inventory")
    for path in paths:
        require(not path.is_symlink(), "symlink in synthetic evidence")
        if path.is_file():
            raw = path.read_bytes()
            require(len(raw) <= 2_000_000, "unbounded synthetic artifact")
            result[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    return result


def rehearse(root: Path, expected_sha: str) -> dict[str, object]:
    """Actual selected-source and hybrid chapter logic; only producers are injected."""
    require(bool(re.fullmatch(r"[0-9a-f]{40}", expected_sha)), "exact source SHA required")
    require(root.is_absolute() and not root.exists(), "fresh absolute root required")
    start = time.monotonic()
    forbidden: list[str] = []
    with offline_guard(forbidden):
        root.mkdir(mode=0o700)
        # Explicitly test-only fixture imports, never production dependencies.
        from clipper.books import (
            chapter,
            chapter_hybrid,
            narration,
            scripting,
            selected_scope,
            whole_book,
        )
        from clipper.books.orchestrator import BookOrchestrator
        from public_book_fixture_support import (
            _hybrid_creative_output,
            _hybrid_wav,
            _injected_hybrid_response,
            _selected_hybrid_source,
        )

        original: list[tuple[Any, str, Any]] = []

        def block(obj: Any, name: str) -> None:
            previous = getattr(obj, name)

            def refused(*_args: Any, **_kwargs: Any) -> Any:
                forbidden.append(f"{obj.__name__}.{name}")
                raise RehearsalStop("forbidden model/source provider")
            setattr(obj, name, refused)
            original.append((obj, name, previous))

        for obj, name in (
            (BookOrchestrator, "mine_candidate"),
            (scripting, "_local_chapter_completion"),
            (scripting, "_script_pipeline"),
            (scripting, "_public_book_context"),
            (narration, "_kokoro_wav"),
        ):
            block(obj, name)

        calls = {stage: 0 for stage in STAGES}
        wav_count = 0

        def producer(name: str) -> Any:
            def produce(request: dict[str, object]) -> str:
                calls[name] += 1
                if name in {"argument", "audit", "reconcile"}:
                    return _injected_hybrid_response(name, request)
                return _hybrid_creative_output(name, request)
            return produce

        def fake_wav(text: str, voice: str) -> bytes:
            nonlocal wav_count
            wav_count += 1
            return _hybrid_wav(text, voice)

        args = {
            "argument_completion": producer("argument"),
            "audit_completion": producer("audit"),
            "reconcile_completion": producer("reconcile"),
            "adaptation_completion": producer("adaptation"),
            "script_completion": producer("chapter_script"),
            "direction_completion": producer("chapter_direction"),
            "narration_producer": fake_wav,
            "token_counter": lambda _messages: 1200,
        }
        try:
            for track in ("pause", "continuation", "malformed", "interruption"):
                (root / track).mkdir()
            runner_a, bundle_a, section_a = _selected_hybrid_source(root / "pause", section_index=1)
            source_a = selected_scope.import_chapter(runner_a.root, bundle_a)
            require(bool(source_a.work.units and source_a.work.skipped), "no public unit/gap")
            require(not source_a.book.parent_original_reverified_on_host, "false parent claim")
            no_producer = chapter.evaluate_hybrid_chapter(runner_a, bundle_a, section_a)
            require(
                no_producer["status"] == "AWAITING_EXPLICIT_CPU_HYBRID_CHAPTER_ARGUMENT",
                "missing producer was not blocked",
            )
            try:
                chapter.evaluate_hybrid_chapter(
                    runner_a, bundle_a, section_a, plan=chapter.ChapterPlan(brief_cpu=True)
                )
            except chapter.ChapterStop as exc:
                require(
                    str(exc) == "hybrid injected-test tokenizer counter is required",
                    "unexpected missing-producer rejection",
                )
            else:
                raise RehearsalStop("missing producer accepted")
            require(
                not (runner_a.root / "book-candidates" / "hybrid_argument").exists(),
                "missing producer reserved a stage attempt",
            )
            require(not any(calls.values()), "unexpected early callback")
            brief_plan = chapter.ChapterPlan(brief_cpu=True)
            pause = chapter.evaluate_hybrid_chapter(
                runner_a, bundle_a, section_a, plan=brief_plan, **args
            )
            require(pause["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW", "review bypass")
            require(pause["automation_claim"] == "INJECTED_TEST", "producer mislabeled")
            require(pause["publish_eligible"] is False, "publish gate bypass")
            before = dict(calls)
            replay_pause = chapter.evaluate_hybrid_chapter(
                runner_a, bundle_a, section_a, plan=brief_plan,
                expected_brief_identity=pause["brief_identity"], **args,
            )
            require(replay_pause["status"] == pause["status"], "review replay differs")
            require(calls == before and wav_count == 0, "pre-review replay did work")

            runner_b, bundle_b, section_b = _selected_hybrid_source(root / "continuation", section_index=1)
            full_plan = chapter.ChapterPlan(
                brief_cpu=True, adaptation_cpu=True, script_cpu=True,
                direction_cpu=True, kokoro_cpu=True,
            )
            pending = chapter.evaluate_hybrid_chapter(
                runner_b, bundle_b, section_b, plan=full_plan, **args
            )
            require(pending["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW", "gate missing")
            home = runner_b.root / "book-candidates"
            ledger_path = next((home / "chapter-evaluations").glob("*/source-ledger.json"))
            brief_path = next((home / "hybrid_reconcile").glob("*/validated.json"))
            ledger = json.loads(ledger_path.read_bytes())
            brief = json.loads(brief_path.read_bytes())["payload"]
            require(
                len(brief["coverage"]) == len(chapter_hybrid.validate_ledger(ledger)),
                "incomplete original unit/gap decisions",
            )
            packet = chapter_hybrid.human_review_packet(brief, ledger)
            require(packet["review_status"] == "PENDING_ACTUAL_HUMAN_ATTESTATION", "auto-approved")
            require(wav_count == 0, "audio before test review")
            review = {
                "schema": chapter_hybrid.REVIEW_SCHEMA,
                "brief_sha256": brief["brief_sha256"],
                "brief_artifact_identity": pending["brief_identity"],
                "reviewer": REVIEW,
                "unit_decisions": [
                    {"id": entry["id"], "review": "accept", "reason": REVIEW}
                    for entry in ledger["entries"]
                ],
                "beat_decisions": [
                    {"beat_id": beat["beat_id"], "review": "support_attested", "reason": REVIEW}
                    for beat in brief["beats"]
                ],
            }
            review_path = root / "continuation" / "test-review.json"
            review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
            final = chapter.evaluate_hybrid_chapter(
                runner_b, bundle_b, section_b, plan=full_plan, review_file=review_path,
                expected_brief_identity=pending["brief_identity"], **args,
            )
            require(
                final["status"] == "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED",
                "synthetic final status misattributed",
            )
            require(final["publish_eligible"] is False, "synthetic completion publishable")
            require(final["audio_provenance"] == "INJECTED_TEST_WAV", "fake PCM mislabeled")
            require(wav_count > 0, "fake audio not exercised")
            before, before_wav = dict(calls), wav_count
            replay = chapter.evaluate_hybrid_chapter(
                runner_b, bundle_b, section_b, plan=full_plan, review_file=review_path,
                expected_brief_identity=pending["brief_identity"], **args,
            )
            require(replay["status"] == final["status"], "postreview cache replay status")
            require(before == calls and before_wav == wav_count, "replay reran producers")

            adversarial: list[str] = []
            changed_review = {**review, "brief_artifact_identity": "0" * 64}
            stale_path = root / "continuation" / "stale-review.json"
            stale_path.write_text(json.dumps(changed_review), encoding="utf-8")
            try:
                chapter.evaluate_hybrid_chapter(
                    runner_b, bundle_b, section_b, plan=full_plan, review_file=stale_path,
                    expected_brief_identity=pending["brief_identity"], **args,
                )
            except chapter.ChapterStop:
                adversarial.append("stale_review")
            else:
                raise RehearsalStop("stale review accepted")

            original_material = whole_book.hybrid_chapter_producer_material
            whole_book.hybrid_chapter_producer_material = lambda: {
                **original_material(), "public_synthetic_producer_variant": "changed"
            }
            try:
                try:
                    chapter.evaluate_hybrid_chapter(
                        runner_b, bundle_b, section_b, plan=full_plan,
                        expected_brief_identity=pending["brief_identity"], **args,
                    )
                except chapter.ChapterStop:
                    adversarial.append("producer_identity_tamper")
                else:
                    raise RehearsalStop("changed producer identity was accepted")
            finally:
                whole_book.hybrid_chapter_producer_material = original_material

            raw_bundle = bundle_b.read_bytes()
            bundle_b.write_bytes(raw_bundle + b" ")
            try:
                selected_scope.import_chapter(runner_b.root, bundle_b)
            except selected_scope.SelectedSourceStop:
                adversarial.append("source_tamper")
            else:
                raise RehearsalStop("source tampering accepted")
            bundle_b.write_bytes(raw_bundle)
            require(bundle_b.read_bytes() == raw_bundle, "source bundle not restored")

            brief_path.write_bytes(brief_path.read_bytes() + b" ")
            try:
                chapter.evaluate_hybrid_chapter(
                    runner_b, bundle_b, section_b, plan=full_plan,
                    expected_brief_identity=pending["brief_identity"], **args,
                )
            except chapter.ChapterStop:
                adversarial.append("raw_artifact_tamper")
            else:
                raise RehearsalStop("raw artifact tamper accepted")

            runner_c, bundle_c, section_c = _selected_hybrid_source(root / "malformed", section_index=1)
            invalid_calls = 0

            def malformed(_request: dict[str, object]) -> str:
                nonlocal invalid_calls
                invalid_calls += 1
                return "{malformed-json"

            invalid = {**args, "argument_completion": malformed}
            one_attempt = chapter.ChapterPlan(brief_cpu=True, max_attempts_per_stage=1)
            for _ in range(2):
                try:
                    chapter.evaluate_hybrid_chapter(
                        runner_c, bundle_c, section_c, plan=one_attempt, **invalid
                    )
                except chapter.ChapterStop:
                    pass
                else:
                    raise RehearsalStop("malformed/failed attempt accepted")
            require(invalid_calls == 1, "failed attempt automatically retried")
            adversarial.append("malformed_reply_no_replay")

            runner_d, bundle_d, section_d = _selected_hybrid_source(root / "interruption", section_index=1)
            interrupt_calls = 0

            def interrupted(_request: dict[str, object]) -> str:
                nonlocal interrupt_calls
                interrupt_calls += 1
                raise RuntimeError("inert synthetic interruption")

            interrupted_args = {**args, "argument_completion": interrupted}
            for _ in range(2):
                try:
                    chapter.evaluate_hybrid_chapter(
                        runner_d, bundle_d, section_d, plan=brief_plan, **interrupted_args
                    )
                except (RuntimeError, chapter.ChapterStop):
                    pass
                else:
                    raise RehearsalStop("interrupted stage accepted")
            require(interrupt_calls == 1, "interrupted producer rerun")
            adversarial.append("interrupted_claim_no_replay")
            require(not forbidden, "prohibited operation was attempted")
            report: dict[str, object] = {
                "schema": "clipper-public-book-model-free-rehearsal-v1",
                "source_sha": expected_sha,
                "source_pin": "EXPLICIT_LOCAL_GIT_CHECK_REQUIRED_FOR_CLI",
                "fixture_label": FIXTURE,
                "status": "SYNTHETIC_CONTRACT_REHEARSAL_PASSED",
                "publish_eligible": False,
                "tracks": {
                    "pause": {"status": pause["status"], "review_supplied": False,
                              "units": len(source_a.work.units),
                              "gaps": len(source_a.work.skipped),
                              "provenance": pause["automation_claim"]},
                    "synthetic_test_review": {
                        "status": final["status"], "review_label": REVIEW,
                        "audio_provenance": final["audio_provenance"],
                        "publish_eligible": False,
                    },
                },
                "production_functions": [
                    "BookOrchestrator.ingest_candidate",
                    "selected_scope.export_chapter",
                    "selected_scope.import_chapter",
                    "whole_book.plan_for",
                    "chapter.evaluate_hybrid_chapter",
                    "chapter._stage",
                    "chapter_hybrid.freeze/request/reconcile/validate_human_review",
                ],
                "injected_callback_calls": dict(calls),
                "injected_pcm_calls": wav_count,
                "malformed_calls": invalid_calls,
                "interrupted_calls": interrupt_calls,
                "real_model_calls": 0,
                "real_provider_calls": 0,
                "network_calls": 0,
                "publication_calls": 0,
                "prohibited_attempts": list(forbidden),
                "zero_call_evidence": {
                    "scope": "IN_PROCESS_PATCHED_PYTHON_APIS_ONLY",
                    "preloaded_forbidden_module_roots": sorted(
                        {name.split(".", 1)[0] for name in sys.modules}
                        & BAN_IMPORTS
                    ),
                    "import_roots_denied": sorted(BAN_IMPORTS),
                    "network_guards": [
                        "socket.connect",
                        "socket.connect_ex",
                        "socket.create_connection",
                        "urllib.request.urlopen",
                    ],
                    "source_provider_guards": [
                        "BookOrchestrator.mine_candidate",
                        "scripting._local_chapter_completion",
                        "scripting._script_pipeline",
                        "scripting._public_book_context",
                        "narration._kokoro_wav",
                    ],
                    "attempts": list(forbidden),
                    "not_a_kernel_network_or_native_io_guard": True,
                },
                "adversarial": adversarial,
                "artifact_sha256": artifact_hashes(root),
                "exclusions": [
                    "CODESPACE_SIGNED_HOST_AUTHORITY", "MODAL_DISTRIBUTED_CAMPAIGN_DAG",
                    "REAL_TOKENIZER_AND_MODEL_QUALITY", "PRIVATE_CHAPTER_SOURCE",
                    "ACTUAL_EDITORIAL_REVIEW", "REAL_TTS_AND_VISUALS",
                    "PROVIDER_EXECUTION_AND_HOST_CRASH_RECOVERY",
                ],
            }
        finally:
            for obj, name, previous in reversed(original):
                setattr(obj, name, previous)
    report["elapsed_seconds"] = round(time.monotonic() - start, 5)
    report["peak_rss_kib_platform"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args(argv)
    try:
        # Local checkout identity only; never contact GitHub or a Modal provider.
        checkout = Path(__file__).resolve().parents[1]
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        require(actual == args.expected_sha, "checkout SHA mismatch")
        require(
            not ({name.split(".", 1)[0] for name in sys.modules} & BAN_IMPORTS),
            "forbidden modules already loaded in CLI process",
        )
        changes = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        require(not changes.strip(), "checkout contains modified or untracked files")
        result = rehearse(args.fresh_root, args.expected_sha)
        result["source_pin"] = "VERIFIED_CLEAN_LOCAL_GIT_HEAD"
        print(json.dumps(result, sort_keys=True))
        return 0
    except (RehearsalStop, OSError, ValueError, subprocess.SubprocessError):
        print(json.dumps({"status": "PUBLIC_SYNTHETIC_REHEARSAL_BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
