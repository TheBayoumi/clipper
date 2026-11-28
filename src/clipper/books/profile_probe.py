"""Model-load probe: does a registered profile load and generate on the pinned image?

Uses only a made-up public sentence, never book text or a private packet. The container is
the same reviewed image the private phase uses (pinned commit, offline pinned snapshot), so a
pass here answers the runtime questions (loader class, torch and transformers pairing,
tokenizer and chat template, speed) before any chapter run. Nothing runs without --execute.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from . import model_profile, private_modal, private_phase

PROBE_PROMPT = "Reply with one short sentence about why the sky looks blue."
PROBE_TOKENS = 24
MAX_TIMEOUT = 1800


def probe_in_container(profile_name: str) -> dict[str, object]:
    """Runs inside the container. Returns facts only; generated text is hashed, not kept."""
    import os

    from . import scripting

    os.environ["CLIPPER_QWEN_DEVICE"] = "cuda"
    started = time.monotonic()
    report: dict[str, object] = {"profile": profile_name}
    with model_profile.use(profile_name):
        profile = model_profile.active()
        report["model"] = profile.model
        report["revision"] = profile.revision
        report["model_class"] = profile.model_class
        for package in ("torch", "transformers", "huggingface_hub", "lm-format-enforcer"):
            report["version_" + package] = version(package)
        torch = importlib.import_module("torch")

        report["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        request = {
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "seed": 1,
            "max_new_tokens": PROBE_TOKENS,
        }
        model, tokenizer = scripting._script_pipeline()
        report["load_seconds"] = round(time.monotonic() - started, 2)
        report["tokenizer_eos_token_id"] = tokenizer.eos_token_id
        report["generation_config_eos_token_id"] = getattr(
            getattr(model, "generation_config", None), "eos_token_id", None
        )
        generation_started = time.monotonic()
        raw = scripting._run_pinned_qwen(request, normalize_script=True)
        report["generate_seconds"] = round(time.monotonic() - generation_started, 2)
        report["reencoded_text_tokens"] = len(
            tokenizer(str(raw), add_special_tokens=False)["input_ids"]
        )
        report["generated_sha256"] = hashlib.sha256(str(raw).encode()).hexdigest()
        report["generated_chars"] = len(str(raw))
        if torch.cuda.is_available():
            report["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["scope"] = "load and basic generate only; not chapter-generator qualification"
    # An empty generation is a failed probe, not a pass.
    report["status"] = "PROBE_OK" if report.get("generated_chars") else "PROBE_EMPTY_GENERATION"
    return report


def build(modal: Any, commit: str, profile_name: str, timeout_seconds: int) -> tuple[Any, Any]:
    """The reviewed private-phase image with the probe as the entry point (L4, no network)."""
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise private_phase.PrivatePhaseError("invalid source commit")
    if not 60 <= timeout_seconds <= MAX_TIMEOUT:
        raise private_phase.PrivatePhaseError("probe timeout out of range")
    profile = model_profile.get(profile_name)
    manifest = {"commit": commit, "accelerator": "L4", "timeout_seconds": timeout_seconds}
    return private_modal.factory(modal, manifest, profile=profile, entry=probe_in_container)


def run(
    commit: str,
    profile_name: str,
    timeout_seconds: int,
    *,
    execute: bool,
    journal_root: Path | None = None,
) -> dict[str, Any]:
    plan = {
        "commit": commit,
        "profile": profile_name,
        "accelerator": "L4",
        "timeout_seconds": timeout_seconds,
        "inputs": "made-up public sentence only",
    }
    if not execute:
        return {"status": "PLAN_ONLY", **plan}
    if journal_root is None:
        raise private_phase.PrivatePhaseError("probe journal root required")
    # Reviewed workspace identity first (same check the private phase uses), then a one-shot
    # marker: the same probe plan cannot start twice.
    modal, client = private_modal.identify()
    try:
        journal_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker = journal_root / (
            hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest() + ".started"
        )
        try:
            descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise private_phase.PrivatePhaseError("this probe plan already started") from None
        os.close(descriptor)
        app, function = build(modal, commit, profile_name, timeout_seconds)
        with app.run(client=client, detach=False, environment_name="main"):
            result = function.remote(profile_name)
    finally:
        private_modal.close_client(client)
    return {**result, "plan": plan}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--journal-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.commit,
            args.profile,
            args.timeout_seconds,
            execute=args.execute,
            journal_root=args.journal_root,
        )
    except Exception as exc:
        print(json.dumps({"status": "PROBE_FAILED", "error": type(exc).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
