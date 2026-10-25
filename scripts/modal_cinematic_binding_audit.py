"""Read-only audit: resolve active Modal account and canonical Qwen/Wan app bindings.

Never attempts to deploy, create a Modal Secret, allocate a GPU, or call a
function. All IDs are fetched from the live backend, never loaded from cache.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import modal
from modal.exception import NotFoundError

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = (
    (
        "qwen",
        "scripts/modal_qwen_keyframes.py",
        "clipper-qwen-cinematic-keyframes",
        "generate_keyframe",
    ),
    ("wan", "scripts/modal_wan_s2v.py", "clipper-wan-s2v", "generate_s2v"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()

    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        raise SystemExit("::error::Cannot audit without explicit Modal token pair")
    env_name = os.environ.get("MODAL_ENVIRONMENT")
    if env_name != "main":
        raise SystemExit("::error::Modal binding audit requires main environment")

    workspace = modal.Workspace.from_context()
    attribute = workspace.name
    name = attribute() if callable(attribute) else attribute
    if name != args.expected:
        raise SystemExit("::error::Current workspace differs from requested workspace")
    print(f"MODAL_BINDING_WORKSPACE={name} ENVIRONMENT={env_name}")

    for kind, script, app_base, function_name in FUNCTIONS:
        script_bytes = (ROOT / script).read_bytes()
        digest = hashlib.sha256(script_bytes).hexdigest()[:16]
        app_name = f"{app_base}-{digest}"
        print(f"MODAL_BINDING_LOOKUP kind={kind} app={app_name} function={function_name}")
        try:
            function = modal.Function.from_name(
                app_name,
                function_name,
                environment_name=env_name,
            )
            function.hydrate()
        except NotFoundError:
            print(f"MODAL_BINDING_RESULT kind={kind} state=ABSENT_IN_CURRENT_WORKSPACE")
            continue
        # A found function is resolved live from this account; any ID reported
        # by the SDK is metadata, not a persisted pointer used for inference.
        function_id = getattr(function, "object_id", None)
        print(f"MODAL_BINDING_RESULT kind={kind} state=FOUND_CURRENT_WORKSPACE")
        if function_id:
            print(f"MODAL_BINDING_LIVE_FUNCTION_ID kind={kind} id={function_id}")


if __name__ == "__main__":
    main()
