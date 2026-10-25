"""Fail-closed Modal workspace identity validation before any deployment/inference.

The identity is resolved by Modal's live API using the token pair in this
process. No function IDs, app IDs, token values or old profile files are used.
"""

from __future__ import annotations

import argparse
import os

import modal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()

    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        raise SystemExit("::error::Explicit Modal API token pair is required")

    workspace = modal.Workspace.from_context()
    name_attribute = getattr(workspace, "name")
    name = name_attribute() if callable(name_attribute) else name_attribute
    if name != args.expected:
        raise SystemExit("::error::Modal active workspace differs from requested backup")

    environment = os.environ.get("MODAL_ENVIRONMENT")
    if environment != "main":
        raise SystemExit("::error::Modal environment must be pinned to main")

    print(f"MODAL_WORKSPACE_CONFIRMED name={name} environment={environment}")


if __name__ == "__main__":
    main()
