"""GPU-free, SDK-native lookup of a deployed, source-versioned Modal function.

Exit 0: deployed function available.
Exit 44: function/application is absent (safe to deploy).
Any other error: authentication, connectivity or API failure (must NOT fail over).
"""

from __future__ import annotations

import argparse
import os
import sys

import modal
from modal.exception import NotFoundError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_name")
    parser.add_argument("function_name")
    args = parser.parse_args()
    try:
        fn = modal.Function.from_name(
            args.app_name,
            args.function_name,
            environment_name=os.environ.get("MODAL_ENVIRONMENT", "main"),
        )
        fn.hydrate()
    except NotFoundError:
        print("MODAL_DEPLOYMENT_NOT_FOUND", file=sys.stderr)
        return 44
    except Exception as exc:
        print(f"MODAL_DEPLOYMENT_LOOKUP_FAILED type={type(exc).__name__}", file=sys.stderr)
        return 1
    print("MODAL_DEPLOYMENT_FOUND")
    return 0


if __name__ == "__main__":
    sys.exit(main())
