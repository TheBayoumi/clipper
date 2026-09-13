from __future__ import annotations

import sys

import mw4_v3_1_contract_legacy as _legacy
from mw4_v3_1_contract_legacy import *  # noqa: F401,F403
from mw4_v3_1_lossless import preflight as _lossless_preflight


def main() -> None:
    if "--self-test" in sys.argv:
        _lossless_preflight()
    _legacy.main()


if __name__ == "__main__":
    main()
