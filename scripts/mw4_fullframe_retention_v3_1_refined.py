from __future__ import annotations

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_production as semantic

# Keep the proven production encoder/effect renderer, while using the hardened
# V3.1 semantic planner with verified opening-first Finishing Move stories.
renderer.semantic = semantic


if __name__ == "__main__":
    renderer.main()
