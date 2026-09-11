from __future__ import annotations

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_refined as refined

# Keep the proven production encoder/effect renderer, but replace its semantic
# planner with the refined V3.1 engagement-driven implementation.
renderer.semantic = refined


if __name__ == "__main__":
    renderer.main()
