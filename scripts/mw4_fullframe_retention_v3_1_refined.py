from __future__ import annotations

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_final as semantic

# Keep the proven production encoder/effect renderer, but route selection and
# editing through the final V3.1 semantic planner.
renderer.semantic = semantic


if __name__ == "__main__":
    renderer.main()
