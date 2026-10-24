# Creative milestone, source-of-truth and blocking defect

User-approved reference images (2026-10-24) are **the creative specification**
not merely a moodboard. Their SHA256 values are pinned in
`niches/atomic-habits-approved-storyboard.json`:
- first storyboard screenshot: `0ce8c2f7898275b6dbc6389eab1850f6ca1c8c19eb307e69be70547fed023181`
- second storyboard screenshot: `d0edab069a137a9eb27db2db6d9bb0d1d5daa48ab34182f917735abc2f00ccbf`
- approved-direction character turnaround: `85561d67fc9b3767f404987ffb907f71a779bd5f738fbfe0b73a4fbe8134486a`

### Current blockers, verified
1. The MakeHuman rig that passed GitHub Actions is visually **not** the
   approved curly-haired glasses-wearing dark-hoodie animated presenter.
   It is a man in a dark casual button-down outfit.
2. The existing Three.js production proof explicitly hardcodes generic
   thoughtful/talk/look clips and three 4-second scenes. It does NOT receive
   a storyboard, prop-interaction contract, or persistent approved face.
3. The earlier motion animatic is still artwork with camera movement.
4. The connected Pika video wallet currently reports **0** credits and the
   available environment has no NVIDIA GPU. Generative image-to-video
   cannot be assumed available. An image collage is not a 3D animation asset.

### Resolution contract
- Replace `male_casualsuit01` and `business-male-01.glb` with a real
  **approved stylized rigged master**: curly dark hair, black rectangular
  glasses, expressive brown eyes, charcoal hoodie, realistic warm lighting.
- Three distinct 0–15s shots: hook (0–3s), book introduction (3–8s),
  scene-specific physical environment/prop action (8–15s). Actions must be
  tied to the character rig and actual audio, not camera zoom or static scenes.
- Script and human reviewer must verify the *exact* 1080×1920 MP4 with
  source/animation manifests, hashes, visible action, proper expressions
  and complete Kokoro speech.
- A 3D joint-motion metric and CI test are necessary technical probes but
  **insufficient for creative acceptance**.
- Keep publication disabled, avoid repeated LLM assessments, preserve
  Qwen3/Kokoro caches and existing PR #10.

The script `scripts/storyboard_acceptance.py` is fail-closed. Passing its
contract validation **does not mean the professional video has been produced**.
Its `audit_render` mode requires an approved master rig hash, matching
physical action proof for all three shots, voice timing, completed video,
and human visual approval still remains separate.

The full 60s storyboard has eight beats defined for subsequent production,
but the next milestone is **one human-approved 15-second sample**.
