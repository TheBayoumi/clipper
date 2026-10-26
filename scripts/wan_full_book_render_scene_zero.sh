#!/usr/bin/env bash
# Render only scene 00 from the current baritone narration and Qwen keyframe.
# No historical video, historical audio, or historical keyframe may satisfy it.
set -euo pipefail

PLAN="${1:-artifacts/full-book/source/plan.json}"
IMAGE="${2:-artifacts/full-book/frames/hook.png}"
AUDIO="${3:-artifacts/full-book/source/scene-00.wav}"
OUT="${4:-artifacts/full-book/rendered/scene-00}"

[[ "${MODAL_ACCOUNT_SLOT:-}" == "3" ]] || {
  echo "::error::Scene-zero isolated rerender must target backup-2 slot 3"; exit 2;
}
[[ -s "$PLAN" && -s "$IMAGE" && -s "$AUDIO" ]] || {
  echo "::error::Current narration plan, scene-zero Qwen PNG and WAV are required"; exit 2;
}
readarray -t FIELDS < <(python - "$PLAN" "$IMAGE" "$AUDIO" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
shot = plan["shots"][0]
audio = Path(sys.argv[3])
if shot["id"] != "scene-00" or shot["keyframe_name"] != "hook":
    raise SystemExit("::error::Source plan is not the locked scene-zero hook")
if plan["voice"]["voice_id"] != "am_adam" or plan["voice"]["style"] != "original_warm_low_cinematic_baritone":
    raise SystemExit("::error::Expected approved open-source low-baritone voice identity")
if shot["audio_sha256"] != hashlib.sha256(audio.read_bytes()).hexdigest():
    raise SystemExit("::error::Scene-zero audio does not match source plan")
if not Path(sys.argv[2]).is_file():
    raise SystemExit("::error::Current Qwen keyframe is missing")
print(shot["seed"])
print(shot["prompt"])
PY
)
[[ "${#FIELDS[@]}" -eq 2 ]] || { echo "::error::Invalid locked scene-zero prompt/seed"; exit 2; }
SEED="${FIELDS[0]}" PROMPT="${FIELDS[1]}"
mkdir -p "$OUT"
# Reuse is allowed only if the image, newer WAV, prompt, seed, revision,
# and candidate video SHA all match this exact source plan.
python scripts/wan_full_book_shot_cache.py --shot 0 \
  --plan "$PLAN" --output "$OUT" --reference "$IMAGE" --audio "$AUDIO"
if PYTHONPATH=scripts python - "$OUT" "$IMAGE" "$AUDIO" "$PROMPT" "$SEED" <<'PY'
import hashlib
import sys
from pathlib import Path
from wan_shared_shot_cache import verified_wan_output

folder, image, audio, prompt, seed = sys.argv[1:]
hash_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
valid = verified_wan_output(
    Path(folder) / "wan-s2v-candidate.mp4",
    Path(folder) / "model-evidence.json",
    hash_file(image), hash_file(audio), prompt, int(seed),
)
print("SCENE_ZERO_CURRENT_VOICE_CACHE", "HIT" if valid else "MISS")
sys.exit(0 if valid else 1)
PY
then
  echo "SCENE_ZERO_NEW_VOICE_GPU_SKIPPED_VERIFIED_CACHE"
else
  [[ -n "${HF_TOKEN:-}" ]] || { echo "::error::Missing HF_TOKEN for Modal Wan"; exit 2; }
  bash scripts/modal_versioned_dispatch.sh \
    scripts/modal_wan_s2v.py generate_s2v speaking \
    --reference "$IMAGE" --audio "$AUDIO" \
    --output "$OUT" --prompt "$PROMPT" --seed "$SEED"
fi
PYTHONPATH=scripts python - "$OUT" "$IMAGE" "$AUDIO" "$PROMPT" "$SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from wan_shared_shot_cache import verified_wan_output

folder, image, audio, prompt, seed = sys.argv[1:]
folder = Path(folder)
hash_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert verified_wan_output(
    folder / "wan-s2v-candidate.mp4", folder / "model-evidence.json",
    hash_file(image), hash_file(audio), prompt, int(seed),
), "No current-source scene-zero Wan video exists"
record = json.loads((folder / "model-evidence.json").read_text())
assert record["audio_sha256"] == hash_file(audio)
assert record["image_sha256"] == hash_file(image)
assert not (folder / "visual-reuse.json").exists(), "Legacy muted B-roll is forbidden"
print("SCENE_ZERO_CURRENT_QWEN_CURRENT_BARITONE_WAN_VERIFIED")
PY
