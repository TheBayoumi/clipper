#!/usr/bin/env bash
# Explicitly redeploy canonical Qwen/Wan Modal app names on backup slot 2.
# Unlike modal_versioned_dispatch.sh, DO NOT probe/reuse a previous deployment.
set -euo pipefail
[[ -n "${MODAL_TOKEN_ID_2:-}" && -n "${MODAL_TOKEN_SECRET_2:-}" ]] || {
  echo "::error::Backup slot 2 credentials are incomplete"; exit 2;
}

# Create a private per-run profile for the actual GitHub backup-1 token pair.
# Modal environment tokens override active profiles, so remove them after
# configuring this profile to ensure 'modal profile activate' is meaningful.
export MODAL_ENVIRONMENT=main
unset MODAL_OAUTH_REFRESH_TOKEN MODAL_OAUTH_CLIENT_ID MODAL_OAUTH_CLIENT_SECRET || true
umask 077
MODAL_PROFILE_TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$MODAL_PROFILE_TEMP_DIR"' EXIT
export MODAL_CONFIG_PATH="$MODAL_PROFILE_TEMP_DIR/.modal.toml"
python - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ["MODAL_CONFIG_PATH"])
config_path.write_text(
    "[mahmoud-bayoumi-bb]\n"
    + "token_id = " + json.dumps(os.environ["MODAL_TOKEN_ID_2"]) + "\n"
    + "token_secret = " + json.dumps(os.environ["MODAL_TOKEN_SECRET_2"]) + "\n",
    encoding="utf-8",
)
config_path.chmod(0o600)
PY
unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET MODAL_PROFILE
echo "MODAL_BACKUP1_PROFILE_ACTIVATION_START"
modal profile activate mahmoud-bayoumi-bb
modal profile current
echo "MODAL_BACKUP1_PROFILE_ACTIVATED"
echo "MODAL_REDEPLOY_REQUESTED_SLOT=2"
modal token info >/tmp/clipper-backup1-modal-token-info
python - <<'PY'
import re
from pathlib import Path

raw = re.sub(r"\x1b\[[0-9;]*m", "", Path("/tmp/clipper-backup1-modal-token-info").read_text())
regexes = (
    r"(?im)^\s*workspace(?:\s+name)?\s*[:=]\s*([a-zA-Z0-9_.-]+)",
    r"(?im)authenticated\s+to\s+workspace\s+([a-zA-Z0-9_.-]+)",
    r"(?im)workspace\s+['\"]([a-zA-Z0-9_.-]+)['\"]",
)
workspace = next((m.group(1) for regex in regexes if (m := re.search(regex, raw))), None)
print(f"MODAL_EFFECTIVE_WORKSPACE={workspace or 'UNRESOLVED'}")
if workspace != "mahmoud-bayoumi-bb":
    raise SystemExit("::error::Current Modal token does not identify mahmoud-bayoumi-bb")
print("MODAL_WORKSPACE_IDENTITY_GUARD_PASSED")
PY

modal app list --json > /tmp/clipper-backup1-apps-before.json
echo "MODAL_BACKUP1_PREDEPLOY_LISTED (JSON stored in runner temp)"
if [[ "${MODAL_PROFILE_ACTIVATION_ONLY:-0}" == "1" ]]; then
  echo "MODAL_BACKUP1_PROFILE_ONLY_PASS workspace=mahmoud-bayoumi-bb environment=main"
  exit 0
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "::error::Missing HF_TOKEN"; exit 2; }
modal secret create clipper-wan-hf "HF_TOKEN=$HF_TOKEN" --force >/dev/null

# Deploy immutable canonical app names, so later Clipper production resolves them.
# NEVER skip deploy due to an existing app/cache. CLI prints real build/deploy logs.
for kind in qwen wan; do
  if [[ "$kind" == qwen ]]; then
    src=scripts/modal_qwen_keyframes.py
    base=clipper-qwen-cinematic-keyframes
  else
    src=scripts/modal_wan_s2v.py
    base=clipper-wan-s2v
  fi
  digest="$(sha256sum "$src" | cut -c1-16)"
  app="${base}-${digest}"
  echo "MODAL_FORCED_REDEPLOY_STARTED kind=$kind app=$app workspace=mahmoud-bayoumi-bb"
  CLIPPER_MODAL_APP_NAME="$app" modal deploy --name "$app" --env main --timestamps "$src"
  echo "MODAL_FORCED_REDEPLOY_COMPLETED kind=$kind app=$app"
  modal app info --env main --json "$app" >"/tmp/clipper-backup1-${kind}-info.json"
  echo "MODAL_REMOTE_DEPLOYMENT_VERIFIED kind=$kind app=$app"
done

echo "MODAL_BACKUP1_A100_INFERENCE_PROBE_START"
modal run --env main --timestamps scripts/modal_a100_access_probe.py
echo "MODAL_BACKUP1_A100_INFERENCE_PROBE_PASS"
