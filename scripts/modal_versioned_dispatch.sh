#!/usr/bin/env bash
# Modal versioned app routing: no GPU deployment when unchanged.
# A different credential is selected ONLY after explicit inference quota exhaustion.
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: modal_versioned_dispatch.sh APP_SCRIPT FUNCTION CLIENT_MODE [client args]" >&2
  exit 2
fi
app_script="$1"; function_tag="$2"; client_mode="$3"; shift 3
[[ -f "$app_script" ]] || { echo "::error::No Modal source script: $app_script"; exit 2; }
[[ "$client_mode" == keyframes || "$client_mode" == speaking ]] || exit 2
case "$client_mode" in
  keyframes)
    [[ "$function_tag" == generate_keyframe ]] || exit 2
    app_base=clipper-qwen-cinematic-keyframes
    ;;
  speaking)
    [[ "$function_tag" == generate_s2v ]] || exit 2
    app_base=clipper-wan-s2v
    ;;
esac

# The app identity changes only when the *remote GPU function definition*
# changes; changing the GitHub workflow or inference client does not redeploy.
source_digest="$(sha256sum "$app_script" | cut -c1-16)"
app_name="${app_base}-${source_digest}"
preferred="${MODAL_ACCOUNT_SLOT:-1}"
[[ "$preferred" =~ ^[123]$ ]] || { echo "::error::Invalid Modal slot"; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "::error::Missing HF_TOKEN"; exit 2; }

for ((slot=preferred; slot<=3; slot++)); do
  if [[ "$slot" == 1 ]]; then
    id_key=MODAL_TOKEN_ID
    secret_key=MODAL_TOKEN_SECRET
  else
    id_key="MODAL_TOKEN_ID_${slot}"
    secret_key="MODAL_TOKEN_SECRET_${slot}"
  fi
  id="${!id_key:-}"
  secret="${!secret_key:-}"
  if [[ -z "$id" && -z "$secret" ]]; then continue; fi
  if [[ -z "$id" || -z "$secret" ]]; then
    echo "::error::Incomplete credentials for Modal slot $slot"
    exit 2
  fi
  # Do not print or write credentials. Every command below is workspace-bound.
  export MODAL_TOKEN_ID="$id" MODAL_TOKEN_SECRET="$secret"
  echo "MODAL_ROUTING_APP=${app_name} WORKSPACE_SLOT=${slot}"
  if ! modal app list --json >/dev/null; then
    echo "::error::Modal authentication failed; NO account failover"
    exit 1
  fi

  probe_log="$(mktemp)"
  if python scripts/modal_deployment_probe.py "$app_name" "$function_tag" >"$probe_log" 2>&1; then
    rm -f "$probe_log"
    echo "MODAL_DEPLOY_REUSED version=${source_digest} slot=${slot}"
  else
    probe_status=$?
    if [[ "$probe_status" -ne 44 ]]; then
      tail -n 35 "$probe_log" >&2
      rm -f "$probe_log"
      echo "::error::Modal deployment lookup failed; NO account failover"
      exit 1
    fi
    rm -f "$probe_log"
    # Backup workspaces are initialized lazily and only when selected.
    if ! modal secret create clipper-wan-hf "HF_TOKEN=$HF_TOKEN" --force >/dev/null; then
      echo "::error::Modal secret provisioning failure; NO account failover"
      exit 1
    fi
    deploy_log="$(mktemp)"
    if CLIPPER_MODAL_APP_NAME="$app_name" modal deploy "$app_script" >"$deploy_log" 2>&1; then
      tail -n 18 "$deploy_log"
      echo "MODAL_DEPLOY_CREATED version=${source_digest} slot=${slot}"
    elif grep -Fq "already taken by a" "$deploy_log" && grep -Fq "deployed app" "$deploy_log"; then
      # A parallel shot raced to deploy this identical immutable version.
      # Never switch account or overwrite it. Wait for the winner to hydrate.
      echo "MODAL_DEPLOY_CONCURRENT_WINNER_DETECTED slot=${slot}"
      ready=false
      for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
        if python scripts/modal_deployment_probe.py "$app_name" "$function_tag" >/dev/null 2>&1; then
          ready=true
          break
        fi
        sleep 5
      done
      if [[ "$ready" != true ]]; then
        tail -n 40 "$deploy_log" >&2
        echo "::error::Concurrent deployment did not become ready; NO account failover"
        rm -f "$deploy_log"
        exit 1
      fi
      echo "MODAL_DEPLOY_REUSED_AFTER_RACE version=${source_digest} slot=${slot}"
    else
      tail -n 40 "$deploy_log" >&2
      echo "::error::Modal deployment failure; NO account failover"
      rm -f "$deploy_log"
      exit 1
    fi
    rm -f "$deploy_log"
  fi

  inference_log="$(mktemp)"
  if python scripts/modal_deployed_client.py "$client_mode" --app "$app_name" "$@" >"$inference_log" 2>&1; then
    cat "$inference_log"
    rm -f "$inference_log"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
      echo "MODAL_ACCOUNT_SLOT=$slot" >> "$GITHUB_ENV"
    fi
    mkdir -p artifacts/cinematic/evidence
    printf '{"app_name":"%s","source_digest":"%s","credential_slot":%s,"publish_eligible":false}\n' \
      "$app_name" "$source_digest" "$slot" \
      > "artifacts/cinematic/evidence/${client_mode}-modal-route.json"
    echo "MODAL_INFERENCE_COMPLETED slot=$slot"
    exit 0
  fi

  # Only a terminal explicit inference-quota message permits fallback.
  # Budget, credits, auth, deployment, timeout, CUDA, model and transient failures
  # are *never* considered quota. Inspect last nonempty line, not warning history.
  last_line="$(awk 'NF { last=$0 } END { print last }' "$inference_log")"
  if printf '%s\n' "$last_line" | grep -Eiq 'quota[[:space:]:_-]*(exceeded|exhausted|reached)|quota[[:space:]:_-]*limit[[:space:]:_-]*(exceeded|reached)|exceeded[[:space:]]+(the[[:space:]]+)?(workspace[[:space:]]+)?(GPU[[:space:]]+)?quota|reached[[:space:]]+(the[[:space:]]+)?(workspace[[:space:]]+)?quota'; then
    tail -n 40 "$inference_log" >&2
    rm -f "$inference_log"
    echo "::warning::Explicit inference quota exhaustion on slot $slot; trying next slot"
  else
    tail -n 80 "$inference_log" >&2
    rm -f "$inference_log"
    echo "::error::Non-quota inference failure; NO account failover"
    exit 1
  fi
done
echo "::error::All configured Modal inference quota slots exhausted"
exit 1
