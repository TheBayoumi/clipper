#!/usr/bin/env bash
# Run on the owner's independent computer OUTSIDE the single approved Codespace.
# Start immediately after Codespace creation, BEFORE any setup in the host.
# Keep this process alive through setup, both chapters, human-review waits and cleanup.
# Never provision another host, change account settings, or fetch private book bytes.
set -euo pipefail
umask 077

: "${CODESPACE_NAME:?Exact already-approved Codespace name is required}"
: "${PRIVATE_CHAPTER_APPROVED_SHA:?Exact post-Codex approved commit is required}"
: "${PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY:?Owner private signing key must remain OUTSIDE Codespace}"
: "${PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY:?Matching public key must be supplied}"
: "${PRIVATE_CHAPTER_ROOT:?Absolute private artifact path in the approved Codespace}"
if [[ "${CODESPACES:-}" == "true" ]]; then
  echo "EXTERNAL_SUPERVISOR_MUST_RUN_OUTSIDE_CODESPACE" >&2
  exit 2
fi
if [[ ! "$PRIVATE_CHAPTER_APPROVED_SHA" =~ ^[a-f0-9]{40}$ ||
      "$PRIVATE_CHAPTER_ROOT" != /* ||
      ! -f "$PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY" ||
      ! -f "$PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY" ]]; then
  echo "EXTERNAL_SUPERVISOR_MISSING_FIXED_IDENTITY_OR_SIGNING_KEY" >&2
  exit 2
fi
for tool in gh openssl python3 timeout; do
  command -v "$tool" >/dev/null || {
    echo "EXTERNAL_SUPERVISOR_REQUIRED_TOOL_MISSING" >&2
    exit 2
  }
done
# The GitHub REST record is the only creation-time clock authority. We never
# accept an invocation-relative timer; 5 min remain for remote stop and deletion.
metadata="$(gh api "user/codespaces/$CODESPACE_NAME")" || {
  echo "EXTERNAL_SUPERVISOR_CODESPACE_REST_UNAVAILABLE" >&2
  exit 2
}
validated="$(printf '%s' "$metadata" | python3 -c '
import datetime, json, os, sys
x = json.load(sys.stdin)
machine = x.get("machine") or {}
expect = (4, 17179869184, 34359738368)
actual = (machine.get("cpus"), machine.get("memory_in_bytes"), machine.get("storage_in_bytes"))
if (x.get("name") != os.environ["CODESPACE_NAME"]
    or type(x.get("id")) is not int or x["id"] < 1
    or x.get("repository", {}).get("full_name") != "TheBayoumi/clipper"
    or actual != expect):
    sys.exit(2)
created = datetime.datetime.fromisoformat(x["created_at"].replace("Z", "+00:00"))
now = datetime.datetime.now(datetime.timezone.utc)
remaining = int((created + datetime.timedelta(seconds=6900) - now).total_seconds())
if created > now or remaining <= 0:
    sys.exit(2)
print(x["id"], x["created_at"], remaining)
')" || {
  echo "EXTERNAL_SUPERVISOR_WRONG_MACHINE_OR_DEADLINE" >&2
  exit 2
}
read -r codespace_id created_at unused_remaining <<< "$validated"
session="$(mktemp -d)"
chmod 700 "$session"
remote="$PRIVATE_CHAPTER_ROOT/book-candidates/private-host"
cleanup_verified=0

terminate_host() {
  # Must execute on *every* exit, including error, operator interrupt and
  # the creation-time cutoff. No reliance on the host's Python finally block.
  trap - EXIT INT TERM
  set +e
  gh api -X POST "user/codespaces/$CODESPACE_NAME/stop" >/dev/null 2>&1
  stop_verified=0
  for _ in 1 2 3 4 5 6; do
    state="$(gh api "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
      python3 -c 'import json,sys; print(json.load(sys.stdin).get("state",""))' 2>/dev/null)"
    if [[ "$state" == "Shutdown" || "$state" == "Stopped" ]]; then
      stop_verified=1
      break
    fi
    sleep 2
  done
  gh api -X DELETE "user/codespaces/$CODESPACE_NAME" >/dev/null 2>&1
  deletion_verified=0
  for _ in 1 2 3 4 5 6; do
    code="$(gh api -i "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
      head -n 1 | grep -oE '404' || true)"
    if [[ "$code" == "404" ]]; then
      deletion_verified=1
      break
    fi
    sleep 2
  done
  rm -rf -- "$session"
  if [[ $stop_verified -eq 1 && $deletion_verified -eq 1 ]]; then
    cleanup_verified=1
    echo "EXTERNAL_SUPERVISOR_HOST_STOP_AND_DELETION_REST_VERIFIED"
  else
    echo "EXTERNAL_SUPERVISOR_CLEANUP_NOT_VERIFIED_OPERATOR_MUST_CHECK" >&2
  fi
}
trap terminate_host EXIT INT TERM

# The owner signing private key never enters the host. Only the public key,
# 45-second heartbeat and its signature are transferred into private storage.
gh codespace ssh -c "$CODESPACE_NAME" -- \
  "mkdir -p '$remote' && chmod 700 '$remote'" >/dev/null
gh codespace cp -c "$CODESPACE_NAME" \
  "$PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY" \
  "remote:$remote/external-supervisor-public.pem" >/dev/null

# Check creation-time cutoff in EACH iteration. External host stopping begins
# at creation+115m, not at creation+120m, reserving cleanup time.
while true; do
  remaining="$(python3 -c '
import datetime, sys
created = datetime.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))
now = datetime.datetime.now(datetime.timezone.utc)
print(int((created+datetime.timedelta(seconds=6900)-now).total_seconds()))
' "$created_at")"
  if (( remaining <= 0 )); then
    break
  fi
  now="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"
  deadline="$(python3 -c '
import datetime,sys
x=datetime.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))
print((x+datetime.timedelta(seconds=7200)).isoformat())
' "$created_at")"
  python3 -c '
import json,sys
obj = dict(
    schema="clipper-external-codespace-supervisor-v1",
    codespace_id=int(sys.argv[1]),
    codespace_name=sys.argv[2],
    created_at=sys.argv[3],
    approved_sha=sys.argv[4],
    observed_at=sys.argv[5],
    hard_deadline_at=sys.argv[6],
    external_watchdog_required=True,
)
with open(sys.argv[7], "wb") as output:
    output.write((json.dumps(obj, sort_keys=True, ensure_ascii=False)+"\n").encode())
' "$codespace_id" "$CODESPACE_NAME" "$created_at" \
  "$PRIVATE_CHAPTER_APPROVED_SHA" "$now" "$deadline" "$session/heartbeat.json"
  openssl dgst -sha256 -sign "$PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY" \
    -out "$session/heartbeat.sig" "$session/heartbeat.json" >/dev/null
  # Fail closed on any transfer failure; trap then tries stop AND deletion.
  gh codespace cp -c "$CODESPACE_NAME" "$session/heartbeat.json" \
    "remote:$remote/external-supervisor-heartbeat.json" >/dev/null
  gh codespace cp -c "$CODESPACE_NAME" "$session/heartbeat.sig" \
    "remote:$remote/external-supervisor-heartbeat.sig" >/dev/null
  sleep "$(( remaining < 15 ? remaining : 15 ))"
done
# Exit trap stops/deletes the SAME known Codespace and verifies REST state.
