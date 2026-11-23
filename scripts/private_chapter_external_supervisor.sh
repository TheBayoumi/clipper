#!/usr/bin/env bash
# OWNER MACHINE ONLY, outside the one approved Codespace. No provisioning.
# Persistent external watchdog: creation-time budget includes installs, human
# review waits and every host operation. Local Python timeout is not a watchdog.
set -euo pipefail
umask 077

: "${CODESPACE_NAME:?Exact existing Codespace name required}"
: "${PRIVATE_CHAPTER_APPROVED_SHA:?Reviewed commit SHA required}"
: "${PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY:?Owner external signing key required}"
: "${PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY:?Owner public key required}"
: "${PRIVATE_CHAPTER_ROOT:?Absolute private Codespace directory required}"
if [[ "${CODESPACES:-}" == "true" ||
      ! "$PRIVATE_CHAPTER_APPROVED_SHA" =~ ^[0-9a-f]{40}$ ||
      "$PRIVATE_CHAPTER_ROOT" != /* ||
      ! -f "$PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY" ||
      ! -f "$PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY" ]]; then
  echo "EXTERNAL_SUPERVISOR_IDENTITY_OR_OWNER_KEY_INVALID" >&2
  exit 2
fi
for binary in gh openssl python3 timeout; do
  command -v "$binary" >/dev/null || {
    echo "EXTERNAL_SUPERVISOR_BOUNDED_EXECUTION_TOOL_MISSING" >&2
    exit 2
  }
done

# Fixed upper bounds, not a caller-overridable new runtime allowance.
# GNU timeout --kill-after prevents commands ignoring SIGTERM forever.
OP_LIMIT=8
KILL_GRACE=2
bounded() {
  timeout --signal=TERM --kill-after="${KILL_GRACE}s" "${OP_LIMIT}s" "$@"
}

# Install cleanup BEFORE the first potentially hung REST/SSH operation.
# If the first identity fetch is unavailable, avoid claiming verified stop;
# owner must independently confirm the exact host identity and deletion.
session="$(mktemp -d)"
chmod 700 "$session"
cleanup_verified=0
terminate_host() {
  trap - EXIT INT TERM HUP
  set +e
  stop_verified=0
  deletion_verified=0
  # Every remote operation in the trap is independently bounded. A hung
  # stop or status call CANNOT suppress the subsequent DELETE attempt.
  bounded gh api -X POST "user/codespaces/$CODESPACE_NAME/stop" >/dev/null 2>&1
  for _ in 1 2 3 4; do
    state="$(bounded gh api "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
      python3 -c 'import json,sys; print(json.load(sys.stdin).get("state",""))' 2>/dev/null || true)"
    if [[ "$state" == "Shutdown" || "$state" == "Stopped" ]]; then
      stop_verified=1
      break
    fi
    sleep 1
  done
  bounded gh api -X DELETE "user/codespaces/$CODESPACE_NAME" >/dev/null 2>&1
  for _ in 1 2 3 4; do
    code="$(bounded gh api -i "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
      head -n 1 | grep -oE '404' || true)"
    if [[ "$code" == "404" ]]; then
      deletion_verified=1
      break
    fi
    sleep 1
  done
  rm -rf -- "$session"
  if [[ "$stop_verified" == 1 && "$deletion_verified" == 1 ]]; then
    cleanup_verified=1
    echo "EXTERNAL_SUPERVISOR_HOST_STOP_AND_DELETION_REST_VERIFIED"
  else
    echo "EXTERNAL_SUPERVISOR_CLEANUP_NOT_VERIFIED_MANUAL_REST_CHECK_REQUIRED" >&2
    return 2
  fi
}
trap terminate_host EXIT INT TERM HUP

metadata="$(bounded gh api "user/codespaces/$CODESPACE_NAME")" || {
  echo "EXTERNAL_SUPERVISOR_CODESPACE_REST_UNAVAILABLE" >&2
  exit 2
}
validated="$(printf '%s' "$metadata" | bounded python3 -c '
import datetime, json, os, sys
x = json.load(sys.stdin)
machine = x.get("machine") or {}
expected = (4, 17179869184, 34359738368)
actual = (machine.get("cpus"), machine.get("memory_in_bytes"), machine.get("storage_in_bytes"))
if (x.get("name") != os.environ["CODESPACE_NAME"]
    or type(x.get("id")) is not int or x["id"] < 1
    or x.get("repository", {}).get("full_name") != "TheBayoumi/clipper"
    or actual != expected):
    sys.exit(2)
created = datetime.datetime.fromisoformat(x["created_at"].replace("Z", "+00:00"))
now = datetime.datetime.now(datetime.timezone.utc)
remaining = int((created + datetime.timedelta(seconds=6900) - now).total_seconds())
if created > now or remaining <= 0:
    sys.exit(2)
print(x["id"], x["created_at"], remaining)
')" || {
  echo "EXTERNAL_SUPERVISOR_WRONG_HOST_OR_DEADLINE" >&2
  exit 2
}
read -r codespace_id created_at initial_remaining <<< "$validated"
remote="$PRIVATE_CHAPTER_ROOT/book-candidates/private-host"

# Remote strings must never allow an operator-provided path to become a
# remote shell command injection. Codespace private root uses a fixed safe set.
if [[ ! "$PRIVATE_CHAPTER_ROOT" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
  echo "EXTERNAL_SUPERVISOR_UNSAFE_REMOTE_PATH" >&2
  exit 2
fi
bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
  "mkdir -p '$remote' && chmod 700 '$remote'" >/dev/null
bounded gh codespace cp -c "$CODESPACE_NAME" \
  "$PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY" \
  "remote:$remote/external-supervisor-public.pem" >/dev/null

# Every iteration bounds both the individual operation and the time remaining.
# Execution stops at host creation +115m; the remaining five minutes are for
# bounded STOP/DELETE and verification. There is NO new time allowance.
while true; do
  remaining="$(bounded python3 -c '
import datetime,sys
created=datetime.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))
print(int((created+datetime.timedelta(seconds=6900)
           -datetime.datetime.now(datetime.timezone.utc)).total_seconds()))
' "$created_at")"
  if (( remaining <= 0 )); then
    break
  fi
  now="$(bounded python3 -c '
import datetime
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
')"
  deadline="$(bounded python3 -c '
import datetime,sys
created=datetime.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))
print((created+datetime.timedelta(seconds=7200)).isoformat())
' "$created_at")"
  bounded python3 -c '
import json,sys
record=dict(
    schema="clipper-external-codespace-supervisor-v1",
    codespace_id=int(sys.argv[1]),
    codespace_name=sys.argv[2],
    created_at=sys.argv[3],
    approved_sha=sys.argv[4],
    observed_at=sys.argv[5],
    hard_deadline_at=sys.argv[6],
    external_watchdog_required=True,
)
with open(sys.argv[7],"wb") as output:
    output.write((json.dumps(record, sort_keys=True, ensure_ascii=False)+"\n").encode())
' "$codespace_id" "$CODESPACE_NAME" "$created_at" \
  "$PRIVATE_CHAPTER_APPROVED_SHA" "$now" "$deadline" "$session/heartbeat.json"
  bounded openssl dgst -sha256 -sign "$PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY" \
    -out "$session/heartbeat.sig" "$session/heartbeat.json" >/dev/null
  bounded gh codespace cp -c "$CODESPACE_NAME" "$session/heartbeat.json" \
    "remote:$remote/external-supervisor-heartbeat.json" >/dev/null
  bounded gh codespace cp -c "$CODESPACE_NAME" "$session/heartbeat.sig" \
    "remote:$remote/external-supervisor-heartbeat.sig" >/dev/null
  sleep "$(( remaining < 10 ? remaining : 10 ))"
done
# The EXIT trap always tries bounded STOP then bounded DELETE and REST checks.
