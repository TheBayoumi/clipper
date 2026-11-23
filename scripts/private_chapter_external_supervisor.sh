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
: "${PRIVATE_CHAPTER_RECOVERY_DIR:?Absolute owner-side private recovery directory required}"
if [[ "${CODESPACES:-}" == "true" ||
      ! "$PRIVATE_CHAPTER_APPROVED_SHA" =~ ^[0-9a-f]{40}$ ||
      "$PRIVATE_CHAPTER_ROOT" != /* ||
      "$PRIVATE_CHAPTER_RECOVERY_DIR" != /* ||
      -L "$PRIVATE_CHAPTER_RECOVERY_DIR" ||
      ! "$CODESPACE_NAME" =~ ^[a-zA-Z0-9_-]+$ ||
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
deadline_epoch=0
recovery_epoch=0
bounded() {
  local limit="$OP_LIMIT" remaining
  if (( deadline_epoch > 0 )); then
    remaining="$(( deadline_epoch - $(date -u +%s) ))"
    if (( remaining <= 0 )); then return 124; fi
    if (( remaining < limit )); then limit="$remaining"; fi
  fi
  if (( recovery_epoch > 0 )); then
    remaining="$(( recovery_epoch - $(date -u +%s) ))"
    if (( remaining <= 0 )); then return 124; fi
    if (( remaining < limit )); then limit="$remaining"; fi
  fi
  timeout --signal=TERM --kill-after="${KILL_GRACE}s" "${limit}s" "$@"
}

# Install cleanup BEFORE the first potentially hung REST/SSH operation.
# If the first identity fetch is unavailable, avoid claiming verified stop;
# owner must independently confirm the exact host identity and deletion.
session="$(mktemp -d)"
chmod 700 "$session"
cleanup_verified=0
verified_identity=0

# Initial identity alone is NOT permission to DELETE a name that has since
# been replaced. Recheck numeric instance ID + repo + name + creation stamp
# immediately before every destructive operation. Unavailable REST fails shut.
identity_matches() {
  bounded gh api "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
    bounded python3 -c '
import json, os, sys
try:
    x = json.load(sys.stdin)
    machine = x.get("machine") or {}
    valid = (
        x.get("id") == int(os.environ["SUPERVISED_CODESPACE_ID"])
        and type(x.get("id")) is int
        and x.get("name") == os.environ["CODESPACE_NAME"]
        and x.get("repository", {}).get("full_name") == "TheBayoumi/clipper"
        and x.get("created_at") == os.environ["SUPERVISED_CREATED_AT"]
        and (machine.get("cpus"), machine.get("memory_in_bytes"), machine.get("storage_in_bytes"))
            == (4, 17179869184, 34359738368)
    )
except (ValueError, TypeError, KeyError, AttributeError):
    sys.exit(2)
sys.exit(0 if valid else 2)
' 2>/dev/null
}

recover_private() (
  # Recovery has its own strict 16-second ceiling *inside* the already reserved
  # external cleanup window. Subshell prevents the cap contaminating STOP/DELETE.
  recovery_epoch="$(( $(date -u +%s) + 16 ))"
  # Owner-side only. Neither status nor private artifacts enter git, logs or
  # ChatGPT; failure here must NEVER postpone STOP/DELETE.
  if ! identity_matches; then return 2; fi
  local destination="$PRIVATE_CHAPTER_RECOVERY_DIR/$CODESPACE_NAME"
  mkdir -p -m 700 -- "$destination" || return 2
  chmod 700 "$destination" || return 2
  if bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
    "python3 -m clipper.books.chapter_host status --artifact-root '$PRIVATE_CHAPTER_ROOT'" \
    >"$session/recovery.json" 2>/dev/null; then
    if bounded python3 -c '
import json, re, sys
with open(sys.argv[1], "rb") as f: raw = f.read(16385)
if len(raw) > 16384: sys.exit(2)
x = json.loads(raw)
if (x.get("schema") != "clipper-private-chapter-safe-recovery-v1"
    or x.get("status") != "PRIVATE_CHAPTER_RECOVERY_CHECKPOINT_ONLY"
    or set(x.get("chapters", {})) != {"1","7"}
    or x.get("publish_eligible") is not False): sys.exit(2)
for item in x["chapters"].values():
    if (item.get("recovery_state") not in
        {"NEVER_STARTED","IN_FLIGHT","INTERRUPTED","AWAITING_REVIEW","READY"}
        or not isinstance(item.get("last_subphase"), str)
        or len(item["last_subphase"]) > 64): sys.exit(2)
' "$session/recovery.json"; then
      cp -- "$session/recovery.json" "$destination/safe-recovery-status.json"
      chmod 600 "$destination/safe-recovery-status.json"
    fi
  fi
  local section
  for section in private-host chapter-evaluations brief adaptation chapter_script chapter_direction; do
    if ! identity_matches; then break; fi
    bounded gh codespace cp --recursive -c "$CODESPACE_NAME" \
      "remote:$PRIVATE_CHAPTER_ROOT/book-candidates/$section" \
      "$destination/" >/dev/null 2>&1 || true
  done
)

terminate_host() {
  original_status=$?
  trap - EXIT INT TERM HUP
  set +e
  if [[ "$verified_identity" != 1 ]]; then
    # NEVER send destructive REST operations to an unverified repository or
    # Codespace name. No safe automatic deletion exists without that identity.
    rm -rf -- "$session"
    echo "EXTERNAL_SUPERVISOR_IDENTITY_UNVERIFIED_OWNER_MUST_CHECK_HOST" >&2
    exit 2
  fi
  stop_verified=0
  deletion_verified=0
  # Recover safe status and bounded private outputs before deletion, within
  # the original shared deadline. Failed retrieval never vetoes cleanup.
  recover_private >/dev/null 2>&1 || true
  if ! identity_matches; then
    rm -rf -- "$session"
    echo "EXTERNAL_SUPERVISOR_REJECTED_HOST_IDENTITY_NO_DESTRUCTIVE_CALL" >&2
    exit 2
  fi
  # Every remote operation in the trap is independently bounded. A hung
  # stop or status call CANNOT suppress the subsequent DELETE attempt.
  bounded gh api -X POST "user/codespaces/$CODESPACE_NAME/stop" >/dev/null 2>&1
  for _ in 1 2 3 4; do
    state="$(bounded gh api "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
      bounded python3 -c 'import json,sys; print(json.load(sys.stdin).get("state",""))' 2>/dev/null || true)"
    if [[ "$state" == "Shutdown" || "$state" == "Stopped" ]]; then
      stop_verified=1
      break
    fi
    sleep 1
  done
  if ! identity_matches; then
    rm -rf -- "$session"
    echo "EXTERNAL_SUPERVISOR_REJECTED_HOST_IDENTITY_NO_DELETE" >&2
    exit 2
  fi
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
    exit "$original_status"
  fi
  echo "EXTERNAL_SUPERVISOR_CLEANUP_NOT_VERIFIED_MANUAL_REST_CHECK_REQUIRED" >&2
  exit 2
}
# Signal handlers must TERMINATE; do not resume an interrupted sign/copy loop
# with traps removed after a Ctrl+C or external TERM.
trap 'exit 2' INT TERM HUP
trap terminate_host EXIT

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
export SUPERVISED_CODESPACE_ID="$codespace_id"
export SUPERVISED_CREATED_AT="$created_at"
deadline_epoch="$(( $(date -u -d "$created_at" +%s) + 7200 ))"
verified_identity=1
remote="$PRIVATE_CHAPTER_ROOT/book-candidates/private-host"

# Remote strings must never allow an operator-provided path to become a
# remote shell command injection. Codespace private root uses a fixed safe set.
if [[ ! "$PRIVATE_CHAPTER_ROOT" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
  echo "EXTERNAL_SUPERVISOR_UNSAFE_REMOTE_PATH" >&2
  exit 2
fi
identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_CHANGED" >&2; exit 2; }
bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
  "mkdir -p '$remote' && chmod 700 '$remote'" >/dev/null
identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_CHANGED" >&2; exit 2; }
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
  # The signature and exact payload are a single bounded envelope. Copy to
  # a staging path, then atomically rename it on the SAME remote filesystem.
  bounded python3 -c '
import base64, json, sys
payload = open(sys.argv[1], "rb").read()
signature = open(sys.argv[2], "rb").read()
assert len(payload) <= 8192 and len(signature) <= 8192
envelope = {
  "schema": "clipper-external-supervisor-envelope-v1",
  "payload_b64": base64.b64encode(payload).decode("ascii"),
  "signature_b64": base64.b64encode(signature).decode("ascii"),
}
with open(sys.argv[3], "wb") as f:
    f.write((json.dumps(envelope, sort_keys=True) + "\n").encode())
' "$session/heartbeat.json" "$session/heartbeat.sig" "$session/envelope.json"
  identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_CHANGED" >&2; exit 2; }
  bounded gh codespace cp -c "$CODESPACE_NAME" "$session/envelope.json" \
    "remote:$remote/external-supervisor-heartbeat-envelope.watchdog.pending" >/dev/null
  identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_CHANGED" >&2; exit 2; }
  bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
    "mv -f -- '$remote/external-supervisor-heartbeat-envelope.watchdog.pending' '$remote/external-supervisor-heartbeat-envelope.json'" >/dev/null
  sleep "$(( remaining < 10 ? remaining : 10 ))"
done
# The EXIT trap always tries bounded STOP then bounded DELETE and REST checks.
