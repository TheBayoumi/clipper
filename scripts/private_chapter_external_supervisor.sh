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
recovery_allowed=0
remote_safe=0
cleanup_only=0

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
  # Recovery is always best-effort and bounded within original cleanup reserve.
  recovery_epoch="$(( $(date -u +%s) + 16 ))"
  if (( recovery_allowed != 1 )) || ! identity_matches; then return 2; fi
  local destination="$PRIVATE_CHAPTER_RECOVERY_DIR/$CODESPACE_NAME"
  bounded mkdir -p -m 700 -- "$destination" || return 2
  bounded chmod 700 "$destination" || return 2
  # One scoped, allowlisted private output TAR includes final WAVs, text
  # validation records, narration reports and safe status; never a whole EPUB.
  if bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
    "python3 -m clipper.books.chapter_host recover --artifact-root '$PRIVATE_CHAPTER_ROOT'" \
    >"$session/recovery-created.json" 2>/dev/null; then
    if identity_matches && bounded gh codespace cp -c "$CODESPACE_NAME" \
      "remote:$PRIVATE_CHAPTER_ROOT/book-candidates/private-host/chapter-recovery.tar" \
      "$destination/chapter-recovery.tar" >/dev/null 2>&1; then
      if bounded python3 -c '
import hashlib, json, pathlib, re, sys, tarfile
created_path, archive_path, report_path, safe_path = map(pathlib.Path, sys.argv[1:])
dig = re.compile(r"[0-9a-f]{64}\\Z")
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
created_raw = created_path.read_bytes()
if len(created_raw) > 2048:
    sys.exit(2)
created = json.loads(created_raw)
if (created.get("schema") != "clipper-private-chapter-recovery-created-v1"
    or not isinstance(created.get("archive_sha256"), str)
    or dig.fullmatch(created["archive_sha256"]) is None):
    sys.exit(2)
if archive_path.is_symlink() or not archive_path.is_file() or archive_path.stat().st_size > 800_000_000:
    sys.exit(2)
h = hashlib.sha256()
with archive_path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        h.update(chunk)
if h.hexdigest() != created["archive_sha256"]:
    sys.exit(2)
with tarfile.open(archive_path, mode="r:") as source:
    members = source.getmembers()
    if len(members) > 514 or any(not x.isfile() for x in members):
        sys.exit(2)
    names = [x.name for x in members]
    if len(names) != len(set(names)) or "private-recovery-manifest.json" not in names or "safe-recovery-status.json" not in names:
        sys.exit(2)
    manifest_member = source.extractfile("private-recovery-manifest.json")
    safe_member = source.extractfile("safe-recovery-status.json")
    if manifest_member is None or safe_member is None:
        sys.exit(2)
    manifest_bytes = manifest_member.read(200_001)
    safe_bytes = safe_member.read(16_385)
    if len(manifest_bytes) > 200_000 or len(safe_bytes) > 16_384:
        sys.exit(2)
    manifest, safe = json.loads(manifest_bytes), json.loads(safe_bytes)
    if (manifest.get("schema") != "clipper-chapter-private-recovery-archive-manifest-v1"
        or digest(manifest_bytes) != created.get("manifest_sha256")
        or manifest.get("runtime_status_sha256") != digest(safe_bytes)
        or safe.get("schema") != "clipper-private-chapter-safe-recovery-v1"
        or set(safe.get("chapters", {})) != {"1", "7"}
        or safe.get("publish_eligible") is not False
        or safe.get("files_retrieved") is not False):
        sys.exit(2)
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) > 512:
        sys.exit(2)
    expected = {"private-recovery-manifest.json", "safe-recovery-status.json"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"chapter", "kind", "relative_path", "sha256", "bytes"}:
            sys.exit(2)
        path = row["relative_path"]
        if (row["chapter"] not in (1, 7)
            or row["kind"] not in {"text","wav","narration_report","review","final_result","source_ledger","human_review"}
            or not isinstance(path, str) or not path.startswith("book-candidates/")
            or any(x in ("", ".", "..") for x in pathlib.PurePosixPath(path).parts)
            or not isinstance(row["sha256"], str) or dig.fullmatch(row["sha256"]) is None
            or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= 300_000_000):
            sys.exit(2)
        expected.add(path)
        member = source.getmember(path)
        if member.size != row["bytes"]:
            sys.exit(2)
        content = source.extractfile(member)
        if content is None:
            sys.exit(2)
        h = hashlib.sha256()
        for chunk in iter(lambda: content.read(1024 * 1024), b""):
            h.update(chunk)
        if h.hexdigest() != row["sha256"]:
            sys.exit(2)
    if set(names) != expected or len(names) != len(expected):
        sys.exit(2)
    for chapter in safe["chapters"].values():
        if (not isinstance(chapter, dict)
            or chapter.get("recovery_state") not in {"NEVER_STARTED","IN_FLIGHT","INTERRUPTED","AWAITING_REVIEW","READY"}
            or not isinstance(chapter.get("last_subphase"), str)
            or len(chapter["last_subphase"]) > 64
            or chapter.get("retrieval_status") != "NOT_ATTEMPTED"
            or not isinstance(chapter.get("last_verified_artifact_identities"), dict)):
            sys.exit(2)
    summary = {
        "schema": "clipper-private-owner-recovery-verification-v1",
        "retrieval_status": "VERIFIED_PRIVATE_ARCHIVE",
        "recovered_file_count": len(rows),
        "output_scope": ("COMPLETE" if all(x["recovery_state"] == "READY"
                                      for x in safe["chapters"].values()) else "PARTIAL"),
        "host_cleanup_status": "NOT_YET_REST_VERIFIED",
    }
    report_path.write_text(json.dumps(summary, sort_keys=True) + "\\n", encoding="utf-8")
    safe_path.write_bytes(safe_bytes)
' "$session/recovery-created.json" "$destination/chapter-recovery.tar" \
  "$destination/retrieval-verification.json" "$destination/safe-recovery-status.json"; then
        bounded chmod 600 "$destination/chapter-recovery.tar" \
          "$destination/retrieval-verification.json" \
          "$destination/safe-recovery-status.json" || true
        return 0
      fi
    fi
  fi
  # Incomplete is distinct from REST cleanup; never report files recovered.
  bounded rm -f -- "$destination/chapter-recovery.tar" || true
  bounded python3 -c '
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"schema":"clipper-private-owner-recovery-verification-v1",
               "retrieval_status":"INCOMPLETE",
               "host_cleanup_status":"NOT_YET_REST_VERIFIED"}, f)
' "$destination/retrieval-verification.json" || true
  # Fallback safe phase-only checkpoint, still within the same retrieval cap.
  if identity_matches; then
    bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
      "python3 -m clipper.books.chapter_host status --artifact-root '$PRIVATE_CHAPTER_ROOT'" \
      >"$session/safe-status.json" 2>/dev/null || true
    bounded python3 -c '
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
raw=p.read_bytes()
if len(raw) > 16384: sys.exit(2)
x=json.loads(raw)
if x.get("schema") != "clipper-private-chapter-safe-recovery-v1" or set(x.get("chapters",{})) != {"1","7"}: sys.exit(2)
for chapter in x["chapters"].values():
    if (not isinstance(chapter,dict)
        or chapter.get("recovery_state") not in {"NEVER_STARTED","IN_FLIGHT","INTERRUPTED","AWAITING_REVIEW","READY"}
        or not isinstance(chapter.get("last_subphase"),str)
        or len(chapter["last_subphase"]) > 64): sys.exit(2)
pathlib.Path(sys.argv[2]).write_bytes(raw)
' "$session/safe-status.json" "$destination/safe-recovery-status.json" || true
  fi
  return 2
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
  if (( recovery_allowed == 1 && cleanup_only == 0 )); then
    recover_private >/dev/null 2>&1 || true
  fi
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
    bounded sleep 1 || break
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
    bounded sleep 1 || break
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

# Recovery is an owner-side private destination, never a git worktree even
# when its requested path traverses symlinks or resolves through "..".
if ! bounded python3 -c '
import pathlib, sys
target = pathlib.Path(sys.argv[1]).resolve()
if any((part / ".git").exists() for part in (target, *target.parents)):
    sys.exit(2)
' "$PRIVATE_CHAPTER_RECOVERY_DIR"; then
  echo "EXTERNAL_SUPERVISOR_RECOVERY_DESTINATION_UNSAFE" >&2
  exit 2
fi

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
if created.tzinfo is None:
    sys.exit(2)
now = datetime.datetime.now(datetime.timezone.utc)
# Identity verification and permission to execute are DISTINCT.
# Even at 116 minutes, an exact approved Codespace must be cleaned up.
if created > now or now >= created + datetime.timedelta(seconds=7200):
    sys.exit(2)
execution_allowed = now < created + datetime.timedelta(seconds=6900)
print(x["id"], x["created_at"], "EXECUTION" if execution_allowed else "CLEANUP_ONLY")
')" || {
  echo "EXTERNAL_SUPERVISOR_WRONG_HOST_OR_TOTAL_DEADLINE" >&2
  exit 2
}
read -r codespace_id created_at lifecycle_mode <<< "$validated"
export SUPERVISED_CODESPACE_ID="$codespace_id"
export SUPERVISED_CREATED_AT="$created_at"
deadline_epoch="$(( $(date -u -d "$created_at" +%s) + 7200 ))"
verified_identity=1

# BEFORE any recovery command or remote shell arguments, validate the path.
# Invalid retrieval configuration must not veto STOP/DELETE for a verified ID.
if [[ "$PRIVATE_CHAPTER_ROOT" =~ ^/[a-zA-Z0-9_./-]+$ &&
      "$PRIVATE_CHAPTER_ROOT" != "/" &&
      "$PRIVATE_CHAPTER_ROOT" != *"/../"* &&
      "$PRIVATE_CHAPTER_ROOT" != *"/./"* &&
      "$PRIVATE_CHAPTER_ROOT" != *"/.." &&
      "$PRIVATE_CHAPTER_ROOT" != *"/." ]]; then
  remote_safe=1
  recovery_allowed=1
else
  echo "EXTERNAL_SUPERVISOR_UNSAFE_REMOTE_PATH_CLEANUP_ONLY" >&2
  exit 2
fi

# Never install keys or refresh heartbeat after the original +115-minute
# execution window. This branch only stops/deletes the exact REST identity.
if [[ "$lifecycle_mode" == "CLEANUP_ONLY" ]]; then
  cleanup_only=1
  echo "EXTERNAL_SUPERVISOR_EXISTING_HOST_CLEANUP_RESERVE" >&2
  exit 0
fi
remote="$PRIVATE_CHAPTER_ROOT/book-candidates/private-host"
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
