#!/usr/bin/env bash
# OWNER MACHINE ONLY, outside the one approved Codespace. No provisioning.
# Persistent external watchdog: creation-time budget includes installs, human
# review waits and every host operation. Local Python timeout is not a watchdog.
set -euo pipefail
umask 077

: "${CODESPACE_NAME:?Exact existing Codespace name required}"
: "${PRIVATE_CHAPTER_APPROVED_SHA:?Reviewed commit SHA required}"
# A restart inside the cleanup-only reserve must not depend on old signing
# keys or a usable retrieval path. Identity recheck precedes those preflights.
PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY="${PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY:-}"
PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY="${PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY:-}"
PRIVATE_CHAPTER_ROOT="${PRIVATE_CHAPTER_ROOT:-}"
PRIVATE_CHAPTER_RECOVERY_DIR="${PRIVATE_CHAPTER_RECOVERY_DIR:-}"
# Only an owner-supplied finite free-inclusion attestation, signed into EVERY
# heartbeat, may select the extended lifecycle. No unsigned duration override.
PRIVATE_CHAPTER_FREE_GRANT_FILE="${PRIVATE_CHAPTER_FREE_GRANT_FILE:-}"
if [[ "${CODESPACES:-}" == "true" ||
      ! "$PRIVATE_CHAPTER_APPROVED_SHA" =~ ^[0-9a-f]{40}$ ||
      ! "$CODESPACE_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
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
bootstrap_epoch=0
execution_epoch=0
optional_epoch=0
heartbeat_expiry_epoch=0
# Owner-side evidence stays outside the ephemeral session. It contains only
# allowlisted operation names, exit classifications, identity and safe phase.
journal_dir=""
journal_ready=0
signal_name=""
last_verified_phase="UNOBSERVED"
last_checkpoint_epoch=0
last_phase_poll_epoch=0
last_progress_epoch=0
owner_stall_since=0
record_operation() {
  local operation="$1" start="$2" ended="$3" rc="$4"
  [[ "$journal_ready" == 1 && -n "$journal_dir" ]] || return 0
  timeout --signal=TERM --kill-after=1s 2s python3 -c '
import datetime, json, os, pathlib, re, sys
root=pathlib.Path(sys.argv[1])
op,start,ended,raw_rc=sys.argv[2:]
if op not in {"rest_metadata","rest_identity","rest_state","rest_stop","rest_delete",
    "rest_deleted_check","key_setup","key_copy","sign_heartbeat","heartbeat_copy",
    "heartbeat_publish","recovery_prepare","recovery_copy","safe_phase_poll",
    "recovery_fallback","archive_verification","signal","heartbeat_build",
    "envelope_build","bootstrap_wait","journal_setup","grant_validation"}:
    sys.exit(2)
rc=int(raw_rc)
if rc<0 or rc>255: sys.exit(2)
phase={"1":"UNOBSERVED","7":"UNOBSERVED"}
saved=root/"owner-safe-phase.json"
try:
    obj=json.loads(saved.read_text(encoding="utf-8"))
    if (obj.get("codespace_id")==int(os.environ["SUPERVISED_CODESPACE_ID"])
        and obj.get("created_at")==os.environ["SUPERVISED_CREATED_AT"]):
        phase={k:v for k,v in obj["chapters"].items()
               if k in ("1","7") and isinstance(v,str)
               and re.fullmatch(r"[A-Z0-9_]{1,64}",v)}
except (OSError,ValueError,KeyError,TypeError): pass
data={"schema":"clipper-owner-supervisor-operation-v1","operation":op,
      "started_at":start,"ended_at":ended,"exit_code":rc,
      "exit_class":"TIMEOUT" if rc==124 else "KILLED" if rc==137
                   else "SIGNAL" if rc>=128 else "FAILURE" if rc else "OK",
      "codespace_id":int(os.environ["SUPERVISED_CODESPACE_ID"]),
      "created_at":os.environ["SUPERVISED_CREATED_AT"],
      "last_verified_safe_phase":phase}
targets=[root/"owner-last-operation.json"]
if rc:
    targets=[root/"owner-last-failure.json",root/"owner-first-failure.json"]
for target in targets:
    if target.name=="owner-first-failure.json" and target.exists():
        continue
    tmp=root/(".operation-"+str(os.getpid())+".tmp")
    tmp.write_text(json.dumps(data,sort_keys=True)+"\n",encoding="utf-8")
    os.chmod(tmp,0o600)
    os.replace(tmp,target)
' "$journal_dir" "$operation" "$start" "$ended" "$rc" >/dev/null 2>&1 || true
}
operation_label() {
  if [[ "${1:-}" == "gh" ]]; then
    if [[ "${2:-}" == "api" ]]; then
      if [[ "${3:-}" == "-X" ]]; then
        [[ "${4:-}" == "POST" ]] && echo rest_stop || echo rest_delete
      elif [[ "${3:-}" == "-i" ]]; then echo rest_deleted_check
      else echo rest_metadata; fi
    elif [[ "${2:-}" == "codespace" ]]; then
      if [[ "${3:-}" == "cp" ]]; then
        case " $* " in
          *heartbeat-envelope.watchdog.pending*) echo heartbeat_copy ;;
          *chapter-recovery.tar*) echo recovery_copy ;;
          *) echo key_copy ;;
        esac
      else
        case " $* " in
          *"chapter_host recover"*) echo recovery_prepare ;;
          *"chapter_host status"*) echo safe_phase_poll ;;
          *heartbeat-envelope.watchdog.pending*) echo heartbeat_publish ;;
          *) echo key_setup ;;
        esac
      fi
    fi
  elif [[ "${1:-}" == "openssl" ]]; then echo sign_heartbeat
  elif [[ -n "${current_operation:-}" ]]; then echo "$current_operation"
  fi
}

critical() {
  local current_operation="$1"
  shift
  bounded "$@"
}
# Local-only operations must not inherit the legacy two-hour execution clock.
# This function never mutates deadline_epoch or any cleanup/REST authority.
# Its fixed eight-second bound, two-second kill grace and sanitized journal
# are independent from the original creation-time cleanup deadline.
bounded_local() {
  local current_operation="$1"
  shift
  local started ended status
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status=0
  timeout --signal=TERM --kill-after="${KILL_GRACE}s" "${OP_LIMIT}s" "$@" || status=$?
  ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  record_operation "$current_operation" "$started" "$ended" "$status"
  return "$status"
}
signal_exit() {
  local label="$1" status="$2" stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # A signal can land between mkdir and chmod while journal_ready is not
  # committed yet. Reuse only an existing safe owner-private directory.
  if [[ "$verified_identity" == 1 && "$journal_ready" != 1 &&
        -n "$journal_dir" && -d "$journal_dir" && ! -L "$journal_dir" ]]; then
    journal_ready=1
  fi
  record_operation signal "$stamp" "$stamp" "$status"
  echo "EXTERNAL_SUPERVISOR_SIGNAL_${label}_CLEANUP_REQUIRED" >&2
  exit "$status"
}
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
  if (( bootstrap_epoch > 0 )); then
    remaining="$(( bootstrap_epoch - $(date -u +%s) ))"
    if (( remaining <= 0 )); then return 124; fi
    if (( remaining < limit )); then limit="$remaining"; fi
  fi
  if (( execution_epoch > 0 )); then
    remaining="$(( execution_epoch - $(date -u +%s) ))"
    if (( remaining <= 0 )); then return 124; fi
    if (( remaining < limit )); then limit="$remaining"; fi
  fi
  # A published heartbeat has a finite 45-second freshness window.
  # Leave room for the host's timestamp precision and GNU timeout grace.
  if (( heartbeat_expiry_epoch > 0 )); then
    remaining="$(( heartbeat_expiry_epoch - $(date -u +%s) ))"
    if (( remaining <= 0 )); then return 124; fi
    if (( remaining < limit )); then limit="$remaining"; fi
  fi
  if (( optional_epoch > 0 )); then
    remaining="$(( optional_epoch - $(date -u +%s) ))"
    if (( remaining <= 0 )); then return 124; fi
    if (( remaining < limit )); then limit="$remaining"; fi
  fi
  local operation started ended status
  operation="$(operation_label "$@")"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status=0
  timeout --signal=TERM --kill-after="${KILL_GRACE}s" "${limit}s" "$@" || status=$?
  if [[ -n "$operation" ]]; then
    ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    record_operation "$operation" "$started" "$ended" "$status"
  fi
  return "$status"
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
owner_recovery_safe=0

# Initial identity alone is NOT permission to DELETE a name that has since
# been replaced. Recheck numeric instance ID + repo + name + creation stamp
# immediately before every destructive operation. Unavailable REST fails shut.
# REST failures are categorized without retaining HTTP bodies, stderr or
# untrusted identifiers. Only a verified exact instance permits host actions.
rest_failure_category() {
  if [[ -f "$session/rest-failure-category" ]]; then
    case "$(cat "$session/rest-failure-category")" in
      TRANSPORT|TIMEOUT|MALFORMED|CONFIRMED_MISMATCH|DISALLOWED_STATE)
        cat "$session/rest-failure-category"; return ;;
    esac
  fi
  printf 'TRANSPORT'
}
record_rest_failure() {
  local category="$1" attempt="$2" evidence_dir=""
  printf '%s' "$category" >"$session/rest-failure-category"
  # Persist owner-private classification even for the FIRST failed REST read:
  # no numeric identity is asserted until the exact REST record is verified.
  if [[ "$journal_ready" == 1 && "$verified_identity" == 1 &&
        $(( deadline_epoch - $(date -u +%s) )) -gt 25 ]]; then
    evidence_dir="$journal_dir"
  elif [[ "$verified_identity" == 0 && "$owner_recovery_safe" == 1 ]]; then
    evidence_dir="$PRIVATE_CHAPTER_RECOVERY_DIR/$CODESPACE_NAME"
    if [[ -L "$evidence_dir" ]]; then return 0; fi
    timeout --signal=TERM --kill-after=1s 2s mkdir -p -m 700 -- "$evidence_dir" \
      >/dev/null 2>&1 || return 0
    if [[ -L "$evidence_dir" ]]; then return 0; fi
  fi
  if [[ -n "$evidence_dir" ]]; then
    timeout --signal=TERM --kill-after=1s 2s python3 -c '
import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); category=sys.argv[2]; attempt=int(sys.argv[3])
if category not in {"TRANSPORT","TIMEOUT","MALFORMED","CONFIRMED_MISMATCH","DISALLOWED_STATE"}:
    sys.exit(2)
verified=sys.argv[4]=="1"
out={"schema":"clipper-owner-rest-failure-v1",
     "operation":sys.argv[5],
     "failure_class":category,"attempt":attempt,
     "identity_verified":verified,
     "codespace_id":int(os.environ["SUPERVISED_CODESPACE_ID"]) if verified else None,
     "created_at":os.environ["SUPERVISED_CREATED_AT"] if verified else None}
tmp=root/".rest-failure.tmp"
tmp.write_text(json.dumps(out,sort_keys=True)+"\n")
os.chmod(tmp,0o600);os.replace(tmp,root/"owner-last-rest-failure.json")
' "$evidence_dir" "$category" "$attempt" "$verified_identity" \
  "${rest_operation:-rest_identity}" >/dev/null 2>&1 || true
  fi
}
# Keep retries inside a small shared read budget. During cleanup's last
# seconds there is only one probe, reserving time for exact-ID DELETE.
rest_read() {
  local operation="$1"
  local attempt rc category max_attempts start_epoch
  rest_operation="$operation"
  max_attempts=2
  if (( deadline_epoch > 0 && deadline_epoch - $(date -u +%s) <= 25 )); then
    max_attempts=1
  fi
  start_epoch="$(date -u +%s)"
  for (( attempt=1; attempt<=max_attempts; attempt++ )); do
    # Never consume the remaining signed heartbeat freshness with retries.
    if (( attempt > 1 )); then
      if (( $(date -u +%s) - start_epoch >= 4 ||
            (heartbeat_expiry_epoch > 0 &&
             heartbeat_expiry_epoch - $(date -u +%s) <= 8) )); then
        return 2
      fi
    fi
    rc=0
    # Each attempt is at most two seconds, still bounded by the shared
    # absolute deadline and optional heartbeat caps in bounded().
    local previous_limit="$OP_LIMIT"
    if (( OP_LIMIT > 2 )); then OP_LIMIT=2; fi
    critical "$operation" gh api "user/codespaces/$CODESPACE_NAME" \
      >"$session/rest-identity-response" 2>/dev/null || rc=$?
    OP_LIMIT="$previous_limit"
    if (( rc == 0 )); then return 0; fi
    category=TRANSPORT
    if (( rc == 124 || rc == 137 )); then category=TIMEOUT; fi
    record_rest_failure "$category" "$attempt"
    if (( attempt >= max_attempts )); then return 2; fi
  done
  return 2
}
identity_matches() {
  local category rc
  rest_read rest_identity || return 2
  rc=0
  bounded python3 -c '
import json,os,sys
try:
    with open(sys.argv[1],"rb") as fd:
        raw=fd.read(16385)
    if len(raw)>16384: sys.exit(22)
    x=json.loads(raw)
    if not isinstance(x,dict): sys.exit(22)
    machine=x.get("machine")
    repo=x.get("repository")
    if (not isinstance(machine,dict) or not isinstance(repo,dict)
        or type(x.get("id")) is not int
        or not isinstance(x.get("name"),str)
        or not isinstance(x.get("created_at"),str)
        or not all(type(machine.get(k)) is int for k in
                   ("cpus","memory_in_bytes","storage_in_bytes"))
        or not isinstance(repo.get("full_name"),str)):
        sys.exit(22)
    valid=(
        x["id"]==int(os.environ["SUPERVISED_CODESPACE_ID"])
        and x["name"]==os.environ["CODESPACE_NAME"]
        and repo["full_name"]=="TheBayoumi/clipper"
        and x["created_at"]==os.environ["SUPERVISED_CREATED_AT"]
        and (machine["cpus"],machine["memory_in_bytes"],
             machine["storage_in_bytes"])==(4,17179869184,34359738368))
    sys.exit(0 if valid else 23)
except (ValueError,TypeError,KeyError,AttributeError,UnicodeDecodeError,OSError):
    sys.exit(22)
' "$session/rest-identity-response" >/dev/null 2>&1 || rc=$?
  if (( rc == 0 )); then
    rm -f -- "$session/rest-failure-category"
    return 0
  fi
  category=MALFORMED
  if (( rc == 23 )); then category=CONFIRMED_MISMATCH; fi
  if (( rc == 124 || rc == 137 )); then category=TIMEOUT; fi
  record_rest_failure "$category" 1
  return 2
}

# Pipe archive bytes through the authenticated, non-PTY Codespace SSH command.
# Both producer and bounded binary receiver belong to the same timed process
# group; pipefail rejects a failed producer even if it emitted a valid prefix.
stream_recovery_archive() {
  local remote_archive="$PRIVATE_CHAPTER_ROOT/book-candidates/private-host/chapter-recovery.tar"
  local target="$1"
  # One bounded parent owns and reaps its SSH child, including on timeout.
  # A shell pipeline killed as a group can orphan unreaped zombie children.
  critical recovery_copy python3 -c '
import json, os, signal, subprocess, sys
name, remote, target, receipt_path = sys.argv[1:]
limit = 800_000_000
with open(receipt_path, "rb") as source:
    receipt = json.loads(source.read(2049))
size = receipt.get("archive_bytes")
if size is not None:
    if type(size) is not int or not 0 < size <= limit:
        sys.exit(2)
    limit = size
proc = None
def end_child():
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=0.7)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=0.7)
def interrupted(signum, _frame):
    end_child()
    sys.exit(128 + signum)
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    proc = subprocess.Popen(
        ["gh", "codespace", "ssh", "-c", name, "--", "-T",
         "exec cat -- " + chr(39) + remote + chr(39)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, close_fds=True,
    )
    if proc.stdout is None:
        sys.exit(2)
    used = 0
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        while True:
            block = proc.stdout.read(min(1024 * 1024, limit - used + 1))
            if not block:
                break
            if used + len(block) > limit:
                sys.exit(2)
            output.write(block)
            used += len(block)
    if used == 0 or (size is not None and used != size):
        sys.exit(2)
    if proc.wait() != 0:
        sys.exit(2)
finally:
    end_child()
' "$CODESPACE_NAME" "$remote_archive" "$target" "$session/recovery-created.json"
}

recover_private() (
  # Recovery is always best-effort and bounded within original cleanup reserve.
  recovery_epoch="$(( $(date -u +%s) + 16 ))"
  if (( recovery_allowed != 1 )) || ! identity_matches; then return 2; fi
  local destination="${SNAPSHOT_STAGE_DIR:-$PRIVATE_CHAPTER_RECOVERY_DIR/$CODESPACE_NAME}"
  bounded mkdir -p -m 700 -- "$destination" || return 2
  bounded chmod 700 "$destination" || return 2
  local transfer_stage
  transfer_stage="$(mktemp -d "$destination/.recovery-stage.XXXXXXXX")" || return 2
  chmod 700 "$transfer_stage" || return 2
  # One scoped, allowlisted private output TAR includes final WAVs, text
  # validation records, narration reports and safe status; never a whole EPUB.
  if critical recovery_prepare gh codespace ssh -c "$CODESPACE_NAME" -- \
    "cd /workspaces/clipper && if [ -x .venv/bin/python ]; then .venv/bin/python -m clipper.books.chapter_host recover --artifact-root '$PRIVATE_CHAPTER_ROOT'; else PYTHONPATH=/workspaces/clipper/src python -m clipper.books.chapter_host recover --artifact-root '$PRIVATE_CHAPTER_ROOT'; fi" \
    >"$session/recovery-created.json" 2>/dev/null; then
    if identity_matches && stream_recovery_archive "$transfer_stage/chapter-recovery.tar" >/dev/null 2>&1 && identity_matches; then
      if critical archive_verification python3 -c '
import hashlib, json, pathlib, re, sys, tarfile
created_path, archive_path, report_path, safe_path = map(pathlib.Path, sys.argv[1:])
dig = re.compile(r"[0-9a-f]{64}\Z")
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
created_raw = created_path.read_bytes()
if len(created_raw) > 2048:
    sys.exit(2)
created = json.loads(created_raw)
if (created.get("schema") != "clipper-private-chapter-recovery-created-v1"
    or not isinstance(created.get("archive_sha256"), str)
    or dig.fullmatch(created["archive_sha256"]) is None
    or ("archive_bytes" in created
        and (type(created["archive_bytes"]) is not int
             or not 0 < created["archive_bytes"] <= 800_000_000))):
    sys.exit(2)
if (archive_path.is_symlink() or not archive_path.is_file()
    or archive_path.stat().st_size > 800_000_000
    or ("archive_bytes" in created
        and archive_path.stat().st_size != created["archive_bytes"])):
    sys.exit(2)
h = hashlib.sha256()
with archive_path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        h.update(chunk)
if h.hexdigest() != created["archive_sha256"]:
    sys.exit(2)
with tarfile.open(archive_path, mode="r:") as source:
    members = source.getmembers()
    # tarfile accepts non-TAR bytes after EOF. This recovery format is created
    # with standard 10 KiB tarfile records; require only zero-filled padding
    # after the last member, including the two mandatory zero EOF blocks.
    trailer_size = archive_path.stat().st_size - source.offset
    if (archive_path.stat().st_size % 10240 != 0
        or not 1024 <= trailer_size <= 10240):
        sys.exit(2)
    with archive_path.open("rb") as raw_archive:
        raw_archive.seek(source.offset)
        trailer = raw_archive.read(trailer_size)
    if len(trailer) != trailer_size or any(trailer):
        sys.exit(2)
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
    chapter_manifests = manifest.get("chapters")
    if (not isinstance(rows, list) or len(rows) > 512
        or not isinstance(chapter_manifests, list) or len(chapter_manifests) != 2):
        sys.exit(2)
    by_chapter = {}
    for chapter_manifest in chapter_manifests:
        if (not isinstance(chapter_manifest, dict)
            or chapter_manifest.get("chapter") not in (1, 7)
            or chapter_manifest["chapter"] in by_chapter
            or chapter_manifest.get("inventory_status") not in {"VERIFIED", "INCOMPLETE", "NOT_STARTED"}
            or not isinstance(chapter_manifest.get("files"), list)
            or len(chapter_manifest["files"]) > 256):
            sys.exit(2)
        by_chapter[chapter_manifest["chapter"]] = chapter_manifest
        if chapter_manifest["inventory_status"] != "VERIFIED" and chapter_manifest["files"]:
            sys.exit(2)
    if set(by_chapter) != {1,7}: sys.exit(2)
    expected = {"private-recovery-manifest.json", "safe-recovery-status.json"}
    by_path = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"chapters","kind","relative_path","sha256","bytes"}:
            sys.exit(2)
        path = row["relative_path"]
        owners = row["chapters"]
        if (not isinstance(owners, list) or not owners
            or owners != sorted(set(owners)) or any(n not in (1,7) for n in owners)
            or row["kind"] not in {"text","wav","narration_report","review","review_record","final_result","source_ledger","human_review"}
            or not isinstance(path, str) or not path.startswith("book-candidates/")
            or any(x in ("", ".", "..") for x in pathlib.PurePosixPath(path).parts)
            or not isinstance(row["sha256"], str) or dig.fullmatch(row["sha256"]) is None
            or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= 300_000_000
            or path in by_path):
            sys.exit(2)
        by_path[path] = row
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
    for number, chapter_manifest in by_chapter.items():
        scoped_files = chapter_manifest["files"]
        claimed = {path for path, row in by_path.items() if number in row["chapters"]}
        if len({r.get("relative_path") for r in scoped_files if isinstance(r, dict)}) != len(scoped_files):
            sys.exit(2)
        if claimed != {r.get("relative_path") for r in scoped_files if isinstance(r, dict)}:
            sys.exit(2)
        for scoped in scoped_files:
            if (not isinstance(scoped,dict) or set(scoped) !=
                {"chapter","kind","relative_path","sha256","bytes"}
                or scoped["chapter"] != number):
                sys.exit(2)
            canonical = by_path[scoped["relative_path"]]
            if any(scoped[field] != canonical[field] for field in ("kind","sha256","bytes")):
                sys.exit(2)
        safe_row = safe["chapters"][str(number)]
        expected_retrieval = (
            "INVENTORY_VERIFIED_NOT_RETRIEVED"
            if chapter_manifest["inventory_status"] == "VERIFIED"
            else "INCOMPLETE"
            if chapter_manifest["inventory_status"] == "INCOMPLETE"
            else "NOT_ATTEMPTED"
        )
        if safe_row.get("retrieval_status") != expected_retrieval:
            sys.exit(2)
    if set(safe) != {"schema","status","chapters","files_retrieved","publish_eligible"}:
        sys.exit(2)
    allowed = {"recovery_state","last_subphase","runtime_journal_status",
               "retrieval_status","last_verified_artifact_identities",
               "bundle_sha256","scope_sha256","committed_phase_event_sha256",
               "output_manifest_sha256"}
    for chapter in safe["chapters"].values():
        if (not isinstance(chapter, dict) or not set(chapter).issubset(allowed)
            or chapter.get("recovery_state") not in {"NEVER_STARTED","IN_FLIGHT","INTERRUPTED","AWAITING_REVIEW","READY","INCOMPLETE"}
            or not isinstance(chapter.get("last_subphase"), str)
            or len(chapter["last_subphase"]) > 64
            or chapter.get("retrieval_status") not in {"NOT_ATTEMPTED","INCOMPLETE","INVENTORY_VERIFIED_NOT_RETRIEVED"}
            or not isinstance(chapter.get("last_verified_artifact_identities"), dict)):
            sys.exit(2)
        if any(not isinstance(value, str) or dig.fullmatch(value) is None
               for value in chapter["last_verified_artifact_identities"].values()):
            sys.exit(2)
    # Archive integrity does not establish chapter completion. Independently
    # attest each READY terminal journal and its complete text+audio inventory.
    required_ids = {
        "brief_identity":"brief",
        "adaptation_identity":"adaptation",
        "script_identity":"chapter_script",
        "direction_identity":"chapter_direction",
        "narration_report_identity":None,
        "final_result_identity":None,
    }
    chapter_results = {}
    for number, manifest_row in by_chapter.items():
        label = str(number)
        state = safe["chapters"][label]["recovery_state"]
        contents = manifest_row["files"]
        if manifest_row["inventory_status"] == "NOT_STARTED":
            chapter_results[label] = "NOT_STARTED"
            continue
        if manifest_row["inventory_status"] != "VERIFIED":
            chapter_results[label] = "INCOMPLETE"
            continue
        if state != "READY":
            chapter_results[label] = "PARTIAL_OUTPUTS" if contents else "CHECKPOINT_ONLY"
            continue
        ids = safe["chapters"][label]["last_verified_artifact_identities"]
        verified_ids = manifest_row.get("verified_artifact_identities")
        if (safe["chapters"][label].get("runtime_journal_status") != "PREPRODUCTION_READY"
            or manifest_row.get("schema") != "clipper-chapter-private-output-manifest-v1"
            or manifest_row.get("publish_eligible") is not False
            or manifest_row.get("bundle_sha256") != safe["chapters"][label].get("bundle_sha256")
            or manifest_row.get("scope_sha256") != safe["chapters"][label].get("scope_sha256")
            or not isinstance(verified_ids,dict)
            or not all(k in verified_ids and verified_ids[k] == ids.get(k)
                       and isinstance(verified_ids[k],str) and dig.fullmatch(verified_ids[k])
                       for k in required_ids)
            or not isinstance(safe["chapters"][label].get("output_manifest_sha256"),str)):
            chapter_results[label] = "INCOMPLETE"
            continue
        original_manifest = {k:v for k,v in manifest_row.items() if k != "inventory_status"}
        original_bytes = (json.dumps(original_manifest,sort_keys=True,ensure_ascii=False)+"\n").encode("utf-8")
        if digest(original_bytes) != safe["chapters"][label]["output_manifest_sha256"]:
            chapter_results[label] = "INCOMPLETE"
            continue
        paths = {r["relative_path"]:r for r in contents}
        source_mode = manifest_row.get("source_mode")
        if source_mode is None:
            # Legacy mode is explicitly the ABSENCE of a source_mode property.
            # Hybrid-only files cannot be relabeled to bypass its source gate.
            expected_stages = ("brief","adaptation","chapter_script","chapter_direction")
            expected_kinds = {"text","narration_report","wav","final_result","source_ledger","human_review"}
            if any(r["kind"] == "review_record" or
                   "/hybrid_" in r["relative_path"] for r in contents):
                chapter_results[label] = "INCOMPLETE"
                continue
        elif source_mode == "DIRECT_SOURCE_HYBRID":
            expected_stages = ("hybrid_argument","hybrid_audit","hybrid_reconcile",
                               "adaptation","chapter_script","chapter_direction")
            expected_kinds = {"text","narration_report","wav","final_result",
                              "source_ledger","human_review","review_record","review"}
        else:
            chapter_results[label] = "INCOMPLETE"
            continue
        expected_text = all(
            any(r["kind"] == "text"
                and r["relative_path"].startswith("book-candidates/"+stage+"/")
                and r["relative_path"].endswith("/validated.json")
                for r in contents)
            for stage in expected_stages
        )
        narration_member_path = (
            "book-candidates/narration-report-"
            + verified_ids["narration_report_identity"]
            + ".json"
        )
        final_path_suffix = "/chapter-result-"+verified_ids["final_result_identity"]+".json"
        report_row = paths.get(narration_member_path)
        if (not expected_text
            or not expected_kinds.issubset({r["kind"] for r in contents})
            or report_row is None or report_row["kind"] != "narration_report"
            or not any(r["kind"]=="final_result" and r["relative_path"].endswith(final_path_suffix)
                       for r in contents)):
            chapter_results[label] = "INCOMPLETE"
            continue
        narration_member_info = source.getmember(narration_member_path)
        if narration_member_info.size > 300_000:
            chapter_results[label] = "INCOMPLETE"
            continue
        narration_stream = source.extractfile(narration_member_info)
        if narration_stream is None:
            chapter_results[label] = "INCOMPLETE"
            continue
        try:
            narration_bytes = narration_stream.read(300_001)
            if (
                len(narration_bytes) > 300_000
                or len(narration_bytes) != narration_member_info.size
            ):
                raise ValueError("narration TAR member size changed")
            report = json.loads(narration_bytes)
            record_list = report["records"]
            if (report.get("schema") != "clipper-chapter-narration-v2"
                or not isinstance(record_list,list) or not 1 <= len(record_list) <= 80):
                raise ValueError("invalid narration record count")
            for audio in record_list:
                path = "book-candidates/"+audio["wav_relative_path"]
                wav = paths.get(path)
                if (wav is None or wav["kind"] != "wav" or wav["sha256"] != audio["sha256"]
                    or not path.endswith(".wav")):
                    raise ValueError("narration WAV reference not recovered")
        except (ValueError,TypeError,KeyError,AttributeError):
            chapter_results[label] = "INCOMPLETE"
            continue
        if source_mode == "DIRECT_SOURCE_HYBRID":
            # Independently verify the exact host-produced source/reviewer/final
            # identity chain, not just member hashes or any convenient text file.
            try:
                def member_json(path, limit):
                    row = paths.get(path)
                    if row is None or row["bytes"] > limit:
                        raise ValueError("required hybrid member missing or oversized")
                    member = source.extractfile(path)
                    if member is None:
                        raise ValueError("required hybrid member unreadable")
                    raw = member.read(limit + 1)
                    if len(raw) != row["bytes"] or len(raw) > limit:
                        raise ValueError("hybrid member changed")
                    return json.loads(raw)

                if set(verified_ids) != {
                    "brief_identity","argument_identity","audit_identities",
                    "producer_identity_sha256","adaptation_identity",
                    "script_identity","direction_identity",
                    "narration_report_identity","final_result_identity",
                }:
                    raise ValueError("hybrid terminal identities incomplete")
                argument_id = verified_ids["argument_identity"]
                audit_ids = verified_ids["audit_identities"]
                producer_id = verified_ids["producer_identity_sha256"]
                if not all(isinstance(x, str) and dig.fullmatch(x) for x in (
                    argument_id, producer_id
                )):
                    raise ValueError("hybrid argument/producer identity invalid")
                if (not isinstance(audit_ids, list) or not 1 <= len(audit_ids) <= 16
                    or len(set(audit_ids)) != len(audit_ids)
                    or not all(isinstance(x, str) and dig.fullmatch(x) for x in audit_ids)):
                    raise ValueError("hybrid local audit identities invalid")

                def stage_artifact(stage, identity):
                    # Reproduce EXACTLY the production chapter text-artifact
                    # contract, independently of any saved host identity label.
                    # Archive/file-level SHA hashes only prove transport, not
                    # the immutable source->response->validated-stage identity.
                    candidates = [
                        r["relative_path"] for r in contents
                        if r["kind"] == "text"
                        and r["relative_path"].startswith("book-candidates/"+stage+"/")
                        and r["relative_path"].endswith("/validated.json")
                    ]
                    if len(candidates) > 128:
                        raise ValueError("hybrid recovered stage inventory oversized")
                    matching = [
                        (path, member_json(path, 250_000)) for path in candidates
                        if paths[path]["bytes"] <= 250_000
                    ]
                    selected = [
                        (path, artifact) for path, artifact in matching
                        if isinstance(artifact,dict) and artifact.get("artifact_identity")==identity
                    ]
                    if len(selected) != 1:
                        raise ValueError("hybrid recovered stage identity absent or ambiguous")
                    path, artifact = selected[0]
                    required = {
                        "request_key","provenance","contract","validator_contract",
                        "payload","owner","attempt","raw_sha256","payload_sha256",
                        "token_receipt","artifact_identity",
                    }
                    optional = {"prior_rejection_sha256"}
                    if (not required.issubset(artifact) or
                        set(artifact)-required-optional or
                        artifact["contract"] != stage or
                        artifact["validator_contract"] != "clipper-chapter-validated-text-artifact-v4" or
                        artifact["provenance"] not in ("ACTUAL_CPU_QWEN","INJECTED_TEST_PRODUCER") or
                        type(artifact["attempt"]) is not int or
                        not 1 <= artifact["attempt"] <= 2 or
                        not isinstance(artifact["owner"],str) or
                        not re.fullmatch(r"[0-9a-f]{32}", artifact["owner"]) or
                        not all(isinstance(artifact[k],str) and dig.fullmatch(artifact[k])
                                for k in ("request_key","raw_sha256","payload_sha256",
                                          "artifact_identity")) or
                        path != ("book-candidates/"+stage+"/"+artifact["request_key"]+
                                 "/validated.json")):
                        raise ValueError("hybrid recovered stage envelope malformed")
                    prior = artifact.get("prior_rejection_sha256")
                    if ((prior is not None and (
                         artifact["attempt"] != 2 or not isinstance(prior,str)
                         or not dig.fullmatch(prior))) or
                        (artifact["attempt"] == 2 and prior is None)):
                        raise ValueError("hybrid recovered stage winning rejection missing")
                    receipt = artifact["token_receipt"]
                    if artifact["provenance"] == "ACTUAL_CPU_QWEN":
                        if (not isinstance(receipt,dict) or
                            set(receipt) != {"raw_sha256","generated_token_count",
                                             "terminal_token_id","eos_decision"} or
                            receipt["raw_sha256"] != artifact["raw_sha256"] or
                            type(receipt["generated_token_count"]) is not int or
                            not 1 <= receipt["generated_token_count"] <= 8192 or
                            type(receipt["terminal_token_id"]) is not int or
                            receipt["terminal_token_id"] < 0 or
                            receipt["eos_decision"] != "EOS_VERIFIED"):
                            raise ValueError("hybrid actual token receipt invalid")
                    elif receipt is not None:
                        raise ValueError("hybrid injected-stage token receipt forged")
                    # evaluation._json_bytes(payload): sorted UTF-8 JSON, with
                    # standard whitespace and one trailing newline.
                    payload_bytes = (
                        json.dumps(artifact["payload"],ensure_ascii=False,sort_keys=True)+"\n"
                    ).encode("utf-8")
                    if hashlib.sha256(payload_bytes).hexdigest()!=artifact["payload_sha256"]:
                        raise ValueError("hybrid recovered stage payload digest changed")
                    # chapter._text_artifact_identity: content_fingerprint uses
                    # compact sorted UTF-8 JSON WITHOUT a trailing newline.
                    identity_fields = {
                        "validator_contract":"clipper-chapter-validated-text-artifact-v4",
                        "stage":stage,
                        "request_key":artifact["request_key"],
                        "provenance":artifact["provenance"],
                        "raw_sha256":artifact["raw_sha256"],
                        "payload_sha256":artifact["payload_sha256"],
                        "token_receipt":receipt,
                    }
                    if prior is not None:
                        identity_fields["prior_rejection_sha256"]=prior
                    actual_identity = hashlib.sha256(
                        json.dumps(identity_fields,ensure_ascii=False,sort_keys=True,
                                   separators=(",",":")).encode("utf-8")
                    ).hexdigest()
                    if actual_identity!=identity or artifact["artifact_identity"]!=identity:
                        raise ValueError("hybrid recovered stage content identity changed")
                    return artifact

                stage_artifact("hybrid_argument", argument_id)
                audit_paths = [r["relative_path"] for r in contents
                               if r["kind"] == "text" and
                               r["relative_path"].startswith("book-candidates/hybrid_audit/")
                               and r["relative_path"].endswith("/validated.json")]
                if len(audit_paths) != len(audit_ids):
                    raise ValueError("hybrid local-audit stage inventory differs")
                for audit_id in audit_ids:
                    stage_artifact("hybrid_audit", audit_id)
                reconcile = stage_artifact("hybrid_reconcile", verified_ids["brief_identity"])
                for stage, field in (("adaptation","adaptation_identity"),
                                     ("chapter_script","script_identity"),
                                     ("chapter_direction","direction_identity")):
                    stage_artifact(stage, verified_ids[field])
                review_path = "book-candidates/private-host/chapter-"+label+"-brief-record.json"
                if paths.get(review_path, {}).get("kind") != "review_record":
                    raise ValueError("hybrid review journal not recovered")
                saved = member_json(review_path, 16_384)
                scope_key = saved["chapter_scope_identity"]
                ledger_path = "book-candidates/chapter-evaluations/"+scope_key+"/source-ledger.json"
                ledger_row = paths.get(ledger_path)
                ledger = member_json(ledger_path, 600_000)
                ledger_id = ledger.get("ledger_sha256")
                canonical_ledger = hashlib.sha256(
                    (json.dumps(
                        {k:v for k,v in ledger.items() if k != "ledger_sha256"},
                        ensure_ascii=False,sort_keys=True,separators=(",", ":")
                    )+"\n").encode("utf-8")
                ).hexdigest()
                if (not isinstance(scope_key, str) or dig.fullmatch(scope_key) is None
                    or ledger_row is None or ledger_row["kind"] != "source_ledger"
                    or ledger_row["sha256"] != saved.get("source_ledger_sha256")
                    or saved.get("chapter_mode") != "DIRECT_SOURCE_HYBRID"
                    or saved.get("chapter") != number
                    or saved.get("bundle_sha256") != manifest_row["bundle_sha256"]
                    or saved.get("scope_sha256") != manifest_row["scope_sha256"]
                    or saved.get("brief_identity") != verified_ids["brief_identity"]
                    or saved.get("argument_identity") != argument_id
                    or saved.get("audit_identities") != audit_ids
                    or saved.get("producer_identity_sha256") != producer_id
                    or saved.get("phase") != "AWAITING_HUMAN_REVIEW"
                    or ledger.get("schema") != "clipper-chapter-hybrid-source-ledger-v1"
                    or ledger_id != canonical_ledger
                    or reconcile.get("payload",{}).get("ledger_sha256") != ledger_id):
                    raise ValueError("hybrid original-source reviewer identity differs")
                packet_path = "book-candidates/private-host/chapter-"+label+"-review-packet.json"
                if paths.get(packet_path,{}).get("kind") != "review":
                    raise ValueError("hybrid review packet not recovered")
                packet = member_json(packet_path, 1_000_000)
                if (packet.get("chapter") != number
                    or packet.get("brief_identity") != verified_ids["brief_identity"]
                    or packet.get("source_ledger_sha256") != ledger_row["sha256"]
                    or packet.get("producer_identity_sha256") != producer_id
                    or packet.get("brief_sha256") != reconcile["payload"].get("brief_sha256")):
                    raise ValueError("hybrid source/review packet detached")
                final_path = ("book-candidates/chapter-evaluations/"+scope_key+
                              "/chapter-result-"+verified_ids["final_result_identity"]+".json")
                if paths.get(final_path,{}).get("kind") != "final_result":
                    raise ValueError("hybrid final result path differs")
                final = member_json(final_path, 500_000)
                final_digest = hashlib.sha256(
                    json.dumps(final,ensure_ascii=False,sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if final_digest != verified_ids["final_result_identity"]:
                    raise ValueError("hybrid final result content identity differs")
                review_id = final["human_review_sha256"]
                human_path = ("book-candidates/chapter-evaluations/"+scope_key+
                              "/human-review-"+review_id+".json")
                if (not isinstance(review_id, str) or dig.fullmatch(review_id) is None
                    or paths.get(human_path,{}).get("kind") != "human_review"
                    or final.get("chapter_scope_identity") != scope_key
                    or final.get("source_mining_provenance") != "DIRECT_SOURCE_NO_MINING"
                    or final.get("source_ledger_sha256") != ledger_row["sha256"]
                    or final.get("producer_identity_sha256") != producer_id
                    or final.get("selected_source_bundle_sha256") != manifest_row["bundle_sha256"]
                    or final.get("argument_key") != argument_id
                    or final.get("audit_keys") != audit_ids
                    or final.get("brief_key") != verified_ids["brief_identity"]
                    or any(final.get(stage_key) != verified_ids[field] for stage_key, field in
                           (("adaptation_key","adaptation_identity"),
                            ("script_key","script_identity"),
                            ("direction_key","direction_identity"),
                            ("narration_report_key","narration_report_identity")))):
                    raise ValueError("hybrid final source or creative identity differs")
                human = member_json(human_path, 250_000)
                review_data = human.get("review")
                if not isinstance(review_data, dict):
                    raise ValueError("hybrid human approval malformed")
                review_digest = hashlib.sha256(
                    (json.dumps(review_data,ensure_ascii=False,sort_keys=True)+"\n").encode("utf-8")
                ).hexdigest()
                if (human.get("review_sha256") != review_id
                    or review_digest != review_id
                    or human.get("schema") != "clipper-chapter-hybrid-human-review-v1"
                    or human.get("publish_eligible") is not False
                    or review_data.get("brief_artifact_identity")
                       != verified_ids["brief_identity"]):
                    raise ValueError("hybrid human approval not recovered")
            except (ValueError, TypeError, KeyError, AssertionError, AttributeError):
                chapter_results[label] = "INCOMPLETE"
                continue
        chapter_results[label] = "READY_OUTPUTS_VERIFIED"
    complete = all(value == "READY_OUTPUTS_VERIFIED" for value in chapter_results.values())
    summary = {
        "schema": "clipper-private-owner-recovery-verification-v1",
        "archive_integrity_status": "VERIFIED",
        "retrieval_status": "VERIFIED_PRIVATE_ARCHIVE" if complete else "VERIFIED_PRIVATE_ARCHIVE_PARTIAL",
        "chapters": chapter_results,
        "recovered_file_count": len(rows),
        "output_scope": "COMPLETE" if complete else "PARTIAL" if rows else "NONE",
        "host_cleanup_status": "NOT_YET_REST_VERIFIED",
    }
    report_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    safe_path.write_bytes(safe_bytes)
' "$session/recovery-created.json" "$transfer_stage/chapter-recovery.tar" \
  "$transfer_stage/retrieval-verification.json" "$transfer_stage/safe-recovery-status.json"; then
        if identity_matches && bounded chmod 600 "$transfer_stage/chapter-recovery.tar" \
          "$transfer_stage/retrieval-verification.json" \
          "$transfer_stage/safe-recovery-status.json"; then
          if [[ -n "${SNAPSHOT_STAGE_DIR:-}" ]]; then
            # Checkpoint staging is a fresh unpublished directory. Its caller
            # atomically renames the entire directory only after verification.
            if bounded mv -- "$transfer_stage/chapter-recovery.tar" "$destination/chapter-recovery.tar" &&
               bounded mv -- "$transfer_stage/retrieval-verification.json" "$destination/retrieval-verification.json" &&
               bounded mv -- "$transfer_stage/safe-recovery-status.json" "$destination/safe-recovery-status.json"; then
              bounded rmdir -- "$transfer_stage" || true
              return 0
            fi
          elif critical archive_verification python3 -c '
import json, os, pathlib, sys, uuid
root, stage = map(pathlib.Path, sys.argv[1:])
names = ("chapter-recovery.tar", "retrieval-verification.json", "safe-recovery-status.json")
link = root / "recovery-current"
def point(name, folder):
    tmp = root / (".recovery-link-" + uuid.uuid4().hex)
    os.symlink(folder, tmp)
    os.replace(tmp, root / name)
def pointer(folder):
    tmp = root / (".recovery-pointer-" + uuid.uuid4().hex)
    os.symlink(folder, tmp)
    os.replace(tmp, link)
# A previously verified direct-file trio is migrated to a hard-linked,
# identical baseline before any reader-facing path changes. Interruptions
# before the final pointer switch still resolve to the old coherent trio.
legacy = [root / name for name in names]
if any(p.is_symlink() for p in legacy):
    if any(p.is_symlink() and os.readlink(p) != "recovery-current/" + p.name
           for p in legacy):
        sys.exit(2)
    if link.is_symlink():
        old = (root / os.readlink(link)).resolve()
        if not old.is_dir() or not all((old / n).is_file() for n in names):
            sys.exit(2)
        # Interrupted initial migration: every remaining regular path must
        # still be the hard-linked old bundle, never a foreign replacement.
        if any(p.exists() and not p.is_symlink()
               and not os.path.samefile(p, old / p.name) for p in legacy):
            sys.exit(2)
    elif any(p.exists() and not p.is_symlink() for p in legacy):
        sys.exit(2)
elif (root / "retrieval-verification.json").is_file():
    try:
        report = json.loads(legacy[1].read_bytes())
        verified = (report.get("archive_integrity_status") == "VERIFIED"
                    and report.get("retrieval_status") in (
                        "VERIFIED_PRIVATE_ARCHIVE", "VERIFIED_PRIVATE_ARCHIVE_PARTIAL"))
    except (OSError, ValueError, TypeError):
        verified = False
    if verified:
        if not all(p.is_file() and not p.is_symlink() for p in legacy):
            sys.exit(2)
        baseline = ".recovery-bundle-" + uuid.uuid4().hex
        prior = root / baseline
        prior.mkdir(mode=0o700)
        for item in legacy:
            os.link(item, prior / item.name)
        pointer(baseline)
# Prepare stable direct reader names. At this point the existing pointer,
# when present, always resolves to the original verified bytes.
for item in legacy:
    if not item.is_symlink():
        point(item.name, "recovery-current/" + item.name)
# An empty or regressed checkpoint cannot demote previously recovered
# positive outputs. The existing positive snapshot remains the reader target.
if link.is_symlink():
    previous = json.loads((root / os.readlink(link) / names[1]).read_bytes())
    proposed = json.loads((stage / names[1]).read_bytes())
    old_count = previous.get("recovered_file_count", 0)
    new_count = proposed.get("recovered_file_count", 0)
    importance = {"PARTIAL_OUTPUTS": 1, "READY_OUTPUTS_VERIFIED": 2}
    old_chapters, new_chapters = previous.get("chapters", {}), proposed.get("chapters", {})
    if (type(old_count) is not int or type(new_count) is not int
        or (old_count > 0 and new_count < old_count)
        or any(importance.get(new_chapters.get(ch), 0) < importance.get(status, 0)
               for ch, status in old_chapters.items())):
        sys.exit(2)
# Stage becomes a complete immutable bundle with one atomic directory rename.
bundle = ".recovery-bundle-" + uuid.uuid4().hex
os.replace(stage, root / bundle)
# This is the sole commit point for the three verified artifacts.
pointer(bundle)
' "$destination" "$transfer_stage" >/dev/null 2>&1; then
            return 0
          fi
        fi
      fi
    fi
  fi
  # Incomplete is distinct from REST cleanup; never report files recovered.
  bounded rm -rf -- "$transfer_stage" || true
  # An unsuccessful new transfer cannot overwrite an earlier verified report.
  if [[ ! -e "$destination/retrieval-verification.json" ]]; then
  bounded python3 -c '
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"schema":"clipper-private-owner-recovery-verification-v1",
               "retrieval_status":"INCOMPLETE",
               "host_cleanup_status":"NOT_YET_REST_VERIFIED"}, f)
' "$destination/retrieval-verification.json" || true
  fi
  # Fallback safe phase-only checkpoint, still within the same retrieval cap.
  if [[ ! -e "$destination/retrieval-verification.json" ]] && identity_matches; then
    bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
      "cd /workspaces/clipper && if [ -x .venv/bin/python ]; then .venv/bin/python -m clipper.books.chapter_host status --artifact-root '$PRIVATE_CHAPTER_ROOT'; else PYTHONPATH=/workspaces/clipper/src python -m clipper.books.chapter_host status --artifact-root '$PRIVATE_CHAPTER_ROOT'; fi" \
      >"$session/safe-status.json" 2>/dev/null || true
    bounded python3 -c '
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
raw=p.read_bytes()
if len(raw) > 16384: sys.exit(2)
x=json.loads(raw)
if x.get("schema") != "clipper-private-chapter-safe-recovery-v1" or set(x.get("chapters",{})) != {"1","7"}: sys.exit(2)
if set(x) != {"schema","status","chapters","files_retrieved","publish_eligible"}:
    sys.exit(2)
allowed={"recovery_state","last_subphase","runtime_journal_status",
         "retrieval_status","last_verified_artifact_identities",
         "bundle_sha256","scope_sha256","committed_phase_event_sha256",
         "output_manifest_sha256"}
for chapter in x["chapters"].values():
    if (not isinstance(chapter,dict) or not set(chapter).issubset(allowed)
        or chapter.get("recovery_state") not in {"NEVER_STARTED","IN_FLIGHT","INTERRUPTED","AWAITING_REVIEW","READY"}
        or not isinstance(chapter.get("last_subphase"),str)
        or len(chapter["last_subphase"]) > 64): sys.exit(2)
pathlib.Path(sys.argv[2]).write_bytes(raw)
' "$session/safe-status.json" "$destination/safe-recovery-status.json" || true
  fi
  return 2
)

# A safe phase snapshot is intentionally separate from the private archive.
# It is bounded, source-free, identity-bound and does NOT claim file retrieval.
poll_safe_phase() (
  (( journal_ready == 1 && recovery_allowed == 1 )) || return 2
  identity_matches || return 2
  local phase_file="$session/phase-snapshot.json"
  critical safe_phase_poll gh codespace ssh -c "$CODESPACE_NAME" -- \
    "cd /workspaces/clipper && if [ -x .venv/bin/python ]; then .venv/bin/python -m clipper.books.chapter_host status --artifact-root '$PRIVATE_CHAPTER_ROOT'; else PYTHONPATH=/workspaces/clipper/src python -m clipper.books.chapter_host status --artifact-root '$PRIVATE_CHAPTER_ROOT'; fi" \
    >"$phase_file" 2>/dev/null || return 2
  identity_matches || return 2
  bounded python3 -c '
import json, os, pathlib, re, sys
source,target=sys.argv[1:]
raw=pathlib.Path(source).read_bytes()
if len(raw)>16384: sys.exit(2)
x=json.loads(raw)
if (not isinstance(x,dict) or x.get("schema")!="clipper-private-chapter-safe-recovery-v1"
    or set(x)!={"schema","status","chapters","files_retrieved","publish_eligible"}
    or x["files_retrieved"] is not False or x["publish_eligible"] is not False
    or not isinstance(x["chapters"],dict) or set(x["chapters"])!={"1","7"}):
    sys.exit(2)
phase={}
for number,chapter in x["chapters"].items():
    if not isinstance(chapter,dict): sys.exit(2)
    value=chapter.get("last_subphase")
    state=chapter.get("recovery_state")
    if (not isinstance(value,str) or not re.fullmatch(r"[A-Z0-9_]{1,64}",value)
        or state not in ("NEVER_STARTED","IN_FLIGHT","INTERRUPTED",
                         "AWAITING_REVIEW","READY","INCOMPLETE")):
        sys.exit(2)
    phase[number]=value
record={"schema":"clipper-owner-private-safe-phase-v1",
        "codespace_id":int(os.environ["SUPERVISED_CODESPACE_ID"]),
        "created_at":os.environ["SUPERVISED_CREATED_AT"],
        "chapters":phase,
        "status_source":"HOST_STATUS_READ_ONLY_NOT_RETRIEVAL"}
dest=pathlib.Path(target)
tmp=dest.with_name("."+dest.name+".tmp")
tmp.write_text(json.dumps(record,sort_keys=True)+"\n",encoding="utf-8")
os.chmod(tmp,0o600)
os.replace(tmp,dest)
' "$phase_file" "$journal_dir/owner-safe-phase.json" || return 2
)

# Periodic checkpoints use the EXISTING strict archive verifier and stage
# privately. Publish a new snapshot only after complete hash/member checks.
# A failed transfer must not replace the last verified snapshot.
checkpoint_private() (
  (( journal_ready == 1 && recovery_allowed == 1 )) || return 2
  (( deadline_epoch - $(date -u +%s) > 60 )) || return 2
  identity_matches || return 2
  local stage now final pointer report hash
  stage="$(mktemp -d "$journal_dir/.snapshot-stage.XXXXXX")" || return 2
  chmod 700 "$stage" || return 2
  if ! SNAPSHOT_STAGE_DIR="$stage" recover_private >/dev/null 2>&1; then
    rm -rf -- "$stage"
    return 2
  fi
  identity_matches || { rm -rf -- "$stage"; return 2; }
  # Do not assign a VERIFIED directory name to a valid empty/status-only TAR.
  # Reject before rename; a prior pointer and its archive remain unchanged.
  if ! critical archive_verification python3 -c '
import json,pathlib,sys
folder=pathlib.Path(sys.argv[1])
report=folder/"retrieval-verification.json"
archive=folder/"chapter-recovery.tar"
if not report.is_file() or not archive.is_file() or archive.is_symlink():
    sys.exit(2)
data=json.loads(report.read_bytes())
count=data.get("recovered_file_count")
status=data.get("retrieval_status")
scope=data.get("output_scope")
if (data.get("schema")!="clipper-private-owner-recovery-verification-v1"
    or data.get("archive_integrity_status")!="VERIFIED"
    or type(count) is not int or count<=0
    or (scope,status) not in {
        ("COMPLETE","VERIFIED_PRIVATE_ARCHIVE"),
        ("PARTIAL","VERIFIED_PRIVATE_ARCHIVE_PARTIAL"),
    }):
    sys.exit(2)
' "$stage" >/dev/null 2>&1; then
    rm -rf -- "$stage"
    return 2
  fi
  now="$(date -u +%s)"
  final="$journal_dir/verified-snapshot-$now"
  if [[ -e "$final" ]] || ! mv -T -- "$stage" "$final"; then
    rm -rf -- "$stage"
    return 2
  fi
  # The only published pointer is owner-private, atomic and refers to a
  # previously verified archive; never trust a bare incomplete TAR.
  critical archive_verification python3 -c '
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1])
name=sys.argv[2]
folder=root/name
archive=folder/"chapter-recovery.tar"
report=folder/"retrieval-verification.json"
if not archive.is_file() or archive.is_symlink() or not report.is_file(): sys.exit(2)
verified=json.loads(report.read_bytes())
# Only verified OUTPUT archives may replace a prior verified pointer.
# A valid safe-status-only EMPTY TAR is not a recovered output snapshot.
if (verified.get("schema") != "clipper-private-owner-recovery-verification-v1"
    or verified.get("archive_integrity_status") != "VERIFIED"
    or verified.get("retrieval_status") not in {
        "VERIFIED_PRIVATE_ARCHIVE", "VERIFIED_PRIVATE_ARCHIVE_PARTIAL"}
    or type(verified.get("recovered_file_count")) is not int
    or verified["recovered_file_count"] <= 0
    or verified.get("output_scope") not in {"PARTIAL", "COMPLETE"}
    or (verified["output_scope"] == "PARTIAL"
        and verified["retrieval_status"] != "VERIFIED_PRIVATE_ARCHIVE_PARTIAL")
    or (verified["output_scope"] == "COMPLETE"
        and verified["retrieval_status"] != "VERIFIED_PRIVATE_ARCHIVE")):
    sys.exit(2)
h=hashlib.sha256()
with archive.open("rb") as stream:
    for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
record={"schema":"clipper-owner-verified-snapshot-pointer-v1",
        "snapshot_dir":name,"archive_sha256":h.hexdigest(),
        "codespace_id":int(os.environ["SUPERVISED_CODESPACE_ID"]),
        "created_at":os.environ["SUPERVISED_CREATED_AT"],
        "retrieval_status":"VERIFIED_OWNER_PRIVATE_SNAPSHOT"}
tmp=root/".latest-verified-snapshot.tmp"
tmp.write_text(json.dumps(record,sort_keys=True)+"\n",encoding="utf-8")
os.chmod(tmp,0o600)
os.replace(tmp,root/"latest-verified-snapshot.json")
' "$journal_dir" "${final##*/}" >/dev/null 2>&1
)

write_owner_cleanup_status() {
  if (( owner_recovery_safe != 1 )); then return 0; fi
  local destination="$PRIVATE_CHAPTER_RECOVERY_DIR/$CODESPACE_NAME"
  bounded mkdir -p -m 700 -- "$destination" || return 0
  bounded python3 -c '
import json,sys
stop,deleted=sys.argv[2:]
data={"schema":"clipper-private-owner-cleanup-verification-v1",
      "host_stop_verified":stop == "1",
      "host_deletion_verified":deleted == "1",
      "retrieval_status_source":"SEPARATE_OWNER_RECOVERY_VERIFICATION"}
with open(sys.argv[1],"w",encoding="utf-8") as out:
    json.dump(data,out,sort_keys=True)
' "$destination/host-cleanup-verification.json" "$stop_verified" "$deletion_verified" || true
  bounded chmod 600 "$destination/host-cleanup-verification.json" || true
}

# Capture the initiating process status BEFORE best-effort recovery and STOP.
# The persistent first/last operation records are distinct from this exit
# classification and must never be overwritten by later cleanup successes.
write_owner_exit_reason() {
  [[ "$journal_ready" == 1 ]] || return 0
  timeout --signal=TERM --kill-after=1s 2s python3 -c '
import json,os,pathlib,sys
folder=pathlib.Path(sys.argv[1])
rc=int(sys.argv[2])
previous=folder/"owner-last-failure.json"
allowed={"rest_metadata","rest_identity","rest_state","rest_stop","rest_delete",
    "rest_deleted_check","key_setup","key_copy","sign_heartbeat",
    "heartbeat_copy","heartbeat_publish","recovery_prepare","recovery_copy",
    "safe_phase_poll","recovery_fallback","archive_verification","signal",
    "heartbeat_build","envelope_build","bootstrap_wait","grant_validation"}
operation="UNIDENTIFIED_EXIT"
if previous.is_file() and previous.stat().st_size < 4096:
    try:
        last=json.loads(previous.read_bytes())
        if (last.get("operation") in allowed
            and last.get("codespace_id")==int(os.environ["SUPERVISED_CODESPACE_ID"])
            and last.get("created_at")==os.environ["SUPERVISED_CREATED_AT"]):
            operation=last["operation"]
    except (ValueError,TypeError,OSError,KeyError): pass
data={"schema":"clipper-owner-supervisor-exit-v1",
      "original_exit_code":rc,
      "exit_class":"TIMEOUT" if rc==124 else "KILLED" if rc==137
         else "SIGNAL" if rc>=128 else "FAILURE" if rc else "NORMAL_EXIT",
      "most_recent_recorded_failed_operation":operation,
      "codespace_id":int(os.environ["SUPERVISED_CODESPACE_ID"]),
      "created_at":os.environ["SUPERVISED_CREATED_AT"],
      "retrieval_status":"NOT_YET_CONFIRMED"}
tmp=folder/".owner-exit.tmp"
tmp.write_text(json.dumps(data,sort_keys=True)+"\n",encoding="utf-8")
os.chmod(tmp,0o600)
os.replace(tmp,folder/"owner-exit-reason.json")
' "$journal_dir" "$1" >/dev/null 2>&1 || true
}
# Emit a distinct sanitized category for exhausted clock versus changed host.
cleanup_identity_failure() {
  local reason="$1" category
  category="$(rest_failure_category)"
  if (( deadline_epoch - $(date -u +%s) <= 0 )); then
    echo "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_MANUAL_REST_CHECK_REQUIRED" >&2
  else
    echo "${reason}_REST_${category}" >&2
  fi
}

terminate_host() {
  original_status=$?
  trap - EXIT
  # An additional INT/TERM/HUP during bounded recovery must not abort the
  # identity-checked DELETE. GNU timeout still bounds hung child processes.
  trap '' INT TERM HUP
  set +e
  # At the hard cutoff, exact-identity DELETE outranks local journaling.
  if (( verified_identity == 1 && deadline_epoch - $(date -u +%s) > 25 )); then
    write_owner_exit_reason "$original_status"
  fi
  # Execution/bootstrap expiry must never block original-clock cleanup.
  bootstrap_epoch=0
  execution_epoch=0
  optional_epoch=0
  heartbeat_expiry_epoch=0
  if [[ "$verified_identity" != 1 ]]; then
    rm -rf -- "$session"
    echo "EXTERNAL_SUPERVISOR_IDENTITY_UNVERIFIED_OWNER_MUST_CHECK_HOST" >&2
    exit 2
  fi
  stop_verified=0
  deletion_verified=0
  delete_dispatched=0
  # Recovery is never allowed to consume the final DELETE reserve.
  if (( recovery_allowed == 1 && cleanup_only == 0 &&
        deadline_epoch - $(date -u +%s) > 45 )); then
    recover_private >/dev/null 2>&1 || true
  fi
  # Near the original hard deadline, DELETE is the FIRST destructive request.
  # It takes priority over graceful STOP, polling, recovery, and local reports.
  # Use short probes to reserve time for the exact-identity-checked DELETE.
  remaining="$(( deadline_epoch - $(date -u +%s) ))"
  if (( remaining <= 0 )); then
    rm -rf -- "$session"
    echo "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_MANUAL_REST_CHECK_REQUIRED" >&2
    exit 2
  fi
  if (( remaining > 22 )); then
    OP_LIMIT=2
    if ! identity_matches; then
      cleanup_identity_failure "EXTERNAL_SUPERVISOR_REJECTED_HOST_IDENTITY_NO_DESTRUCTIVE_CALL"
      rm -rf -- "$session"
      exit 2
    fi
    # Only attempt STOP while enough time remains to probe and DELETE later.
    if (( deadline_epoch - $(date -u +%s) > 18 )); then
      bounded gh api -X POST "user/codespaces/$CODESPACE_NAME/stop" >/dev/null 2>&1 || true
      for _ in 1 2; do
        if (( deadline_epoch - $(date -u +%s) <= 12 )); then break; fi
        state="$(bounded gh api "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
          bounded python3 -c 'import json,sys; print(json.load(sys.stdin).get("state",""))' 2>/dev/null || true)"
        if [[ "$state" == "Shutdown" || "$state" == "Stopped" ]]; then
          stop_verified=1
          break
        fi
      done
    fi
  fi
  # Reserve the last seconds for a fresh exact numeric identity probe and
  # DELETE. Never send DELETE on a changed name/ID/repository/machine.
  # Owner evidence already persisted; avoid filesystem journal delay here.
  journal_ready=0
  OP_LIMIT=1
  if ! identity_matches; then
    cleanup_identity_failure "EXTERNAL_SUPERVISOR_REJECTED_HOST_IDENTITY_NO_DELETE"
    rm -rf -- "$session"
    exit 2
  fi
  if (( deadline_epoch - $(date -u +%s) <= 0 )); then
    rm -rf -- "$session"
    echo "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_MANUAL_REST_CHECK_REQUIRED" >&2
    exit 2
  fi
  # Even a hung STOP cannot consume the last DELETE slot: near expiry STOP
  # is skipped altogether. Keep the DELETE call bounded to the ORIGINAL clock.
  OP_LIMIT=2
  delete_dispatched=1
  bounded gh api -X DELETE "user/codespaces/$CODESPACE_NAME" >/dev/null 2>&1 || true
  OP_LIMIT=1
  if (( deadline_epoch - $(date -u +%s) > 0 )); then
    for _ in 1 2; do
      code="$(bounded gh api -i "user/codespaces/$CODESPACE_NAME" 2>/dev/null |
        head -n 1 | grep -oE '404' || true)"
      if [[ "$code" == "404" ]]; then
        deletion_verified=1
        break
      fi
      if (( deadline_epoch - $(date -u +%s) <= 0 )); then break; fi
    done
  fi
  write_owner_cleanup_status
  rm -rf -- "$session"
  if (( deletion_verified == 1 )); then
    cleanup_verified=1
    if (( stop_verified == 1 )); then
      echo "EXTERNAL_SUPERVISOR_HOST_STOP_AND_DELETION_REST_VERIFIED"
    else
      echo "EXTERNAL_SUPERVISOR_HOST_DELETION_REST_VERIFIED_STOP_NOT_ATTESTED"
    fi
    exit "$original_status"
  fi
  if (( deadline_epoch - $(date -u +%s) <= 0 )); then
    echo "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_DELETE_DISPATCHED_VERIFICATION_INCOMPLETE" >&2
  elif (( delete_dispatched == 1 )); then
    echo "EXTERNAL_SUPERVISOR_DELETE_DISPATCHED_REST_VERIFICATION_INCOMPLETE" >&2
  fi
  exit 2
}
# Signal handlers must TERMINATE; do not resume an interrupted sign/copy loop
# with traps removed after a Ctrl+C or external TERM.
trap 'signal_exit INT 130' INT
trap 'signal_exit TERM 143' TERM
trap 'signal_exit HUP 129' HUP
trap terminate_host EXIT

# This is a retrieval capability check only. An invalid owner path must
# NEVER veto independent REST STOP/DELETE after the numeric host ID matches.
if [[ "$PRIVATE_CHAPTER_RECOVERY_DIR" == /* &&
      ! -L "$PRIVATE_CHAPTER_RECOVERY_DIR" ]] &&
   bounded python3 -c '
import pathlib, sys
target = pathlib.Path(sys.argv[1]).resolve()
if any((part / ".git").exists() for part in (target, *target.parents)):
    sys.exit(2)
' "$PRIVATE_CHAPTER_RECOVERY_DIR"; then
  owner_recovery_safe=1
fi

if ! rest_read rest_metadata; then
  echo "EXTERNAL_SUPERVISOR_CODESPACE_REST_$(rest_failure_category)" >&2
  exit 2
fi
metadata="$(cat "$session/rest-identity-response")"
# Verify REST numeric identity and original creation clock independently of
# ANY optional execution allowance. An invalid grant must never suppress the
# already-approved host's identity-checked STOP/DELETE in the legacy safe clock.
initial_rc=0
validated="$(printf '%s' "$metadata" | bounded python3 -c '
import datetime,json,os,sys
try:
    x=json.load(sys.stdin)
    if not isinstance(x,dict): sys.exit(22)
    repo=x.get("repository")
    machine=x.get("machine")
    if (not isinstance(repo,dict) or not isinstance(machine,dict)
        or type(x.get("id")) is not int or x["id"]<1
        or not isinstance(x.get("name"),str)
        or not isinstance(x.get("created_at"),str)
        or not isinstance(repo.get("full_name"),str)
        or not all(type(machine.get(k)) is int for k in
                   ("cpus","memory_in_bytes","storage_in_bytes"))):
        sys.exit(22)
    created=datetime.datetime.fromisoformat(x["created_at"].replace("Z","+00:00"))
    if created.tzinfo is None: sys.exit(22)
    if created>datetime.datetime.now(datetime.timezone.utc):
        sys.exit(23)
    if (x["name"]!=os.environ["CODESPACE_NAME"]
        or repo["full_name"]!="TheBayoumi/clipper"
        or (machine["cpus"],machine["memory_in_bytes"],machine["storage_in_bytes"])
           !=(4,17179869184,34359738368)):
        sys.exit(23)
    print(x["id"],x["created_at"])
except (ValueError,TypeError,KeyError,AttributeError,UnicodeDecodeError):
    sys.exit(22)
')" || initial_rc=$?
if (( initial_rc != 0 )); then
  initial_category=MALFORMED
  if (( initial_rc == 23 )); then initial_category=CONFIRMED_MISMATCH; fi
  if (( initial_rc == 124 || initial_rc == 137 )); then initial_category=TIMEOUT; fi
  record_rest_failure "$initial_category" 1
  echo "EXTERNAL_SUPERVISOR_CODESPACE_REST_$initial_category" >&2
  exit 2
fi
read -r codespace_id created_at <<< "$validated"
export SUPERVISED_CODESPACE_ID="$codespace_id"
export SUPERVISED_CREATED_AT="$created_at"
total_seconds=7200
deadline_epoch="$(( $(date -u -d "$created_at" +%s) + total_seconds ))"
verified_identity=1
# Owner-safe journaling is prepared before grant validation, so a bad grant
# itself is recorded but its failure cannot hide the identity-checked cleanup.
if (( owner_recovery_safe == 1 )); then
  journal_dir="$PRIVATE_CHAPTER_RECOVERY_DIR/$CODESPACE_NAME"
  # Local bounded setup never changes the cleanup clock, even if
  # SIGTERM arrives while an operation is in progress.
  if [[ ! -L "$journal_dir" ]] &&
      bounded_local journal_setup mkdir -p -m 700 -- "$journal_dir" &&
      bounded_local journal_setup chmod 700 "$journal_dir"; then
    journal_ready=1
  fi
fi
if [[ -n "$PRIVATE_CHAPTER_FREE_GRANT_FILE" ]]; then
  # Read-only validation receives a local eight-second operation cap,
  # without mutating the legacy cleanup deadline or its signal path.
  grant_seconds="$(printf '%s' "$metadata" | bounded_local grant_validation python3 -c '
import datetime,json,os,pathlib,sys
x=json.load(sys.stdin)
path=pathlib.Path(os.environ["PRIVATE_CHAPTER_FREE_GRANT_FILE"])
if not path.is_absolute() or path.is_symlink() or not path.is_file():
    sys.exit(2)
if any((part/".git").exists() for part in (path.parent,*path.parents)):
    sys.exit(2)
if path.stat().st_size>4096 or path.stat().st_mode & 0o077:
    sys.exit(2)
try:
    grant=json.loads(path.read_bytes())
    created=datetime.datetime.fromisoformat(x["created_at"].replace("Z","+00:00"))
    checked=datetime.datetime.fromisoformat(grant["verified_at"].replace("Z","+00:00"))
    total=grant.get("total_seconds")
    remaining=grant.get("remaining_included_core_seconds")
    if (set(grant)!={"schema","codespace_id","codespace_name","repository",
          "created_at","approved_sha","machine","verified_at",
          "remaining_included_core_seconds","total_seconds",
          "storage_included_verified","paid_overage_block_verified"}
        or grant["schema"]!="clipper-owner-free-allowance-v1"
        or type(grant["codespace_id"]) is not int or grant["codespace_id"]!=x["id"]
        or grant["codespace_name"]!=x["name"]
        or grant["repository"]!="TheBayoumi/clipper"
        or grant["created_at"]!=x["created_at"]
        or grant["approved_sha"]!=os.environ["PRIVATE_CHAPTER_APPROVED_SHA"]
        or grant["machine"]!={"cpus":4,"memory_in_bytes":17179869184,
                              "storage_in_bytes":34359738368}
        or type(total) is not int or not 7200<total<=42900
        or type(remaining) is not int or remaining<total*4
        or grant["storage_included_verified"] is not True
        or grant["paid_overage_block_verified"] is not True
        or checked.tzinfo is None
        or checked<created-datetime.timedelta(minutes=30)
        or checked>created+datetime.timedelta(minutes=5)
        or checked>datetime.datetime.now(datetime.timezone.utc)
                   +datetime.timedelta(seconds=5)):
        sys.exit(2)
except (ValueError,TypeError,KeyError,AttributeError,OSError):
    sys.exit(2)
print(total)
')" || {
    echo "EXTERNAL_SUPERVISOR_FREE_GRANT_REJECTED_NO_EXECUTION" >&2
    exit 2
  }
  total_seconds="$grant_seconds"
  deadline_epoch="$(( $(date -u -d "$created_at" +%s) + total_seconds ))"
fi
if (( $(date -u +%s) >= deadline_epoch )); then
  echo "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_NO_AUTOMATED_DELETE" >&2
  exit 2
fi
lifecycle_mode="EXECUTION"
if (( $(date -u +%s) >= deadline_epoch - 300 )); then
  lifecycle_mode="CLEANUP_ONLY"
fi

# BEFORE any recovery command or remote shell arguments, validate the path.
# Invalid retrieval configuration must not veto STOP/DELETE for a verified ID.
if [[ "$PRIVATE_CHAPTER_ROOT" =~ ^/[a-zA-Z0-9_./-]+$ &&
      "$PRIVATE_CHAPTER_ROOT" != "/" &&
      "$PRIVATE_CHAPTER_ROOT" != *"/../"* &&
      "$PRIVATE_CHAPTER_ROOT" != *"/./"* &&
      "$PRIVATE_CHAPTER_ROOT" != *"/.." &&
      "$PRIVATE_CHAPTER_ROOT" != *"/." &&
      "$owner_recovery_safe" == 1 ]]; then
  remote_safe=1
  # Retrieval is armed only after safe SSH setup and public-key transfer.
  recovery_allowed=0
else
  echo "EXTERNAL_SUPERVISOR_UNSAFE_RETRIEVAL_CONFIG_CLEANUP_ONLY" >&2
  exit 2
fi

# Never install keys or refresh heartbeat after the original +115-minute
# execution window. This branch only stops/deletes the exact REST identity.
if [[ "$lifecycle_mode" == "CLEANUP_ONLY" ]]; then
  cleanup_only=1
  echo "EXTERNAL_SUPERVISOR_EXISTING_HOST_CLEANUP_RESERVE" >&2
  exit 0
fi
if [[ ! -f "$PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY" ||
      ! -f "$PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY" ]]; then
  echo "EXTERNAL_SUPERVISOR_EXECUTION_SIGNER_UNAVAILABLE_CLEANUP_ONLY" >&2
  exit 2
fi
# Bootstrap consumes the ORIGINAL creation clock, never an additional allowance.
# Ten minutes at most for provisioning and transient SSH readiness; do not
# start dependencies or issue a signed heartbeat until BOTH remote steps pass.
bootstrap_epoch="$(( $(date -u -d "$created_at" +%s) + 600 ))"
execution_epoch="$(( deadline_epoch - 300 ))"
if (( bootstrap_epoch > execution_epoch - 60 )); then
  bootstrap_epoch="$(( execution_epoch - 60 ))"
fi
bootstrap_budget() {
  (( $(date -u +%s) < bootstrap_epoch && $(date -u +%s) < execution_epoch - 60 ))
}
bootstrap_identity() {
  if ! bootstrap_budget; then
    echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_CLOCK_EXPIRED" >&2
    return 2
  fi
  if ! identity_matches; then
    echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_REST_$(rest_failure_category)" >&2
    return 2
  fi
}
# Bind the readiness state to the SAME REST object carrying the exact
# approved instance identity. Bracketing probes alone are insufficient:
# a replaced ID could supply Available between two matching identity reads.
bootstrap_state() {
  bootstrap_identity || return 2
  local state rc category
  rest_read rest_state || {
    echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_STATE_REST_$(rest_failure_category)" >&2
    return 2
  }
  rc=0
  state="$(bounded python3 -c '
import json,os,sys
try:
    with open(sys.argv[1],"rb") as f: raw=f.read(16385)
    if len(raw)>16384: sys.exit(22)
    obj=json.loads(raw)
    if not isinstance(obj,dict): sys.exit(22)
    machine=obj.get("machine")
    repo=obj.get("repository")
    state=obj.get("state")
    if (not isinstance(machine,dict) or not isinstance(repo,dict)
        or type(obj.get("id")) is not int
        or not isinstance(obj.get("name"),str)
        or not isinstance(obj.get("created_at"),str)
        or not isinstance(repo.get("full_name"),str)
        or not all(type(machine.get(k)) is int for k in
                   ("cpus","memory_in_bytes","storage_in_bytes"))
        or not isinstance(state,str)):
        sys.exit(22)
    if (obj["id"]!=int(os.environ["SUPERVISED_CODESPACE_ID"])
        or obj["name"]!=os.environ["CODESPACE_NAME"]
        or repo["full_name"]!="TheBayoumi/clipper"
        or obj["created_at"]!=os.environ["SUPERVISED_CREATED_AT"]
        or (machine["cpus"],machine["memory_in_bytes"],machine["storage_in_bytes"])
            !=(4,17179869184,34359738368)):
        sys.exit(23)
    # A correctly pinned host in Shutdown is not a foreign host.
    if state not in {"Available","Created","Starting","Provisioning","Queued","Rebuilding"}:
        sys.exit(24)
    print(state)
except (ValueError,TypeError,KeyError,AttributeError,UnicodeDecodeError,OSError):
    sys.exit(22)
' "$session/rest-identity-response")" || rc=$?
  if (( rc != 0 )); then
    category=MALFORMED
    if (( rc == 23 )); then category=CONFIRMED_MISMATCH; fi
    if (( rc == 24 )); then category=DISALLOWED_STATE; fi
    if (( rc == 124 || rc == 137 )); then category=TIMEOUT; fi
    record_rest_failure "$category" 1
    echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_STATE_REST_$category" >&2
    return 2
  fi
  bootstrap_identity || return 2
  printf '%s\n' "$state"
}
remote="$PRIVATE_CHAPTER_ROOT/book-candidates/private-host"
while true; do
  state="$(bootstrap_state)" || {
    echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_REST_UNAVAILABLE_OR_TERMINAL" >&2
    exit 2
  }
  if [[ "$state" == "Available" ]]; then break; fi
  bootstrap_budget || { echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_EXPIRED" >&2; exit 2; }
  bounded sleep 2 || { echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_EXPIRED" >&2; exit 2; }
done
# SSH may lag behind REST Available. Retry only within the same finite window,
# with fresh exact REST identity/state before every attempt. A hung operation
# (GNU timeout 124/137) is terminal, never a reason to start another host.
for stage in ssh cp; do
  ready=0
  while bootstrap_budget; do
    state="$(bootstrap_state)" || {
      echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_REST_UNAVAILABLE_OR_TERMINAL" >&2
      exit 2
    }
    if [[ "$state" != "Available" ]]; then
      echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_READY_STATE_LOST" >&2
      exit 2
    fi
    rc=0
    if [[ "$stage" == "ssh" ]]; then
      critical key_setup gh codespace ssh -c "$CODESPACE_NAME" -- \
        "mkdir -p '$remote' && chmod 700 '$remote'" >/dev/null || rc=$?
    else
      critical key_copy gh codespace cp -c "$CODESPACE_NAME" \
        "$PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY" \
        "remote:$remote/external-supervisor-public.pem" >/dev/null || rc=$?
    fi
    if (( rc == 0 )); then
      bootstrap_identity || exit 2
      ready=1
      break
    fi
    if (( rc == 124 || rc == 137 )); then
      echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_REMOTE_HUNG" >&2
      exit "$rc"
    fi
    bootstrap_budget || break
    bounded sleep 2 || break
  done
  if (( ready != 1 )); then
    echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_EXPIRED_NO_HEARTBEAT" >&2
    exit 2
  fi
done
state="$(bootstrap_state)" || {
  echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_FINAL_REST_UNAVAILABLE" >&2
  exit 2
}
if [[ "$state" != "Available" ]]; then
  echo "EXTERNAL_SUPERVISOR_BOOTSTRAP_READY_STATE_LOST" >&2
  exit 2
fi
bootstrap_epoch=0
recovery_allowed=1
last_phase_poll_epoch="$(date -u +%s)"
last_checkpoint_epoch="$last_phase_poll_epoch"

# Every iteration bounds both the individual operation and the time remaining.
# Execution stops at host creation +115m; the remaining five minutes are for
# bounded STOP/DELETE and verification. There is NO new time allowance.
while true; do
  remaining="$(bounded python3 -c '
import datetime,sys
created=datetime.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))
print(int((created+datetime.timedelta(seconds=int(sys.argv[2])-300)
           -datetime.datetime.now(datetime.timezone.utc)).total_seconds()))
' "$created_at" "$total_seconds")"
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
print((created+datetime.timedelta(seconds=int(sys.argv[2]))).isoformat())
' "$created_at" "$total_seconds")"
  critical heartbeat_build python3 -c '
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
if sys.argv[8]:
    with open(sys.argv[8],"rb") as grant_file:
        record["free_allowance"]=json.load(grant_file)
with open(sys.argv[7],"wb") as output:
    output.write((json.dumps(record, sort_keys=True, ensure_ascii=False)+"\n").encode())
' "$codespace_id" "$CODESPACE_NAME" "$created_at" \
  "$PRIVATE_CHAPTER_APPROVED_SHA" "$now" "$deadline" "$session/heartbeat.json" \
  "$PRIVATE_CHAPTER_FREE_GRANT_FILE"
  critical sign_heartbeat openssl dgst -sha256 -sign "$PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY" \
    -out "$session/heartbeat.sig" "$session/heartbeat.json" >/dev/null
  # The signature and exact payload are a single bounded envelope. Copy to
  # a staging path, then atomically rename it on the SAME remote filesystem.
  envelope_sha="$(critical envelope_build python3 -c '
import base64, hashlib, json, sys
payload = open(sys.argv[1], "rb").read()
signature = open(sys.argv[2], "rb").read()
assert len(payload) <= 8192 and len(signature) <= 8192
envelope = {
  "schema": "clipper-external-supervisor-envelope-v1",
  "payload_b64": base64.b64encode(payload).decode("ascii"),
  "signature_b64": base64.b64encode(signature).decode("ascii"),
}
raw = (json.dumps(envelope, sort_keys=True) + "\n").encode()
with open(sys.argv[3], "wb") as f:
    f.write(raw)
print(hashlib.sha256(raw).hexdigest())
' "$session/heartbeat.json" "$session/heartbeat.sig" "$session/envelope.json")"
  [[ "$envelope_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo "EXTERNAL_SUPERVISOR_HEARTBEAT_ENVELOPE_HASH_INVALID" >&2
    exit 2
  }
  identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_RECHECK_REST_$(rest_failure_category)" >&2; exit 2; }
  # One authenticated CLI SSH connection transports the entire signed envelope
  # over stdin, verifies exact bytes on the host, atomically promotes and
  # returns ONE content-bound ACK. No private signing key leaves the owner.
  # On partial/duplicate input, no promotion or ACK is possible.
  remote_pending="$remote/external-supervisor-heartbeat-envelope.watchdog.pending"
  remote_published="$remote/external-supervisor-heartbeat-envelope.json"
  publish_command="set -eu; umask 077; cat > '$remote_pending'; test \"\$(sha256sum '$remote_pending' | cut -d ' ' -f1)\" = '$envelope_sha'; mv -f -- '$remote_pending' '$remote_published'; test \"\$(sha256sum '$remote_published' | cut -d ' ' -f1)\" = '$envelope_sha'; printf 'ACK:%s\\n' '$envelope_sha'"
  transport_ack="$(critical heartbeat_publish gh codespace ssh -c "$CODESPACE_NAME" -- \
    -T "$publish_command" < "$session/envelope.json")"
  if [[ "$transport_ack" != "ACK:$envelope_sha" ]]; then
    record_operation heartbeat_publish "$now" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 2
    echo "EXTERNAL_SUPERVISOR_HEARTBEAT_PUBLICATION_ACK_INVALID" >&2
    exit 2
  fi
  identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_RECHECK_REST_$(rest_failure_category)" >&2; exit 2; }
  signed_issued_epoch="$(date -u -d "$now" +%s)"
  # The signed observed_at timestamp, not local loop start or publish finish,
  # is the freshness authority. Fail closed if transport already consumed it.
  if (( $(date -u +%s) >= signed_issued_epoch + 43 )); then
    echo "EXTERNAL_SUPERVISOR_HEARTBEAT_FRESHNESS_BUDGET_EXHAUSTED" >&2
    exit 124
  fi
  heartbeat_expiry_epoch="$(( signed_issued_epoch + 43 ))"
  # ALL optional tasks share only the first eight seconds after observed_at;
  # the rest is reserved for safe signed renewal and owner DELETE.
  optional_epoch="$(( signed_issued_epoch + 8 ))"
  # Poll source-free phase every minute and verify owner-private output
  # archive approximately every five minutes. Both are best effort and must
  # yield to heartbeat and DELETE; never treat phase-only status as retrieval.
  if (( journal_ready == 1 && deadline_epoch - $(date -u +%s) > 60 )); then
    checked_now="$(date -u +%s)"
    if (( checked_now < optional_epoch &&
          checked_now - last_phase_poll_epoch >= 60 )); then
      last_phase_poll_epoch="$checked_now"
      phase_rc=0
      poll_safe_phase || phase_rc=$?
      if (( phase_rc != 0 )); then
        record_operation safe_phase_poll "$now" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$phase_rc"
      fi
    fi
    if (( $(date -u +%s) < optional_epoch &&
          checked_now - last_checkpoint_epoch >= 300 )); then
      last_checkpoint_epoch="$checked_now"
      checkpoint_rc=0
      checkpoint_private || checkpoint_rc=$?
      if (( checkpoint_rc != 0 )); then
        record_operation archive_verification "$now" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$checkpoint_rc"
      fi
    fi
  fi
  optional_epoch=0
  # No fixed ten-second sleep after slow successful optional calls. Cap idle
  # time to an absolute timestamp from the signed heartbeat, preserving the
  # remaining freshness budget for the next signed renewal.
  until_renew="$(( signed_issued_epoch + 10 - $(date -u +%s) ))"
  while (( until_renew > 0 )); do
    # One idle wait may exceed the fixed eight-second operation cap.
    # Divide only this finite signed-time wait; never extend its deadline.
    sleep_slice="$until_renew"
    if (( sleep_slice >= OP_LIMIT )); then sleep_slice="$(( OP_LIMIT - 1 ))"; fi
    if (( remaining < sleep_slice )); then sleep_slice="$remaining"; fi
    bounded sleep "$sleep_slice" || {
      echo "EXTERNAL_SUPERVISOR_HEARTBEAT_RENEWAL_BUDGET_EXHAUSTED" >&2
      exit 124
    }
    until_renew="$(( signed_issued_epoch + 10 - $(date -u +%s) ))"
  done
done
# The EXIT trap always tries bounded STOP then bounded DELETE and REST checks.
