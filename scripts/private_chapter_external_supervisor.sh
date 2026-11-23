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
owner_recovery_safe=0

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
dig = re.compile(r"[0-9a-f]{64}\Z")
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
                if (not isinstance(audit_ids, list) or not 1 <= len(audit_ids) <= 128
                    or len(set(audit_ids)) != len(audit_ids)
                    or not all(isinstance(x, str) and dig.fullmatch(x) for x in audit_ids)):
                    raise ValueError("hybrid local audit identities invalid")

                def stage_artifact(stage, identity):
                    candidates = [r["relative_path"] for r in contents
                                  if r["kind"] == "text"
                                  and r["relative_path"].startswith("book-candidates/"+stage+"/")
                                  and r["relative_path"].endswith("/validated.json")]
                    matching = [path for path in candidates
                                if member_json(path, 250_000).get("artifact_identity") == identity]
                    if len(matching) != 1:
                        raise ValueError("hybrid stage identity absent or ambiguous")
                    return member_json(matching[0], 250_000)

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
                    or reconcile.get("payload",{}).get("ledger_sha256") is None):
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
                    (json.dumps(review_data,ensure_ascii=False,sort_keys=True)+"\\n").encode("utf-8")
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

# Emit a distinct sanitized category for exhausted clock versus changed host.
cleanup_identity_failure() {
  local reason="$1"
  if (( deadline_epoch - $(date -u +%s) <= 0 )); then
    echo "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_MANUAL_REST_CHECK_REQUIRED" >&2
  else
    echo "$reason" >&2
  fi
}

terminate_host() {
  original_status=$?
  trap - EXIT INT TERM HUP
  set +e
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
trap 'exit 2' INT TERM HUP
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
if created > now:
    sys.exit(2)
if now >= created + datetime.timedelta(seconds=7200):
    sys.exit(3)
execution_allowed = now < created + datetime.timedelta(seconds=6900)
print(x["id"], x["created_at"], "EXECUTION" if execution_allowed else "CLEANUP_ONLY")
')" || {
  code=$?
  if (( code == 3 )); then
    echo "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_NO_AUTOMATED_DELETE" >&2
  else
    echo "EXTERNAL_SUPERVISOR_WRONG_HOST_IDENTITY" >&2
  fi
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
remote="$PRIVATE_CHAPTER_ROOT/book-candidates/private-host"
identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_CHANGED" >&2; exit 2; }
bounded gh codespace ssh -c "$CODESPACE_NAME" -- \
  "mkdir -p '$remote' && chmod 700 '$remote'" >/dev/null
identity_matches || { echo "EXTERNAL_SUPERVISOR_IDENTITY_CHANGED" >&2; exit 2; }
bounded gh codespace cp -c "$CODESPACE_NAME" \
  "$PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY" \
  "remote:$remote/external-supervisor-public.pem" >/dev/null
recovery_allowed=1

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
