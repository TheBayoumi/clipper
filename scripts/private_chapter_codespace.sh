#!/usr/bin/env bash
# Execute only inside the one explicitly approved Codespace. No provisioning.
set -euo pipefail
umask 077

if [[ "${1:-}" != "inspect" && "${1:-}" != "brief" && "${1:-}" != "resume" &&
      "${1:-}" != "hybrid-brief" && "${1:-}" != "hybrid-resume" ]]; then
  echo "Usage: bash scripts/private_chapter_codespace.sh inspect|brief|resume|hybrid-brief|hybrid-resume" >&2
  exit 2
fi
mode="$1"
: "${PRIVATE_CHAPTER_ROOT:?Set an absolute private artifact directory outside the worktree}"
: "${PRIVATE_CHAPTER_MANIFEST:?Set the private two-chapter manifest path}"
: "${CODESPACE_NAME:?Run in the single authorized Codespace}"
: "${PRIVATE_CHAPTER_APPROVED_SHA:?Exact Codex-reviewed checkout must be pinned}"
: "${PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY_SHA256:?Owner external signer must be pinned}"
if [[ "${CODESPACES:-}" != "true" ]]; then
  echo "PRIVATE_CHAPTER_HOST_NOT_CODESPACE" >&2
  exit 2
fi
if [[ "$(git rev-parse --abbrev-ref HEAD)" != "review/unified-tts-cleanup" ||
      "$(git rev-parse HEAD)" != "$PRIVATE_CHAPTER_APPROVED_SHA" ]]; then
  echo "PRIVATE_CHAPTER_UNEXPECTED_BRANCH" >&2
  exit 2
fi
if [[ "$PRIVATE_CHAPTER_ROOT" != /* || "$PRIVATE_CHAPTER_MANIFEST" != /* ]]; then
  echo "PRIVATE_CHAPTER_PATHS_MUST_BE_ABSOLUTE" >&2
  exit 2
fi
repo_root="$(git rev-parse --show-toplevel)"
mkdir -p "$PRIVATE_CHAPTER_ROOT"
private_root="$(realpath "$PRIVATE_CHAPTER_ROOT")"
case "$private_root/" in
  "$repo_root/"*|//) echo "PRIVATE_CHAPTER_WORKTREE_STORAGE_FORBIDDEN" >&2; exit 2 ;;
esac
# The Codespace REST record, not the time the command was first run, starts
# the single cumulative 2-hour clock. Do not override or cache creation time.
if ! codespace="$(gh api "user/codespaces/${CODESPACE_NAME}")"; then
  echo "PRIVATE_CHAPTER_CODESPACE_METADATA_UNAVAILABLE" >&2
  exit 2
fi
if ! clock="$(printf '%s' "$codespace" | python -c '
import datetime, json, sys
obj = json.load(sys.stdin)
machine = obj.get("machine") or {}
expected = (4, 17179869184, 34359738368)
actual = (machine.get("cpus"), machine.get("memory_in_bytes"), machine.get("storage_in_bytes"))
if obj.get("repository", {}).get("full_name") != "TheBayoumi/clipper" or actual != expected:
    sys.exit(2)
created = datetime.datetime.fromisoformat(obj["created_at"].replace("Z", "+00:00"))
if created.tzinfo is None:
    sys.exit(2)
now = datetime.datetime.now(datetime.timezone.utc)
deadline = created + datetime.timedelta(seconds=7200 - 300)
remaining = int((deadline - now).total_seconds())
if created > now or remaining <= 0:
    sys.exit(2)
print(created.isoformat(), remaining)
')"; then
  echo "PRIVATE_CHAPTER_CODESPACE_SHAPE_OR_DEADLINE_REJECTED" >&2
  exit 2
fi
read -r codespace_created_at remaining_seconds <<< "$clock"
# REST metadata and branch do not prove an independent hard supervisor:
# require a fresh externally signed heartbeat BEFORE Python or any providers.
if ! python -m clipper.books.chapter_lifecycle \
  --artifact-root "$private_root" \
  --codespace-created-at "$codespace_created_at"; then
  echo "PRIVATE_CHAPTER_LIVE_EXTERNAL_SUPERVISOR_REQUIRED" >&2
  exit 2
fi
# Exactly one shared Hugging Face cache under the private root; same pinned
# revision for public tokenizer preflight and actual Qwen model weights.
export HF_HOME="$private_root/model-cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$HF_HOME"
# GNU timeout bounds only this Python invocation. It is NOT a lifecycle
# supervisor. The independent owner-side watchdog covers install, human waits,
# Python hangs, stop and verified deletion from original host creation time.
set +e
timeout --signal=TERM --kill-after=15s "${remaining_seconds}s" \
  python -m clipper.books.chapter_host "$mode" \
    --artifact-root "$private_root" \
    --manifest "$PRIVATE_CHAPTER_MANIFEST" \
    --codespace-created-at "$codespace_created_at"
status=$?
set -e
if [[ $status -eq 124 || $status -eq 137 ]]; then
  echo "PRIVATE_CHAPTER_SHARED_DEADLINE_STOP_NO_AUTO_RESTART" >&2
  exit 2
fi
exit "$status"
