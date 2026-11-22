# Dormant public synthetic Qwen CPU smoke

**Status: STAGED ONLY — no model inference has run.** This is not a
private-book, editorial lesson-quality, or video-generation acceptance.

## Fixed public input and actual production path

The only source input is `acceptance/book-local-cpu-smoke/public-original-synthetic.txt`,
an original two-paragraph test with label
`PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK` and SHA-256
`eaa39fee22abeeec8df706d64a67d41f6a3c00a91add2c947ac02a1a6cce9d3d`.
The harness creates a deterministic EPUB from that text, passes it through
the existing recorded **in-memory fixture transport**, and then obtains an
actual immutable candidate analysis reference from `BookOrchestrator.ingest_candidate`.
The synthetic Archive-looking URL is an inert compatibility identifier,
not a real publisher source and never fetched. There are **exactly two**
eligible content-bearing, non-heading faithful units, and **at most two**
new model producer calls. No production reviewed registry is changed.

The future real test calls the existing `mine_candidate(ref, local_cpu=True,
max_new_units=2)`, *without injecting a producer or fake model*. It preserves
all pinned settings, generation and raw validation. Model:
`Qwen/Qwen3-1.7B`, exact revision
`8d4744f9e13072f4920c326350fa81eedb74eae9`, default auto dtype,
CPU-only, seed 42, max_new_tokens 960, existing temperature/top-p and
other request settings unchanged. The source and rights remain
unverified, and every outcome stays `MINED_UNREVIEWED` and
`publish_eligible=false`.

The ephemeral standard GitHub-hosted `ubuntu-24.04` runner uses Python
3.11, `torch==2.6.0+cpu`, `transformers==4.57.3`,
`huggingface-hub==0.36.2`, minimal PyYAML and a `--no-deps` editable
project install. No Kokoro/Modal/CUDA packages or paid/GPU runners.
No secrets, upload-artifact or cache actions. The pinned public weight
snapshot may download only in the **later gated real execution**. Both
pip and HF model caches are confined to ephemeral runner storage.
The smoke workflow sets HF_HOME via the runner's RUNNER_TEMP shell variable
and GITHUB_ENV in a step **after** the request gate. The runner context is
not permitted in job-level env expressions; only step context/shell execution
can access the runner temp location. The job has a 45-minute hard timeout, a shorter 34-minute model-process
timeout and up to 2 minutes for a fresh-process completed-cache replay.

The runtime checks at least 6 GiB available RAM and 10 GiB free ephemeral
disk before the first model call. These are minimum prerequisites, **not a
guarantee** that Qwen will load: a prior roughly 2 GiB environment failed
to mmap the first 3.44 GB shard. Loading/generation or the job may time out
on a standard runner. A resource/timeout failure is evidence of a failed
test, not a model completion.

## Strict dormant trigger, not yet created

`.github/workflows/book-local-cpu-smoke.yml` is **push-only**, restricted
to branch `review/unified-tts-cleanup` and changes to the sole path
`acceptance/book-local-cpu-smoke/run-request.json`. There is no schedule,
PR, workflow_dispatch, other-workflow piggyback, or automatic inference
on harness, workflow, test or documentation commits. GitHub's UI manual
dispatch requires a workflow on the default branch; that is deliberately
not being added. **The run-request file MUST remain absent in this staging
pass.**

After the owner reviews the staged final full SHA and its green exact-head
synthetic CI, a separate one-off commit can **add only** that request file,
with this exact schema (replace the placeholder with the **full final green
reviewed code SHA**):

```json
{
  "schema": "clipper-public-book-local-cpu-smoke-run-request-v1",
  "intent": "REAL_EXECUTION",
  "branch": "review/unified-tts-cleanup",
  "reviewed_code_sha": "REPLACE_WITH_FINAL_REVIEWED_40_HEX_SHA",
  "fixture_label": "PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK",
  "fixture_sha256": "eaa39fee22abeeec8df706d64a67d41f6a3c00a91add2c947ac02a1a6cce9d3d",
  "max_new_units": 2
}
```

The future trigger must start from that exact SHA on the **same branch**,
add only this single file, commit once, and push that one commit. The gate
requires `github.sha == git HEAD`, exactly one parent equal to the
request's reviewed SHA **and** the GitHub push-before SHA, and exactly one
added diff path (the run request). It validates strict JSON keys, no
duplicates, fixed fixture bytes and SHA, exact branch, budget, identity and
intent *before* dependency installation or model invocation. Extra paths,
private input, other model settings, another parent, multiple commits,
edits to an existing request or another branch all fail. No main edit,
merge, PR or other workflow is part of this procedure.

## Phase observability, failure and replay

The harness emits `PUBLIC_BOOK_CPU_SMOKE_PHASE` markers for fixture
preparation and initial model work. The existing pinned producer additionally
emits public-only start/finish markers for snapshot resolution, tokenizer
initialization, CPU weight loading and actual generation when the
harness sets `CLIPPER_PUBLIC_BOOK_CPU_SMOKE=1`. These markers do not include
prompts, completions, environmental secrets or private context.

Before inference it records resource availability, package versions, pinned
model identity, canonical requests and request/source fingerprints. During
model work it samples memory and available disk, boundedly. It retains
raw generated completions, DAG stage/output provenance, original progress,
failure records and optional provisional export on **ephemeral local disk**.
A valid empty model reply counts only as a completion-mechanics result,
not a useful lesson. Missing, malformed or invalid results never become
a fabricated PASS. Incomplete work exits nonzero.

On a complete first run, the workflow launches a new Python process to
replay the exact same production candidate-ref path, verifying zero new
model initializations and identical raw/progress/export bytes and output
identity. A timeout, resource failure or mismatch fails the job. No
selector, narration, storyboard, visuals, assembly or publishing are run.

## Full exact evidence in bounded job logs, without artifact uploads

The always-run collector reads **only** the known public smoke output root,
allowlisting public evidence files, bounded generated completion/failure/
progress/export files, and the two mining DAG stage/output files. It
enforces at most 90 files, 96,000 raw bytes per file and 512,000 total
raw bytes. Collection **and replay hashing** use one confined reader:
every root/path ancestor must be non-symlink, the strict resolved path
must stay under the dedicated public root, the source must be a regular
file, and sizes must be checked **before** opening or allocating bytes.
An opened file is stat-checked again and read at most remaining-budget+1
bytes; growth, replacement, nonregular inputs, symlinks and total/file
overruns fail, not leak or truncate output. No traversal of unrelated
artifacts, the HF cache, source repositories or private inputs occurs.
No upload-artifact, actions/cache, setup-python cache or committed output
files are used. A failed collection does **not** prove a completed model run.

Each evidence record is a JSON log line tagged
`PUBLIC_BOOK_CPU_SMOKE_EVIDENCE_CHUNK_V1` with a gate-derived `execution`
and `outcome`, the public label, relative filename, 0-based `index`,
`chunk_count`, original `byte_count`, full-file `sha256` and `data_b64`.
Only a verified branch, one-file trigger, parent, and matching on-disk
trigger trace can earn `execution=GATED_PUBLIC_CPU_ATTEMPT`. Synthetic,
dry-run and failed-before-gate collection is labeled
`SYNTHETIC_OR_UNGATED_NO_MODEL` (or a gated request **without**
a model-execution trace). Neither asserts an actual completion.
The outcome independently records preflight, snapshot/load start,
generation start, invalid/incomplete reply, process timeout, validated
zero-candidate completion or valid unreviewed candidates, and whether
a fresh-process cache replay was verified. These statuses do **not**
assert semantic lesson quality. Model loading/generation phase events
are source-free and recorded privately in `model-phases.jsonl`.
Each base64 chunk encodes **at most 600 raw bytes**, staying under normal
GitHub log-line limits. This is exact original evidence, not a summary.
An Actions job summary is written **only** after gate verification; synthetic
pytest runs cannot append fake execution summaries to it. No storage artifact
is uploaded.

After downloading the later job's **raw text log** as `job.log`, use this
standard-library reconstruction procedure:

```python
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path

groups = defaultdict(list)
for line in Path("job.log").read_text(encoding="utf-8").splitlines():
    offset = line.find('{"byte_count":')
    if offset < 0:
        continue
    item = json.loads(line[offset:])
    if item.get("kind") == "PUBLIC_BOOK_CPU_SMOKE_EVIDENCE_CHUNK_V1":
        assert item["label"] == "PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK"
        assert item["execution"] == "GATED_PUBLIC_CPU_ATTEMPT"
        assert item["outcome"] != "NO_VERIFIED_GATED_RUN"
        groups[item["path"]].append(item)

root = Path("public-smoke-reconstruction")
for name, chunks in groups.items():
    path = Path(name)
    assert not path.is_absolute() and ".." not in path.parts
    ordered = sorted(chunks, key=lambda x: x["index"])
    head = ordered[0]
    assert [x["index"] for x in ordered] == list(range(head["chunk_count"]))
    assert all(
        (x["chunk_count"], x["byte_count"], x["sha256"])
        == (head["chunk_count"], head["byte_count"], head["sha256"])
        for x in ordered
    )
    data = b"".join(base64.b64decode(x["data_b64"], validate=True) for x in ordered)
    assert len(data) == head["byte_count"]
    assert hashlib.sha256(data).hexdigest() == head["sha256"]
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
print(f"Verified {len(groups)} exact public synthetic files")
```

A log truncated by GitHub or missing expected chunks cannot support a
claim of exact output reconstruction. The initial green synthetic CI in
this staging pass does **not** constitute real inference, private-book
acceptance, lesson quality approval or a complete book-to-video pipeline.
