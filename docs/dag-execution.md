# DAG execution, cache identity and interrupted-run recovery

## Shared contract

`src/clipper/dag.py` provides one content-addressed `DagStore` for ordinary
pipeline stages and the book front half. A `StageIdentity` binds stage name,
source hash, structural contract hash, ordered dependency-output fingerprints,
model revision, decoding parameters and relevant-policy hash. The SHA-256 cache
key is derived from canonical JSON with sorted keys, not from Python dictionary
equality. The persisted **complete** identity must revalidate against the
requested identity using the same canonical fingerprint. In particular,
JSON `false`, number `0` and `0.0` are distinct; nested values are not
coerced. Malformed identity evidence is not eligible for a cache hit.

A reusable `PASS` needs a complete record (identity, status, positive attempt,
start/completion, usage, nonnegative finite cost, output fingerprint, empty
error fields) and the corresponding fingerprint-valid JSON output. JSON
`null`, `false`, `0`, `[]`, `{}` and `""` are valid outputs:
`execute(identity, operation)` carries a separate internal cache-hit bit.
The public `cached_output(identity)` returns the value or `None`, so its
historical `None` ambiguity is retained for direct callers. Do not infer
a miss from `None` when using `execute`.

## Local and older callback coordination

Local stages atomically publish `output.json` and `stage.json` in their
identity-scoped directory. The stage record's replacement is the local PASS
publication boundary, not the helper's return. When its post-replace
cleanup fails or is cancelled, the primary error still propagates and a
fully valid PASS is not rewritten as FAILED. An incomplete, foreign or
corrupt record is never reused. Local claim release checks the owner;
secondary cleanup/reporting faults do not replace an active primary error.

Local write locks use a persistent `.write-lock-file` with an OS-managed
nonblocking advisory lock (POSIX `fcntl.flock`, Windows CRT
`msvcrt.locking`). The open descriptor closes on cancellation/process
termination; do not delete the persistent lock file to release a lock.
Only documented contention errors are retried (POSIX EACCES/EAGAIN,
Windows CRT EACCES/EDEADLOCK or native lock violation 33). Other
`OSError` values propagate immediately; a busy lock times out after a
finite deadline. An existing legacy `.write-lock` directory is **not**
assumed stale or removed: acquisition refuses it pending owner resolution.
Private stage directories remain mode 0700 and files/lock files 0600
on the supported POSIX filesystems. This assumes an owner-controlled
artifact root; it is not hostile-ancestor containment or Windows/remote
filesystem security certification.

`DagLeaseCoordinator` still accepts the older `claim`, `renew`,
`release`, `commit` and `reload` callbacks without paired authority
callbacks. That **compatibility path** retains historical lease/reload
semantics and best-effort cleanup; it does not independently provide
owner-fenced durable PASS or an authoritative non-flushing committed read.
Do not extrapolate the stronger production adapter guarantee to arbitrary
legacy callbacks.

## Authoritative coordinated terminal publication

The production wiring in `scripts/modal_pipeline.py` pairs
`read_committed(identity)` with
`publish_terminal(identity, owner_id, record, output)`. Claimed attempts
receive a `StageClaim` counter; the existing `dag_execution_lease` endpoint
is the **only new writer** of terminal envelopes under
`_dag-authority/<sha256(stage_name + NUL + cache_key)>.json`. The
endpoint checks active owner, unexpired lease, attempt count and a complete
canonical identity **whose independently reconstructed cache key agrees
with the request**, before accepting either PASS or FAILED. PASS additionally
requires complete terminal metadata and a matching output digest. Invalid
requests do not create a terminal; existing bad committed evidence is not
deleted, overwritten, silently repaired or declared successful.

The authoritative `read` action reads committed bytes via
`artifact_volume.read_file`, without calling `reload` on the writer's
dirty mount. Modal Volume reload can implicitly commit outstanding changes,
and background/shutdown commits can occur. A worker therefore does not
write canonical PASS into its mutable mount: only the serialized endpoint
fences and publishes terminal state. A previously valid committed PASS
remains reusable even if the response acknowledgement is lost. That
acknowledgement error still propagates; a failed commit, inaccessible
read or uncertain durability is **not** silently upgraded to success.
Owner-bound release is finite/best effort, and a foreign owner's claim
is never deleted. Old complete persisted records may be *read* through
the legacy fallback after reconstructing and validating canonical identity;
old bytes are not rewritten or migrated.

Renewal runs in a background worker with synchronized termination handoff.
Ordinary renewal failures/false returns cause lease-loss handling;
`KeyboardInterrupt`/`SystemExit` are propagated at supported foreground
checks and after bounded shutdown. An active primary operation/publication
exception remains primary even when cleanup fails. Valid committed PASS
survives a later cancellation. The renewal worker join budget is finite
(one second in the current implementation); an uncooperative callback
cannot be forcibly killed.

## Preconditions, runtime limits and verification

The endpoint currently specifies `max_containers=1` and assumes requests
are processed one at a time by the lease authority. This code setting is
**not** independent live evidence of global serialization or transaction
isolation, and must be verified in any authorized deployment. Local lease
default is 86,400 seconds; the production adapter requests 120 seconds.
The endpoint has a 60-second invocation timeout. JSON RPC transports
complete results: payload limits, latency, lease/renewal timing, follower
polling costs and persistence behavior require separate live qualification.
Busy/unavailable storage, partial authority writes after failed commit,
mixed-version writers, inaccessible filesystems, unsupported locking
semantics and Windows runtime behavior remain open runtime constraints.
Do not claim that file locks alone fence independent coordinator processes.

Reproducible CPU-only checks from an existing checkout:

```bash
ruff check .
ruff format --check .
mypy
pytest
coverage report --fail-under=95
pytest -o addopts="" -q tests/test_dag_authoritative_publication.py \
  tests/test_dag_concurrency_regression.py tests/test_dag_resume.py
```

`pyproject.toml` enforces at least 95% measured **package** coverage; CI
checks it separately from script coverage (see
[GPU-driving script coverage](gpu-worker-script-coverage.md)).
Synthetic and source-extracted endpoint tests do **not** invoke Modal
production functions, certify live lease behavior, verify private books,
source fidelity/rights, recovered image text, human creative review,
deployed models/videos or publishing eligibility.

For book-specific rights and provenance limits, see
[candidate acquisition and faithful ingest](book-candidate-acquisition.md)
and the [creative review contract](book-creative-review.md).
