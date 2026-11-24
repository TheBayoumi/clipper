# Modal CPU-only infrastructure qualification (offline implementation)

**Status:** Code and synthetic tests only; not a live provider qualification.
The Codespace chapter recovery supervisor at a4016288 remains **PAUSED/BLOCKED**.
No chapter adapter, model, TTS, inference, private text, upload, deployment or production
pipeline is part of this experiment.

## Ownership boundary

Implementation: src/clipper/modal_infra_qualification.py.
Tests: tests/test_modal_infra_qualification.py.

The module does not import modal, initialize credentials, create a client or acquire a
resource at import. It is not wired into a CLI or production/chapter route.
There is **no automatic live entry point**. SDKBackend accepts
an explicitly constructed Modal SDK module and client from a separate intentional
caller. Its pinned API contract is Modal Python SDK **1.5.5**, as recorded in
pyproject.toml. Any live call additionally needs explicit run intent,
target workspace, environment and immutable source SHA. The constructor does not
turn these inputs into spend authorization.

Only one explicit client is used by Workspace.from_context, App.run,
Volume.ephemeral, and Sandbox.create; no ambient workspace checks or external
resource handles are accepted. Pinned SDK 1.5.5 exposes Workspace.name as an
Optional[str] property, not a method; missing or mismatched identity stops
before any app/volume context entry. Its synchronous hydrate() is executed
inside asyncio.to_thread, not in the event-loop thread. App.run() and
Volume.ephemeral() expose async-capable context managers; app context return
is correlated via nonempty App.app_id, whereas Volume and Sandbox use
object_id. Any missing app/volume/sandbox ID is a fail-closed
unresolved-allocation outcome; known acquired handles remain subject to
teardown and no second allocation is permitted. Neither an empty ID nor source SHA
constitutes provider recovery or remote identity proof. Sandbox environment is inherited from App.run,
not passed via deprecated Sandbox.create environment_name.

**Important runtime limitation:** The owner is a retained asyncio Task.
observe() has finite *decision* and *reconciliation observation* windows,
not a hard timeout or provider operation deadline. On expiry the durable ledger
records UNRESOLVED with the current active awaited operation and any separate
settled unknown allocation IDs/outcomes, plus
OWNER_MONITORING_CONTINUES_IN_ACTIVE_LOOP, and the
owner continues waiting and cleaning up inside the **same running event loop**.
Callers must explicitly await terminal() before closing the loop; calling
asyncio.run(owner.observe()) alone is unsafe and unsupported. A process crash
or event-loop destruction may leave an unknown provider allocation. No implicit
retry, provider discovery, external supervisor or cross-process continuation is
claimed. Journal intent before allocation is an audit marker; it does not
identify unknown provider allocations.

The ledger uses private file permissions, append-only JSON records and fsync.
Records separate run/source identity, acquisition intent, returned IDs,
active awaited operations, durable unknown-allocation outcomes, first
primary error, separate cleanup/evidence errors, sandbox terminal observations,
exact bytes, entered-context closures and provider_removal=UNKNOWN (not verified).
A failed __aenter__ is never followed by __aexit__ for that context; the
unknown allocation outcome remains recorded, not asserted reversed.
Any journal error latches STOP and forbids new acquisitions; it must not
suppress teardown of previously returned handles. A ledger failure can
make a locally completed synthetic run terminally BLOCKED.
No local observation or journal failure proves provider non-acquisition.

## Synthetic exercise

Payload: bytes(range(256)) * 256 = **65,536 bytes**. Fixed SHA-256 is
computed by code; no private inputs. Two CPU-only sandboxes share one anonymous
ephemeral Volume, execute serially, then context managers exit. The writer
writes /data/infra-check.bin, flushes and fsyncs its file, then emits
one bounded readiness line and stays alive
until explicitly terminated. Only after terminate(wait=True) and a terminal
poll() is the fresh reader created. Reader compares byte count/hash and
remains alive for explicit termination. Readiness only proves a local
write/flush and expected hash expression, NOT provider Volume persistence.
The persistence experiment occurs strictly after first sandbox termination,
through a fresh reader and then exact bytes and SHA from Volume.read_file
before context exit. No unspecified Volume commit or sync API is assumed.

Requested sandbox limits: CPU (0.25, 0.25), memory (256, 256) MiB,
gpu=None, no secrets or ports, block_network=True, timeout=60.
The image may incur preparation/build time or charges outside that timer;
configuration is **not** a free-tier or spend guarantee.

Readiness validates only the **first complete line**, with a finite prefix cap;
trailing output is ignored even when newline and large suffix share one chunk.
Volume bytes are accumulated with a 64 KiB local cap and exact comparison.
Local caps are not global SDK prefetch or process memory bounds.
Iterators have explicit close handling, but a noncooperative SDK call or
iterator close can delay the owner beyond either observation deadline.

## What remains unknown

Offline fakes do **not** prove provider persistence, allocation, resource
deletion, billing, remote termination latency, workspace availability, actual
image preparation, reliable expiry at 60 seconds, or behavior after a real
hard timeout. No optional expiry case has run; commit-after-hard-timeout
remains unknown. App and Volume context exit are reported separately and do
not establish provider removal. Independent review and exact-SHA CI are
required before any live evaluation or run authorization.

## Candidate 2215: immutable identity, reconciliation, and evidence acceptance

**UNQUALIFIED until executable regression checks, exact-SHA CI and independent
review.** The SDK adapter is frozen and its environment must exactly match the
owner's immutable intent before any resource context entry; all SDK operations
still use the explicitly injected client, not ambient SDK identity.

Cancellation of **either** `observe()` or `terminal()` sets STOP, preserves
the retained owner task and prohibits the next acquisition. The caller must
keep the event loop running, await a subsequent `terminal()`, and reconcile
late handles. A returned sandbox ID is retained in
`known_unconfirmed_sandboxes` and `unresolved_ids` until positively observed
terminal, with terminate/poll attempts and errors. These are distinct from
unknown allocations and currently active operations. Context closure never
clears known unconfirmed sandbox identities.

A journal `TERMINAL` record is deliberately **not a success certificate**.
It contains `status=UNCONFIRMED` and
`evidence_durability=UNCONFIRMED_FROM_FILE` even if its bytes are readable.
A successful live owner `terminal()` returns `OWNER_ACKNOWLEDGED` only
after that journal write's successful fsync. If terminal publication raises,
including **after writing bytes but before confirming fsync**, the live result
is BLOCKED with evidence durability UNKNOWN. A restarted process or file-only
reader cannot reconstruct fsync acknowledgement or elevate visible bytes to
SUCCESS. No persistent external receipt, crash-recovery guarantee, resource
removal, cost, image readiness or provider qualification is implied.


## Candidate 2305: result isolation and finite observation

**UNQUALIFIED pending exact-SHA tests, CI and independent review.**
Every owner snapshot isolates nested evidence collections, and each public
`observe()` / `terminal()` return is independent of cached task results.
Mutating a returned verdict, sandbox terminal history, known-resource cleanup
errors or unknown outcomes cannot mutate owner state, journal evidence,
other callers, or a later terminal verdict. Pending and completed observations
are both covered.

Both observation budgets must be finite and strictly positive: NaN,
positive/negative infinity, zero and negative durations are rejected at owner
construction, before any provider entry. The windows remain observations only,
not provider deadlines, cross-process supervision or removal guarantees.
