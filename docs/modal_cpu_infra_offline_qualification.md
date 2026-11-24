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

## Candidate 2337: explicit synchronous driver (UNQUALIFIED)

New entrypoint: `clipper.modal_infra_driver.run_infrastructure_experiment`.
The previously qualified owner module is unchanged. This *driver* has no CLI,
implicit Modal import, credential setup, client discovery, model initialization,
account selection or auto-execution. An independently approved and authenticated
host wrapper would have to supply the **already-instantiated** pinned SDK 1.5.5
module, explicit client, named workspace/environment, exact source SHA and
explicit run intent. Passing strings and handles does **not** grant provider or
spending authorization. Do not place tokens or book text in the journal or repo.

Contract illustration, **not an executable credential/bootstrap recipe**:

```python
from pathlib import Path
from clipper.modal_infra_driver import run_infrastructure_experiment, result_exit_code
from clipper.modal_infra_qualification import Intent

# sdk and client are existing, separately authorized objects from a host wrapper;
# no SDK/client construction, network calls or provider activation is shown here.
intent = Intent(
    expected_workspace="APPROVED_WORKSPACE",
    environment="APPROVED_ENVIRONMENT",
    source_sha="EXACT_APPROVED_40_HEX_GIT_SHA",
    explicit_run_intent="modal-cpu-synthetic-persistence",
)
result = run_infrastructure_experiment(
    sdk, client, intent, Path("/PRIVATE_OWNER_PATH/modal-infra.jsonl")
)
code = result_exit_code(result)  # 0 only for live owner-acknowledged local SUCCESS
```

The actual call **can allocate provider resources** if a future independently
authorized caller supplies real SDK/client and invokes it. None has been called
during offline implementation. The driver is POSIX main-thread only and refuses
nested asyncio loops or unsupported threads before entering provider contexts.
It owns one persistent `asyncio.Runner` event loop and private Ledger, uses
SDKBackend+QualificationOwner directly, calls `observe()` and always awaits
`terminal()` before leaving that loop. SIGINT/SIGTERM temporarily request
STOP (including repeated signals), never cancel the owner, and previous signal
handlers are restored on return/failure. Ledger closure follows owner terminal;
pre-start exceptions close the ledger and do not claim a provider result.
A permanent or stalled remote operation can hold the caller indefinitely.
No hard operation/cost deadline, process-crash, SIGKILL, or cross-process
recovery guarantee is introduced.

The returned structured result preserves status, OWNER_ACKNOWLEDGED vs UNKNOWN
durability, known-unconfirmed IDs, unknown allocations, active operations and
provider_removal=UNKNOWN. `result_exit_code` is pure: returns 0 only for
locally SUCCESS + OWNER_ACKNOWLEDGED and 2 otherwise. No machine JSON or
success receipt is published from file bytes.

Offline E2E tests inject a test-only SDK-shaped object through the **real**
SDKBackend and owner; POSIX signal fixtures may send signals to their own test
process, never a provider host. The driver, unlike its owner baseline, requires
new independent execution, exact-SHA CI + unchanged 95% coverage gate and
separate final review before any claim of qualification. Live image preparation,
cost/free eligibility, persistence/expiry and resource removal remain unknown.


## Candidate 0021: driver cancellation and completion handoff (UNQUALIFIED)

The driver, not the qualified owner module, handles cancellation of its public
observation and repeated cancellations while awaiting `terminal()`: STOP is
latched, only the driver waiter removes its own cancellation state, and the
retained owner task remains alive until known late handles have had terminate
and poll attempts. Unexpected observation failures are re-raised only **after**
owner reconciliation rather than being turned into a success. No new provider
acquisition is attempted after STOP.

The owner-returned `status` is preserved as `owner_status` in the driver's
copy. The driver adds `driver_cancelled`, `driver_interrupted` and
`driver_disposition`. Any SIGINT/SIGTERM processed before signal-handler
restoration, including during final journal fsync or ledger close, produces
driver `status=BLOCKED` and a nonzero `result_exit_code` even if the owner
already produced a valid local `OWNER_ACKNOWLEDGED`. Journal records and owner
acknowledgement are not retroactively rewritten as an evidence failure.
Completion handoff linearizes after ledger closure and restoration of previous
signal handlers; signals after restoration are the caller's responsibility.
This remains in-process only: SIGKILL, process death and noncooperative SDK
calls have no recovery or deadline guarantee.


## Candidate 0036: completed-await cancellation checkpoint (UNQUALIFIED)

The driver terminal waiter now makes a scheduler checkpoint after receiving a
terminal result, before breaking its reconciliation loop. This closes the case
where cancellation was requested on the driver task but an already-completed
owner await returned without delivering it. Such cancellation is latched as
driver interruption, STOP is requested, and the driver continues through
terminal reconciliation. An unexpectedly **cancelled owner task** raises an
error instead of spinning forever on repeated `CancelledError`.

Offline regression timing holds cleanup through both intended terminal-waiter
cancellations, rather than relying on a fixed helper timeout to force release.
A separate owner-already-completed fixture tests pending cancellation after
local owner SUCCESS and requires driver BLOCKED/nonzero without altering the
owner's journal evidence. The owner baseline remains untouched.
