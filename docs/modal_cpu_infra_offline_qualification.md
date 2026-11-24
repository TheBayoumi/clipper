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
object_id. A missing app ID is a fail-closed evidence error; the acquired
app context must still exit, and neither an empty ID nor source SHA
constitutes provider recovery or remote identity proof. Sandbox environment is inherited from App.run,
not passed via deprecated Sandbox.create environment_name.

**Important runtime limitation:** The owner is a retained asyncio Task.
observe() has finite *decision* and *reconciliation observation* windows,
not a hard timeout or provider operation deadline. On expiry the durable ledger
records UNRESOLVED and OWNER_MONITORING_CONTINUES_IN_ACTIVE_LOOP, and the
owner continues waiting and cleaning up inside the **same running event loop**.
Callers must explicitly await terminal() before closing the loop; calling
asyncio.run(owner.observe()) alone is unsafe and unsupported. A process crash
or event-loop destruction may leave an unknown provider allocation. No implicit
retry, provider discovery, external supervisor or cross-process continuation is
claimed. Journal intent before allocation is an audit marker; it does not
identify unknown provider allocations.

The ledger uses private file permissions, append-only JSON records and fsync.
Records separate run/source identity, acquisition intent, returned IDs,
sandbox terminal observations, exact bytes, context closures and
provider_removal=UNKNOWN (not verified). Failed journal records MUST NOT be
interpreted as provider non-acquisition; known handles remain subject to teardown.

## Synthetic exercise

Payload: bytes(range(256)) * 256 = **65,536 bytes**. Fixed SHA-256 is
computed by code; no private inputs. Two CPU-only sandboxes share one anonymous
ephemeral Volume, execute serially, then context managers exit. The writer
writes /data/infra-check.bin, flushes and fsyncs, executes sync /data
to commit the mounted Volume, emits one bounded readiness line and stays alive
until explicitly terminated. Only after terminate(wait=True) and a terminal
poll() is the fresh reader created. Reader compares byte count/hash and
remains alive for explicit termination. The owner also checks exact bytes and
SHA from Volume.read_file before context exit.

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
