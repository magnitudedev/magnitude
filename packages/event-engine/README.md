# Event Engine

`@magnitudedev/event-engine` settles an event log through projections and releases it to workers.
One engine instance holds one log.

It provides:

- application-selected event discriminators and typed refinement selectors;
- standalone schema-backed root and child address spaces;
- value-equal addresses, encoded address documents, targets, and explicit matching;
- singleton and address-partitioned projections with deterministic settlement;
- explicit singleton, addressed, current-address, and full-entry reads;
- unpartitioned workers and independently admitted address-partitioned worker lanes, whose
  handlers may require Effect services, provided where the engine is built or opened;
- ordered work within each lane, concurrent progress across lanes, and interruption by address,
  address match, or worker;
- atomic transactions that read settled state, decide, and accept any number of events as one;
- engine-family signal definitions with explicit projection publisher capabilities;
- synchronous projection signal cascades with typed payloads and source/address provenance;
- signal-triggered projections and workers, including signal-routed worker-lane admission;
- ambients: live values outside the log that workers update and projections read and fold;
- one settlement barrier before event or signal work is released to workers and observers;
- persistence through a storage-agnostic, per-instance event store port;
- Effect-native facades with capability-separated mutations and queries;
- current and reactive query evaluation over settled projection state.

It does not implement snapshots, offloaded projection fields, dependency-tracked query
invalidation, typed event request/reply, public introspection, or management of several instances;
the host opens and closes instances.

## Signals

Signals are ephemeral derivation occurrences, not events or retained state. A projection declares
the signal definitions it may emit with `emits`, and signal subscribers use the same ordered
handler DSL as event subscribers. Signal payloads contain only domain data; the runtime carries the
actual source projection, optional source address, and timestamp separately. An accepted event
settles its complete projection signal cascade before workers or observers receive any trigger.
Signals are never stored.

## Ambients

`Ambient.define({ name, initial })` defines a value outside the log, with event-core's semantics.
Projection handlers read current values synchronously with `ambient.get(definition)`, and fold
changes with `on.ambient(definition, handler)`. A worker's `update(definition, value)` (or
`runtime.update`) settles under the acceptance lock like an event: ambient handlers fold the value,
the signals they emit cascade and are released, and nothing is stored. `runtime.update` is rejected
with `EngineClosed` once closing has begun or a handler has failed; a worker's `update` is not, like
its `emit`. Every instance starts from `initial`.

## Persistence

`EventStore<Event, E>` is the port: `read`, a stream of the log in commit order, and
`commit(events)`, which appends one acceptance's events atomically. `EventStore.memory()` is an
in-memory store. The engine stores events only: every projection's state is what the log replays
to, and a host that keeps state beside the log owns it itself.

`definition.open({ store, recover })` builds an instance over a store:

1. it replays the stored events through the projections, with workers inactive and no signals
   released;
2. it runs `recover` with a reader and an emit that commits;
3. it activates workers with an emit that commits.

Each accepted event is settled, every worker route it takes is resolved, and the event is
committed. Only then is it released. The whole step, the commit included, is uninterruptible, so a
committed event is always released. Its release reaches workers only if they were active and closing
had not begun when it settled: an event settled before activation, such as one `recover` emits, or
once closing has begun, reaches none. A failed settlement or commit restores every projection's state
and ambient values, and releases nothing.

`definition.build` builds an instance without a store, which keeps accepted events nowhere. Its
workers are inactive until `runtime.activate()`, which activates them with the engine's own emit,
or `runtime.activate({ emit })` with another. The engine's own emit, which workers of an opened
instance use, fails only as a settlement or the store fails: it is never rejected as closed, since
closing waits for the lanes. Once the engine has ended no lane runs, so only a fiber that escaped
its handler (forked as a daemon or into another scope) can emit or update: that is a defect, and
the call dies without storing or settling anything.

## Transactions

`runtime.transact(({ read }) => Effect.succeed({ events, interrupts, result }))` runs its body under
the acceptance lock. Its events settle in order, commit in one store commit and release together,
or, if the body, a settlement or the commit fails, none of them do. Its `interrupts` (addresses,
address matches, or workers) apply after the commit and before the release, in the same
uninterruptible step as the commit, so interrupting the caller during the commit loses neither.
They reach the work, running or queued, that existed before the transaction, and not the work its
own events release, so one transaction can stop running work and start new work.
`runtime.accept(event)` accepts one event the same way.

The engine tracks which fiber holds its acceptance lock: fibers forked while it is held hold it
too, until it is released, and the engine's own lane fibers never do. Such a fiber reads through
any of the engine's readers directly; any other call that takes the lock (`accept`, `transact`,
`activate`, `interrupt`, `update`, `awaitIdle`, `close`, the engine's own emit, or a worker's
`update`) dies as a defect instead of waiting on itself. So a body that calls back into its own engine cannot deadlock; a body that waits on other
work that needs the lock, such as a worker's emit, still can.

## Workers and closing

A worker handler's failure closes the engine as `Failed` with its cause, even when it fails while
the engine's scope is closing. No admitted work vanishes:
interrupting a lane gives every queued task the interruption, which it observes when it runs, and
interrupts its running handler without waiting for it, so the handler's interruption handling may
emit. A task keeps the first interruption it receives, and reads it through `interruption`:
`Interrupted`, `Closed`, or `Removed`. An unpartitioned worker has no address, so it is interrupted
by naming the worker. `awaitIdle` completes once no lane holds work,
including work that work admitted meanwhile and lanes that are still finishing.

Once a handler has failed, accepts, transactions and public updates are rejected with
`EngineClosed`, even before the close it starts has begun.

`runtime.close("closed" | "removed")` starts the close, unless one has started, and waits for
`runtime.closed`. The close rejects further accepts, transactions and public updates with
`EngineClosed`, interrupts every lane with `Closed` or `Removed`, waits for the lanes, which may
still emit, ends reactive queries and signal streams, and completes `runtime.closed` with the
closure. It runs once, in its own fiber: interrupting a caller of `close` stops only its wait, and a
handler that closes its own engine is interrupted by the close rather than waited on. Closing the
instance's scope closes it as `Closed`.

## Facades

```ts
const models = yield* ModelsFacade.Tag.pipe(
  Effect.provide(ModelsFacade.host({ runtime })),
)

yield* models.mutations.sync(modelId)
yield* models.queries.localStatus.get(modelId)
models.queries.localStatus.changes(modelId)
```

Mutation definitions receive `emit` (the runtime's `accept`, unless the host gives another),
`read`, the runtime's `interrupt`, and the runtime's `transact`. Inside a transaction's body,
`read` reads directly and the others die as a defect (see Transactions). Query definitions receive
only `read`; hosting turns each query into a current `get` Effect and a reactive `changes` Stream.

A hosted query runs under one acquisition of the acceptance lock, so all its reads see one settled
state, and it holds the lock for as long as it runs. `changes` reads on subscribing and after
settlements, latest first: a subscriber that falls behind has at most one read pending, never one
per settlement. It reads once more after the engine has closed, so it always ends on the final
state, and emits no two consecutive `Equal` values.
