# Client State Patterns

## Reactive State Policy

### Declarative reactive vs imperative side effect

**Declarative reactive** means expressing output as `f(inputs)` where the framework handles synchronization — atom derivations, React props, `reactivityKeys`. The relationship is declared once; propagation is automatic. There is no trigger, no cleanup, no race window. State has a single source of truth and changes flow through the system without manual wiring.

**Imperative side effect** means reacting to a value changing and then manually doing work — `useEffect`, ref-diff, async IIFEs, callback ref deps. The trigger is manual, cleanup is manual, and the execution timing relative to render is implicit. Every instance duplicates the framework's propagation mechanism with hand-rolled code that can be wrong, incomplete, or racy.

Prioritize declarative reactive because it eliminates entire classes of bugs — stale closures, missing cleanup, race conditions, state desynchronization — by construction rather than by discipline.

### Why useAtomMount over useEffect / ref-diff / callback refs

When a side effect is unavoidable (no declarative mechanism exists, no single user action is the sole trigger), `useAtomMount` with Effect is the exclusive pattern — not `useEffect`, not ref-diff, not callback ref dep arrays. These are all semantically equivalent ("react to value changing, do work") but `useAtomMount` provides:

- **Structured cleanup** via `Effect.addFinalizer` — guaranteed to run on unmount or fiber interruption, not best-effort like `useEffect` return
- **Fiber management** — the Effect runtime can interrupt, cancel, or timeout pending work
- **Error channel** — failures go through Effect's typed error handling, not swallowed or thrown into React's error boundary
- **Composability** — multiple side effects compose with `Effect.zip`, `Effect.timeout`, `Effect.retry`

Callback ref dep arrays are `useEffect` in disguise — they use React's dep mechanism to re-run work on value changes. Plain refs for element capture are fine; dep-array re-runs for side effects are not.

### Patterns to follow

These follow directly from the principles above — declarative first, imperative only when unavoidable, and when imperative, use the safest mechanism.

**1. Declarative** — atom derivation, React props, `reactivityKeys`. Always the first choice. If this applies, imperative patterns are a violation.

**2. Event-source** — imperative call in a user action handler, when the user action is the sole cause of the state change. This eliminates the reactive trigger entirely — the work happens when the user acts, not when state updates.

**3. `useAtomMount`** — Effect-scoped side effect with `Effect.addFinalizer`. The only sanctioned pattern when a side effect is inherent: no declarative mechanism exists, and no single user action is the sole trigger (state comes from server, timers, agent activity, or multiple sources).

This decision applies to component-owned effects. A connection- or registry-lifetime resource owned
by a client service is acquired by that service's `Layer.scoped`, as defined in
`design/patterns/client-di.md`; React must not mount it to manufacture service lifetime.

**Decision:** Can output be `f(inputs)` with platform sync? → Declarative. Is a user action the sole trigger? → Event-source. Otherwise → `useAtomMount`.

**Prohibited:** `useEffect` with side effects, ref-diff (`prevRef !== value → doWork`), async IIFEs for server state, `useState` + `useEffect` sync, callback ref dep arrays for side effects.

## State Ownership

Classify state before implementing it:

- **Server state** — sessions, provider/model configuration, operation progress, durable settings, and daemon status. The RPC query atom is the source of truth. Never copy it into `useState` or another writable atom.
- **Shared client state** — client-only state used across components or surfaces. Put it in an Effect Atom in client-common.
- **Presentation state** — a local selection, open panel, input draft, or route. A component-local value or presentation atom is appropriate, but it must not duplicate a server fact.

Keep domains independent. A flow may compose several query atoms in a pure view model; do not create a combined server RPC, snapshot, or controller merely because one screen uses those domains together.

## Effect Atom Lifetimes

Every writable client atom MUST choose the correct lifetime:

- `Atom.make(initial)` — when the last consumer unmounts, the registry deletes the
  current value after its idle TTL. **All writes are lost.** The next read recreates
  the atom as `initial`. Use this only when that reset is intended.
- `Atom.keepAlive(Atom.make(initial))` — the registry retains the current value even
  with zero consumers. **Writes survive** gates, route changes, and component
  unmounts until the registry is disposed. Use this whenever state must survive them.

`useAtomInitialValues` only writes an initial value; it does not retain it. Never
root-mount an atom to simulate durability—declare it with `Atom.keepAlive`.

## RPC state patterns

`AgentClient` owns the shared ACN transport. Ordinary client code must not build a raw RPC client or maintain a parallel request cache.

### Effect Query adoption

A subsystem may adopt `@magnitudedev/effect-query` without migrating unrelated domains. Its
queries, mutations, and subscriptions are defined in the ACN contract (`packages/acn-protocol`)
through `Acn.query`, `Acn.mutation`, and `Acn.subscription` from `@magnitudedev/effect-query/rpc`:
each definition is a core Effect Query definition that also carries its Rpc, so the wire group and
the client consume the same value. Contract definitions carry the command's `scope` and
`synchronize` postcondition; client-common never re-declares keys, fetch effects, or invalidation.
`createAgentClient` creates one connection-scoped Effect Query client over the shared transport
(`Acn.Client`), and domain services materialize definitions through
`client.effectQuery.query(...)`, `.mutation(...)`, and `.subscription(...)`. Do not construct
domain-local query runtimes in feature code.

Once adopted, that subsystem uses Effect Query as its only query cache and mutation-state authority;
do not retain an AtomRpc query or writable pending/error atom for the same operation. The Effect
Query client owns cache identity, mutation history, and the service runtime for one connection.

Use semantic mutation scopes for resource-specific concurrency, typed mutation-state selectors for
pending and rejection presentation, and mutation synchronization for promised query visibility.
Long-running resource progress still comes from the authoritative query. Migration is vertical:
move a domain's query, mutations, and freshness together, then remove its direct-mirror ownership.

Freshness of Effect Query domains is owned by the connection, not by domains: `StreamChanges` is one
`Acn.subscription` whose events name a query (`{ query, key?, revision? }`), and
`state/changes.ts` drains it into `QueryClient.invalidate` by name. A domain service writes no
invalidation code. A direct-mirror (AtomRpc) domain still invalidates through `Reactivity`; during
migration `createAgentClient` maps the same change events onto Reactivity keys.

### Queries

- Read server state with `useAtomValue(client.query(...))`.
- Give a query stable, domain-owned `reactivityKeys` when mutations or event streams can change it.
- Render loading, success, and failure directly from the query's `Result`.
- Derive transformed views with pure functions, `useMemo`, or derived atoms. Do not copy query results into writable state.
- A query must be observational. Reading or mounting it must not cause installation, downloads, process startup, or other product mutations.

### Mutations

- Trigger a mutation from the user event that causes it with `useAtomSet(client.mutation(...))`.
- Pass every affected domain key through `reactivityKeys`; let AtomRpc invalidate the corresponding queries after success.
- If the UI displays mutation pending/failure state, read the mutation atom's `Result` with `useAtomValue`. Do not wrap the call with `busy`/`error` `useState` or `try/finally` bookkeeping.
- Prefer value mode. Use `{ mode: "promise" }` only when the event handler genuinely needs the returned success value for immediate one-shot control flow. A promise is not a state store and must not be used to mirror loading, errors, progress, or query data.
- For long-running work, the mutation should acknowledge or return an operation ID. Progress and terminal state belong to a query, not to the mutation promise.

### Streams and invalidation

- If a stream announces changes to state available from a query, treat the stream only as an invalidation channel. Consume it in an Effect owned by `useAtomMount`, call `Reactivity.invalidate(...)`, and continue rendering from the query atom.
- The preceding `Reactivity` rule applies to direct-mirror/AtomRpc queries. Effect Query domains are
  kept fresh by the connection's `StreamChanges` drain described above and write no invalidation code;
  a keyed stream that feeds one domain is an `Acn.subscription` drained in that domain's scoped Layer.
- Do not copy stream events into React state when the same facts exist in a query snapshot.
- Use `Effect.addFinalizer` or interruption-safe stream scope for cleanup. Interruption on unmount is normal; handle other failures through Effect's error channel.
- A raw `RpcClient` is permitted only inside such an Effect-scoped bridge when AtomRpc's query/mutation abstraction cannot express the resident stream lifecycle. Keep that bridge in client-common when more than one client surface can use it.

## Shared Boundaries

Reusable query atoms, mutation actions, stream bridges, state derivations, and domain hooks belong in client-common. CLI, web, and desktop should provide rendering and platform interaction, not separate RPC state systems.
