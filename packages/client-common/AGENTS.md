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

**Decision:** Can output be `f(inputs)` with platform sync? → Declarative. Is a user action the sole trigger? → Event-source. Otherwise → `useAtomMount`.

**Prohibited:** `useEffect` with side effects, ref-diff (`prevRef !== value → doWork`), async IIFEs for server state, `useState` + `useEffect` sync, callback ref dep arrays for side effects.

## State Ownership

Classify state before implementing it:

- **Server state** — sessions, provider/model configuration, operation progress, durable settings, and daemon status. The RPC query atom is the source of truth. Never copy it into `useState` or another writable atom.
- **Shared client state** — client-only state used across components or surfaces. Put it in an Effect Atom in client-common.
- **Presentation state** — a local selection, open panel, input draft, or route. A component-local value or presentation atom is appropriate, but it must not duplicate a server fact.

Keep domains independent. A flow may compose several query atoms in a pure view model; do not create a combined server RPC, snapshot, or controller merely because one screen uses those domains together.

## AtomRpc Patterns

`AgentClient` is the standard RPC interface. Ordinary client code must not build a raw RPC client, manually run RPC Effects, or maintain a parallel request cache.

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
- Do not copy stream events into React state when the same facts exist in a query snapshot.
- Use `Effect.addFinalizer` or interruption-safe stream scope for cleanup. Interruption on unmount is normal; handle other failures through Effect's error channel.
- A raw `RpcClient` is permitted only inside such an Effect-scoped bridge when AtomRpc's query/mutation abstraction cannot express the resident stream lifecycle. Keep that bridge in client-common when more than one client surface can use it.

## Shared Boundaries

Reusable query atoms, mutation actions, stream bridges, state derivations, and domain hooks belong in client-common. CLI, web, and desktop should provide rendering and platform interaction, not separate RPC state systems.
