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
