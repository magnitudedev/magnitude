---
applies_to:
  - packages/client-common/**
  - cli/**
  - web/**
  - desktop/**
---

# Client query and mutation abstraction

## Purpose

Effect Query is the client-common mechanism for observing and changing backend-owned state: every
client↔ACN interaction is a boundary-group query, mutation, or subscription materialized through the
connection's Effect Query client. A feature exposes the product capability built with the mechanism
rather than propagating mechanism-level representations.

```text
backend authority
      |
      | RPC snapshot / command / invalidation
      v
+---------------------------+
| ACN-backed client service |
|                           |
| private                   |
| - materialized atoms      |
| - keyed watch dependency  |
| - mutation-state lookup   |
|                           |
| public                    |
| - observed domain state   |
| - domain operations       |
| - semantic command status |
+---------------------------+
      |
      | values and domain meaning
      v
+---------------------------+       +----------------+
| composed client service   | ----> | React hook     |
| use-case state + Effects  |       | UI adaptation  |
+---------------------------+       +----------------+
                                              |
                                              v
                                        presentation
```

The abstraction is successful when a caller can use the domain without knowing whether its state
came from Effect Query, another cache, or a direct in-memory source.

Service identity, construction, dependencies, and lifetime follow
[Client dependency injection](./client-di.md). This document defines the state-mechanism boundary;
the DI document defines how the services at that boundary are acquired and composed.

## Ownership

```text
core operation definitions (Query / Mutation / Subscription)
              |
              v
   connection Effect Query client (AgentClient)
              |
   +----------+-----------------------------+
   |                                        |
   v                                        v
domain service (when warranted)       hook materializing a definition directly
   |                                        |
   +------------- semantic API -------------+
```

The ACN change stream (`StreamChanges`) carries pokes that name a query. The Effect Query client
drains it once per connection and invalidates the named query entries; domains own no invalidation
code for poked state. A keyed watch that keeps a domain's queries fresh is a dependency of those
query atoms inside the domain's service. There is no second client state mechanism for ACN data.

## When Query and Mutation exist

Query and Mutation represent interaction with an independently owned authority. They are not the
default vocabulary for every asynchronous function or every layer of an application.

### Query

A Query is warranted when all of the following are true:

1. The value is observed rather than owned by the caller.
2. Obtaining it has an asynchronous lifecycle that callers must represent: unavailable, fetching,
   successful, refreshing, or failed.
3. Equivalent observations have stable input identity and may share cached data or in-flight work.
4. The observation can become stale and has a defined refresh or invalidation relationship.

```text
external owner + keyed asynchronous observation + reusable lifecycle + invalidation
                                   = Query
```

A Query is not warranted for:

- a pure derivation from already-observed values;
- client-owned in-memory state;
- an event or one-shot Effect whose result is consumed only by its caller;
- a command merely because the command eventually affects observed state; or
- a second representation of an existing lower Query.

### Mutation

A Mutation is warranted when all of the following are true:

1. A command requests change from an independently owned authority.
2. Submission has an asynchronous acknowledgement or rejection lifecycle useful beyond the
   current Effect call.
3. Invocations need cache-level behavior such as semantic concurrency scope, retry, observable
   pending/rejection status, retention, or synchronization with affected Queries.

```text
command across ownership boundary + tracked asynchronous invocation + query synchronization
                                      = Mutation
```

A Mutation is not warranted for:

- changing client-owned Atom state;
- composing existing domain operations into a larger Effect;
- adapting an Effect to a click handler;
- representing authoritative long-running resource progress; or
- recreating a lower Mutation at a higher abstraction boundary.

If a command crosses an ownership boundary but no shared invocation lifecycle is needed, an
ordinary Effect may be sufficient. If retained client state has meaningful modes and transitions,
it remains Atom state or a state machine; wrapping it in Query and Mutation would add a cache and
mutation registry without adding an authority.

### Boundary rule

Whether Query and Mutation are needed is decided where authority is crossed. A higher service that
uses an existing query or mutation does not cross that authority again. It therefore consumes the
lower service's state and Effects and does not recreate Query definitions, Mutation definitions,
cache entries, or mutation history.

```text
UI -> composed client service -> ACN-backed client service -> ACN
                                      ^                       ^
                                      |                       |
                              Query/Mutation mechanism   authority boundary

No new authority boundary exists between UI, hook, and composed service.
```

## Responsibilities

### ACN

ACN owns application truth, durable state transitions, validation, admitted work, and authoritative
operation progress. It exposes observational queries, mutations, and invalidation-only watches.

### Effect Query definitions

Query, Mutation, and Subscription definitions live in the ACN boundary groups in
`packages/acn-protocol` and are constructed with the core Effect Query primitives. Cache identity
derives from the payload; a mutation's scope and synchronization postcondition are declared with
the command. The ACN RPC adapter derives RPCs from the root `AcnBoundary` group and supplies an
implementation layer to clients. Definitions capture no client instance, RPC, Atom registry,
React lifecycle, or feature workflow.

When a domain has a client service (the client-di criteria: connection-lifetime state, a resident
resource such as a keyed watch, reusable operations with dependencies, or a stateful use case), its
definitions are consumed only by that service; UI and composed services see the service's semantic
surface. When no service is warranted (sessions, projects, provider auth, usage, skills), a hook
materializes the definition directly through the connection client; that hook is the domain's
terminal adapter and still returns values and callbacks, not atoms or clients.

### ACN-backed client service

One service instance exists per client connection and owns materialization of a backend domain. It:

- materializes the domain's boundary operations with that connection's Effect Query client;
- exposes one read-only Atom containing the domain query Result;
- exposes domain operations as functions returning Effects;
- exposes semantic derived state or status selectors when presentation needs mutation intent; and
- keeps materialized query atoms, mutation atoms, mutation history, and synchronization atoms
  private.

The service does not copy backend state into writable atoms. Its state Atom is a read-only view of
the canonical query entry.

```text
                   private implementation
                  +-----------------------+
ACN notification ->| QueryClient           |
snapshot RPC ------>| query atom ----+      |
command RPC ------->| mutation atom  |      |
                  +----------------|------+
                                   |
                      public       v
                    state Atom + Effect operations + semantic selectors
```

### Composed client service

A use case spanning multiple domains may be a client service when it owns meaningful in-memory
state, concurrency, or cancellation. It consumes the lower services' observed state and Effect
operations. It never reaches into their Query definitions, Mutation definitions, materialized
atoms, caches, or invalidation lifecycle.

Each command is one Effect program. Values returned by one operation are passed directly into the
next operation. Authoritative long-running progress is awaited through lower service observation
using the exact admitted identity. No step infers causality from the latest mutation, a matching
resource discovered later, or timing.

```text
choose exact configuration
          |
          v
install(configurationId) -> admission(providerModelId, modelDownloadId)
          |                              |
          | await exact download         +---- cancellation addresses exact modelDownloadId
          v
assign(exact providerModelId) -> exact selection
          |
          v
load(exact selection) -> instanceId
          |
          | await exact instanceId
          v
complete onboarding
```

### Hook

A hook is the terminal React adapter. Inside the same domain implementation boundary, it may set
and observe a private Mutation atom. It returns callbacks and presentation-relevant command
outcomes, never that atom or another writable proxy around it. A composed Effect is adapted to a
user event as an ordinary runtime action, not as another Mutation.

A hook does not return query atoms, mutation atoms, query clients, mutation history, invalidation
bridges, or watch atoms. It does not require a component to mount synchronization plumbing.

### UI

The UI chooses when a user action occurs and renders state. It may consume semantic command status
such as `isInstalling(configurationId)` or the outcome of `assign`, but it does not inspect a
mutation registry, select the latest invocation, execute dependent operations, or synchronize
backend state.

## Conforming example

This example is illustrative; names are domain-specific rather than prescribed infrastructure.

```ts
interface LocalModels {
  readonly state: Atom<Result<LocalModelsState, LocalModelsError>>
  readonly reconcile: (
    identity: CatalogIdentity,
  ) => Effect<CatalogModelReconciliationAdmission, ReconciliationError>
  readonly reconciliationFor: (
    identity: CatalogIdentity,
  ) => Atom<Result<CatalogModelReconciliationAdmission, ReconciliationError>>
}
```

The service privately implements `reconcile` with `Mutation.execute`. A composed setup service can
flat-map its returned admission into slot assignment. A hook can adapt the same Effect to a click
handler and expose `reconciliationFor(identity)` to render pending or rejected command intent. Neither
caller receives the underlying Mutation atom or its global history.

Non-conforming shapes include:

```ts
// Mechanism bundle: callers must understand and coordinate the implementation.
{ queryAtom, resultAtom, installMutation, mutationStatesAtom, bridgeAtom, watchAtom }

// Correlation: causality is guessed from unrelated global history.
mutationStates.at(-1)
```

## Composition rules

1. A backend domain has one canonical query cache and one materialization owner per connection.
2. Query and mutation atoms terminate at that owner. They are not passed into another service or
   exported from the package barrel.
3. Dependencies between operations are represented by Effect composition over exact outputs.
4. Query observation supplies authoritative state; mutation state supplies only command intent and
   outcome.
5. A public selector expresses domain meaning and hides registry ordering and mutation identity.
6. Freshness of poked state is owned by the connection: one scoped `StreamChanges` subscription
   invalidates queries by name. Domain services register no invalidation for it; public state
   mounting is observational, and callers never mount synchronization separately.
7. A keyed watch is a dependency of the query atoms it keeps fresh, inside the owning service: open
   while observed, closed when unobserved. No hook or component mounts a watch.
8. A query remains observational. Creating or mounting a service cannot install, assign, load,
   stop, delete, or complete anything.
9. A composed service exists only for a stateful cross-domain use case. One-shot composition is a
   plain domain Effect, not a new controller or workflow abstraction.

## Why this boundary

Effect Query's low-level objects are deliberately powerful: they expose cache lifecycle,
invocation lifecycle, synchronization, and registry-wide history. Propagating them makes every
caller responsible for those mechanics and permits independent interpretations of causality.

Terminating them in the ACN-backed service provides one place where cache and invalidation
correctness are established. Exposing Effect operations preserves native composition: output and
error types connect dependent steps without flags, polling utilities, or inferred relationships.
Exposing semantic state keeps React declarative without copying server facts.

The result is less public API, fewer valid call arrangements, and a direct correspondence between
the type-level dependency chain and the behavior the user observes.

## Conformance

- No client or composed service imports a domain's Query definition, Mutation definition, or
  materialized atom.
- No public client-common barrel exports domain materialization internals.
- No caller mounts invalidation or watch atoms; keyed watches are dependencies of service query atoms.
- No caller reads registry-wide mutation history to determine domain state or causal identity.
- Every dependent command consumes the exact prior command output.
- Every long-running wait is correlated by the identity returned at admission.
- Mounting and remounting observed state performs no product mutation.
- Query invalidation and reconnect converge on the same canonical query Result.
