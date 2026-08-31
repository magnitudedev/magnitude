---
applies_to:
  - cli/**
  - desktop/**
  - web/**
  - packages/client-common/**
  - packages/effect-query/**
  - packages/sdk/**
  - packages/acn/**
  - packages/acn-protocol/**
  - packages/icn/**
  - packages/agent/**
  - packages/event-core/**
---

# Query, mutation, and state architecture

## Core model

```text
                    owns
              +---------------+
              |               v
caller --mutation--> OWNER --> STATE
                         ^        |
                         |        |
observer <----query------+--------+
```

| Concept | Meaning | Owns |
| --- | --- | --- |
| State | Information retained over time | Current truth |
| Owner | Component responsible for state invariants | Valid changes and lifetime |
| Mutation | Request crossing an ownership boundary | Submission, rejection, acknowledgement |
| Query | Observation or derivation | Fetching, caching, observational failure |

```text
change crosses boundary as mutation
truth crosses boundary as query
```

State is the primitive. Ownership is a relationship, not another state object. There is no
universal controller primitive.

## Authority

```text
one fact ──> one owner
              |
              +── queries
              +── derived views
              +── disposable caches
```

- A copy cannot become independently authoritative.
- A derived value remains derived while its inputs fully determine it.
- A cache affects cost and latency, never meaning.
- Presentation may retain presentation state, never a duplicate server fact.
- External facts remain owned by the external system; Magnitude owns only its observation.

## Choosing the state model

```text
Is information retained?
  no  -> value or event
  yes -> STATE
          |
          +-- meaningful modes and legal transitions? -> state machine
          +-- committed history is truth?              -> event-sourced state
          +-- otherwise                                 -> plain state
```

| Pattern | Use when |
| --- | --- |
| Plain state | The current value is sufficient |
| State machine | Named modes and legal transitions express real behavior |
| Reducer | An event synchronously computes the next value |
| Event sourcing | Committed history is authoritative |
| Activity/worker | Scoped work produces an outcome or stream |
| Supervisor | A parent owns a child runtime |
| Reconciler | Durable intent and observed reality require a continuing feedback loop |

Patterns are selected by semantics. Not every state needs a state machine, service, workflow, or
RPC.

## Mutations

```text
request
   |
   v
validate -> reject
   |
   v
commit change -----------------------> acknowledge completion
   |
   +-> admit owned work -> publish nonterminal state -> acknowledge admission
```

Every mutation defines:

| Question | Required answer |
| --- | --- |
| Target | Which owner receives it? |
| Validation | What current state permits it? |
| Concurrency | Join, serialize, reject, or supersede? |
| Identity | Which exact occurrence does it address or create? |
| Acknowledgement | Rejected, committed, admitted, or terminal? |
| Cancellation | Caller-owned or domain-owned after admission? |

Mutation state describes the command:

```text
idle -> submitting -> rejected / synchronizing / succeeded
```

It does not describe the resource:

```text
mutation pending  !=  model Loading
mutation success  !=  operation success
transport success !=  query visibility
```

When submitted input completely determines an immediate user-facing value, presentation may
project the latest pending mutation over the authoritative query result. This optimistic value is
a derived view of command intent, not copied server state and not a query-cache write. Rejection
removes the pending projection and reveals the unchanged query value. Successful synchronization
makes affected query entries stale before publishing mutation success. When the caller immediately
continues from that state, synchronization also awaits one fresh read of the owner's
already-committed value; it never polls a domain predicate to prove the acknowledgement.

The command owner defines the acknowledgement boundary. A committed command returns only after
its durable or in-memory owner state has been updated. An admitted long-running command returns
after the external owner accepts the exact occurrence; later progress arrives through ordinary
resource observation. The client-side synchronization hook invalidates affected queries and may
await their replacement snapshot when immediate cache consistency is part of the client operation.
It does not perform a post-mutation proof read, retry a domain predicate, or create a second
completion protocol.

### Client-side services

A client-owned use case may span several backend domains without becoming an ACN operation. When
that use case has meaningful in-memory state, concurrency, and cancellation, client-common owns one
client-lifetime service. When it is a semantic command whose pending, failure, scope, or retry
lifecycle is useful beyond its immediate caller, client-common defines one Effect-backed Mutation
in the application operation graph. A one-shot composition whose result matters only to its caller
remains an ordinary Effect.

Each command is one Effect program. Every step consumes the exact value returned by the preceding
step; it never infers identity from mutation recency or an unrelated query fact. The public state
atom joins only service-owned causal identities with canonical lower queries. It is observational,
and it never copies lower lifecycle or progress into writable client state.

Client-owned in-memory state is not put into Effect Query merely to make the API resemble a remote
domain. Effect-backed definitions use the existing query cache and mutation registry; they never
create a second runtime or transport side channel. A concrete service remains justified only by a
real stateful client-owned use case.

The materialization boundary and responsibilities of lower ACN-backed services, composed services,
hooks, and UI are defined by
[Client query and mutation abstraction](../patterns/query-atom-abstraction.md). Their identity,
dependency graph, and renderer lifetime are defined by
[Client dependency injection](../patterns/client-di.md).

## Queries

Queries observe state or compose observations. They never request product change.

```text
authoritative state
        |
        v
      query ----> cache ----> derived query ----> presentation
        |
        +---- observational failure
```

A query defines:

- source and authority;
- input/key identity;
- freshness and retention;
- retry and cancellation;
- composition; and
- observational failure.

Query cache lifetime is distinct from observer lifetime. A canonical query entry owns its retained
result, fetch status, freshness/invalidation state, and in-flight coordination as one lifecycle.
The canonical entry is the query object returned to consumers: keyed lookup must retain and return
that same object, never an unretained wrapper whose collection can silently recreate the query.
Unmounting the last observer schedules collection according to the entry's retention policy; it
must not discard only part of the entry or recreate an empty observation on remount. While retained,
equivalent reads reuse the same result and concurrent fetches join the same work. Invalidation is
idempotent with respect to active replacement work: repeated notifications may advance freshness,
but they do not fan out parallel snapshot reads.

Fetch and prefetch obey freshness: they return or preserve a fresh retained result, join active
work, and execute only for missing or stale state. Refetch is the explicit operation that forces
new observational work. A mutation that changes query authority invalidates the affected query
before publishing mutation success. Mounted consumers then refetch through normal query
observation; imperative consumers fetch when they next require the value.

An imperative fetch operation awaits an internal ticket for the fetch generation it joined or
started. It does not infer completion from public presentation state such as `idle`, `paused`, or
the presence of retained data: those states deliberately collapse multiple execution histories and
cannot identify the work being acknowledged. A sanctioned invalidation replacement carries the
ticket forward until a result accepted for the latest invalidation settles. Explicit cancellation,
removal, and runtime termination terminalize the ticket. Invalidation generations and tickets are
private cache mechanics; consumers see only result, fetch status, and freshness.

A batch refetch admits all matching fetches before awaiting their tickets concurrently. This keeps
one slow query from preventing unrelated matching queries from starting and makes the batch result
the collection of the exact work admitted by that call.

```text
mount / read / refetch / watch
              X
        must not mutate
```

Advisory queries may perform effects. Their result informs a later mutation; it does not authorize
that mutation against stale reality. The owner validates again at admission.

An expensive shared derivation may be retained as a disposable materialized projection. Source
notifications are coalesced invalidations: the projection rereads the latest authoritative
snapshots, rebuilds serially, and publishes only semantic changes. Its current-state read returns
the retained value and never performs reconciliation or expensive derivation. Invalidation depends
only on facts that can change the result; high-frequency incidental changes such as transfer bytes
cannot trigger unrelated projection work. The retained value remains a cache and is fully
rebuildable from its authorities.

## Observation

Current truth and change notification have different roles:

```text
subscribe to invalidation
          |
          v
read current snapshot ----> render
          ^                   |
          |                   |
          +-- invalidated ----+
```

- Snapshot: current authoritative value.
- Watch: bounded notification that current truth may have changed.
- Revision: ordering within the authority that owns the snapshot.
- Reconnect: resubscribe, then reread current truth.

Watch events are not another state store or an event log. Missed/coalesced notifications are safe
because the snapshot is authoritative.

## Owned nonterminal state

```text
NONTERMINAL STATE
       |
       +-- must have one live owner
       +-- must have work capable of terminalizing it
       +-- must cover every Effect Exit
```

The owner handles:

```text
success | typed failure | cancellation | interruption | defect | owner shutdown
                                      |
                                      v
                              terminal state/outcome
```

Invalid:

```text
public Loading + no live load owner
public Stopping + no live removal owner
client disconnect -> abandoned server operation
```

Admission publishes the nonterminal state and establishes its owner before returning success.

## Identity and completion

```text
admit occurrence -> retain exact ID -> observe / stop / replace same occurrence
```

- Canonical model identity does not substitute for instance occurrence identity.
- “Latest” does not substitute for the admitted occurrence.
- A delayed command cannot affect a replacement.
- The state owner decides completion once.
- Consumers do not add refresh, inventory, presentation, or timer conditions to redefine it.

## Failure ownership

| Failure | Lives in |
| --- | --- |
| Command rejected | Mutation result |
| Command accepted but not yet visible | Mutation synchronization |
| Snapshot unavailable | Query result |
| Background work failed | Authoritative domain state |
| Connection lost | Transport/connection state |
| Local interaction failed | Presentation state |
| Internal invariant violated | Defect |

```text
observation failure -X-> domain failure
mutation rejection   -X-> resource failure
connection loss      -X-> empty authoritative state
```

## Lifecycle

Lifecycle is ordinary owned state-machine architecture:

```text
stop mutation
     |
     v
LIFECYCLE OWNER <---- lifecycle query ---- neighbor
     |
     +---- admission gate
     +---- scoped work
```

For a long-lived service:

```text
Starting(activity) -> Ready -> Stopping(reason) -> exact exit observed by owner
                          |
                          +-- ordinary work admitted only here
```

- Lifecycle and admission share one authority.
- Entering `Ready` opens admission atomically.
- Entering `Stopping` closes admission before publication.
- Stop is monotonic and idempotent.
- A process cannot publish its own completed death; its owner observes exact exit.
- Direct neighbors observe lifecycle; the system does not broadcast every lifecycle globally.

Ownership chain:

```text
client SDK --observes--> ACN
client holding fenced JIT replacement claim --removes--> ACN
ACN --owns/removes--> ICN
ICN --owns/removes--> native worker
```

Removal is finite:

```text
request stop
   -> bounded cooperative cleanup
   -> terminate exact process
   -> bounded wait
   -> force kill and reap
```

Timeout triggers escalation. It does not prove death, permit ownership theft, or authorize a second
active owner.

## Applications across Magnitude

| Area | Authority | Mutations | Queries | Critical rule |
| --- | --- | --- | --- | --- |
| Component UI | Component/local atom | User event | Props/derivation | No server copies |
| Shared client | client-common atom/service | Client action | Derived atom | Declare lifetime |
| Remote product state | ACN domain service | RPC command | Query/server atom | React has no server behavior |
| Sessions | Event log/runtime owner | Append/admit work | Projection | History remains authority |
| Display | Requested-shape owner; agent projection | Shape command | Display snapshot | Requested and accepted state stay distinct |
| ACN process | ACN lifecycle authority | Stop | Health/lifecycle | Admission and lifecycle are one authority |
| ICN process | ICN lifecycle authority | ACN-owned stop | ICN lifecycle/exit | ACN owns exact child removal |
| Local model intent | ACN `ModelSlotController` | Assign/load/stop request | Slot snapshot | Product intent is not physical truth |
| Physical model | ICN `ModelInstanceController` | Load/stop exact instance | Instance snapshot/watch | ICN alone authors physical lifecycle |
| Hardware | ICN hardware authority | Refresh/probe where explicit | Topology/fit/advisory plan | Advice never authorizes stale admission |
| Files/providers/VCS | External system | Adapter command | Adapter query | External facts remain externally owned |
| Persistence | Domain store/event log | Durable commit | Read/replay | Acknowledgement follows commit |
| Cache | Cache owner | Invalidate/refresh | Cached query | Deletion changes cost, not truth |
| Diagnostics | Telemetry source | Instrumentation only | Logs/traces/metrics | Diagnostics explain; never authorize |

### Frontend and client-common

```text
server query atom -> Result -> pure derived view -> React
user event         -> mutation atom -> owner
local interaction  -> presentation atom
```

- Server, shared-client, and presentation state remain separate.
- Independent domains retain independent `Result` values.
- Declarative derivation replaces synchronization effects.
- Shared atoms choose disposable or keep-alive lifetime intentionally.
- CLI, web, and desktop share state behavior through client-common.

Effect Query is the client cache and command-state authority for every first-party client
interaction with ACN and every client-common semantic command requiring a shared invocation
lifecycle.
Each interaction is defined once with the core Effect Query primitives: `Query.make`,
`Query.fromStream`, `Mutation.make`, or `Subscription.make`. Domain definitions compose through
`Group.make`, and `Group.extend` recursively adds use-case operations under existing domain groups.
`AcnBoundary` is the declared transport graph; the application graph extends it with Effect-backed
client-common definitions before constructing `AgentClient`. The ACN RPC adapter accepts only the
declared transport graph and supplies exact compile-time implementation coverage. Queries declare
freshness, while mutations declare
their scope, recovery policy, and synchronization postcondition on the same values. One
connection-scoped Effect Query client (the `AgentClient`) is made for the application graph with the
RPC implementation Layer and client-common service Layers. Client construction fails to typecheck
unless those Layers close every declared implementation and embedded Effect requirement. It carries
every member of the group, materialized,
at its name — a query member
is `(input) => QueryAtom` (`client.Sessions.GetSession(input)`), a mutation member is its
`MutationAtom` (`client.Agent.SendMessage`), a subscription member is `(input) => SubscriptionAtom`
(`client.Display.StreamDisplayView(input)`). Members are the canonical atoms: equal inputs return
the same atom, and a member is identical to what the generic materializer returns for the same
definition. Components consume those atoms; they do not pass definitions to a materializer,
construct runtimes, or wrap them in parallel request atoms or writable status state.

Mutation states are retained per invocation and keyed by the mutation definition and, when
concurrency is resource-specific, a semantic scope. A model-scoped installation therefore supports
concurrent installations of different models while serializing duplicate commands for the same
canonical `ModelId`. Pending and failure presentation is selected from those exact mutation states.
It is never represented by a singleton `installingId`, `busy`, or error side channel.

Mutation success includes invalidation of every canonical query whose visibility may have changed.
The command owner, not a client reread, defines acknowledgement: a committed command returns after
its owner commit. Model sync is model-addressed at the client boundary; ICN returns a private
`CatalogInstallationOperationId` that ACN observes but does not expose to the client.
Progress, physical completion, and serving readiness remain authoritative query state. Mutation
synchronization never polls a query until it appears to prove the command response.

There is one canonical query cache and one mutation-state registry per connection; a second cache,
request system, or command-state mechanism is prohibited. Definers know their semantic boundary,
not the transport; callers know the three primitives, not the transport.

### AgentClient and RPC

```text
component
   +-- ACN domain member ------> Effect Query client ----> ACN RPC
   +-- composed member --------> Effect Query client ----> client-common services
                                           ^
                                           +-- StreamChanges invalidates query
```

- The ACN transport service executes RPCs and owns no query or mutation state. Its implementation
  (the JIT runtime, recovering HTTP protocol, subscription framing) is the only client-side place
  that knows Effect RPC. Native ICN management and observation remain private to ACN.
- One Effect Query client per connection owns the Atom runtime, query cache, mutation history, and
  the domain service Layers.
- Declared definitions require generic operation implementations through Effect DI; Effect-backed
  definitions contribute their own Effect requirements. The connection closes both sets.
- Freshness is owned by the connection: `StreamChanges` invalidates ACN Queries. Reconnect rereads
  every covered authoritative snapshot.
- A keyed subscription that keeps queries fresh (`WatchFile`, `WatchProjectFiles`) is a dependency
  of the query atoms it serves, owned by that domain's client service: open while observed, closed
  when unobserved. Components and hooks mount no watches and own no stream fibers.
- Components do not own RPC clients, request caches, retries, or invalidation wiring.
- Mutation receipts acknowledge owner commit or exact admission; they do not create another
  resource state or await a proof reread.
- Reconnection preserves client state and rereads authoritative ACN state.

### Sessions, events, projections, and workers

```text
mutation -> append committed event -> event log
                                      |
                                      +-> projection query
                                      +-> worker trigger -> later event mutation
```

- The event log owns durable session history.
- Projections are queries over committed history.
- Projection output cannot mutate its own inputs.
- Workers own their execution; durable outcomes return through committed events.
- Session residency is runtime state, not session history.

### Display views

```text
client requested shape --subscription(sessionId, shape)--> ACN --> agent view
agent accepted shape + projections --> display events --> client
```

- Requested shape and accepted display state have different owners: the client's requested shape is
  the subscription argument; a different shape is a different subscription.
- Opening a subscription materializes the view for its shape; it cannot change another view's shape.
- Resync and retry reopen the subscription, which rereads a full accepted snapshot.
- The subscription's status is the connection phase.
- Display state does not become a second session or agent authority.

### Service and process lifecycle

```text
client SDK --query/watch--> ACN lifecycle
client holding fenced JIT replacement claim --stop/remove--> exact ACN

ACN --query/watch--> ICN lifecycle
ACN --stop/remove--> exact ICN child

ICN --observe/remove--> exact native workers
```

- Only direct neighbors observe lifecycle.
- Health is a query over lifecycle, not an independent readiness fact.
- Unexpected child exit terminalizes the state owned by its parent.
- Cleanup is bounded and cannot veto eventual process removal.
- Clients recover from ACN loss; they do not inherit daemon failure as client lifecycle.

### Local inference

```text
ACN ModelSlot                         ICN Model / Instance
durable product selection             physical availability / occurrence
         |                                      |
         +---------- client-side join ----------+

hardware query ----> fit/load advice ----> fresh validation at mutation admission
```

- Slot selection, model residency, hardware topology, downloads, and catalog state remain
  independent authorities.
- ACN stores no requested or observed physical residency in a Slot.
- Explicit warm load addresses a canonical model ID; exact stop addresses an ICN Instance ID.
- Chat Completions and Responses acquire residency automatically through that same ICN coordinator.
- Client mutation pending and response progress never substitute for authoritative residency.
- Loading repeats current hardware and package validation at ICN admission.

### Providers, files, skills, and external systems

```text
Magnitude adapter --query--> external authority
Magnitude adapter --mutation--> external authority
```

- Reads may be cached but retain external provenance.
- File watches and provider events invalidate queries; they do not become parallel state stores.
- Installation, deletion, authentication, and configuration are mutations.
- Mounting a screen or query never starts those mutations.

### Persistence, caching, and recovery

```text
mutation -> durable commit -> acknowledgement -> query visibility
                         |
                         +-> cache invalidation

restart -> read/replay authority -> rebuild projections and caches
```

- Durable acknowledgement follows the domain's commit point.
- Recovery rereads state or replays committed history.
- Recovery never reconstructs mutation intent from progress or presentation.
- Corrupt/missing disposable caches become misses or query failures, not new domain truth.

### Observability

```text
authority + mutation + query + lifecycle
                  |
                  +-> traces / logs / metrics
```

Telemetry records identity, owner, transition, operation, and failure category. It may diagnose a
broken authority but cannot become a health gate, completion condition, or recovery authority.

## Client presentation

Keep three sources separate:

| Source | May show | Must not show |
| --- | --- | --- |
| Query | Resource truth, lifecycle, progress | Command submission |
| Mutation | Pending/rejected/synchronizing | Invented resource lifecycle |
| Presentation state | Focus, draft, open panel | Duplicate server truth |

```text
resource view = f(query)
control view  = f(mutation)
local view    = f(presentation state)
```

Components compose these views; they do not merge them into another authority.

A client-owned composite workflow exposes its canonical operation state directly to presentation.
Consumers must not decompose that state and reconstruct its phase from mutation waiting and later
query snapshots. The workflow state may present that a command has been requested immediately;
authoritative resource progress and lifecycle still come only from the correlated query state.

Client synchronization is declarative:

```text
output = f(inputs)                         preferred
user event -> mutation                     when event is the cause
scoped Effect mount -> external lifecycle  only when inherently effectful
```

Server query results are never copied into writable atoms or React state.

## Transport and recovery

```text
transport carries semantics
transport does not own semantics
```

- Request cancellation follows operation ownership.
- Stream termination invalidates observation, not domain state.
- Reconnection rereads authoritative state.
- Recovery never reconstructs mutation intent from cached progress.
- Processes, connections, observers, and caches are replaceable.
- Client-local and presentation state survive server reconnection according to their own lifetime.

## Prohibited architectures

```text
query -> mutation
render -> product mutation
server state -> copied writable client state
mutation pending -> fabricated resource phase
watch events -> parallel state store
workflow -> duplicate domain lifecycle
timeout -> ownership theft
PID/socket/HTTP 200 -> inferred readiness
```

Also prohibited:

- universal controller or workflow registries;
- a state machine for state without meaningful transitions;
- one aggregate Result hiding independent domain results;
- component mount accidentally owning server work;
- redundant identities, revisions, generations, or lifecycle copies; and
- graceful cleanup capable of preventing eventual process removal.

## Conformance questions

For every domain:

```text
[ ] What facts are retained?
[ ] Who owns each fact?
[ ] Which requests are mutations?
[ ] Which reads are queries?
[ ] What is derived rather than stored?
[ ] What exact identity addresses the occurrence?
[ ] Who owns each nonterminal state?
[ ] How does every exit terminalize it?
[ ] What does acknowledgement mean?
[ ] Which failures belong to mutation, query, domain, transport, or presentation?
[ ] What scope owns state, work, cache, stream, and process?
[ ] Can replacement and recovery complete in finite time?
```

## Related domain contracts

- [Operation ownership](./operation-ownership.md)
- [JIT ACN spawning](../acn/lifecycle/jit-spawning.md)
- [ACN client lifecycle](../acn/lifecycle/client-lifecycle.md)
- [ACN service lifecycle](../acn/lifecycle/service-lifecycle.md)
- [Session runtime lifecycle](../acn/lifecycle/session-runtime.md)
- [ACN subscriptions](../acn/subscriptions.md)
- [Root work activity](../agent/work-activity.md)
- [ICN process lifecycle](../icn/lifecycle.md)
- [Model instance lifecycle](../model-management/instance-lifecycle.md)
