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

```text
mount / read / refetch / watch
              X
        must not mutate
```

Advisory queries may perform effects. Their result informs a later mutation; it does not authorize
that mutation against stale reality. The owner validates again at admission.

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

- Configuration identity does not substitute for occurrence identity.
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

### AgentClient and RPC

```text
component
   +-- query definition ----> AgentClient ----> ACN query
   +-- mutation definition -> AgentClient ----> ACN mutation
                                      ^
                                      +-- watch invalidates query
```

- One AgentClient owns transport recovery, query state, and mutation state.
- Components do not own RPC clients, request caches, retries, or invalidation wiring.
- Mutation receipts may await query visibility; they do not create another resource state.
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
client requested shape --mutation--> agent
agent accepted shape + projections --> display query --> client
```

- Requested shape and accepted display state have different owners.
- Opening a stream cannot mutate display shape.
- Resync rereads a full accepted snapshot.
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
ACN ModelSlot                         ICN ModelInstance
durable product intent --command--> physical admitted occurrence
         ^                                      |
         +---------- projected query -----------+

hardware query ----> fit/load advice ----> fresh validation at mutation admission
```

- Slot selection, instance lifecycle, hardware topology, downloads, and catalog state remain
  independent authorities.
- ACN projects exact ICN instance state; it never authors `Loading`, `Ready`, or `Stopping`.
- Load/stop addresses the retained exact instance identity.
- Mutation pending and response progress never substitute for instance lifecycle.
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
- [Mirrored state](../misc/mirrored-state.md)
- [JIT ACN spawning](../acn/lifecycle/jit-spawning.md)
- [ACN client lifecycle](../acn/lifecycle/client-lifecycle.md)
- [ACN service lifecycle](../acn/lifecycle/service-lifecycle.md)
- [Session runtime lifecycle](../acn/lifecycle/session-runtime.md)
- [ACN subscriptions](../acn/subscriptions.md)
- [Root work activity](../agent/work-activity.md)
- [ICN process lifecycle](../icn/lifecycle.md)
- [Model instance lifecycle](../model-management/instance-lifecycle.md)
