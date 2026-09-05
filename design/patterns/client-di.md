---
applies_to:
  - packages/client-common/**
  - packages/effect-query/**
  - cli/**
  - web/**
  - desktop/**
---

# Client dependency injection

## Purpose

Client-common contains stateful capabilities whose lifetime is neither a React component lifetime
nor a process-global lifetime. A client service is one such capability acquired for a renderer's
ACN connection and Effect Atom registry. Effect `Context.Tag`, `Layer`, and scope are the authority
for its identity, dependencies, acquisition, and release.

This pattern governs construction and lifetime. It does not introduce another state mechanism:
backend state still uses either Effect Query or direct mirrors, client-owned state still uses
Effect Atoms, and one-shot behavior remains an Effect.

| Layer | Responsibility |
| --- | --- |
| Renderer composition root | Pair the connection and Atom registry, assemble the Layer graph, and own its scope |
| State mechanism | Provide canonical query/mutation materialization and registry-local caching |
| ACN-backed service | Terminate mechanism details behind domain state and Effects |
| Composed client service | Own client state and compose semantic lower services |
| React hook | Project a service into render values and user-event callbacks |
| UI | Render values and choose when user actions occur |

ACN, its RPC protocol, and backend ownership do not change merely because client construction
changes. A new backend operation is warranted only by a backend-owned semantic operation, never by
the client DI graph.

```text
renderer scope
  |
  +-- ACN connection
  +-- Atom registry
  +-- Effect runtime
        |
        +-- transport and cache infrastructure
        +-- ACN-backed domain services
        +-- composed client services
              |
              v
        terminal React hooks -> UI
```

## Service model

A client service is warranted when a capability has at least one of these properties:

- connection- or registry-lifetime in-memory state shared by multiple consumers;
- a resident resource whose acquisition and release must be scoped, such as a subscription;
- reusable domain operations with dependencies that must be supplied consistently; or
- a stateful client-owned use case with concurrency or cancellation.

A pure derivation, one-shot Effect composition, presentation-only value, Query definition, or
Mutation definition is not thereby a service.

Every service has one semantic interface and one `Context.Tag` with the same name. Its interface
contains only the capability callers use: passive read-only Atoms, domain Effects, and semantic
selectors. It does not expose transport clients, Query or Mutation definitions, materialized
mutation atoms, cache clients, runtimes, registries, invalidation fibers, or another service bag.

```ts
interface ModelSlots {
  readonly state: Atom<Result<ModelSlotsState, ModelSlotsError>>
  readonly assign: (slotId: SlotId, selection: SlotSelection) =>
    Effect<Assignment, AssignmentError>
}

const ModelSlots = Context.GenericTag<ModelSlots>("client/ModelSlots")
```

The example is illustrative. Domain types and errors remain specific; the Tag-and-interface shape
is the pattern.

## Construction and dependency graph

Each implementation is a Layer. Its requirements are its complete direct dependencies. A
composed service obtains lower services by yielding their Tags; it does not call constructors,
accept an aggregate client object, or look services up by object identity.

```text
ACN RPC + QueryClient + AtomRegistry
             |
       +-----+----------------+
       |                      |
       v                      v
 LocalModels Layer       ModelSlots Layer       OnboardingPersistence Layer
       |                      |                           |
       +----------------------+---------------------------+
                              |
                              v
                  OnboardingModelSetup Layer
```

```ts
const OnboardingModelSetupLive = Layer.effect(
  OnboardingModelSetup,
  Effect.gen(function* () {
    const localModels = yield* LocalModels
    const modelSlots = yield* ModelSlots
    const onboarding = yield* OnboardingPersistence

    return makeOnboardingModelSetup({ localModels, modelSlots, onboarding })
  }),
)
```

The renderer composition root assembles the complete Layer graph into the connection's existing
Effect Atom runtime. `createAgentClient(sdk)` provides that same `MagnitudeClient` to the
application operation graph and its composed client services. Client-common definitions call SDK
operations through DI; there is no RPC adapter or transport reconstruction in Effect Query. The graph is acquired once for the paired
connection and Atom registry. A new user-owned SDK connection requires a new registry scope; daemon recovery within that SDK does not. Release of that
scope interrupts all scoped resources and discards all client-owned Atom state.

There is no process-global client Layer and no second domain runtime. Infrastructure needed by
Effect Query, direct mirrors, and client services is provided by the composition root. Adding
client DI does not justify a new Query or Mutation composition primitive.

Effect Atom's `RuntimeFactory.addGlobalLayer` name is broader than its actual scope. It may be used
on a fresh, private `Atom.context(...)` factory owned by one client connection; the layer then
applies only to runtimes created by that factory. Calling it on the process-default `Atom.runtime`
for connection-specific state is prohibited.

## Resource ownership

Resident work belongs in `Layer.scoped` or `Layer.scopedDiscard`. Acquisition must establish the
resource before the service is available, and scope release must stop it. A keyed backend watch is
owned by the ACN-backed service Layer that exposes the queries it keeps fresh — as a dependency of
those query atoms, open exactly while one of them is observed — not by a mounted component or a
hidden resident Atom.

```text
connection: StreamChanges (one Subscription) -> QueryClient.invalidate by query name -> canonical Query
service:    WatchX(key) (one Subscription per observed key) -> QueryClient.invalidate -> that service's queries
```

The change subscription is owned by the connection (`state/changes.ts`); domain services own no
invalidation for poked state.

## React boundary

React is a terminal adapter, not the dependency container. The renderer provider carries the
already-defined connection runtime and paired registry. A domain hook evaluates or projects its
service Tag through that runtime, observes the service's public Atoms, and adapts its Effects to
user-event callbacks.

A hook may memoize a lightweight projection or action Atom because that is component adaptation.
It must not construct a service, acquire a resource, maintain a service singleton, or expose a raw
service lookup to UI code. Components receive values and callbacks with domain meaning.

## Prohibited ownership shapes

The following are non-conforming:

```ts
const servicesByClient = new WeakMap<object, Service>()
const serviceFor = (client: AgentClient) => servicesByClient.get(client) ?? makeService(client)

const service = useMemo(() => makeService(client), [client])

Atom.runtime.addGlobalLayer(connectionSpecificLayer) // process-wide default runtime

Object.assign(client, { domainA, domainB })
```

These shapes are service locators or ambient lifetime management. They hide dependency edges,
cannot express acquisition failure, do not deterministically release resources, and permit
different call paths to construct different graphs.

Implementation caches internal to a state mechanism are different. A cache may key canonical
Query entries, Mutation materializations, or registry state by identity when caching is its stated
responsibility. Such a cache must not be used to manufacture domain-service identity or replace
Layer scope.

The members of the group client are also different: `Client.make(AcnQueries, …)` carries every
boundary operation, materialized, at its group name (`client.Sessions.GetSession(input)`,
`client.Agent.SendMessage`). Those are the state mechanism's own surface, derived from the boundary
group the client is made for — not domain services attached to a client object. Domain services
remain Tags acquired through the client's runtime (`client.runtime.atom(Tag)`).

## Onboarding model setup

The onboarding model setup capability is the proof case for this pattern:

- local models, model slots, and onboarding persistence are separate ACN-backed service Tags;
- each service privately owns its Effect Query materialization; the connection owns the change
  subscription and invalidation;
- onboarding model setup is a Layer requiring those three Tags;
- its client-owned execution state is retained in keep-alive Atoms in the connection registry;
- each command is one Effect program passing exact outputs to dependent operations; and
- the onboarding hook observes and invokes the composed service without receiving lower Query or
  Mutation machinery.

Construction does not change product behavior. Acquiring, reading, mounting, or remounting any of
these services is observational and cannot install, assign, load, stop, or complete onboarding.

## Conformance

- Every stateful client capability has one Tag and one Layer-owned implementation per renderer
  connection and Atom registry.
- Every service dependency appears in its Layer requirements; domain code performs no ambient or
  identity-keyed service lookup.
- Connection-specific Layers are not installed globally.
- Every resident fiber or subscription is acquired and released by Effect scope.
- Hooks and components never construct or cache service instances.
- ACN-backed services terminate Query and Mutation machinery at their semantic boundary.
- Composed services depend on semantic lower services and carry exact operation outputs through
  ordinary Effect composition.
- Internal mechanism caches remain private and are not used as a client DI system.
- Replacing or releasing the renderer connection disposes the old graph and cannot retain its
  client-owned state or resources.
