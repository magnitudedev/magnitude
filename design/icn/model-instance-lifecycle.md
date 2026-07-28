---
applies_to:
  - inference/crates/icn-contracts/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - packages/icn/src/instances/**
  - packages/icn/src/provider/**
  - packages/acn/src/local-inference-hardware.ts
  - packages/acn/src/model-slot-controller.ts
  - packages/protocol/src/schemas/model-state.ts
  - packages/client-common/src/utils/current-local-model.ts
  - packages/client-common/src/utils/model-slots.ts
  - cli/src/features/local-inference/**
  - cli/src/features/model-menus/**
---

# Canonical model slots and instances

## Ownership

Magnitude has two parallel model domains:

```text
ACN ModelSlotController              ICN ModelInstanceController
controls ModelSlot                   controls ModelInstance
product intent                       physical truth
```

`ModelSlot` owns durable selection, renderable identity, provider availability, selected local
load readiness, an optional exact instance projection, and product actions. `ModelInstance` owns
the physical lifecycle of one admitted occurrence. Controllers are behavioral services; there is
no controller-state domain object.

Hardware observation owns topology, capacity, and current availability. It never owns model
identity or lifecycle. A client may join hardware and a slot's actual allocation for presentation,
but must not infer selection, loading, readiness, or actions from hardware.

## Identity

`SlotId` identifies a durable product role. `ModelServingConfigurationId` identifies what ICN is
asked to realize. `ModelInstanceId` identifies one admitted physical occurrence and is the only
public physical lifecycle key.

ACN creates a fresh branded instance ID at the load command boundary. ICN admits that ID with one
exact configuration. Repeating the same pair is idempotent; reusing the ID for different data is a
conflict. Stop addresses only the exact instance ID, so a delayed Stop cannot affect a replacement.
ICN retains the canonical instance entry as a resource-free terminal tombstone for the controller
lifetime; there is no parallel admission ledger. The instance snapshot exposes those same entries,
including terminal tombstones, so exact-ID observation and idempotent replay cannot lose an
accepted occurrence. Private worker epochs never leave ICN.

## Native lifecycle

`ModelInstanceController` owns every admitted instance through:

```text
Loading -> Ready -> Stopping -> Stopped
    \------------------------> Failed
```

Loading is canonical instance state, not merely a response-stream event. Loading carries stage,
semantic progress, and optional planned allocation. Ready always carries the complete actual
allocation. Stopping carries the release reason and either planned or resident allocation
evidence. Stopped and Failed are terminal and own no live resources.

The canonical entry enforces those transitions. Once Stop is requested, later Loading publications
are ignored; Ready checks the stop flag while atomically installing the leaseable resource record
and lifecycle value. Terminal entries cannot be reopened or overwritten.

`ModelInstancesSnapshot` is the complete revisioned native observation. It contains every admitted
instance, including resource-free terminal tombstones retained for exact-ID observation and
idempotent replay. Its revision advances only when an observable instance value changes. Get
answers current truth; Watch is a coalescing invalidation stream. The TypeScript `IcnInstances`
observer admits Watch before Get, retries failed refreshes, re-admits terminated watches, and
never converts read failure into an empty collection.

The load response stream may repeat progress for the initiating caller, but it is not authoritative.
Caller cancellation cannot cancel an admitted instance. Success, typed failure, defect, Stop,
worker loss, pressure release, idle release, and controller shutdown all terminalize the instance
through the controller.

## Resource and lease invariant

Every worker, backend, lease runtime, package claim, allocation, load operation, and failure is
attributed to exactly one `ModelInstanceId` entry. Singleton residency is only an index into those
entries. Hardware assessment may hold a weak observation handle, but cannot extend backend
lifetime. Cleanup operates on the owned instance. There is no public backend registry, residency
generation, parallel lifecycle record, or anonymous worker result.

`ModelInstanceLease` is an internal Rust capability. Acquisition atomically verifies exact instance
ID, exact configuration ID, Ready lifecycle, open admission, and the matching backend. Stop and
replacement close admission before draining accepted leases. A successful inference response owns
the lease until completion, failure, or cancellation.

Chat requests carry the exact instance ID selected by `ModelSlotController`. ACN installs that
binding in the Effect request scope; the local provider refuses to construct an ICN request without
it. ICN performs the atomic lease acquisition. ACN has no second physical admission gate.

## Slot aggregate

The public slot schema is deliberately small:

```text
Unassigned
ConfiguredRemote(selection, descriptor, availability, actions)
ConfiguredLocal(selection, descriptor, availability, readiness, optional instance, actions)
```

This union is the ACN-owned `ModelSlot` FSM defined with the repository `defineFSM`. Assignment,
reassignment, clearing, degradation, and recovery use its typed `transition` or `hold` operations
directly; production code does not construct replacement slot variants or route transitions through
an untyped helper.

An assigned slot always retains its selection and renderable descriptor. Transient catalog,
provider, preview, or instance-observation failure changes evidence, never assignment.

For local slots:

- readiness predicts whether a new load is currently selectable;
- the optional instance is a direct projection of one exact native instance;
- Loading, Ready, Stopping, Stopped, and Failed are never authored by ACN;
- actual allocation appears only in the native lifecycle states that own it; and
- actions are a pure, schema-validated consequence of availability, readiness, and lifecycle.

Actions are presentation affordances, never command authorization. Slot commands validate current
selection, installation, readiness, and exact instance state directly.

`ModelSlotController` reduces stored selections, provider/catalog evidence, installed packages,
selected readiness, and `ModelInstancesSnapshot`. One atomic aggregate commit publishes both the
new `ModelSlotsState` and the corresponding agent model configuration. Each projection has its own
semantic revision and filtered stream, so a change relevant to only one does not create a false
change in the other. Clients read `ModelSlotsMirror`; they do not query load preview or reconstruct
lifecycle.

That aggregate also retains the private exact serving configuration derived for each local slot.
Load commands take their immutable target from this same commit rather than resolving an offering
through a second race-prone read. A serving-configuration change can therefore update command
intent even when the public slot and agent projections are otherwise equivalent.

Local assignment validates installed packages and previews the exact configuration before its
durable commit. Success publishes the slot as loadable before returning. Catalog publication is
not command authority for an already retained exact local configuration.

## Request acquisition

For a local request, the slot controller verifies the current selection, admits or joins the exact
bound instance, observes that ID to Ready, rechecks the slot binding, and installs the exact
instance/configuration binding in the request scope. The ICN provider includes the instance ID in
the request. ICN then acquires `ModelInstanceLease` atomically.

Equivalent ACN load commands share one finite controller-wide command keyed by the exact serving
configuration, even when different slots request it. Selection checks and slot mutations use one
short controller boundary; transport and observation run outside that boundary in controller
scope. The command remains current while any slot selects its exact configuration. A superseded
command stops only its own exact instance. This is command deduplication, not a second physical
lifecycle or inference-admission gate.

`waiting_for_model` is session request progress, not model lifecycle. Model wait and prefill remain
distinct from productive generation work.

## Conformance

- ICN exposes no residency-only contract beside model instances.
- ACN owns no physical model FSM, worker, lease, or model admission gate.
- Every nonterminal instance has a live ICN owner.
- Every Ready instance has a complete actual allocation and exact lease boundary.
- Every assigned slot has a renderable identity even when dependencies fail.
- A new local assignment is committed only when it is load-admissible.
- Slot transitions use `defineFSM` directly, and slot actions never authorize commands.
- Slot and agent configuration advance from one aggregate commit and independent semantic
  revisions.
- Clients obtain current identity, lifecycle, readiness, allocation, and actions from one slot
  snapshot.
- Hardware can alter memory presentation but cannot create or erase model lifecycle.
- No legacy endpoint, union, adapter, alias, or preview hook remains beside this architecture.
