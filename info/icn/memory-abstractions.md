# ICN Memory Abstractions

ICN memory planning separates two facts that native inference reports tend to mix together:

- **where an allocation was made**; and
- **how much memory that allocation requires**.

Hardware inventory is the sole authority for translating a native allocation location into a
physical memory pool. Native fitting and runtime reports supply allocation evidence only.

## Architectural relationship

```text
HardwareSnapshot
      |
      v
MemoryTopology
  - physical memory domains and capacities
  - canonical native device identities
  - device-to-domain bindings
      ^
      | resolves MemoryLocation
      |
MemoryCharge
  - MemoryChargeOwner
  - MemoryLocation
  - MemoryBreakdown
      |
      v
MemoryAccountant
  - aggregates by resolved physical domain
  - applies device-local limits
  - produces MemoryAccounting
      |
      +-- fitting policy --> HardwareAssessment
      |
      +-- runtime observation --> ModelInstanceAllocation
```

## Entities

### `HardwareSnapshot`

A point-in-time hardware observation. It records system memory, native device views, physical
memory domains, capacities, and the bindings discovered between them. Planning and load admission
must use the exact snapshot supplied by the owning ICN process; a worker must not rediscover a
different topology while interpreting an allocation.

### `NativeDeviceIdentity`

The canonical identity of a device view in the process-global native registry. It contains:

- canonical backend identity;
- optional physical-device identity; and
- native registry index.

Backend aliases are canonicalized here and nowhere else. Hardware discovery uses this identity
when it constructs stable device IDs and groups backend views that are proven to share a physical
device.

### `NativeDeviceLocator`

The device evidence supplied by one native allocation report. It is deliberately distinct from
device identity because different native APIs expose different amounts of evidence:

- an exact locator supplies backend, optional physical identity, and native index;
- an index locator supplies only the process-global native index.

A locator cannot select a memory domain. It can only be matched against an identity already owned
by `MemoryTopology`.

### `MemoryLocation`

The unresolved location of one allocation:

- host memory; or
- a native device locator.

This type is the boundary between native allocation evidence and topology-owned physical memory
semantics.

### `MemoryTopology`

An immutable, validated interpretation of one `HardwareSnapshot`. It owns:

- physical memory-domain identity;
- stable and total capacities;
- canonical native device identities;
- native-device-to-domain bindings; and
- optional device-local memory constraints.

`MemoryTopology.resolve` is the only operation that may translate a `MemoryLocation` into a
physical memory domain. Unknown or contradictory device evidence fails closed as a topology
mismatch. Code outside topology must not infer shared memory from platform, backend, device name,
or architecture.

### `MemoryBreakdown`

The complete memory requirement at one allocation location, divided into:

- model storage;
- context state;
- compute workspace; and
- auxiliary allocations.

It owns category-wise aggregation and total-byte derivation. This prevents every planner,
accountant, and resident observer from reimplementing four parallel counters or recomputing totals
with subtly different rules.

### `MemoryCharge`

One source's complete memory claim at one unresolved location. It combines:

- the owner of the allocation;
- its `MemoryLocation`; and
- its complete `MemoryBreakdown`.

A charge is not split into category-shaped fragments. Its owner and location are resolved once,
then the whole breakdown is aggregated together. Source-specific rules belong at normalization:
for example, bundled MTP keeps its working-memory breakdown but removes duplicate model storage.

### `MemoryAccountant`

The single consumer of topology and charges. For every charge it:

1. resolves the location through `MemoryTopology`;
2. adds the breakdown to the resolved physical-domain ledger;
3. adds the same breakdown to an applicable device-local constraint;
4. derives totals from the aggregate breakdowns; and
5. compares those totals with topology-owned capacities.

The accountant does not receive platform, architecture, backend, unified-memory, or
component-specific placement switches.

### `MemoryAccounting`

The factual result of accounting: aggregate breakdowns and usable capacities for every charged
physical domain and device-local limit. It contains no fitting verdict and no runtime presentation
state. Fitting policy converts it into a hardware assessment; resident observation converts its
domain allocations into model-instance state.

## Invariants

1. A physical allocation is charged exactly once.
2. Every charge is resolved through the snapshot-derived topology.
3. Native reports describe allocations; they never create memory domains.
4. Backend aliases have one canonicalization authority.
5. The native registry index is unique within a topology.
6. Unknown or inconsistent location evidence fails closed.
7. Category totals are derived from `MemoryBreakdown`, not maintained independently.
8. Planning, admission, and resident accounting use the same abstractions and resolution rules.
