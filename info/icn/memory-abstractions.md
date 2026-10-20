# ICN Memory Abstractions

ICN memory planning separates three facts that native inference reports tend to mix together:

- **where bytes live** (a store);
- **how an executor reaches it** (an access path, under its driver's grant); and
- **how much a load requires there**.

The device topology is the sole authority for the first two. Fit decides the third against it.
Native allocation reports supply charges only; they never create stores.

## Architectural relationship

```text
DeviceTopology (seismic-topology)
  - physical devices and the executors that drive them
  - stores: host RAM, device memory, volumes
  - access paths (executor -> store, with a grant) and placements
      |
      +-- Status: each store's headroom, what is left of each grant
      |
      v
Fit (engine platform policy)
  - store fit: capacity less the planning reserve; live ceiling: headroom less it
  - a path takes the smaller of its store's room and its grant's (less the same reserve)
      |
      +-- assessment: does a model fit, and what room each store has for it
      +-- admission: DeviceHeap claims from a fresh status sample
      +-- tuning room, the hardware view, downloads
      |
      v
Seismic ledgers (runtime)
  - every allocation charged along its access paths to its stores
  - enforced store, path and storage limits
      |
      v
Store census (engine)
  - each store's charge by class: model, context, compute, auxiliary
```

## Entities

### `DeviceTopology`

An immutable revision of the machine as its APIs report it. Physical devices are identified by the
identity their drivers share, so a GPU seen through CUDA and Vulkan is one device. Each executor
(one backend's handle on a device) has a placement per storage kind (device and upload storage): an
ordered list of access paths. A placement may span stores, and several paths may reach one store.
Host RAM is one store however many executors use it. Identifiers belong to one revision; stable
selectors and store identities cross processes.

### Grant

What an executor's driver lets this process place along one access path: CUDA's usable memory, a
Vulkan heap's budget, Metal's recommended working set. A grant bounds its store as the store's
headroom does, so it keeps the store's planning reserve inside it: a grant filled to its end fails
the device's own work beside the charges.

### `Status`

One timestamped sample of each store's headroom for this process and what is left of each grant,
from the OS and drivers without an opened executor, plus what only opened executors know (their
ledgers' charges and CUDA's free memory).

### `Fit`

The engine's single interpretation of stores, grants and reserves. Stable fit uses capacities;
live fit uses the sample. Direct host demands (tables, staging) and executor storage share one
budget per store and one budget per grant. Nothing outside fit infers a placement from a backend
name, platform or device kind.

### Claims and the `DeviceHeap`

A loaded engine claims each physical operation's peak before it allocates, from a fresh sample, in
the Normal band of every store it uses. Seismic's enforced limits are refreshed from the same
sample. A claim may also take the free space of blocks the device already holds.

### Store census

The classification of a device's charges by store and class, each allocation counted once
whatever holds it. Store identities cross the worker boundary; host-resident tables, which the
runtime does not allocate, are added to host RAM.

## Invariants

1. Every byte is charged to the store it lives in, exactly once.
2. Every fit question goes through `Fit`; reserves are subtracted once per store.
3. Native reports describe charges; they never create stores or paths.
4. A placement is what the backend actually allocates from.
5. Planning, admission, tuning and resident accounting read the same topology and fit rules.
