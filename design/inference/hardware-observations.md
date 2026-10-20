---
applies_to:
  - inference/seismic/topology/**
  - inference/service/server/src/hardware.rs
  - inference/service/server/src/memory_domains.rs
  - inference/service/contracts/src/inventory.rs
  - packages/acn-protocol/src/schemas/inference-projection.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/utils/hardware-memory.ts
  - cli/src/commands/inference-format.ts
  - web/src/components/hardware-details.ts
---

# Descriptive hardware observations

ICN projects the Seismic topology and the engine's fit calculation. ACN projects that wire graph
into product hardware observations. Renderers do not query operating systems, initialize drivers,
measure hardware, or compute allocation capacity. Native desktop enclosure identity describes the
client machine independently and supplies no inference capacity or ranking input.

The graph distinguishes physical devices, backend executors, stores, access paths and transfer
paths. Executors retain their native selectors, which load plans also use. Physical identity is
independent of those selectors: CUDA and Vulkan views of one GPU remain one physical device. Device
and upload placements may each name several access paths. Shared host RAM is one store even when
several executors access it; physical identity must never be inferred by grouping memory domains.

Every executor carries either available stable placement capacity or an explicit unavailability
reason. Compatibility admission remains Seismic's responsibility, including numerical qualification
where API reports cannot establish it. Performance measurements are explicit operations and are not
invoked by a hardware snapshot. Unavailable executors remain in the wire graph; only usable GPU
executors enter product accelerator cards. Backend and OS-adapter discovery diagnostics retain
their source.

The snapshot names the executor the engine's device selection takes for the service's request, the
one every load runs on, from the engine's own selection rule, or no executor when none can run.
Clients read the backend that runs from it and never restate or approximate the selection rule.

Stores expose reported physical capacity, its typed basis (the interface that reported it), and
capacities and reserves from the engine's single fit owner. The snapshot holds host RAM, device
memory and volumes as distinct fields, so every snapshot has exactly one host RAM store, and host
RAM and volumes always carry the OS's free reading. Host RAM carries both what the OS reports
available to every program and what this process may still take within its limits. Access grants
remain distinct from store capacity. Host RAM has the product memory-domain name `system`; other
domains use the topology's stable store identity. The same identity joins assessment results,
resident allocations and hardware observations across processes. No native ordinal or
process-local graph identifier becomes a persistent identity.

ACN resolves every executor's device placement once, and both the product hardware and the
startup hardware summary read that resolution. Device storage is placed only in a memory domain,
so a placement naming a missing path, another executor's path, or a path to a volume is a
contract violation, never skipped. A product memory domain is host RAM (`System`, or
`UnifiedMemory` when a usable accelerator places on it) or a device's own memory
(`PhysicalDevice`); whether a domain is host RAM is its kind, not a separate flag. The memory
domains an accelerator places on are every domain any of its executors places on, a rule clients
share rather than restate.

The hardware provider samples unopened observations from its shared inventory. Only an opened
worker observes dedicated device memory's live headroom, so a resident worker's fresh reading is
merged by stable store identity. Host RAM always keeps the service's own sample. A dedicated store
without a resident observation has unknown free bytes, not zero. Observational failure is explicit.

Physical CPU topology is optional descriptive metadata, cached per ICN process. Machine-wide
physical cores and scheduler-available threads are distinct quantities. Quotas and affinity limits
are not installed core counts. Missing or incomplete physical topology stays unknown; logical
processor records cannot substitute for it. Virtual machines describe guest topology only.

The desktop shows one card per physical GPU. Only the GPU holding the selected executor names a
backend, and it names that executor's backend alone. A placement spanning dedicated memory and host
RAM shows both. Published chip facts remain presentation
metadata and cannot override observed RAM, VRAM or fit. Configurable chip facts
require evidence for a unique published variant; enclosure names, photographs and scheduling
parallelism are insufficient.

Conformance covers multiple backend views of one GPU, distinct GPUs sharing host RAM, placements
spanning stores, unavailable executors, absent free-memory observations, and physical core counts
larger than process parallelism.
