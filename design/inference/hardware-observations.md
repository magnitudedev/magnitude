---
applies_to:
  - inference/service/server/src/hardware.rs
  - inference/service/server/src/memory_domains.rs
  - inference/service/contracts/src/inventory.rs
  - packages/acn-protocol/src/schemas/inference-projection.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - desktop/src/hardware-details.ts
---

# Descriptive hardware observations

ICN owns observations of the inference host. ACN projects them into the client contract; renderers do not query operating systems, initialize GPU drivers, or benchmark hardware to populate a hardware card. Native desktop enclosure identity identifies the client machine independently and never supplies inference capacity or ranking inputs.

Physical CPU topology is optional descriptive metadata. It is cached per ICN process and does not affect scheduling, placement, capacity, ranking, or topology fingerprints. Machine-wide physical cores and scheduler-available CPU threads are distinct quantities. A scheduling quota or affinity limit must not be presented as an installed core count. Missing or incomplete physical topology remains unknown; counting logical processor records is not an acceptable substitute. Virtual machines expose guest topology and cannot establish the host's physical configuration.

Each execution device is identified by its serialized Seismic device selector (`metal:<registry-id>`, `cuda:<uuid>`, `vulkan:<uuid>`, `host-cpu`), the same identity a load plan names as its device; there is no native index or separate physical identity. Its backend is one of `cpu`, `metal`, `cuda`, or `vulkan`, and clients choose the display label. A device that discovery found but that is below its backend's floor carries its unavailability reason, so an old driver is reported instead of silently falling back to CPU; absence of a reason means the device is usable.

The snapshot is the service process's own Seismic topology and host memory status; the service
opens no device for it. Host RAM is the `system` memory domain (`unified_memory` when a GPU
allocates from it); a dedicated device's memory is a domain named by that device's selector, the
same names assessments and instance allocations use. Every domain's stable capacity is its
capacity, bounded by process limits and, on Metal, the recommended working set, less that domain's
planning reserve; `assessReserve` and `abortReserve` are the system-RAM planning and emergency
reserves. System-RAM free bytes are the limit-bounded headroom. Only a process with the device
open observes a dedicated domain's live free bytes, so while a model is loaded the snapshot merges
the resident worker's fresh reading of its device's domain; otherwise a dedicated domain reports no
live free bytes. System RAM always keeps the service's own sample. Domain names come from the
topology's pools, so a GPU visible through two backends is one domain whichever backend loaded it.
The desktop hardware card shows one GPU per dedicated domain and lists the available backends
under that GPU; backend views do not become additional physical GPU cards.

An unavailable descriptive field must not make the rest of the hardware observation unavailable. Published chip facts are separate from observations and cannot override observed RAM or VRAM. Configurable chip facts require sufficient evidence to select a unique published variant; an enclosure name, product photograph, or scheduling parallelism is insufficient.

Conformance includes SMT and multiple-socket topology, incomplete topology, missing observations through the wire projection, and a machine whose physical core count exceeds its process parallelism. GPU descriptions represent inference-visible devices and do not guarantee enumeration of every installed graphics adapter.
