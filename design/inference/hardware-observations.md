---
applies_to:
  - inference/crates/icn-hardware/src/cpu_topology.rs
  - inference/crates/icn-hardware/src/lib.rs
  - inference/crates/icn-contracts/src/inventory.rs
  - packages/acn-protocol/src/schemas/inference-projection.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - desktop/src/hardware-details.ts
---

# Descriptive hardware observations

ICN owns observations of the inference host. ACN projects them into the client contract; renderers do not query operating systems, initialize GPU drivers, or benchmark hardware to populate a hardware card. Native desktop enclosure identity identifies the client machine independently and never supplies inference capacity or ranking inputs.

Physical CPU topology is optional descriptive metadata. It is cached per ICN process and does not affect scheduling, placement, capacity, ranking, or topology fingerprints. Machine-wide physical cores and scheduler-available CPU threads are distinct quantities. A scheduling quota or affinity limit must not be presented as an installed core count. Missing or incomplete physical topology remains unknown; counting logical processor records is not an acceptable substitute. Virtual machines expose guest topology and cannot establish the host's physical configuration.

An unavailable descriptive field must not make the rest of the hardware observation unavailable. Published chip facts are separate from observations and cannot override observed RAM or VRAM. Configurable chip facts require sufficient evidence to select a unique published variant; an enclosure name, product photograph, or scheduling parallelism is insufficient.

Conformance includes SMT and multiple-socket topology, incomplete topology, missing observations through the wire projection, and a machine whose physical core count exceeds its process parallelism. GPU descriptions represent inference-visible devices and do not guarantee enumeration of every installed graphics adapter.
