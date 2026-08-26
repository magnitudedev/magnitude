---
applies_to:
  - packages/client-common/src/utils/format-bytes.ts
  - packages/client-common/src/local-models/failure-messages.ts
  - packages/client-common/src/state/notification-area-state.ts
  - cli/src/**
  - packages/acn/src/local-model-ranking-policy.ts
  - packages/agent/src/tools/web-fetch-tool.ts
  - packages/agent/src/window/inbox/render.ts
  - packages/release/src/**
  - packages/acn-dashboard/src/**
  - packages/inference-benchmark*/src/**
---

# User-facing byte units

Serialized and domain values remain exact byte counts. Unit conversion belongs only to presentation
and cannot change memory assessment, admission, reserve, download, or storage semantics.

Product-facing memory quantities follow hardware-industry convention: RAM, unified memory, VRAM,
model footprints, caches, and live memory use divide by powers of 1024 and carry the familiar
`MB`, `GB`, or `TB` hardware labels. They render with at most one fractional digit and omit a zero
fraction. Minimum requirements round upward at that precision so guidance never understates the
bytes required. This convention applies only to memory, not files or transfer quantities.

Product-facing storage quantities select decimal `B` through `PB`, and transfer rates use decimal
`MB/s`. Explicitly technical and diagnostic output uses binary scaling with IEC `KiB`, `MiB`,
`GiB`, or `TiB` labels. A binary-scaled file quantity or technical value cannot carry a decimal-unit
label.

Model-selection actions do not show artifact size alongside predicted memory. Download byte counts
remain visible when operationally relevant: active transfer progress and storage-capacity failures.

## Conformance

- Domain and protocol contracts carry bytes, not preformatted units.
- Product-facing memory renders hardware-conventional values through a memory-specific formatter.
- Minimum-memory guidance rounds upward at the displayed precision.
- Product-facing storage, downloads, and transfer rates use decimal SI units.
- Technical and diagnostic binary-scaled quantities use IEC labels.
- Model-menu download actions contain no artifact-size suffix.
- Transfer progress and disk-space failures may show decimal artifact sizes.
