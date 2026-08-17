---
applies_to:
  - packages/client-common/src/utils/format-bytes.ts
  - packages/client-common/src/local-models/failure-messages.ts
  - packages/client-common/src/state/notification-area-state.ts
  - cli/src/components/hardware-memory-domain.tsx
  - cli/src/features/local-inference/**
  - cli/src/features/model-menus/**
  - cli/src/features/model-setup/**
  - packages/acn/src/local-model-recommendation-policy.ts
---

# User-facing byte units

Serialized and domain values remain exact byte counts. Unit conversion belongs only to presentation
and cannot change memory assessment, admission, reserve, download, or storage semantics.

Customer-facing memory quantities use decimal gigabytes, where `1 GB = 1,000,000,000 bytes`, and
render with one fractional digit. Minimum requirements round upward at that precision so guidance
never understates the bytes required. Customer-facing transfer quantities use decimal MB or GB,
and transfer rates use decimal MB/s.

Binary GiB and MiB remain valid for internal policy definitions and explicitly technical output,
but a value divided by a binary unit cannot carry a decimal-unit label.

Model-selection actions do not show artifact size alongside predicted memory. Download byte counts
remain visible when operationally relevant: active transfer progress and storage-capacity failures.

## Conformance

- Domain and protocol contracts carry bytes, not preformatted units.
- User-facing memory renders decimal `GB` values through the shared client formatter.
- Minimum-memory guidance rounds upward to one decimal `GB`.
- Model-menu download actions contain no artifact-size suffix.
- Transfer progress and disk-space failures may show decimal artifact sizes.
