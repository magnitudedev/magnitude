---
applies_to:
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/agent/src/ambient/config-ambient.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - packages/client-common/src/local-models/**
  - packages/client-common/src/utils/model-presentation.ts
  - cli/src/features/local-inference/**
  - cli/src/features/model-*/**
---

# Local-model product projection

This document defines the ACN-owned local-model read model and its client contract. Terms follow
[Model-management terminology](./terminology.md).

## Boundary

ACN joins release catalog, filesystem-derived package inventory, artifact inspection, assessment,
process-local acquisition, provider publication, recommendations, and live memory observations into
one product model. Clients consume that model directly and do not correlate parallel collections.

`LocalModelsState.models` contains one row for every active catalog bundle and every independently
servable installed target. A row may be uninstalled, downloading, installed, assessing,
recommended, selectable, or unavailable. These are facets of one row, not persisted entities.

## Canonical row

Each `LocalModel` contains its exact bundle, presentation, download size, catalog membership,
acquisition state, and serving state. State machines use tagged unions.

`acquisitionState` is `NotInstalled`, `Downloading`, `Failed`, `Cancelled`, or `Installed`.
Downloading, failed, and cancelled states may carry the exact process-local `ModelDownloadId`.
Package-attempt membership is private to ICN. Restart clears non-installed acquisition occurrences;
the next snapshot derives installed state from current files.

`servingState` is `Resolving`, `Assessing`, `Failed`, or `Assessed`. An assessed row contains its
current exact configuration, capabilities, assessment, availability, and recommendations.
Assessment is `Fits`, `DoesNotFit`, or `Incompatible`. Advisory live headroom is nested with its
stable memory evidence but never authorizes loading.

`availabilityState` is `Installable`, `Preparing`, `Selectable`, or `Unavailable`. Once a current
configuration resolves, its deterministic local provider-model identity is preserved through
preparation and unavailability so durable slot selection can correlate it. `Selectable` requires
both installed files and a current provider offering.

## Membership and configuration

Target-package identity coalesces sources into one row. Catalog removal from active discovery does
not hide an installed target. A separately packaged speculative draft appears through its catalog
target rather than as an independently servable row unless inspection proves independent
servability.

Catalog membership is explicit presentation and recommendation evidence; it is not inferred from
installation, assessment, or availability. `MagnitudeStore` and `HuggingFaceCache` are equally
runnable origins, with ownership-sensitive operations determined by origin.

ACN resolves one current configuration for a row:

1. the exact issued catalog configuration for that target, including a deprecated entry; otherwise
2. the ICN-issued standard configuration for an installed inspected standalone package, using the
   canonical stable standard-profile rule.

Configurations are derived and never persisted in model state. Catalog and standard assessment
demand may coexist internally but never create parallel product rows.

## Lifecycle and consistency

Reconciliation reads all contributing authorities and commits one complete snapshot. Missing files
change installed acquisition immediately after inventory convergence. Restoring valid files restores
the row without another state mutation. Invalid artifacts remain visible with their inspection
failure when they form an independently identified candidate.

When ACN observes a download complete, it refreshes installed inventory before projecting terminal
completion. Historical completion never substitutes for current files. Catalog or recommendation
failure cannot erase installed non-catalog rows.

Discovery reports `Loading`, `Ready`, or `Failed` for portfolio production. A successful empty
collection is distinct from discovery failure. Restart does not reconstruct download occurrences;
it reconstructs rows from catalog and current artifacts.

## Client behavior

Onboarding renders fitting installed or recommended rows. Catalog renders fitting active catalog
rows. Models renders installed rows and preserves their assessment or availability failures.
Persistent download UI reads acquisition state from these rows and stores no second counter or
lifecycle.

Clients render presentation fields directly and never infer identity, variant, quantization, or
availability from display strings. Recommendation and memory guidance remain independent evidence.

## Conformance

- Exact target identity produces one current row.
- Every active catalog bundle and independently servable installed target is represented.
- Completed files, not acquisition history or cached state, determine `Installed`.
- Serving configurations and provider offerings are reconstructed, not persisted.
- Process restart clears acquisition occurrences without hiding installed artifacts.
- Catalog failure cannot hide independently servable installed targets.
- Clients consume acquisition, serving, recommendation, availability, and memory from one row.
- Advisory memory state never authorizes loading.
- Download observation and cancellation correlate the exact process-local download identity.
