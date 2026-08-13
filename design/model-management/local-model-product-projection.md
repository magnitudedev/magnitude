---
applies_to:
  - packages/acn/src/local-models.ts
  - packages/acn/src/local-model-presentation.ts
  - packages/acn/src/local-model-assessor.ts
  - packages/acn/src/local-model-configuration-resolver.ts
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

Inventory, artifact inspection, assessment, acquisition, recommendation, provider publication, and
live memory observations are independent backend mechanisms. ACN joins them into one product model.
Clients consume that model directly and do not correlate parallel candidate, download, guidance, or
provider-catalog collections.

```text
catalog and retained configurations --+
installed package inventory -----------+
artifact inspection and assessment ----+--> ACN reconciliation --> LocalModelsState
download attempts ---------------------+                           models[]
recommendation portfolio --------------+                           discoveryState
provider offerings --------------------+                           inventoryState
normalized memory observation --------+
resident replacement credit -----------+
```

`LocalModelsState.models` is the single collection of local models known to the product. One row
may be uninstalled, downloading, installed, unresolved, assessed, recommended, selectable, or
temporarily unable to load. Those are facets of the same model, not different entities.

## Canonical row

Each `LocalModel` is identified by exact servable-bundle structure and contains:

- the complete bundle;
- presentation and download size; presentation keeps the base display name, short variant label,
  user-facing precision label, and exact quantization independent;
- one explicit `catalogMembershipState`;
- one explicit `acquisitionState`;
- one explicit `servingState`.

Every property whose value is a state machine is suffixed `State`. State is represented by tagged
unions, not by combinations of nullable fields or absence.

`acquisitionState` is exactly one of `NotInstalled`, `Downloading`, `Failed`, `Cancelled`, or
`Installed`. Active and terminal model-download identity and progress live here. There is no
parallel public download collection.

For every admitted single- or multi-package bundle, `Downloading.downloadId`, `Failed.downloadId`,
and `Cancelled.downloadId` identify the same durable ICN `ModelDownload` occurrence. Package
components may complete independently, but their internal attempt membership never changes the
identity exposed to ACN or clients.

`servingState` is exactly one of `Resolving`, `Assessing`, `Failed`, or `Assessed`. An assessed row
contains the decided configuration, capabilities, assessment, `availabilityState`, and zero or
more recommendation annotations. Recommendation annotations label the row; they do not copy its
bundle, configuration, assessment, or presentation.

The assessment is `Fits`, `DoesNotFit`, or `Incompatible`. A `Fits` assessment owns its stable
memory and performance evidence. Its memory value also contains ACN's advisory system-use result
and `currentHeadroomState`, computed from one current normalized memory observation and any exact
resident replacement credit. This volatile evidence is nested on the same atomic model snapshot;
it does not create a second model identity or mirror.

`availabilityState` is `Installable`, `Preparing`, `Selectable`, or `Unavailable`. It is the
client-ready result of joining acquisition and current local-provider publication. Clients do not
look up the local model again in `ProviderModelCatalog`. Once a provider-model identity exists,
preparing and unavailable states preserve it so durable slot selection still correlates the row.

## Membership and configuration

A row exists for every release-catalog bundle and every independently servable installed bundle.
Exact bundle identity coalesces those sources into one row. Catalog removal cannot hide an
installed bundle; package removal removes a non-catalog row but leaves a catalog row as uninstalled.
Packages that are members of a catalog or retained bundle do not create additional standalone rows;
the bundle is the product identity. In particular, a separately packaged speculative draft appears
through its speculative bundle, not as an ordinary model beside that bundle. A package containing
only draft-role payload is not evidence of standalone servability.

`catalogMembershipState` is `NotInCatalog` or `InCatalog`. `InCatalog` carries the complete catalog
data required by product presentation: intelligence score and source, fidelity rank,
quantization-aware-training status, and quality notes. This membership is an explicit source fact
because it is not derivable from acquisition, assessment, availability, or recommendation state. It
does not create another model collection or identity.

Both `Magnitude` and `HuggingFaceCache` are installed origins and are equally runnable. Origin
affects ownership-sensitive operations, not eligibility.

The backend decides at most one serving configuration for a bundle, in this order:

1. retained configuration;
2. exact catalog configuration;
3. ICN-issued configuration for ACN's standard profile decision, for an installed inspected
   non-catalog standalone bundle.

Catalog, retained, and standard assessment demand may coexist internally. They never become
parallel client rows or multiple product choices for one bundle. Source provenance is not product
identity.

## Lifecycle and consistency

Reconciliation reads all contributing authorities and commits one complete snapshot. A client can
therefore render acquisition, recommendation, assessment, availability, and current memory guidance
from one row without a temporal join between mirrors.

Stable assessment and recommendation identity depend only on their semantic evidence. Live memory
availability updates only the nested advisory headroom result; it does not rerun or relabel stable
assessment or recommendation. A later load remains authoritative and revalidates current safety in
ICN. When an in-progress model-instance transition makes resident replacement credit indeterminate,
the advisory current-headroom result is `NotObserved`; unknown residency never becomes evidence of
insufficient headroom.

Inventory reconciliation reuses inspection and assessment results when semantic keys are unchanged.
Work is serialized, invalidations are coalesced, and completion rechecks its semantic key before
publication. Missing files update acquisition or membership; invalid installed artifacts remain
visible with their exact failure.

When ACN first observes an exact model download complete, it refreshes installed-package inventory
before projecting the terminal occurrence. ICN publishes inventory before package completion, so
the refreshed join becomes `Installed`. If the package is currently absent, including after later
deletion or external removal, historical completion does not prove presence and the row is
`NotInstalled`.

Discovery reports `Loading`, `Ready`, or `Failed` with progress. It describes portfolio production,
not whether `models` is authoritative or empty. A successful empty model collection is distinct
from discovery failure.

## Client behavior

Onboarding renders fitting installed rows and fitting recommended rows from `models`. The Catalog
page renders fitting `InCatalog` rows, including uninstalled rows. The Models page renders only
installed rows and preserves assessment or availability failures on those rows. Persistent download
status reads `acquisitionState` from the same rows. All surfaces retain references to the canonical
`LocalModel`; client wrappers may add interaction-only state but must not copy domain facts that
need synchronization.

Clients render a local product name as `base display name (variant label)`. Generic provider and
slot projections carry the same optional structured label. The protocol's pure composition helper
applies the same rule to agent attribution, menus, settings, status, and exit notices. Exact
quantization and precision remain detail metadata. Consumers never parse names or infer a variant
from fidelity rank.

Recommendation labels and explanations remain visible independently of memory guidance. The
selected row's detail pane presents stable system-use guidance and current load headroom on the
right-hand side. Current insufficiency never changes recommendation rank and never becomes load
authorization.

## Conformance

- Exact bundle identity produces at most one `LocalModel` row.
- Every catalog bundle and every independently servable installed bundle is represented.
- Catalog membership is explicit and an `InCatalog` row carries complete catalog data.
- Acquisition, serving, recommendation, availability, and memory guidance are facets of that row,
  not parallel client-facing entity collections.
- `Assessed` always contains one configuration, capabilities, one terminal assessment, and one
  explicit `availabilityState`.
- `Fits` contains both stable memory evidence and explicit `currentHeadroomState`.
- Recommendation annotations never duplicate model facts.
- Clients neither join local models with the provider catalog nor manufacture local-model domain
  state in client-common.
- Models contains only installed rows; Catalog contains only fitting `InCatalog` rows; onboarding
  contains only fitting installed or recommended rows.
- Live availability changes do not invalidate stable assessment or recommendation evidence.
- Indeterminate residency during an instance transition produces `NotObserved`, not a false
  sufficient or insufficient result.
- Advisory memory state never authorizes loading; ICN revalidates at the load boundary.
- Download observation and cancellation correlate the exact model-download identity from
  `acquisitionState`.
