---
applies_to:
  - cli/src/features/model-menus/**
---

# Catalog presentation

The model-selection surface renders the entry and action semantics defined by
[Local-model product projection](./local-model-product-projection.md). It does not reconstruct local
identity from provider-catalog entries.

## Entry presentation

A Models-page local row exists only for a downloaded bundle. Every downloaded bundle remains
visible even when inspection, assessment, or provider availability prevents execution; the row
shows that reason and becomes selectable only through an exact `Fits` configuration. Catalog-only,
non-downloaded bundles never appear in Models.

The Catalog page filters eligible assessed catalog rows from `LocalModelsState.models`. An
uninstalled row that fits may be installed there. A catalog configuration that does not fit does not appear in either
product list unless its bundle is already installed, in which case Models shows the installed row
and its failure.

An installed standalone row presents physical inventory and inspection status before assessment.
When its exact configuration fits, the same row becomes selectable; incompatible and over-capacity
results remain visible with their reason. Selection styling is applied only when the row contains
the same concrete provider-qualified offering identity as the selected slot. While `AssignSlot` is
pending, selection styling projects its latest slot-scoped submitted identity over the
authoritative slot query. Rejection reveals the prior query value; successful query synchronization
replaces the projection without an intermediate stale selection.

The catalog detail view owns the complete descriptive, recommendation, calibration,
quantization-fidelity label, license, source repository, and actions for one eligible catalog
configuration. Catalog-maintainer scoring data is not presented to users. Hugging Face
repositories are presented as themed, underlined links that open in the user's browser and provide
pointer-hover feedback.

After a user requests installation, Catalog presents `Starting download…` for that exact
configuration while the `InstallModel` mutation awaits ACN admission and synchronized query
visibility. The action is unavailable for that configuration only; other configurations may be
installed concurrently. Once the canonical local-model query publishes acquisition, its
`Downloading` progress replaces command status. A typed admission rejection settles the scoped
mutation state as a failure; it never escapes as an unhandled Promise rejection.
Models and Catalog render installation rejection from that retained mutation state and selection
rejection from the canonical `AssignSlot` mutation result. They do not copy either failure into
presentation state or discard it in a fire-and-forget workflow.

Initial observation renders loading. Refresh retains the prior successful rows. Catalog, slot,
hardware, or recommendation observation failure may degrade affected metadata or actions but cannot
erase successful local-model entries.

Live allocation headroom may be sampled more frequently than model-list meaning changes. Models and
Catalog consume read-only semantic selectors over the canonical Effect Query local-model snapshot. Byte-only
headroom changes do not rebuild either list while its displayed headroom category is unchanged;
category transitions, acquisition progress, assessment, recommendation, membership, and identity
changes remain observable. Cursor, detail identity, ordering, and scroll position are presentation
state and survive these source updates. A Models menu captures the complete ordering projection on
entry, including the selected-model exception used for eligibility; assignment, recency, favorite,
and favorite updates affect rendering but cannot reorder or remove a captured candidate during the
open interaction. Independent source membership and availability changes remain observable.

## Responsive information hierarchy

The client derives presentation solely from its measured local content width. Width changes do not
create or copy catalog, recommendation, download, offering, or slot state.

The list preserves information in this order:

1. entry identity, including quantization or component role;
2. acquisition or availability status;
3. recommendation and required memory;
4. calibrated speed; and
5. intelligence and quality data.

Wide layouts may show all data as columns. As space decreases, intelligence moves to the detail
view first, followed by quality at the next narrower boundary. When a table can no longer preserve
a useful model identity, each candidate becomes a fixed two-line row. At the narrowest supported
widths speed also moves to the detail view.

Entry identity and status never disappear. Text is display-width truncated or deliberately wrapped;
layout-engine column compression must not create accidental multi-line table cells.

An installed model from Magnitude's managed store is labeled `Installed`; one discovered in a
Hugging Face cache is labeled `Installed (HF)`. Missing memory evidence is not model unavailability
and contributes no status label. Only a specific assessment, inspection, runtime, or provider
failure may present an unavailable state.

## Conformance

- Resizing chooses a pure presentation layout from the measured local width.
- Local-model product state and slot state retain their distinct authorities and client
  query/mutation paths; catalog presentation is a filter over local-model rows.
- Local-model rows preserve package, configuration, and assessment semantics at every width.
- Installed-row visibility does not depend on provider availability, assessment completion, or
  slot-query success; Models-page membership does require the bundle to be downloaded.
- Uninstalled and non-fitting catalog rows do not appear in Models.
- Only a concrete provider-qualified offering identity can render selected state.
- Refresh and remount preserve the latest successful local-model rows.
- Byte-only live-headroom polling cannot reset or redraw the complete Models or Catalog list.
- Every list layout exposes details and preserves all existing catalog actions, even when compact
  help copy omits secondary shortcuts.
- Download admission is correlated to the submitted configuration and cannot be mistaken for
  another catalog row's acquisition state.
- Installation pending and rejection presentation comes only from configuration-scoped Effect Query
  mutation states; no singleton client installation atom exists.
- Assignment pending and rejection presentation comes only from slot-scoped Effect Query mutation
  states; pending input may change selection styling but never the authoritative slot query cache.
- Selecting a model cannot reorder or remove candidates until the Models menu is closed and opened
  again.
- Keyboard cursor movement keeps the focused candidate inside the visible scrollbox viewport.
- Narrow detail views retain every fact by reflowing content vertically.
- Table rows do not wrap, overlap, or render beyond their allocated width.
