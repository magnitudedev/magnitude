---
applies_to:
  - cli/src/features/model-menus/**
---

# Catalog presentation

The local-model catalog list supports comparison and navigation. The catalog detail view owns the
complete descriptive, recommendation, calibration, quality, license, and action evidence for one
candidate.

## Responsive information hierarchy

The client derives presentation solely from its measured local content width. Width changes do not
create or copy catalog, recommendation, download, offering, or slot state.

The list preserves information in this order:

1. model identity, including quantization;
2. acquisition or availability status;
3. recommendation and required memory;
4. calibrated speed; and
5. intelligence and quality evidence.

Wide layouts may show all evidence as columns. As space decreases, intelligence moves to the detail
view first, followed by quality at the next narrower boundary. When a table can no longer preserve
a useful model identity, each candidate becomes a fixed two-line row. At the narrowest supported
widths speed also moves to the detail view.

Model identity and status never disappear. Text is display-width truncated or deliberately wrapped;
layout-engine column compression must not create accidental multi-line table cells.

## Conformance

- Resizing chooses a pure presentation layout from the measured local width.
- Catalog and slot server state retain their existing authorities and client query/mutation paths.
- Every list layout exposes details and preserves all existing catalog actions, even when compact
  help copy omits secondary shortcuts.
- Keyboard cursor movement keeps the focused candidate inside the visible scrollbox viewport.
- Narrow detail views retain every fact by reflowing content vertically.
- Table rows do not wrap, overlap, or render beyond their allocated width.
