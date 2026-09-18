# Sequence state

**The engine owns logical history and acceptance; Seismic owns the storage and
completion lifetime of its physical values.**

## State components

| Component | Meaning |
| --- | --- |
| Logical position | Accepted extent of model advancement |
| Visible history | Explicit logical ranges available to subsequent computation |
| History claims | Shared ownership of backing extents, independent of physical adjacency |
| Recurrent values | Accepted component versions consumed by the next advance |
| Tentative advance | Reserved destinations and successor values for unaccepted work |
| Checkpoint | Reconciled numerical state and required input continuation |

## Advance lifecycle

```text
accepted state → reserve private destinations → submit tentative work
    → completion → accept → install successor state
                 → reject → release tentative claims
```

- A sequence has at most one unresolved advance.
- Submission does not advance logical position.
- Acceptance requires completion and valid request-level results.
- Abort preserves accepted history and waits before recycling destinations that
  submitted work can still write. Allocation retention alone does not reserve a range.
- Checkpoints cannot capture unresolved advances.
- A generation checkpoint captures reconciled numerical state together with the
  retained input layout, accepted output, pending token, grammar, publication
  cursor, recovery boundary, sampling options, and terminal outcome. Forks have
  distinct proposal identities and independent matcher/output state. A checkpoint
  during replay retains the remaining recovery obligation without resampling.

Service checkpoints are opaque, owner-scoped handles. Their logical and numerical
snapshots remain on the execution worker; host handles cannot transfer live device
or matcher objects between owners. Capture requires reconciled state and matching
logical/numerical positions. Fork admission observes the normal request limit,
creates a distinct request identity, and preserves queued output, terminal errors,
and any recovery protection. A fresh deferred numerical sequence can be captured
without allocating model state. Evicted or unresolved state cannot be captured.

Snapshot count is bounded separately from live requests, using the configured
request-count limit. Dropped or abandoned handles release through reserved lifecycle
delivery; shutdown disposes every retained snapshot. Numerical alias accounting
includes snapshots when pricing eviction. Releasing one can wake capacity waits.

## Sharing and visibility

- A checkpoint retains existing claims; a fork copies ownership descriptions rather
  than the history tensor bytes.
- Appends use an exclusive tail or a new extent. They cannot extend or overwrite
  a checkpoint's visible history.
- Recurrent successors have independent handles and compatible component schemas.
- Visibility comes from explicit metadata, never from allocation capacity.
- Fragmented storage changes views and access geometry, not attention semantics.
  Attention combines ordered visible spans under one normalization and then
  consumes the fresh causal rows. Physical gaps, empty spans, and unrelated
  requests' storage do not become visible; logical rotary positions remain
  independent of physical row destinations.
- Trimming one sequence does not change retained checkpoint history.

## Reclamation and recovery

| Responsibility | Owner |
| --- | --- |
| Choose which requests to evict | Service |
| Price resources released by closing a set of sequences | Model/state owners |
| Count aliased physical storage once and retain submitted uses | Seismic runtime |
| Preserve accepted tokens, queued output, and grammar progress | Generation |
| Reconstruct state through replay | Generation and model executor |

Reclamation requires both the end of logical claims and completion of physical uses.
A checkpoint retains numerical state for inexpensive branching; eviction releases it
and preserves the logical record needed for reconstruction. Input continuation
follows [inputs](inputs.md); generation recovery follows [generation](generation.md).
