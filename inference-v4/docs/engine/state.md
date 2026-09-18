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

## Sharing and visibility

- A checkpoint retains existing claims; a fork copies ownership descriptions rather
  than the history tensor bytes.
- Appends use an exclusive tail or a new extent. They cannot extend or overwrite
  a checkpoint's visible history.
- Recurrent successors have independent handles and compatible component schemas.
- Visibility comes from explicit metadata, never from allocation capacity.
- Fragmented storage changes views and access geometry, not attention semantics.
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
