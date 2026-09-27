---
applies_to:
  - inference/engine/batching/**
  - inference/engine/state/**
  - inference/engine/executor/**
---

# Validated numerical launches

Batching owns device-independent row semantics. It validates slot order, row mappings,
coordinates, visibility, destinations, demands, selection controls, and physical capacity class
once, then produces an opaque domain-specific batch. A row's visible history spans keep the
history's logical order and must not share a row; their addresses need not ascend. The state
store coalesces logically adjacent spans within one slab, and batching preserves its boundaries
so no span crosses a slab. The span class limit is the largest span bound of the loaded store's
history domains, each `ceil(row limit / rows per slab) + 16` (see
[state transactions](state-transactions.md)).
Its row tables are the only upload source;
programs encode each graph's inputs from them directly, with no separate packed control image.
Selection masks are shared with their producer rather than copied into the batch, and an
unconstrained row carries no mask.

Execution joins a validated batch with owned state advances, conditioning, workspace, and output
leases into one opaque launch. Its constructor checks only cross-domain facts: resource identity,
slot-to-advance cardinality and row alignment, conditioning references, and planned lease class.
Programs accept the launch as one value and retain it in their submission until completion.

Head feature projection is a distinct checked launch because it consumes retained features and
selection controls without advancing sequence state. Its requests, feature extents, domain,
selection masks, and aggregate row capacity are validated together. It uses the head workspace
and output leases and follows the same submit, complete, and reconcile lifecycle as head forward.

Batching does not depend on execution or contain device values. Execution does not reinterpret
independent arrays to reconstruct row alignment inside a native program.

## Acceptance criteria

- Invalid row or slot relationships cannot be represented by a validated batch.
- No visible span crosses a slab boundary, and a batch cannot exceed the loaded model's span bound.
- A launch cannot pair a row with another request's state or resource lease.
- All program inputs remain owned until physical completion and reconciliation.
- No program accepts parallel unchecked row, state, conditioning, or demand arrays.
