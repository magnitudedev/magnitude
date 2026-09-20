# Seismic execution

This document connects the logical source language to selected physical execution. Source meaning
is defined by [Language](language.md); target realization is defined by
[Backends](backends.md).

## Reference semantics

The interpreter executes function bodies and is the semantic oracle. `for` visits its half-open
range in ascending order. `parallel for` is executed sequentially by the interpreter but means that
visits are independent: source cannot observe an order between them.

An implementation must preserve source value, ownership, mutation, and ordering effects. Numerical
operations additionally carry exact, bounded, or qualification-required contracts. A production
execution is admitted only when its composed evidence satisfies the entry policy.

## Logical loops and state

| Source form | Meaning |
| --- | --- |
| `for i in range` | Ordered ascending visits. Captured `let mut` state may be updated. |
| `parallel for i in range` | Independent visits. Captured scalar or owned local state may not be updated. |

An exclusive tensor may be written in `parallel for` only when the checker proves different visits
write disjoint places, or through an explicitly atomic operation. Lexical position fixes production:
a binding outside a loop is evaluated once; a binding inside is evaluated once per visit.

The compiler may block, vectorize, fuse, stage, pipeline, or distribute loops. These are physical
mapping choices, not additional source constructs. Ordered dependence and parallel independence are
hard constraints on every mapping.

## Ownership during execution

- An owned tensor argument transfers ownership into the call.
- A shared borrow permits reads and may overlap other shared borrows.
- An exclusive mutable borrow permits mutation and may not overlap another live borrow.
- Slices are borrows of their backing allocation.
- `to_owned` and `clone` create new logical ownership; a physical copy may be elided only when this
  is unobservable.
- Returned tensors and tuple members are owned. The ABI may use hidden destinations or legal reuse.

Invocation validation checks shapes, bounded indices and ranges, allocation extents, representation,
and exclusive-borrow non-aliasing before work is submitted.

## Calls and candidate selection

A static call occurrence selects one applicable definition from its function family. For backend
`B`, portable bodies and applicable `lower ... for B` bodies are alternatives. Backend-specific
helpers are callable only from definitions for the same backend. Capability requirements and exact
typed intrinsic uses filter unsupported candidates before selection.

Instantiation is a deterministic function of the checked program, target profile, workload, and
witness. It does not choose or repair. The selected logical body is lowered into compiler-owned
execution IR containing physical loops, allocations, transfers, launches, and synchronization.

## Physical execution units

Execution units and fusion intervals belong to compiler IR. They are derived from logical
statements and calls in authored order. A backend may offer a prescribed realization for a
contiguous interval when it proves:

1. source order and dependencies are preserved;
2. loop coordinates correspond;
3. ownership, mutation, and value lifetimes are preserved;
4. numerical effects remain admissible;
5. synchronization and participation are complete; and
6. all target resource limits hold.

Legality admits an alternative; it does not rank it. The solver chooses an exact cover of the
available intervals together with function implementations and other finite mapping decisions.

## Physical IR boundary

Physical blocks, local arrays, participant groups, launch phases, barriers, and staging buffers may
appear after logical checking. They are never authored as source types or source control flow. The
execution IR is verified before backend realization, and realized resource use is checked again
before native compilation.

## Current migration limits

- Logical `range[N]` values need a dedicated checked/execution representation; the temporary front
  end bridge retains only the existing domain form.
- `parallel for` currently bridges through the existing ordered range node after front-end
  independence checks; explicit logical-parallel IR and physical mapping are still required.
- Borrow analysis is lexical and conservative; non-lexical lifetime and disjoint mutable-slice
  proofs remain to be implemented.
- Owned-result ABI binding and hidden destinations are not yet complete across every runtime path.
- Existing compiler IR still contains physical constructs inherited from the prior source model.
  Those are migration internals, not supported source syntax.

Each unsupported boundary is diagnosed. It does not silently select a different algorithm.
