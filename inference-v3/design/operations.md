# Operations

**An operation is a contract over logical tensors. Which schedule serves a shape
is a decision made once from declared facts and recorded; scratch is declared,
never owned.** The model composes operations; it never sees a kernel.

## The contract

| Property | What must be explicit |
|---|---|
| Computation | Outputs, arithmetic, and the numerical contract a substitute must match |
| Inputs | Operand geometry and dtypes, and which representation a weight operand is in |
| State effects | What is read from and written to state, and when it becomes visible |
| Support | Shapes, precisions and capabilities under which any schedule applies |
| Lifetime | Scratch required and resources that must outlive submission |

Replacing a schedule preserves the contract. Two schedules that agree over the
reals but not in finite precision are not substitutes without qualification.

## Preparation

```text
validate operands ──► plan for this shape (cached) ──► views, indices, scratch regions
                                                          │ all owned by the preparation
                                                          ▼
                                             prepared commands, or full unwind
```

An operation prepares; it never submits. Everything it creates while preparing
is owned by the preparation and released on failure in reverse, so a failed
preparation leaves nothing behind and nothing partially submitted. Capacity
failures surface here, before any device work, with what was required.

## Selection

```text
candidate: name · applies(shape, precision, capability, representation) · rank · scratch(shape) · build
select:    applicable ──► by rank ──► scratch fits? ──► winner
realize:   build once ──► plan: executables · scratch · recorded choice
```

| Rule | Reason |
|---|---|
| A condition is stated in the four facts and nothing else | A backend or container name is a proxy, and a proxy is wrong for the next backend or container |
| Rank orders preference among the applicable | Applicability is a fact; preference is a choice; keeping them apart makes both testable |
| Scratch is declared and checked before anything is allocated | A schedule that cannot fit is skipped, not attempted and unwound |
| Selection runs once per shape and the plan is fixed | The per-step path carries no choice; invocation is binding operands to a known tuple |
| The choice is recorded on the plan | A measurement can separate a selection change from a kernel change |
| Schedule policy stays with the schedule | How attention partitions history or which dtype it stores scores in is its business; the operation sees partial counts and scratch |

The choice is a pure function of declared facts, so it is tested without a
device: for every input the engine actually sees, the table's answer is asserted.

## Fusion

A fusion is a candidate of the composite it fuses, never a special case inside a
component.

```text
GatedLinear:  fused single pass            applies: planes, native rounding, few rows
              fused hierarchical pass      applies: compatible compact hierarchy, native rounding, few rows
              two projections + gate       applies: always; declares the packed buffer
```

The composite reads which row came back and prepares accordingly. A component
never learns it is being fused past; the composite owns that knowledge.

## Scratch

A plan declares the regions it needs by name and size as a function of shape.
The bound program owns one arena: it merges declarations by name, the largest
winning, and allocates once per invocation geometry. Operations receive views of
regions inside their preparation and never allocate. Two consequences:

- Attention's scores are the same bytes in every layer, and the arena is sized
  once, not per operation and not per layer.
- Growth of any region is a change of geometry; it invalidates the captured
  sequence, and nothing else does.

## Weight-backed operations

One binding builds every weight-backed operation for any container. What a
resident weight is in decides what its operations can be; the binding does not
know which container produced it. Projections that share an input are a grouping
opportunity: whether they become one contraction is residency's decision, made
before either is asked for alone, so that a compatible row-concatenated group is
one allocation rather than a copy. Grouping eligibility follows from the resident
representation, never from the container class.
