# Kernels

**A kernel is a portable schedule: one way of arranging a numerical operation,
specialized at trace time on what it computes over, and lowered per target by
the compiler.** Its name says how it computes, never where it runs.

## Trace time and run time

```text
factory(geometry, dtypes, precision, capability, weight layout)
    │  constants folded while the program is built
    ▼
program ──► compiler fork ──► target source ──► executable(operands…)
                                                  positions, visibility, offsets, tensors
```

| Folded at trace time | Stays an operand |
|---|---|
| Extents, dtypes, tile shapes | Positions and coordinates |
| Rounding mode and per-role storage dtypes | Visibility metadata: which segments may be read |
| Capability: lanes, group width, matrix hardware | Write offsets and run capacities within a class |
| Representation parameters: code interpretation, group and supergroup geometry, coefficient scheme | Every tensor |

Folding buys straight-line code; every folded value multiplies the specializations.
Geometry that varies continuously is bounded into classes (a history capacity is a
power of two) before it reaches a factory. A value that changes per invocation is
an operand, even when a constant would be faster.

## Numerical invariants

| Invariant | Reason |
|---|---|
| Reductions, statistics, recurrent state, decay and logits are FP32 | Their error compounds; storage width is a role, accumulation width is not |
| Every rounding boundary is explicit in the schedule | Equivalence over the reals does not give finite-precision equivalence; a cast that is not written is a cast the compiler chooses |
| Bit views are reinterpretations, never conversions | Reading encoded bits as a number is a decode, and a decode is arithmetic that must be written |
| A row's arithmetic does not depend on its peers | Sharing a weight tile across rows never merges their accumulators; padding never enters a reduction |
| The same factory emits the same source on every target that lowers it | A backend-specific body is a second schedule under one name, and it will drift |

## Precision

Precision is one value with a storage dtype per role and a rounding mode.

| Role | What it stores |
|---|---|
| activation | Matrix inputs, elementwise results, attention queries |
| residual | The residual stream and readout input |
| recurrent | Recurrent projections and the mixed output |
| kv | Attention history |

Two rounding modes exist. FP32-internal keeps one FP32 intermediate across an
expression and rounds at storage. Native rounds after every operation the
reference implementation rounds after, so that a kernel reproduces the
reference bit for bit. A schedule that must round differently between modes
branches while building its program; the two branches share one structure and
differ in casts. A mode never selects a different factory. The one schedule whose
whole arithmetic exists only for native rounding declares that as a condition.

## Capability

A schedule asks three questions of the endpoint and no others: can lanes
exchange values, how wide is a group, is there matrix hardware. A schedule that
needs thirty-two lanes states it; it does not assume a backend implies it. On the
host the answer is one lane, and the same iteration space is a loop; the choice
is a branch at trace time with one shared body.

## Why several schedules exist

| Operation | Strategy | Wins when | Needs |
|---|---|---|---|
| Projection | Subgroup fold over canonical weights | One or few rows; weights stream once | Lanes that reduce |
| | Packed hierarchical-affine fold | Compact canonical codes and coefficients; input values are reused across output rows | 32 lanes |
| | Vector fold over direct affine fields | Few rows; canonical Q4 and BF16 coefficients | 32 lanes |
| | Matrix tiles with partitioned contraction | Eight rows or more; narrow outputs cannot fill the device alone | Matrix hardware |
| Attention | Streaming over history spans | Prefill; scores never materialize | Matrix hardware, 32 lanes |
| | Materialized scores | Very wide prefill; three stages over shared scratch | Scratch that fits |
| | Online decode, KV loads shared across grouped heads | Decode against long history | 32 lanes, 256-wide heads |
| | Bounded matrix partitions | Decode against short history | 32 lanes |
| Recurrence | A channel per lane, state in registers | Any endpoint with lanes | Power-of-two lanes |

A schedule is admitted by its conditions and ranked among those that apply. The
conditions are stated in shape, precision, capability and representation, and the
table that holds them belongs to the operation, not to the kernel.

Compact hierarchical schedules interpret canonical local coefficients in
registers and apply the superblock correction around the accumulated code dot
product and input sum, once per owned group fragment. All schedules for this
family use one representation-level eligibility predicate. Direct-affine
schedules read the same canonical layout function as the generic readers. No
inference schedule imports a format codec, materializes dequantized weights, or
expands coefficient planes. Matrix schedules decode only the tile reused by the
matrix contraction.

## Portability

```text
allowed:    tiles, fragments, gemm, warp reductions, fast transcendentals, bit views
forbidden:  a call naming a target function; a target dialect import; compiler
            internals; compile flags; a body that differs by backend
```

When a backend needs what the language cannot say, the compiler fork gains a
language operation, a lowering or a pipeline default, with tests for that target,
and the schedule uses it portably. The boundary test enforces the rule and carries
the one recorded exception by file name; a second exception is a decision, not an
edit.

## Adding a schedule

A schedule arrives with: the factory; its row in the operation's table, stating
conditions, rank and scratch; its case in the selection test, which asserts the
table's choice without compiling; and, where its numbers differ from an existing
schedule, its own validation against an independent reference. Emitting different
source is not evidence of anything.
