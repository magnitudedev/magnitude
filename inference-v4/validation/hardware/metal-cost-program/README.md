# Target-closed Metal cost-program proof

This standalone crate tests the prerequisite behind an exhaustive analytical
evaluator: one target-closed program must be the exclusive source for both
Metal rendering and physical-demand derivation. It does not change production.

Run it with:

```sh
cargo test --manifest-path inference-v4/validation/hardware/metal-cost-program/Cargo.toml
```

## Result

The shared cost-program abstraction is representable, but the current
`ScalarEmissionFamily` is not that abstraction and cannot construct a complete
program. This proof does not establish that the complete analytical estimator
is tractable. Native-realization bounds, cache/residency laws, path/address
envelopes, useful interval width, and accuracy remain unproved gates. The
proof's whole-vocabulary gate fails with an aggregate of precise obligations.

`ClosedCostProgram` contains only sealed operations whose bodies are one typed
`PhysicalNode` AST. The renderer fixture emits code from those nodes. The
demand fixture maps those same nodes to physical primitives. There are no
parallel render and cost fields which can silently disagree. Neither consumer
reconstructs an emission family, chooses a helper, or introduces
performance-bearing work. There is no gap, unknown, optional-cost,
fallback-service, or empty-demand representation in a successfully closed
program.

The proof constructs two concrete operations from that shape:

- native U32 addition renders an MSL add directly from its `U32Add` node;
- a fully selected finite-normal, exact-rounding path of the repository's
  actual `f32_mul` helper is expanded into executable AST nodes for decode,
  guards, significand formation, 64-bit multiply, jam shift, normalization and
  round-pack integer work. Both fixtures traverse those same nodes.

The strict fixture does not leave an opaque multiply or round-pack service. It
proves the recursive shape for one real helper path. It does not claim that the
entire 3,700-line helper library is already expanded; production must encode
every reachable helper path before it can claim closure.

## Exact production types required

Production needs these closed types, with names chosen by the owning crates:

1. `MetalCostProgram`: sealed target program attached to each closed kernel.
2. `MetalCostOperation`: exact operation discriminant plus one authoritative
   physical/control body.
3. `MetalPhysicalNode`: a sealed typed AST from which rendering and physical
   demand are both derived exhaustively.
4. `MetalPhysicalValue<T>` and typed block/value identities, replacing the
   proof fixture's readable string names so invalid operand types and undefined
   values are impossible to construct.
5. `MetalPhysicalProgram`: nodes, dependencies, finite path
   graph, lifetime intervals, memory accesses, synchronization, and phases.
6. `MetalInvocationFacts`: launch geometry, SIMD grouping, workgroup storage,
   matrix tile/staging cardinalities, and repeat bounds.
7. `MetalPathFacts`: reachable helper paths and lane-cohort relationships, or a
   sound all-path envelope when runtime values are unknown.
8. `MetalAddressFacts`: access functions, transaction/coalescing relations,
   reuse relationships, and atomic destination equivalence classes.
9. `MetalNativeRealizationBounds`: sound pre-selection bounds for instruction
   selection, register liveness/allocation, spill traffic, occupancy and native
   scheduling. These are toolchain-specific laws, not measured candidate facts.
10. `MetalProgressContract`: a finite progress rule for retrying mechanisms.
11. `CostProgramConstructionFailure`: an aggregate of typed obligations. It is
    the only alternative to a fully closed program and never reaches evaluation.

Construction must recurse through every helper routine until only characterized
physical primitives remain. Characterization and the device profile are defined
against that same closed primitive vocabulary.

## Exhaustiveness findings

The proof depends directly on production `seismic_ir::metal::ScalarEmissionFamily`.
`audit_scalar_family` has an exhaustive match with no wildcard, so adding a
variant breaks compilation until classified. `audit_mechanism` does the same for
the proof's representative schedule, global-memory, native-atomic, weak-CAS,
subgroup, and matrix vocabulary. The tests execute construction; they do not
compare independently maintained expected-name lists.

The current scalar family is lossy:

- `F32AddSub`, `F32MinMax`, and `F32Comparison` erase the exact operation;
- `F32ToInteger`, `IntegerToF32`, and `NativeControl` erase source/destination
  types or the actual control operation;
- `NativeIntegerBit` merges arithmetic, division/remainder control, bit ops,
  casts, and comparisons;
- strict arithmetic and conversion families name helpers whose physical paths
  are added later by `render.rs` and `softfloat.metal`.

Those are typed construction failures. The production cost program must be
built from exact closed operations before this information is erased. It should
not be built by enriching `ScalarEmissionFamily` with more broad categories.
`BF16ToF32` is the one current scalar family that is itself exact enough to
expand pre-native; exact native operations are also representable when built
from the closed operation rather than the lossy family.

## Where closure facts originate

| Mechanism | Required source |
|---|---|
| Exact scalar operation and helper expansion | pre-native target closure |
| Register/lifetime, spill, occupancy, native scheduling | sound native-realization bounds |
| Launch counts, SIMD/workgroup cohorts, staging and repeat bounds | invocation facts |
| Strict helper branches and lane divergence | path/cohort facts |
| Memory transactions, reuse and atomic contention | address-relation facts |
| Weak compare-exchange termination | explicit progress contract or a different algorithm |

These sources are inputs to construction. The evaluator cannot infer a missing
source using a coefficient, heuristic, default path, or generous penalty.

## Fatal blocker under current semantics

The float F32/F16/BF16 atomic renderer uses an unbounded `while (true)` around
`atomic_compare_exchange_weak_explicit`. Metal's weak operation may fail
spuriously, and the current IR contains no retry/fairness/progress contract.
Consequently no finite sound worst-case latency bound exists for every legal
execution. A measured retry count cannot prove one.

Exhaustive useful analytical bounds therefore require one semantic change:
replace the loop with a mechanism having a finite declared bound, or add a
target/runtime progress contract strong enough to prove one. If neither is
acceptable, the required finite exhaustive estimator is impossible for the
current full legal domain. This is the only demonstrated logical blocker to
finite exhaustive latency in this proof. Native realization, invocation,
path/cohort, cache/residency, and address laws are substantial unproved
engineering and research gates; representable contract shapes do not establish
that their bounds will be tight enough to optimize usefully.
