# Language and libraries

**A Seismic program declares numerical meaning and semantic decomposition.**
The compiler derives implementation choices and their consequences from that meaning.

## Libraries and scope

| Layer | Declares |
| --- | --- |
| Toolchain | Fundamental primitive semantics and backend intrinsic definitions |
| Standard library | General constructs, their lowerings, and portable kernel compositions |
| User library | Application functions and model topology; additional constructs when justified |

A compilation resolves declarations across its supplied libraries. Duplicate names,
unresolved calls, and illegal scope crossings fail explicitly. Directory layout is
organization for authors, not semantic dispatch.

| Scope | Available behavior |
| --- | --- |
| Portable | Typed computation, logical shapes, precision, effects, and semantic decomposition |
| Backend | Portable vocabulary plus that backend's intrinsics and implementation helpers |

Portable programs cannot name or query vendors, backend instructions, thread/lane
geometry, hardware memory spaces, register budgets, or chosen implementation tile
sizes. They cannot branch on device capabilities. Only backend lowering code and
its helpers can access backend-specific operations; this scope restriction is
enforced during checking. Compiler-chosen partitioning is also unobservable in
portable computation: no piece extents, piece counts, partition-local coordinates,
or inferred helper dimensions may carry it into source arithmetic or control flow.
Renaming, inferring, or hiding such values behind helpers does not satisfy this rule.

Logical shape queries remain valid. Explicit algorithmic windows and their local
coordinates are valid when their boundaries are part of the computation, independent
of execution choices. Backend lowerings may express physical streaming and intrinsic
geometry to implement that computation; these details cannot change its declared
numerical or effect contract or escape into portable semantics.

## Semantic vocabulary

| Entity | Meaning |
| --- | --- |
| Primitive | Fundamental computation or decomposition with reference semantics |
| Function / kernel | Composition whose body defines its meaning; no independent backend lowering |
| Construct | General semantic operation with a portable definition and admitted backend implementations |
| Lowering | Backend implementation of a construct over a derived applicability domain |
| Intrinsic | Terminal backend operation with numerical, participation, hardware, and emission contracts |

Language-feature and construct admission starts with existing composition. Add
vocabulary only for necessary semantics or implementation expressiveness that the
existing facilities cannot represent; a faster spelling of an equivalent supported
composition does not justify a separate feature.
Use the most general suitable semantics, cover supported backends, and remove
redundancy when composition becomes sufficient. A model or benchmark name is not a
semantic contract.

A missed optimization of an expressible composition is a compiler gap, not by
itself a reason to admit a construct. A new general decomposition is justified when
the desired implementation strategy cannot be stated through existing facilities.

## Values and computation

| Facility | Meaning |
| --- | --- |
| Tensor | Externally bound storage with explicit shape and representation |
| Tile | Logical block of values; placement and participant ownership are implementation choices |
| Scalar | Typed arithmetic value with defined conversion and exceptional-value behavior |
| View | Indexing, slicing, or layout mapping; does not by itself imply a copy |
| Packed representation | Central definition of stored planes, codes, metadata, decode meaning, and precision |
| Parallel / owned iteration | Independent work and element ownership |
| Load / store | Logical value snapshots and observable publication; physical movement is derived |
| Reduction / atomic | Explicit aggregation, ordering permissions, and update effects |
| Bounded control flow | Iteration and predicates retained in checking and resource analysis |

Logical tiles, whole-domain loads, and intermediate values do not mandate full
physical allocation or materialization. The compiler derives bounded streaming,
producer-consumer fusion, and retained state from their uses and dependencies.
Parallel/owned iteration describes logical independence and coordinates, not
physical participant identity. Matmul, reductions, and recurrences retain useful
semantic structure; minimal vocabulary does not require scalarizing them.

Finite precision is part of semantics. Accumulation type, reassociation, fused
arithmetic, fast math, and publication rounding cannot change merely to reach a
faster instruction. Representation decoding must preserve coefficient precision.

Rounding and publication are distinct: a required conversion can remain when an
internal store/load disappears. Externally observable writes, state updates, and
aliases remain effects. Exact representation decoding does not by itself authorize
factoring or reassociating a contraction.

## Lowering authoring contract

**Authors state implementation strategy; the compiler derives its realization.**
A structured lowering must be able to express the desired legal composition of
admitted intrinsics, including its ownership, movement, and completion relationships.
This expressiveness requirement covers the declared backend mechanisms and workload
domain; it does not require discovery of arbitrary equivalent algorithms.

| Author states | Compiler derives or selects |
| --- | --- |
| Algorithm, arithmetic, conversions, numerical permissions | Applicable implementation alternatives and transformations |
| Intrinsics and fixed instruction geometry | Repetition, grouping, and compatible operand realization |
| Independent work, reductions/recurrences, staging and streaming strategy | Piece extents, work partition, ownership, and supported merge realization |
| Logical intermediates and their uses | Storage placement, materialization, replication, recomputation, and lifetime reuse |
| Observable effects and required completion relationships | Movement, synchronization, buffering, and executable ordering |

Logical dimensions and fixed instruction dimensions are legitimate source values.
Machine-tuned tile sizes, lane groupings, and buffer counts are not ordinary author
parameters. When a strategy requires repeating such a choice through bounds, shapes,
and indices, supply the missing general decomposition facility. A diagnostic fixed
assignment uses the same execution machinery without changing this authoring rule.

The supported structured form includes:

- Bounded iteration with explicit independent, ordered, and reduction relationships.
- Typed views and supported normalized access maps, including bounded indirect and
  segmented accesses with their actual safety and alias conditions.
- Logical value versions and uses, derived through local mutation and helper calls;
  authors need not manually write SSA or physical allocation lifetimes.
- Stream and ownership facilities that retain decomposition relationships rather
  than require handwritten scheduling loops.
- Intrinsics with checked participation/effects and, where asynchronous, issue,
  completion, visibility, and storage-lifetime semantics.

A coupled reduction or scan requires an admitted semantic merge interpretation and
numerical permissions. Arbitrary loop-carried state is not assumed associative.
Unknown indexed locality or disjointness is not inferred from index bounds alone.

A lowering can intentionally constrain strategy through its arithmetic, ordered
dependencies, intrinsic choice, or external publications. Distinct algorithms may
require alternative lowerings. The compiler is responsible for all freedom promised
by the [execution form](execution.md#legal-execution-forms), not for inventing an
unprovided algorithm. Such constraints must be explainable from source semantics.

Source admission and optimization coverage are separate judgments. Tooling reports
the derived applicability domain, fixed commitments, remaining choice domains, and
any unsupported analysis at its source. A mathematically valid program outside a
supported analysis domain must not be labeled semantically invalid or silently
receive a complete-optimization claim. No author-maintained cost model,
“optimizable” annotation, or special source spelling supplies missing analysis.

The [compiler](compiler.md#source-stability) defines the source refactorings that
preserve optimization coverage. Missing support within that promised domain is
fixed in the compiler rather than imposed as an author workaround.

## Checking and lowering coverage

- Check declarations, shapes, types, initialization, access bounds, effects, and
  ownership within the supported decidable analysis domain.
- Derive lowering preconditions from signatures and body constraints.
- Cover each construct's domain across the configured supported backends, through
  specialized lowerings or an applicable portable implementation.
- Keep valid alternative implementations available for joint compiler selection.
- Validate intrinsic participation and capability requirements in backend scope.
- Retain runtime-dependent conditions explicitly; safety cannot rely on an
  unsupported static assertion.

The interpreter supplies reference execution. [Execution](execution.md) defines
transformation and storage invariants; [compiler](compiler.md) defines composition,
binding generation, and specialization.
