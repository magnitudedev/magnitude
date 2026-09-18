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

Portable programs do not name vendors, thread/lane geometry, register budgets, or
chosen implementation tile sizes. A decomposition yields a piece whose extent the
body can inspect; the compiler chooses its size. Logical tensor dimensions and
fixed intrinsic atom dimensions are not tuning parameters.

## Semantic vocabulary

| Entity | Meaning |
| --- | --- |
| Primitive | Fundamental computation or decomposition with reference semantics |
| Function / kernel | Composition whose body defines its meaning; no independent backend lowering |
| Construct | General semantic operation with a portable definition and admitted backend implementations |
| Lowering | Backend implementation of a construct over a derived applicability domain |
| Intrinsic | Terminal backend operation with numerical, participation, hardware, and emission contracts |

Construct admission starts with existing composition. Add a construct only when a
required performant implementation or decomposition cannot be expressed cleanly.
Use the most general suitable semantics, cover supported backends, and remove
redundancy when composition becomes sufficient. A model or benchmark name is not a
semantic contract.

## Values and computation

| Facility | Meaning |
| --- | --- |
| Tensor | Externally bound storage with explicit shape and representation |
| Tile | Logical block of values; placement and participant ownership are implementation choices |
| Scalar | Typed arithmetic value with defined conversion and exceptional-value behavior |
| View | Indexing, slicing, or layout mapping; does not by itself imply a copy |
| Packed representation | Central definition of stored planes, codes, metadata, decode meaning, and precision |
| Parallel / owned iteration | Independent work and element ownership |
| Streaming load / store | Value availability, axis decomposition, and publication |
| Reduction / atomic | Explicit aggregation, ordering permissions, and update effects |
| Bounded control flow | Iteration and predicates retained in checking and resource analysis |

Finite precision is part of semantics. Accumulation type, reassociation, fused
arithmetic, fast math, and publication rounding cannot change merely to reach a
faster instruction. Representation decoding must preserve coefficient precision.

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
