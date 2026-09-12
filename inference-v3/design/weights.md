# Weights

**Magnitude interprets artifacts and model roles; Magnitensor owns numerical
representations and physical tensor resources. Container identity and packing
never enter operation lowering or portable kernels.**

## Boundary

```text
artifact container
    │ Magnitude parses names, geometry, codec and numerical meaning
    ▼
model-role tensors with residency constraints
    │ Magnitensor imports into typed physical resources
    ▼
representation and layout visible to graph lowering
    │ selected portable TileLang kernel
    ▼
numerical use
```

| Owner | Responsibility | Must not decide |
|---|---|---|
| Format adapter | Parsing, validation, artifact identity, source geometry and codec | Execution strategy or target layout |
| Model description | Weight roles, transforms and architecture geometry | Container byte layout or kernels |
| Magnitude residency policy | Which weights may be resident, streamed or grouped under the memory policy | Backend-specific numerical implementation |
| Magnitensor | Numerical representation, selected physical layout, import computation, resources and views | Container names, model topology or eviction policy |
| Portable kernel | Decode and consume the selected representation | Source format or vendor identity |

## Numerical representation

A representation describes mathematical interpretation, not provenance. Dense,
affine and codebook representations state their dtypes, code interpretation,
groups, coefficients and lookup values. Containers with equal numerical
parameters use the same operations and candidate families.

A new wire packing does not create a representation. A representation is new
only when existing descriptions cannot express the stored values' mathematical
meaning. Source codecs disappear after import.

## Layout selection

Logical representation and physical layout are different facts. A representation
may have several lossless layouts suitable for different algorithms. Magnitensor
selects layout together with graph regions and kernel candidates, charging any
conversion and additional residency explicitly.

```text
representation: what values the bytes mean
layout:         where those bytes are for a selected implementation
codec:          how the source container stored them before import
```

One canonical compact layout is always available for supported encoded weights.
Additional execution layouts exist only when their enclosing performance benefit
justifies their memory cost. They are not selected by container or backend name,
and they never silently create an uncharged dequantized copy.

## Import

The format adapter presents bounded source tiles and a typed decoding contract.
Magnitensor performs required permutation or conversion as ordinary tensor work,
compiled through TileLang. Import publishes only complete target resources; a
failure releases staging and unpublished allocations.

Lossless relayout preserves encoded numerical meaning. Widening, dequantization
or coefficient precomputation changes representation and must be declared as
such. Import and execution may overlap only through explicit completion and
resource ownership.

## Identity, grouping and sharing

Artifact identity remains attached to every bound role so weights from different
artifacts cannot be combined accidentally. Compatible projections may share or
group physical storage when declared before import and when representation,
layout, transforms and row boundaries permit it.

Grouping is a compiler-visible tensor relationship, not an operation object. It
may enable one projection or fused region without requiring a later copy.
Equal-shaped weights share compiled code but remain distinct resources.

## Extension

A new container adds a Magnitude format adapter and source codec. A new numerical
representation adds its Magnitensor contract, reference interpretation, import
path and portable lowering support. A new fast layout or kernel candidate changes
neither container parsing nor model equations.
