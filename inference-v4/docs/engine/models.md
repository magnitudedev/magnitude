# Models and weights

**A model executor connects artifact-defined architecture and logical sequence
advances to model-owned Seismic programs.**

## Artifact and numerical boundaries

| Concept | Meaning | Owner |
| --- | --- | --- |
| Artifact | Identified source snapshot, metadata, and stored tensors | Loader / format adapter |
| Model description | Architecture geometry, layer topology, weight roles, and input requirements | Model library / adapter |
| Source codec | Container byte encoding before import | Format adapter and typed import contract |
| Numerical representation | Values represented by codes, coefficients, and metadata | Seismic representation semantics |
| Physical layout | Placement and arrangement for an implementation | Seismic compiler/runtime |
| Residency policy | Which weights may be resident or streamed under a budget | Engine |

## Loading and weight ownership

- Validate geometry, role coverage, shapes, encoding, and artifact identity before
  admitting numerical execution.
- GGUF and MLX/Safetensors adapters describe source data; they do not select kernels.
- Express imports, decoding, conversion, and relayout through typed Seismic operations.
  Preserve encoded meaning and coefficient precision; account for transient storage.
- Use bounded source reads and explicit transfer lifetimes. Publish complete imported
  resources; failures release unpublished allocations and staging.
- Equal-shaped roles may share compiled code but retain distinct artifact resources.
- Resident and streamed policies describe the same numerical values. Recurring
  streaming costs belong to execution; initial import belongs to preparation.

## Numerical composition

Model programs own topology and equations. The standard library owns reusable
projection, normalization, attention, recurrence, routing, codec, and sampling
compositions. Seismic owns their physical implementations.

- Preserve accumulation types, stored activation precision, and publication order.
- Fusion can eliminate storage without removing observable rounding boundaries.
- Packed weights and KV retain their declared decode semantics through computation.
- Fresh K/V participates in the current computation before persistent encoding;
  committed history is interpreted through its selected codec.
- Codec identity includes packing, metadata precision, and any rotation/codebook
  definition. A similarly named algorithm does not establish equivalent values.

## Packed execution

```text
requests + tentative state views + conditioned inputs
    → row-local positions, visibility, routes, masks, and output selection
    → prepared Seismic execution
    → per-request outputs + shared completion
    → independent acceptance of sequence advances
```

| Contract | Behavior |
| --- | --- |
| Row isolation | Padding and peer requests never become visible history or reduction operands |
| Specialization | Stable capacity classes bound compiled geometry; actual history and visibility remain dynamic inputs |
| Readout | State-only, last/all logits, selected vocabulary, and sampled output have explicit meanings |
| Selected vocabulary | Ordered vocabulary and validation belong to the request; raw selected logits are not implicitly normalized or sampled |
| Preparation | A batch acquires tentative resources together and unwinds failed preparation without publishing state |
| Completion | Includes forward execution and any deferred control transfer or sampling |
| Acceptance | Advances commit independently after completion and request-level validation |

The executor reports reclaimable resources for sets of sequences. It does not choose
victims. Prepared native execution belongs to [Seismic runtime](../seismic/runtime.md);
input continuation belongs to [inputs](inputs.md), and history ownership to [state](state.md).
