# Component performance

Production implementation, theoretical formulation and measurement share one captured
component graph. The graph is inspected from actual loaded objects. Theory and tooling
consume its serialized facts; production does not depend on performance tooling.

## Ownership and component records

| Owner | Responsibility |
|---|---|
| [assembly.py](../performance/assembly.py) | Actual implementations, parameters, named children, shared weights/state and source bindings |
| [theory/](../performance/theory/) | Executable requirements, dimension contracts and theoretical bounds |
| [benchmarks/](../performance/benchmarks/) | Ordinary component measurements, controls and persistent Python cases |
| [runner.py](../performance/runner.py) | Completion boundaries, validation, incremental samples and automatic recording |
| [assessment.py](../performance/assessment.py) | Evidence compatibility, selection and both graph evaluations |
| [store.py](../performance/store.py) | Immutable raw runs, remote import and atomic `state.json` publication |
| [presentation.py](../performance/presentation.py) | Shared tree rendering for documents and the Textual TUI |

[Component identification](components.md) defines semantic IDs. The
[catalog](performance/catalog.md) locates behavioral contracts and explanatory derivations.
Executable formulas are authoritative for calculated values. Documents explain their
assumptions; they do not maintain another set of numerical assessments.

## Dimensions and parameter binding

Every dimension is `FAMILY:COMPONENT/DIMENSION`, including single-dimension types.
[The executable catalog](../performance/theory/catalog.py) defines allowed dimensions,
units and metric meanings once. Split dimensions only for independently meaningful
outcomes. Context, batch, query width and prefill/decode usually select operating points.

Inputs come from four places:

- Architecture: captured tensor headers, encoding, geometry, sharing and selected operations.
- Workload: prepared inputs, histories, output budget, residency and observation boundary.
- Platform: capacity upper bounds and residency constraints, with provenance in `Profile`.
- Conditioning: explicitly recorded routing, acceptance or output behavior.

No measured reference speed becomes a theoretical capacity. Unknown capacity bindings
remain explicit. `assessment.preflight(graph, workload, profile)` evaluates every node
without measuring or loading a model; resolve required modeling inputs before a campaign
whose purpose is to populate efficiency percentages.

## Two evaluations of the same component graph

Theoretical evaluation unions unavoidable input information, retains shared resource
identities, removes internal transfers and permits ideal legal MLX/Metal fusion and reuse.
Apply capacity bounds after composition. Conventional arithmetic is an explicit optional
assumption, not a universal lower bound on all algorithms. Fixed encoded representation
is part of the mathematical contract. Prefer an optimistic performance upper bound over
a false claim that an implementation has reached its limit.

Implementation evaluation selects matching observations. A measured parent owns its
actual metric; child times are not added to it. An explicitly serial execution region may
sum compatible child costs; declared independent parallel work may use their maximum.
Joint or fused regions require a joint observation. Child invocations and observation
bindings must be explicit. The system does not infer production overhead from isolated
kernel timings or substitute theoretical-best times for missing observations.

Percentages are calculated at each node:

```text
EXEC or latency efficiency = 100 × theoretical minimum time / observed time
MEM efficiency = 100 × required retained bytes / observed physical backing
higher-is-better efficiency = 100 × observed outcome / theoretical upper bound
```

Percentages are never averaged up the tree. Zero lower bounds, unbounded rates, missing
bindings and inconsistent values above 100% remain distinct. Raw costs remain visible
when no meaningful percentage exists. Saved-reference restoration and bookkeeping may
have a valid zero floor; additional timing samples cannot make that floor positive.

The current service relaxation permits ideal parameter reuse over the entire workload.
It deliberately omits unproved launch, fence and repeated-transfer costs. Publication GAP
has a zero floor when buffering is permitted. Stronger bounds require a stronger declared
contract, not an empirically chosen denominator.

## Evidence and current assessments

Each recording starts before benchmark-owned preparation and automatically saves warmups,
raw samples, failures, source snapshots, the captured graph, workload, originating hardware
and formula evaluation. Reset/validation are outside timing; required completion is inside.
Finalized bundles are immutable. Interrupted journals remain recoverable on their original
host after the process exits.

The store selects the latest completed valid observation **per dimension**, with deterministic
completion-time/run-ID ordering. Matching requires component source/weight/configuration,
hardware/runtime, full operating point, boundary and contract version. A memory sample
cannot erase a still-applicable timing sample. Raw history remains available.

Unchanged components share assessments across compositions. Changed implementations
invalidate themselves and dependent ancestors; unchanged children retain applicable
observations. A formula change rebuilds history without remeasuring and does not imply
an implementation speedup. Different hardware remains separate within the same composition;
no universal utilization factor or undocumented interpolation transfers observations.

A run can provide explicit `bindings` for descendant observations: each entry gives the
complete child workload, boundary and contract version. This connects compatible isolated
measurements to a parent view without treating the parent's workload or timer as child
measurements. Absent bindings, matching is exact. Conditional work must be bound explicitly.

## Tree annotations and benchmark references

Both the TUI and document exporter read the same published assessments. The TUI has one
composition selector, the actual component tree on the left and selected-component details
on the right. It opens the current implementation with the latest matching evidence per
dimension. The store publishes these selections as references to existing assessments;
the TUI does not calculate its own scores. Each selected value retains its hardware and
workload in the details pane. Values at different nodes may describe different operating
points; they are never pooled or used to recompute a parent. Document exports select an
explicit operating point. Shared dependencies appear as references. Selecting a node
exposes raw run paths, theoretical terms, assumptions and missing prerequisites.

```text
IMPLEMENTATION_ID    [PERCENT @benchmark.identity]
IMPLEMENTATION_ID    [DIM: PERCENT @benchmark.identity]
```

Use exactly four spaces before annotations. One dimension omits its label; multiple
dimensions use their uppercase codes. `~` marks an explicit execution estimate. Unavailable
or inconsistent percentages receive no documentation annotation or citation. The TUI shows
their raw costs and reasons. Assembly sections contain only generated trees and numbers.

`@benchmark.identity` is the semantic name recorded by the measurement function, independent
of its filename, implementation variant and run ID. It resolves through the selected
assessment to current applicable raw records. Export provenance stays in the generated
store's export artifact; do not add a document-side table or setup paragraph.

## Storage and verification

`runs/performance/` contains immutable run bundles and one rebuildable `state.json`.
Import validates hashes, deduplicates identical runs and rejects conflicting content.
Remote pull transfers data only. Publication uses a process lock and atomic replacement;
the TUI never reads a partial generation. Session-bench automatically contributes completed
HTTP observations at their HTTP boundary, preserving its existing native raw records.

Run `python -m performance tui`, `rebuild`, `check`, `import`, `pull` or `render` from
`inference-v2/`. [The README](../README.md#benchmarks) gives executable examples.
Session cycle logs stay under `./sessions/YY-MM-DD/` relative to the monorepo root.
