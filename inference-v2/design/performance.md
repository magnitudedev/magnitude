# Component performance

Production implementation, theoretical formulation and measurement share one captured
component graph. The graph is inspected from actual loaded objects. Theory and tooling
consume its serialized facts; production does not depend on performance tooling.

## Ownership and component records

| Owner | Responsibility |
|---|---|
| [Production contracts](../src/magnitude_engine/components.py) | Typed component contracts and execution declarations |
| [schemas/](../performance/schemas/) | Typed readers of actual runtime children, dependencies, operands and use geometry |
| [assembly.py](../performance/assembly.py) | Capture those bindings, materialize tensor facts and fingerprint executable dependencies |
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

## Binding the executed composition

`@component` on an execution class or operation declares its contract, source and variant.
The decorator returns the original object: no wrapper, registry lookup or reporting call is
inserted into execution. Model roots reference the same `ModelDefinition` that constructs
the production default. `@blueprint` owns serialization and construction independently.

A typed schema reads the selected live object's existing fields. `Fields` separates local
operands from child components and shared dependencies; each `Use` points to the actual
object. Owner-supplied geometry belongs to that use, so one reusable attention operation
can serve layers with different head widths. Capture traverses these references and preserves
sharing. It does not build an architecture from names, blueprint types or benchmark cases.

Schemas live by component domain under `performance/schemas/`; they never construct runtime
components or choose defaults. Tensor inspection and serialization are shared infrastructure.
Unmodifiable upstream operations have explicit adapters. Lifetime wrappers expose existing
typed bindings through a small `bindings()` method; they do not build performance records.
Unknown execution types or mismatched parameter contracts fail capture explicitly.

Benchmarks accept an actual object or a binding from this capture. Theory consumes the typed
serialized facts. Neither has its own implementation catalog or copy of the model hierarchy.
Changing a selected child changes the next capture automatically. Changing schema-derived
facts changes evidence compatibility; editing a reporting declaration alone does not change
the numerical source fingerprint. Changing an implementation ID still changes its identity.

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

A contract pairs a stable ID with its parameter type. Execution declarations, typed binding
schemas and registered formulas share that contract; workloads are validated separately. IDs are rendered for records
and presentation, never used to guess Python classes or attributes.

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
a false claim that an implementation has reached its limit. A ceiling may be too high;
it must never be too low under its premises. Tighten it only with a mathematical proof
that the added resource demand is unavoidable for every permitted implementation.

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

Display these ratios as **≥ efficiency floors**, conditional on the recorded premises and
the accuracy of the observed metric. They do not measure the remaining achievable speedup.

Percentages are never averaged up the tree. Zero lower bounds, unbounded rates, missing
bindings and inconsistent values above 100% remain distinct. Raw costs remain visible
when no meaningful percentage exists. Saved-reference restoration and bookkeeping may
have a valid zero floor; additional timing samples cannot make that floor positive.

The current service relaxation permits ideal parameter reuse over the entire workload.
It deliberately omits unproved launch, fence and repeated-transfer costs. Publication GAP
has a zero floor when buffering is permitted. Stronger bounds require a stronger declared
contract, not an empirically chosen denominator. The dependent-phase read theorem requires
a proof that each phase must access its information after its causal barrier. Autoregression
alone does not establish that: speculative execution, precomputation and replay must remain
legal unless excluded by the actual contract.

## Stable compositions and evidence

Production `ModelDefinition` objects own both the stable architecture identity and its
default executor factory. A default selection validates against that same factory before
construction. Explicit experimental executors are candidates. There is no benchmark model
registry or duplicate default configuration.

A persistent composition is definition + scope + artifact + intentional configuration.
Its implementation topology and source belong to snapshots within that entry. Hardware and
workload select evidence, not composition identities. Only a production-default capture can
advance an established default; a newer candidate cannot promote itself. Without a default
capture, the entry is explicitly candidate or historical. The TUI shows the latest captured
default, not an assertion that unobserved local edits have been measured.


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
observations. Previous values for changed nodes remain in details as historical evidence,
never as current percentages. A formula change rebuilds history without remeasuring and does not imply
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
IMPLEMENTATION_ID    [≥PERCENT @benchmark.identity]
IMPLEMENTATION_ID    [DIM: ≥PERCENT @benchmark.identity]
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

Schema-1 conversion is explicit: `python -m performance migrate` preserves original bundles
under `archive/schema-1/` and writes an audited schema-2 conversion. Old reflective bindings
remain historical; they cannot silently establish equivalence with typed production captures.

Run `python -m performance tui`, `rebuild`, `check`, `import`, `pull` or `render` from
`inference-v2/`. [The README](../README.md#benchmarks) gives executable examples.
Session cycle logs stay under `./sessions/YY-MM-DD/` relative to the monorepo root.
