# Component performance models

**Each component type owns a stable, parameterized performance model. Models
compose along the component graph to derive theoretical ceilings and estimate
implementation performance. Percentages are calculated from those two evaluations.**

[Component identification](components.md) defines types, implementations and assembly
relationships. The [catalog](performance/catalog.md) locates their authoritative
records. Reusable mathematics lives in the derivation library:
[resources](performance/derivations/resources.md),
[neural operations](performance/derivations/neural.md),
[state](performance/derivations/state.md) and
[service](performance/derivations/service.md).

## Ownership and component records

A type's owning document defines its contract and performance model together.
Shared model types live in [composability](models/composability.md), architecture
specific types in their architecture document, and engine types in
[engine components](engine/components.md). The catalog contains links, not a second
definition. Architecture trees select implementations of these types.

Each component record has the same fields:

| Field | Owns |
|---|---|
| Contract | Required behavior, outputs, state, numerical and lifetime guarantees |
| Parameters | Architecture/configuration and workload bindings; explicit origins and constraints |
| Composition | Child types, invocation multiplicities, shared dependencies and the demand expression |
| Dimensions | Full IDs, metric/unit, observation boundary and bound expression using reusable derivations |
| Implementations and controls | Variant identities, execution arrangement, references and independent validation |

Records inherit the platform parameters and assessment rules below. A local record
states exceptions explicitly. Formula symbols refer to the linked derivation and
bound artifact geometry; they are not new runtime objects or mandatory dispatches.

## Dimensions and parameter binding

Every dimension has the full ID `FAMILY:COMPONENT/DIMENSION`, independent of source
and variant. Codes are concise uppercase names, defined by the component record.
For example, `MODEL:ATTENTION/EXEC` is assessed on a specific
`MODEL:ATTENTION:MAG:PAGED` revision. The theoretical model belongs to the contract;
its implementation fingerprint belongs to the observation and estimate.

Use one dimension when one objective suffices. Split only for meaningful independent
outcomes or trade-offs. A single-dimension tree node displays one percentage;
multiple dimensions display their codes and percentages. Never average dimensions.
Bandwidth, arithmetic and dispatch are explanations of execution cost, not automatic
extra dimensions. Prefill/decode mode and context/batch sizes normally select
operating points within a dimension.

Parameters have explicit origins:

| Origin | Binding |
|---|---|
| Architecture/configuration | Mathematical equations, shapes, precision/encoding, sharing, state format, optional branches and observable outputs; from artifact/configuration and the contract |
| Workload | Batch/query widths, histories, initial residency, request arrivals, required checkpoints, output allowances and external readiness; from the trial specification |
| Platform | Theoretical capacity upper bounds, memory hierarchy and MLX/runtime constraints; from a versioned profile with documented provenance |
| Data-dependent conditions | Routes, acceptance and stopping outcomes; from an explicit distribution, optimistic range or labeled trace-conditioned sample |

For neural records, `m=bq` denotes consumed rows, `h` hidden width and `m_out`
requested logit rows. Weight symbols identify actual encoded tensors, including
required metadata and sharing. Each record supplies its remaining geometry. Engine
records bind a request workload and its service constraints instead.

No implementation timing sets a theoretical capacity. Data-dependent bindings may
use observed routes or acceptance only as explicit workload conditioning; they must
not silently change the standard between variants. Changed precision, semantics or
boundary residency is a changed operating point. Unknown parameters stay symbolic.

## Two evaluations of the same component graph

**Theoretical evaluation:** instantiate unavoidable demands and compose them with
[resource rules](performance/derivations/resources.md#evaluation-algebra). Permit
optimal legal MLX/Metal code, packing, fusion, reuse and overlap. At each parent,
union shared information and remove eliminable internal transfers before applying
capacity constraints. Add arithmetic/dependency refinements only under justified
assumptions. Reference rates never define the ceiling. Prefer an overly optimistic
bound over a false declaration of saturation.

**Implementation evaluation:** select the actual variants and execution arrangement.
Use matched measurements to estimate region costs, actual copies, resource contention,
submission and completion behavior. Compose those costs through the selected plan
using [execution estimation](performance/derivations/resources.md#execution-estimation).
The local component or parent measurement calibrates and checks the estimate.
Neither observed overhead nor a fitted execution cost changes the theoretical bound.

Both evaluations follow the component relationships, but their execution boundaries
may differ. A fused region crossing two components needs a joint cost observation;
standalone child timings cannot identify its production cost. Missing observations
remain unknown rather than being replaced with the theoretical best case.

What propagates upward is demand, dependencies and execution behavior. A parent
can combine child execution, memory and restoration properties without inheriting
all their display dimensions. Shared accounting follows a graph even when the view
is a tree. Parent percentages are derived at the parent, never averaged from children.

## Efficiency and bottleneck importance

For useful work `u`, a time lower bound `L`, observed time `T`, a footprint lower
bound `M_min` and observed footprint `M`:

```text
throughput ceiling = u / L
execution efficiency = 100 * L / T
footprint efficiency = 100 * M_min / M
higher-is-better outcome efficiency = 100 * observed / theoretical_upper
```

An estimated time `T_hat` gives an estimated percentage by the same expression.
The component record defines the exact metric, work unit, boundary and population
statistic. A mean lower bound cannot score a p95 observation. Separate dimension
optima need not be jointly attainable under the same constraints.

A loose theoretical bound understates efficiency; the gap is not a promise of
recoverable performance. A zero or unresolved denominator yields no meaningful
percentage. A result exceeding 100% challenges the bound, its binding or the
measurement; never clamp it. No positive floor is invented for bookkeeping that
can legally disappear into its owner.

Efficiency is distinct from bottleneck importance. Report the parent's predicted
change under an explicit change to a child's cost/behavior, holding the workload
fixed. Preserve effects on other dimensions and memory feasibility. Such sensitivity
is a model prediction requiring parent evidence, not a speedup obtained by subtracting
child percentages. A small inefficient component may have little parent impact.

## Evidence and current assessments

A ceiling evaluation records the full dimension ID, component-record and reusable-
formula revisions, all bound inputs, platform profile, assumptions and resulting
symbolic/numerical bound. It can exist without a measurement.

A measurement records the implementation fingerprint, artifact/workload/profile,
observation boundary, raw metric samples and relevant child/execution arrangement.
An assessment joins a matching measurement or supported estimate to a ceiling
evaluation. It preserves both identities, the result, coverage and uncertainty.
These are information requirements for using recorded results, not separate stores
or a new evidence registry.

To update a current estimate:

1. Match the new samples to the dimension, operating point and implementation fingerprint.
2. Evaluate that dimension's theoretical formula using the matching bindings.
3. Add the samples to the supported region of the implementation's performance estimate.
4. Reevaluate affected parent predictions using their composition rules and evidence.
5. Refresh affected tree percentages and their benchmark references together;
   remove references that no longer support the displayed value.

Interpolation is an explicit estimation method with a stated supported region and
uncertainty. Do not mix samples from different variants, numerical contracts or
platforms without a justified transfer model. Samples do not establish a universal
score over all contexts/batches. A tree view chooses an explicit workload/profile
and distinguishes measured, estimated, unmeasured and unresolved values.

**Any implementation change resets all its current dimension assessments to
`unmeasured`**, transitively through actual parent compositions using it. Preserve
stable IDs and historical evidence; unaffected components retain their assessments.
Fingerprints cover implementation content, selected children and performance-relevant
configuration/dependencies. A performance-neutral assumption does not qualify a new
revision. New evidence qualifies only its covered points.

Changing only a derivation permits reevaluation of matching raw measurements without
remeasuring. Preserve the old assessment; until reevaluated, its percentage is
historical. A new denominator must not appear as a code speedup. Implementation
changes do not automatically invalidate a contract's theoretical derivation.

## Tree annotations and benchmark references

The tree represents the implementation hierarchy. Its structure changes when the
assembly changes; its annotations change as evidence changes. Put exactly four
spaces after each implementation ID, followed by brackets. Do not align columns.

```text
IMPLEMENTATION_ID    [PERCENT @benchmark.identity]
IMPLEMENTATION_ID    [DIM: PERCENT @benchmark.identity @another.identity, DIM: unmeasured]
```

This is syntax, not a performance claim. A single-dimension component omits the
dimension label; multiple dimensions use every code defined by its type. Write
`78%` for a percentage computed from a matched measurement, `~78%` for a supported
implementation estimate, and `unmeasured` when applicable evidence is absent.
`unresolved` means the theoretical binding yields no usable denominator, including
a zero bound; it does not mean a benchmark is missing. Notes such as `fallback`
follow the closing bracket with one space.

Before displaying numbers, state the artifact, workload and platform context next
to the tree. Each occurrence binds its own shapes, state and invocation scope.
Repeated IDs do not justify copying a layer's percentage to other layers or taking
an average. A collapsed repeated node needs a defined aggregate boundary or remains
unmeasured. Parent values follow the composition rules above.

`@` references are exact existing benchmark `Experiment.identity` values, such as
`@operator.delta-owned` or `@state.append-runs`. Preserve their spelling; these are
benchmark identities, distinct from component IDs and from result/run/request IDs.
List multiple references beside a dimension when they jointly support its value.
An implementation's validation description names relevant benchmark controls and
their boundaries; a tree annotation cites only those supporting its current value.
Do not introduce evidence aliases or a provenance table.

Each reference resolves to the most recent applicable completed, valid recorded
results of that benchmark. Applicability requires matching implementation content
and selected children, benchmark definition, artifact/workload, hardware/runtime,
and the dimension's metric and observation boundary. A name or clean revision alone
is insufficient; use the recorded implementation/composition digests and environment.
Do not pool distinct contexts, rejected runs or incompatible revisions. If recorded
metadata cannot establish applicability, the value remains unmeasured.

An enclosing benchmark is not automatically evidence for a child. For example,
`operator.attention-metal-16k` times append plus attention, so it cannot directly
score attention alone. An explicit, supported decomposition may produce an estimate;
the implementation's validation description must explain the method and limits.
Reference implementations supply comparisons, never theoretical denominators.

When new applicable results are adopted, recompute the percentage and review its
references in the same update. A newer incompatible result does not supersede an
applicable one. An implementation change clears affected percentages and their
supporting annotations until matching evidence exists. Keep only currently useful
references; history remains in recorded benchmark results and version control.

## Storage and verification

Durable documents contain definitions, derivations and current tree annotations.
Existing recorded benchmark results retain samples and full provenance; do not copy
them into a second evidence store or maintain a separate current-evidence index.
Session logs remain `./sessions/YY-MM-DD/<name>.md`, relative to the monorepo root,
with supporting analysis under that date's evidence directories. Existing evidence
paths remain valid. Preserve content revisions/hashes and historical records when
documentation moves.

Static verification checks that every current implementation resolves to one type,
every dimension has a definition/binding, formula links resolve and composition
identities hold. Numerical ceilings require complete geometry/capacity inputs;
percentages additionally require measurements or explicitly supported estimates.
These documentation contracts do not create a runtime registry or a new benchmark
framework. Engine integration and component correctness remain separate qualification
requirements under [optimization](models/optimization.md).
