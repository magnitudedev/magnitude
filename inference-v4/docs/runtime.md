# Seismic compilation and execution runtime

The runtime owns devices, artifacts, memory bindings, submission, and completion.
It applies the results of the [compiler](compiler.md); it does not acquire an
independent tuning policy or reinterpret selected execution decisions.

## Compilation request

A request binds a semantic program, entry point, target backend, workload domain,
execution form, hardware/model contract, objective, and compiler configuration.
Shapes, representations, layouts, alias conditions, and relevant runtime predicates
are explicit inputs or retained invocation conditions.

Device discovery supplies facts with the meaning guaranteed by its API. It does
not promote architecture names, thread limits, or storage capacities into a complete
throughput profile. Hardware/model admission follows [Accounting](accounting.md).

The runtime composition root supplies concrete backend implementations to the
compiler through shared interfaces. Candidate construction, model derivation, and
selection have no dependency on native compilation or device execution.

## Artifact lifecycle

```mermaid
flowchart LR
    R[Compilation request] --> T[Verified Tuned IR]
    T --> C[Target code]
    C --> N[Native artifact]
    N --> Q[Applicable qualification]
    Q --> B[Bound invocation]
    B --> X[Submitted execution]
    X --> D[Completion and status]
```

The diagram describes the qualified execution path. Diagnostic compilation may
produce explicitly unqualified artifacts, but cannot promote them into the qualified
path by relabeling them.

| Artifact | Retained information |
| --- | --- |
| Tuned artifact | Actual selected execution, model/account, checked objective and optimality evidence, conditions, and semantic identities. |
| Native artifact | Generated code identity, native code, ABI, target/toolchain settings, and correspondence to Tuned IR. |
| Qualification | Mapping/model claims supported for that target, compiler configuration, and condition domain. |
| Invocation | Concrete bindings and checked evidence that relevant artifact conditions hold. |

Modeled optimality, physical lower bounds, qualification, and current applicability
remain separately represented. A native executable can exist without possessing all
of those claims. Production policy must not silently equate existence with qualification.

Native compilation consumes the selected execution once. Failure does not trigger
trial compilation of other candidates. A changed compilation request is an explicit
new request, not an invisible fallback.

## Invocation applicability

Before relying on an artifact, establish its required conditions:

- Shape, dtype, representation, layout, strides, offsets, and alignment.
- Buffer sizes, canonical allocation relationships, and required disjointness.
- Scalar domains, predicates, and relevant content/version conditions.
- Target capabilities, compiler/mapping assumptions, and operating domain.
- Any enforced residency or initial-state conditions and the objective boundary.

Checks use actual allocation identity, not parameter names or exposed pointer
coincidence. Aliased views remain views of the same allocation. Reused addresses do
not establish content identity. Mutation invalidates assumptions tied to prior
content versions.

Static proofs eliminate checks only when their assumptions hold. Invocation checks
establish binding properties. Necessary data-dependent checks remain in the selected
execution with defined failure behavior and accounted costs. The runtime does not
insert a second per-access policy behind the compiler's model.

Residency claims require an enforced state or conservative envelope. If residency
is only an unverified condition, a corresponding bound remains conditional rather
than certified applicable to the current invocation.

## Memory and execution ownership

Buffers and views retain their allocations for every operation that can access them.
Invocation scratch follows the selected allocation and lifetime plan. Aliasing,
retained views, asynchronous work, and cross-launch handoffs prevent premature reuse
or deallocation.

Submission preserves the selected launch and dependency graph. Fusion, batching,
command grouping, or overlap introduced by the runtime must either be represented
in that graph/model or be proven irrelevant to the declared objective and semantics.
Runtime policy cannot silently serialize or overlap work behind the compiler's account.

Completion establishes both physical completion and the required status/result
visibility. A submission error does not prove that already submitted work stopped.
A dynamic kernel failure may follow earlier observable writes; failure results must
not imply transaction rollback. Resources remain retained until safe release is
established.

## Cache identity and invalidation

Compilation, proof, qualification, and executable caches have distinct entries but
share explicit dependency identities. Relevant keys include:

- Semantic program, entry, legal execution form, and implementation definitions.
- Workload domain and retained conditions.
- Hardware/model and native mapping contracts.
- Objective, timing boundary, units, and proof-rule versions.
- Compiler configuration, backend features, toolchain, and artifact ABI.

Display names and timestamps are not semantic identities. Hashes establish identity,
not correctness. Proof deserialization creates unverified proposals; only checking
restores verified authority.

A hardware, mapping, or rule update creates a new dependency version; it does not
mutate the meaning of an old certificate. Reuse under different conditions requires
a checked implication. Reusing native code and reusing a proof are separate decisions;
compatibility of one does not establish the other.

## Failures and result classification

| Outcome | Required interpretation |
| --- | --- |
| Invalid source/request | A semantic or request violation; no execution guarantee. |
| Proven infeasible form | Checked absence of a legal feasible realization under the stated conditions. |
| Incomplete tuning/checking | Retained progress and unresolved coverage; no exact-optimality claim. |
| Compiler or native compilation failure | Implementation/toolchain failure, not evidence that all alternatives are illegal. |
| Inapplicable binding | Invocation conditions do not hold; do not use the conditional claim. |
| Qualification mismatch | Withhold affected claims and preserve diagnostic evidence. |
| Execution failure | Report completion/status and possible partial effects accurately. |

An incomplete or failed path does not authorize heuristic selection, candidate
benchmarks, altered numerical semantics, or hidden interpreter fallback.

## Engine integration and performance

[V3](../../inference-v3/) is the behavioral and numerical reference for the engine
port: model/GGUF loading, scheduling and batching, multimodality, cache policies,
quantization behavior, templating, thinking, tool calling, and constrained generation
must be preserved unless a divergence is explicitly adopted. The new compiler and
runtime provide execution; they do not redesign those policies.

Qualification includes Qwen3.5-35B-A3B and Qwen3.5-4B on Metal and CUDA and the required
CPU execution paths. Preserve the optimized V3 operations, including quantized KV
cache and TurboQuant behavior where used. Compare correctness and enclosing prefill,
decode, latency, memory, compilation, and tuning behavior under compatible workloads.

The V3 session benchmark may drive the Rust engine through an adapter that preserves
its workload and measurement semantics. Compilation/tuning, model loading, host
submission, device execution, and end-to-end latency remain distinct measurements;
warm-cache reuse must be declared. Engine performance acceptance is V3 parity or
better on the agreed workloads, not merely an isolated kernel speedup.

The compiler qualification boundary closes before manual model-kernel performance
tuning. Engine-level discrepancies then remain evidence to investigate, including
compiler, model, runtime, and integration causes. All measurements retain correctness,
artifact identity, conditions, and the timing boundary they actually measure.
