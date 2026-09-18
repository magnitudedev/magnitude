# Seismic compiler

**The compiler turns a checked library composition into a selected execution for a
workload and device.** It owns semantics-preserving transformations and joint
implementation selection; the runtime binds and executes its result.

## Compilation flow

| Stage | Responsibility |
| --- | --- |
| Load and check | Resolve libraries, types, symbolic shapes, effects, and lowering coverage |
| Canonicalize | Apply deterministic, idempotent normalization and recognize valid operation compositions |
| Compose | Inline and transform enclosing computations while preserving precision and effects |
| Lower | Expose applicable backend implementations and unresolved typed choices |
| Derive and optimize | Propagate legality/resource constraints, resolve choices and schedules, validate the objective |
| Emit | Translate the resolved execution through its implementation definitions |
| Native compile | Compile selected target code under its backend/toolchain contract |

Canonicalization covers supported rewrites, not arbitrary program equivalence.
Fusion and decomposition are choices about whole executions: eliminated transfers
can trade against longer lifetimes, lower concurrency, or added synchronization.

## Representations

| Representation | Contents |
| --- | --- |
| Portable IR | Checked computation, values, symbolic shapes, representations, control, numerical permissions, and effects |
| Lowered IR | Shared computation extended with backend implementations, dependent choices, and legality constraints |
| Tuned IR | Actual selected operations, allocations, layouts, mappings, checks, synchronization, and launches |
| Target code | Backend code-generation input implementing the selection |
| Executable | Native code, binding interface, retained conditions, and relevant identities |

Stages share semantic definitions and explicit transformation relationships.
Resource and dependency models are derived views. Optimization refines this execution;
accounting derives its consequences; emission consumes it. There is no separate
computation graph for bounds, authored companion cost model, or proof-artifact pipeline.

Tuned IR resolves compilation decisions into actual execution structure. Runtime
values and declared dynamic extents may remain; emitter-selected tuning may not.

## Ownership

| Owner | Authority |
| --- | --- |
| [Language](language.md) | Source semantics, primitive definitions, libraries, and checking |
| [Execution](execution.md) | Legal forms, operation contracts, choices, dependencies, allocations |
| [Accounting](accounting.md) | Derived resource constraints, exact quantities, timing relationships, sound relaxations |
| [Tuning](tuning.md) | Search, propagation, scheduling, and exact selection within the declared form |
| [Backends](backends.md) | Concrete implementations, emission, native mappings, hardware inputs |
| [Runtime](runtime.md) | Device/artifact ownership, binding, submission, completion, and caching |
| Tooling | Read-only explanations of these structures |

Shared contracts do not depend on the engine or runtime. Concrete backends are
composed through compilation interfaces; the runtime composition root does not
acquire optimization policy.

## Build-time and device-time work

| Boundary | Work |
| --- | --- |
| Application build | Check standard/model libraries, embed parsed/canonical programs, generate typed Rust bindings |
| Device compilation | Specialize shapes and conditions, select implementations, derive storage and ABI, emit native input |
| Warm invocation | Validate dynamic bindings and reuse prepared execution |

Signatures define host shapes, representations, parameters, and effects. Generated
bindings eliminate independently maintained ABI layouts. Missing declarations,
incompatible calls, and coverage errors fail library checking. Accelerator presence
is not required for device-independent checking.

Development source overrides pass the same checks as embedded programs. Relevant
program, workload, hardware, and implementation identities govern reuse.

## Optimization principles

- Preserve operation meaning, value versions, numerical permissions, and effects.
- Expose performance-relevant choices in the legal execution space.
- Derive resources from the same implementation definitions used by emission.
- Select from IR and hardware contracts; candidate benchmarks, native resource
  queries, heuristic scores, and compile-and-try fallback are not selection inputs.
- Distinguish model optimality, sound physical bounds, hardware fidelity, and
  invocation applicability. A private constructor cannot establish physical truth.
- Keep construction and validation on the same path for explicit diagnostic choices.
- Preserve complete search status; unfinished work cannot establish an optimum.

## Authoring and inspection

| Capability | Required information |
| --- | --- |
| Check / interpret | Source locations, typed semantics, violated conditions, reference outputs |
| Inspect lowering | Chosen and unresolved implementations, shapes, ownership, sizes, dependencies |
| Inspect performance | Resource terms, limiting constraints, choice explanations, scope and assumptions |
| Inspect emission | Target code, native diagnostics, applicable mapping information |
| Reproduce | Bounded source/library identities, inputs or references, options, device/model conditions |

The CLI operates independently of the engine. Editing a kernel or lowering affects
only dependent artifacts. When authors must contort a natural program to work around
compiler behavior, repair the responsible language, analysis, lowering, or tooling.
