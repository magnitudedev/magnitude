# Seismic

**Seismic is a kernel language, compiler, and runtime packaged as a Rust library.**
Authors write numerical kernels with their logical execution structure. The compiler
selects among the authored implementations, contiguous fusion groups, and numerical
dimensions, and realizes the selection through prescribed backend mappings. It never
invents execution structure.

The governing specification is
`specs/26-09-18/seismic-structured-authoring-spec.md`. These documents state the
contract of the implemented system.

## Principles

1. **Structure is authored; numbers are selected.** An implementation fixes its
   algorithm, independent domains, producer and state scopes, stage order, and
   combination order. Authors never write widths, candidate lists, or hardware
   constants. Portable code cannot observe a selected width, piece count, or
   physical mapping.

2. **Optimization is selection over a finite supplied family.** The decisions are:
   one applicable implementation per active static call occurrence, one exact cover
   of contiguous intervals per active sequence of execution units, and one value per
   active numerical site. There is no search over graphs, producer placement,
   reduction trees, layouts, storage placement, or schedules.

3. **One authority.** The selected witness drives checking, the estimate, and
   emission. Nothing downstream of selection chooses, repairs, or re-tiles. A
   reconstruction that disagrees with the witness is a compiler defect, not an
   infeasible candidate.

4. **No hidden search and no hidden policy.** A backend hook is a deterministic
   function of its inputs. Every performance preference lives in the solver
   objective as an explicit local cost factor. Anything a backend fixes, it fixes
   by one documented rule.

5. **Select in IR, before native compilation.** Selection evaluates arithmetic over
   the family. Native source is generated once, for the selected witness. No
   compile-and-benchmark loop exists.

6. **Numerics and effects are preserved.** Casts, FMA, accumulation dtype,
   publication rounding, reduction order, producer multiplicity, state order, and
   observable writes are exactly what the selected bodies say. Selection cannot
   change them.

7. **The compiler is generic.** No model names, model dimensions, or pattern
   recognition of particular kernels exist in the compiler or a backend. Model
   knowledge lives in authored sources and engine bindings.

8. **Unknown is not zero; unsupported is not infeasible.** Missing analysis, a missing
   mapping, missing target coverage, proved infeasibility, and an exhausted budget
   are distinct outcomes. None triggers a fallback.

9. **Honest results.** A selected execution is *feasible* unless the solver proved it
   optimal over the stated family under the stated estimate model. Estimates are
   labelled estimates; the current Metal estimate model is labelled unqualified.

10. **Fix the system, not the kernel.** When a natural structure cannot be expressed
    or realized, the owning layer changes: the language, a mapping, or a library
    body. Sources do not work around compiler defects.

## Responsibilities

| Owner | Establishes | Does not |
| --- | --- | --- |
| Kernel or library author | Algorithm, regions, producers, state, stages, merges, numerical contract, alternative portable bodies, target lowerings, backend-specific helpers | Declare widths, candidate lists, or hardware constants |
| Lowering author | A target implementation within the ownership its signature grants | Restructure the caller or widen its scope |
| Checker | Types, shapes, bounds, modes, aliasing, slice opacity, region results, stages, partial obligations, target coverage declarations | Prove bodies equivalent |
| Family construction | Applicable candidates per occurrence, numerical sites, execution-unit sequences, obligations | Enumerate compositions; drop what it cannot analyze |
| Backend mapping | Site domains, hard limits, legal intervals, local cost factors, a constructive seed, deterministic realization | Rank, filter by profitability, or search |
| Solver | The joint assignment under a budget, with decomposition and proof reuse | Invent calls, stages, groupings, or sites |
| Instantiation and emission | Exactly the witness | Any second tiling, fusion, staging, or placement policy |
| Runtime | Native compilation of a checked selection, binding, validation, submission, completion | Select, substitute, or fall back |

## Packages

| Package | Responsibility |
| --- | --- |
| `seismic-lang` | Syntax, checker, structured IR, reference interpreter, joint family, instantiation to the execution IR; the symbolic prover, representations, intrinsics, ABI. |
| `seismic-compiler` | Joint selection: solver export, seed validation, budgeted search, witness audit, replay, search analysis; the `Backend` contract; the target-neutral structural walk and mapping helpers every backend shares. |
| `magnitude-solver` | Generic exact and neighborhood search with guards, residual decomposition, and proof reuse. Knows nothing about Seismic. |
| `seismic-realization` | Target-neutral realization contracts shared by backends: invocation ABI and conditions, launch phases, tile placement, and the local storage type rule. |
| `seismic-metal` | The Metal mapping, realized execution, MSL emission, device runtime. |
| `seismic-runtime` | Devices, buffers, compilation of a selected execution, plan compiler, invocation validation. |
| `seismic-cli` | `check`, `print`, `select`, `emit`, `analyze-search`, `bindings`. |
| `seismic-std` | The standard kernel library and its Metal lowerings, authored in Seismic. |
| `seismic-cpu` | The CPU mapping, scalar realized execution, Cranelift native compilation, worker threads and host buffers. |
| `seismic-cuda` | The CUDA mapping, scalar realized execution printed as PTX, driver runtime (loaded dynamically). |

Model topology belongs to user libraries. Artifacts, residency, logical state,
scheduling, and serving belong to the host application.

## Flow

```text
plain `.seismic` sources (portable functions, target lowerings, backend-specific helpers)
    -> checked closed program: structured IR and contract families
    -> joint family for (entry, target, workload)
    -> backend: site domains, limits, intervals, cost factors, seed
    -> budgeted joint selection -> audited witness
    -> instantiation: concrete execution IR, verified
    -> backend realization: deterministic mapping rules
    -> emission -> native compilation -> bound, validated invocation
```

The reference interpreter executes the same structured IR under any caller-supplied
partition and defines the semantics every backend must reproduce, including
finite-precision behavior.

## Outcomes

| Outcome | Meaning |
| --- | --- |
| Invalid source | Type, effect, ownership, or declaration error. |
| Missing target coverage | No applicable portable body, target lowering, or backend-specific helper has a complete supported dependency tree for a reached call. |
| Unsupported structural mapping | The backend has no mapping for a meaningful structure. |
| Incompatible composition | Required interfaces or hard capacities cannot agree. |
| Infeasible | The exported family is proved to have no solution. |
| Selection incomplete | The budget ended without a checked configuration. |
| Analysis unavailable | A required quantity or estimate has no supported derivation. |
| Reconstruction defect | A witness, seed, or instantiation disagreed with the family. Compiler defect. |
| Selected, feasible | Complete checked execution; estimated performance only. |
| Selected, model-optimal | Additionally optimal over the stated family and estimate model. |

## Current scope

- Metal is the only backend. Qwen3.5-4B prefill and decode run through this pipeline.
- Width domains offer only divisors of static extents, so tail pieces do not occur.
- `pipeline` has one mapping: synchronous, same participant, ring depth one.
- A region result cannot cross a launch.
- The estimate model is unqualified. No performance claim follows from a selection.

## Further reading

| Document | Focus |
| --- | --- |
| [Language](language.md) | Source surface and its rules |
| [Compiler](compiler.md) | Stages, authoritative representations, permitted transformations |
| [Execution](execution.md) | Region semantics as realized, execution units, instantiation rules |
| [Tuning](tuning.md) | Joint selection, search, proof status, replay |
| [Backends](backends.md) | The `Backend` contract and the Metal mapping |
| [Runtime](runtime.md) | Executable boundary, binding, validation, reuse |
| [Accounting](accounting.md) | Derived quantities, estimates, and their authority |
| [Inference V4](../overview.md) | Enclosing inference-engine architecture |
