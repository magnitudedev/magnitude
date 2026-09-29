# Compiler architecture

The compiler constructs legal physical implementations of an authored entry, evaluates their performance, and publishes a reusable selection policy. Each boundary owns a distinct decision. Correctness follows from how those boundaries construct their outputs.

## Pipeline

```text
source / bundle
      |
      v
CheckedModule
      | entry + concrete element types
      v
LogicalEntry <---------- target/deployment + precision policy
      |
      v
RefinementSession
  defines construction rules and constructs general reference candidate
      |
      v
CandidateDomain
  owns families, physical/emission choices and construction cache
      ^
      | requests                         EvaluationSession
      |                                  owns shared services/resources
+-----+---------------- CandidateEvaluator -----------------------+
|                                                                 |
| explore or transform coordinates through CandidateNavigator     |
|      |                                                          |
|      v                                                          |
| source-directed construction                                    |
|   values + effects + control + storage + numerical meaning      |
|      |                                                          |
|      v                                                          |
| ClosedExecutableIr                                              |
|   actual execution and derived resources                        |
|      |                                                          |
|      +----> analytical estimates                                |
|      |                                                          |
|      v                                                          |
| specified emission request -> native formation/reconciliation   |
|      |                                                          |
|      v                                                          |
| applicable candidate executable ----> feedback measurements     |
|      |                                                          |
|      v                                                          |
| retention and total invocation decision                         |
+------+----------------------------------------------------------+
       |
       v
SelectionPolicy -> exact finalization -> PreparedKernel
                                              |
                                     actual arguments
                                              |
                                              v
                             binding -> workflow -> execution
```

Construction and native compilation happen on demand during evaluation. The evaluator controls exploration and performance decisions; shared candidate navigation applies transformations through the domain's constructors. Physical and permitted emission alternatives participate in the same search. Analytical exploration can inspect selected structure before native formation. Timing and retained execution use the same applicable, natively reconciled executable boundary.

## Entities and cardinality

| Entity | Meaning for one prepared source entry |
| --- | --- |
| `CheckedModule` | Checked definitions, semantic registry bindings and authored bodies. Can serve many entries. |
| `LogicalEntry` | One instantiated entry and reachable computation, invocation contract and expression owner. Runtime dimensions can remain symbolic. |
| `CandidateDomain` | One complete definition of permitted implementations for this entry, target and policy. |
| `CandidateFamily` | A parameterized construction, initially grouped by an applicable authored root body. |
| Candidate coordinate | Immutable structured selection of active body, physical and emission choices. Partial requests support exploration; derived facts are omitted. |
| Candidate | One whole-entry construction with selected emission choices and derived applicability. Can contain many launches and control regions. |
| Physical IR | The constructed physical execution. One physical contract may admit several emission alternatives; cached materializations do not define coordinate identity. |
| `SelectionPolicy` | A retained portfolio and a total deterministic decision among its applicable members. |
| `PreparedKernel` | One immutable published policy with its invocation contract and native resources. |

There is neither one physical IR for the source function forever nor a fresh IR per invocation. One domain can construct many candidates; many candidates can share native kernels. Evaluation can publish successive prepared snapshots without changing previous snapshots.

## Boundary contracts

| Owner | Input -> output | Guarantee |
| --- | --- | --- |
| [Checked semantics](checked-semantics.md) | Source, registry, entry bindings -> `LogicalEntry` | One meaning, type, effect and exact value origin for every admitted operation. |
| [Target contracts](target-contracts.md) | Device, backend, toolchain, deployment -> immutable target | Actual supported operations, limits, ABI and numerical environment. |
| [Candidate domain](candidate-domain.md) | Entry, target, policy -> construction space | Membership independent of search effort; no independently chosen derived facts. |
| [Construction](construction.md) | Checked operations, bound operands, choices -> physical execution | Complete values and effects, source correspondence, legal scopes and general reference coverage. |
| [Physical IR](physical-ir.md) | Constructed regions -> closed executable | Actual work, storage, synchronization, publication and resources remain one owned structure. |
| [Numerics](numerics.md) | Actual source/physical semantics, policy -> applicability | Whole-entry guarantee over all admitted contents and permitted outcomes. |
| [Native emission](native-emission.md) | Fixed physical contract and selected emission request -> native artifacts | Faithful formation and reflected requirements reconciled before executable exposure. |
| [Evaluation](evaluation.md) | Domain, services, objective, effort -> `SelectionPolicy` | Search, costs, retention and total selection have one owner. |
| [Prepared kernel](prepared-kernel.md) | Policy and exact selected artifacts -> published handle | Finalization packages the decision without recompiling, trimming or reselecting. |
| [Execution](../execution/overview.md) | Prepared handle and actual arguments -> completion | Complete binding, resource ownership and dependency-preserving execution. |

## Guarantees by construction

Every semantic producer constructs all its results and effects from its actual operands. Physical consumers receive those results, including stored/computed realization and value version. They cannot recover omitted producers or supply an independent symbolic meaning.

Expression APIs admit only operands available in their environment. Region constructors close local values, carries and resources. Executable closure derives launch requirements from the normalized structure it consumes. Selection constructs its decision from its actual retained portfolio. Binding constructs one complete invocation from actual arguments and that selection.

These are constraints on normal APIs, not a request for a final checker to reject malformed intermediate objects. External inputs still require checking, and native reflection must reconcile real artifact facts. Diagnostic assertions and tests help identify implementation defects; they cannot grant missing semantic properties. Private fields prevent bypass, but the constructors themselves still need a preservation argument for every admitted case and composition.

## Distinct completeness claims

| Claim | What must be established |
| --- | --- |
| Reference completeness | Every admitted source operation and composition has a general legal realization across the entry domain. |
| Physical-domain completeness | Every permitted source-preserving native physical contract is representable by the construction rules. |
| Emission coverage | The permitted formation routes can obtain an optimum among the realizations of each fixed physical contract. |
| Numerical resolution | Applicability of the relevant candidates has been resolved under the whole-entry policy. |
| Search completion | All relevant alternatives have been resolved and compared under the stated objective and scope. |
| Optimality | Domain coverage, emission quality and selection together satisfy the [optimality equation](../overview.md#the-optimality-equation). |

An exhaustive enum match establishes the operation cases to handle, not the correctness of those cases. A general implementation does not establish exhaustive optimization. A measured winner does not establish numerical validity or global optimality.
