# Selection policy and prepared kernel

`SelectionPolicy` is the evaluator's decision. `PreparedKernel` is its immutable, resource-owning published form for repeated invocation.

| Boundary | Input -> output |
| --- | --- |
| Policy construction | Applicable retained candidates and evaluator comparisons -> portfolio with a total deterministic selection decision |
| Finalization | Policy, entry invocation contract and exact selected native artifacts -> `PreparedKernel` |

## Selection policy

The policy owns its candidates and decision together. Each decision leaf identifies a member of that portfolio, and each specialized branch lies inside that member's established applicability. The general reference candidate is the default for the remaining legal entry domain.

```text
invocation metadata
        |
        +-- preferred region A and candidate A applicable -> A
        |
        +-- preferred region B and candidate B applicable -> B
        |
        +-- remaining legal domain ----------------------> general
```

Branch precedence and ties are deterministic. Selection uses only metadata available through binding; it cannot inspect an unavailable device scalar or assume sampled tensor contents. The decision must be total over the legal invocation domain, not merely the shapes visited during evaluation.

The selector is built from owned candidate references. Callers cannot supply an unrelated index-returning function beside a portfolio and ask a later stage to reconstruct whether they match.

## Exact finalization

Finalization packages the chosen executable policy, invocation schema, native artifacts and owned requirements. It does not compile again, trim candidates, alter guards or make another performance decision. If reconciliation changes what is usable, that work belongs in evaluation before policy construction.

`PreparedKernel` is reusable across legal invocation dimensions and ordinary arguments. It is not a bound call and does not own a future caller's tensor contents. It retains all immutable compilation resources needed for binding and execution.

Continuation can publish another snapshot with a different portfolio or broader established applicability. Existing snapshots keep their prior decision, guards and resource ownership. Destroying an evaluation session cannot invalidate an already published kernel.

## What runtime may assume

For a successfully bound legal invocation, runtime can rely on source correspondence, numerical applicability, complete executable structure, compatible native artifacts and derivable resource requirements. It does not requalify candidates, repair missing values, invoke a compiler fallback or repeat numerical selection.

Runtime still checks actual arguments, alias/access rights, device compatibility and real capacity. It can encounter device loss or native service failure. These are [execution boundaries](../execution/overview.md), not missing compile-time construction guarantees.

## Identity and caches

Prepared identity includes checked semantic content and entry bindings, target/deployment compatibility, precision policy, selected physical and emission choices, ABI and native dependencies. Performance characterization belongs in evaluation records; changing a timing model does not mutate a published executable's meaning.

Persistent caches may store reusable artifacts, but bytes are external input. Reconstruction must restore valid owned relationships through the appropriate constructors and compatibility checks. A cache key, schema tag or previously successful test is not correctness authority.

The explicit authored [direct-native route](../language/functions-and-capabilities.md#explicit-direct-native-execution) remains separate. It does not produce this compiler-derived selection policy or numerical guarantee.

