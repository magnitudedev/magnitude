# Feedback evaluation

Feedback evaluation chooses candidates using controlled performance measurements of actual applicable executables. It shares construction, binding and runtime with ordinary execution.

| Input | Output |
| --- | --- |
| Domain, executable services, measurement workload/objective and effort budget | Observations, retained candidates, selection policy and continuation |

## Trial path

```text
physical and emission coordinate
       |
       v
constructed applicable candidate + reconciled native artifacts
       |
       v
private trial inputs -> ordinary invocation binding
       |
       v
workflow -> admission -> submission -> completion
       |
       v
measurement record -> comparison and retention
```

The entry-owned binder consumes the chosen candidate and actual trial arguments, producing the same complete bound invocation used by ordinary calls. Trials use this operation directly without constructing a selection policy.

Only numerically applicable candidates execute for performance measurement. Private compilation may occur while analysis is pending, but empirical execution cannot promote an unresolved candidate into correctness.

## Controlled observations

Trials own private inputs and outputs. Writable state is reset to the intended initial state before each comparable run, or follows an explicitly measured repeated-invocation protocol. Aliases, representations and dimensions match the declared invocation. Compilation, warmup, reset, transfer, synchronization and execution costs are included or excluded according to an explicit measurement endpoint.

Submission-through-completion excludes admission and fixture setup. Candidate work cannot be moved into fixture setup to make it free; acquisition or initialization benefits require an objective that includes them. Measurements compare complete entries, including their host control and native launches.

Timing completion must correspond to the actual work being measured, including asynchronous commands. Reusing an allocation or command object does not imply all prior readers have finished. Trial resources follow the ordinary [lifetime rules](../execution/resources-and-lifetimes.md).

The evaluator records conditions needed to interpret measurements: candidate and exact artifact identities, invocation and content/state case, target/environment, endpoint, repetition and aggregation policy, and observed variability. Close or unstable comparisons may need additional measurements. A finite workload samples performance; it does not establish the best implementation across all legal inputs.
Each observer request carries the exact preparation candidate ID whose retained native handles are executed. A semantic implementation digest alone cannot distinguish separate formation outcomes.

Compile and executable caches may be shared across trials when their compatibility contract matches. Measurements have a separate identity that includes their environment and endpoint. Timing records cannot authorize artifact reuse under an incompatible ABI.

## Correctness observations

Optional diagnostics compare completed native outcomes with complete interpreter outcomes. The interpreter outcome owns returns, final input backing and the allowed-result relation. Read-only bytes are exact; floating tolerance applies only where the policy permits. Integer/index/range transport preserves typed values.

A disagreement with one representative of a nondeterministic reference can be unsupported to compare rather than definitely incorrect. Missing observations, unsupported encoded-result comparison and exhausted checking work cannot become successful comparisons.

A definite violation by an admitted candidate is a compiler/backend defect and stops that path with a diagnostic. It cannot be hidden by removing the candidate as an ordinary tuning loser. Successful comparisons are debugging results, never numerical admission or a certificate for other contents.

## Search

Feedback uses [shared candidate navigation](candidate-domain.md#transformations-and-navigation) for local changes, region regeneration, compatible recombination and fresh systematic exploration across physical and emission choices. Feedback owns parent selection, operator allocation, screening, confirmation and retention. Evolution is a search strategy over these operations, not a second candidate representation or transformation system.

The evaluator interleaves exploration with confirmation of useful candidates, retaining a general default. Budget exhaustion returns a valid current portfolio and resumable state. Pending construction remains schedulable; bounded construction caches and observation archives do not permanently close unvisited choices. Failure to measure a candidate says nothing about its semantic membership. Device loss, allocation failure and cancelled work remain real execution outcomes rather than scores.

The result is a [selection policy](prepared-kernel.md), not a claim that every possible candidate was measured or that the global optimum was found.
