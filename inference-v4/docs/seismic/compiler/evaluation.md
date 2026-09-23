# Candidate evaluation

`CandidateEvaluator` decides which permitted implementations are worth retaining and how to select among them for future invocations. It owns exploration, performance comparison, retention and the final selection decision.

| Input | Output |
| --- | --- |
| Candidate domain, execution/compilation services, objective, workload information and effort budget | `SelectionPolicy`, evaluation report and resumable search state |

`EvaluationSession` owns the domain by composition and supplies shared services, native artifacts, resource accounting and cancellation. The domain remains the sole owner of physical construction caches. Session state does not become another planner with independent decisions.

The public `CandidateEvaluator` boundary supports downstream strategies through
`prepare_with_evaluator`. A session constructs its domain together with the opened device,
registry and native services. Strategies can inspect and advance that domain, request native
preparation of canonical coordinates, observe admitted executables and native work, and build
their retained selection through the session. They cannot replace applicability or insert native
artifacts. Candidate references are opaque and belong to one preparation.

Native preparation distinguishes ready candidates, established rejections, unresolved numerical
analysis and exhausted native effort. The evaluator returns its selection policy, evaluation
identity and planning report together. Common finalization checks their preparation/device
ownership and packages that exact policy. Analytical and continuing feedback preparation use the
same orchestration; there is no mutable result slot for a later finalizer to discover.

## One evaluation loop

```text
general candidate / retained incumbent
                 |
                 v
CandidateNavigator: transform / traverse
                 |
                 v
request construction and applicability ---- pending -> resume later
                 |
                 v
estimate cost / form selected native request
                 |
                 v
compare estimates or controlled measurements
                 |
                 v
retain useful applicable candidates
                 |
                 v
construct total selection decision -> publish snapshot
```

[Analytical evaluation](analytical-evaluation.md) predicts cost from physical demand and target service behavior. [Feedback evaluation](feedback-evaluation.md) observes concrete executions. Both use the same domain, executable preparation and policy construction boundaries. Neither can admit an invalid implementation through its own result type.

The [domain's shared navigation](candidate-domain.md#transformations-and-navigation) supplies candidate transformation mechanics. The evaluator chooses how to use them. Physical changes and emission alternatives compete in one search; native formation handles one specified request. Performance comparisons cover the complete entry, since a locally faster lowering can change spills, overlap or host cost elsewhere.

Analytical search may inspect parameterized closed structure before native compilation. Before retention, native artifacts and resource facts must be reconciled. A feedback trial starts from that same executable and binds through the ordinary invocation contract.

## Objective and metadata

The objective identifies what “best” means: for example a particular invocation's latency or expected latency over a supplied workload distribution. Compare candidates over the same declared scope; a candidate applicable only to a subset cannot represent the whole scope without its guarded fallback. Dimensions, representations and ordinary scalar arguments may guide selection when the binder makes them available.

Optimization ranges and measurement points describe search effort. They do not narrow the entry domain, imply workload probabilities, or authorize content-dependent guards over unread tensor data. Device-produced metadata requires an explicit completion/read boundary before it can guide a later public invocation.

## Retention and selection

Retain candidates with established applicability and useful performance tradeoffs. Construct the decision from the actual retained portfolio: every specialized branch intersects its candidate's applicability, and a general default covers the remaining legal domain. Ties and overlapping branches have a deterministic rule.

A selector consumes the admitted candidate references and their guards as its actual operands.
Policy construction resolves that operand order into retained executable bodies, validates their
preparation ownership, and checks the general first candidate, duplicates and resource budget.
There is no independently supplied retained list whose ordinals can disagree with the selector.

The resulting [SelectionPolicy](prepared-kernel.md) owns both portfolio and decision. There is no post-evaluation planner that independently recompiles, prunes or reconstructs their relationship. Finalization packages exactly that result.

## Incomplete work and failures

Effort exhaustion returns the current valid incumbent/portfolio and continuation. It does not mean the domain is exhausted, a candidate is infeasible, or the winner is globally optimal. The general candidate is constructed independently of optional search.

Reports preserve distinct outcomes:

- A supported construction is still pending or numerically unresolved.
- A particular choice is excluded by an established capability, resource or policy condition.
- The search budget or measurement budget expired.
- Actual compilation, device, service or capacity failure prevented work.
- An admitted candidate violated its contract, indicating a compiler/backend defect.

Permanent exclusion caches cannot contain timeouts, interrupted work or noisy timings. Cache identity includes the facts relevant to the cached result; performance records cannot stand in for numerical or execution compatibility.

Published prepared snapshots own their resources independently. Continuing search may publish a new snapshot, but cannot change an old snapshot's guards, artifacts or behavior.
