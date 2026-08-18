---
applies_to:
  - packages/agent/src/compaction/**
  - packages/agent/src/projections/compaction*.ts
  - packages/agent/src/window/**
  - packages/agent/src/projections/turn.ts
  - packages/agent/src/workers/**
  - packages/agent/src/session-work-status.ts
  - packages/agent/src/coding-agent.ts
  - packages/event-core/src/worker/define*.ts
---

# Context compaction

Compaction incrementally reduces one fork's context window until its next ordinary request fits the
configured operating threshold. It is an event-sourced convergence process.

## Guarantees

- Compaction eventually succeeds while a compatible model remains available.
- A context-rejected ordinary turn waits and resumes only after compaction succeeds.
- A context-rejected compaction request is replanned with less input.
- Estimation error and model changes cannot terminate compaction.
- Switching to a smaller compatible model uses normal sequential compaction.
- Summary quality may degrade when necessary, but every committed step makes progress.
- No new ordinary turn starts while the window requires reduction.
- Every nonterminal compaction state is replayable and recoverable.

The convergence guarantee applies to any finite amount of accepted context. Unbounded input uses
normal admission and backpressure.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Window projection | Logical context, prompt cost, model-fit status, provider rejection evidence, and rewrites |
| Compaction projection | Compaction step identity, lifecycle, retry state, and prepared result |
| Turn projection | Pending triggers and ordinary-turn lifecycle |
| `TurnInitiator` | Ordinary-turn admission |
| `CompactionWorker` | Compaction admission, execution, retry, and recovery |
| TurnExecutor | Ordinary-turn execution |

Window means the context window. It remains one projection authority, internally modularized into
ledger, costing, fit-policy, and rewrite logic as needed. Compaction depends on Window and contains
only compaction-specific state.

```text
durable events
    |
    +--> WindowProjection --------+
    +--> CompactionProjection ----+--> settled projection truth
    `--> TurnProjection ----------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
              TurnInitiator                   CompactionWorker
                     |                                 |
                     v                                 v
                turn_started                 compaction events
                     |
                     v
                   TurnExecutor
```

The event log is authoritative. Projections are replayable queries. Workers own effects but no
durable truth. Signals communicate synchronous derivation or wake workers to reread projections;
they never own work.

## Model compatibility

Window computes one explicit precondition:

```text
MIN_CONTEXT_WORKSPACE_TOKENS = 16,384

fixedEnvelopeTokens =
    required system content
  + effective tool definitions
  + permanent session or fork context
  + fixed framing and safety overhead

requiredContextWindow = fixedEnvelopeTokens + 16,384

compatible = modelContextWindow >= requiredContextWindow
```

The workspace allocation is:

```text
16,384 workspace
  = 8,192 variable input
  + 8,192 output reserve

8,192 variable input
  = up to 4,096 existing summary
  + at least 4,096 new compactable context
```

Every prepared summary, including retained file material, is bounded to 4,096 tokens. Any conforming
summary can therefore participate in the next compaction step.

The soft cap must also fit the fixed envelope plus one maximum summary. Invalid threshold policy is
a configuration error rather than an endless compaction condition.

Window exposes one fit result:

```text
fits
reductionRequired(reason, observedBound?)
incompatible(required, available, deficit, source)
```

`source` distinguishes calculated incompatibility from provider rejection of the minimum request.
Changes to the model, system content, tools, permanent context, or threshold policy recompute fit.

## Window accounting

Every accepted prompt-bearing item enters Window's logical ledger and accounting immediately.
Rendering, grouping, and coalescing are derived views.

One cost implementation serves ordinary admission, compaction planning, rewrite validation, and
post-rewrite evaluation. It includes fixed content, tools, framing, permanent context, text, images,
output reservation, and safety margin.

The soft cap is the convergence target. The hard cap is a strict request-admission boundary.
Provider usage calibrates estimates. Provider context rejection overrides an optimistic estimate
for the rejected request shape.

## Compaction state

CompactionProjection contains no duplicate window-fit or turn-blocking booleans.

```text
idle ---- admitted ----> compacting ---- generated ----> prepared
 ^                           |   ^                          |
 |                           |   | due and admitted         | rewrite applied
 |                           v   |                          |
 |                        backoff                           |
 +----------------------------------------------------------+
```

`idle` plus `Window.reductionRequired` means a new step is needed. `backoff` records attempt and
eligibility time. `prepared` contains the validated rewrite until Window applies it. Incompatible
Window state admits no compaction work.

Session work is derived from Window needing reduction or Compaction being non-idle. Incompatibility
is terminal for the current configuration and does not claim active work.

## Ordinary context rejection

```text
ordinary request rejected for context
        |
        +--> Window records reductionRequired and observed bound
        +--> Turn returns idle with the chain trigger retained
        `--> CompactionWorker admits a compaction step

Window returns fits
        `--> TurnInitiator resumes the pending chain
```

New user activity may extend Window while turns wait. It cannot bypass fit admission.

## Sequential compaction

Each durable step reduces the oldest useful prefix that fits a bounded request.

```text
initial:
[permanent] [turn 0 .... turn n] [turn n+1 .... newest]

step 1:
compact(permanent, turn 0 ... turn n)
    -> [permanent] [summary 1] [turn n+1 .... newest]

step 2, if Window still requires reduction:
compact(permanent, summary 1, turn n+1 ... turn m)
    -> [permanent] [summary 2] [turn m+1 .... newest]

repeat until Window reports fits
```

The same Window costs and thresholds apply to gradual growth, inaccurate estimates, provider
rejection, and model changes.

Each admitted step records its identity, frozen boundary, model and toolkit snapshot, bounded input,
replacement budget, retry state, and any rejection bound. Later appends remain outside the boundary.
Only outcomes matching the active step are accepted.

## Compaction request rejection

```text
request rejected for context
      |
      +--> Window records the rejected bound
      +--> planner removes optional recent context
      +--> planner reduces variable prefix input monotonically
      `--> CompactionWorker admits the smaller step
```

An oversized entry is fragmented or represented lossily. Provider rejection at the minimum request
makes Window `incompatible` with a typed observed-precondition error.

## Admission and concurrency

TurnInitiator is the sole publisher of `turn_started`. It reads settled Window, Compaction, and
Turn state:

```text
pending turn trigger
        |
        +-- Window fits + Compaction idle --> turn_started
        `-- otherwise ---------------------> wait
```

CompactionWorker is the sole publisher of compaction admission and outcome events. Window fit changes
wake it to reread Window and Compaction. Its admission path is short; admitted model work executes in
the worker's per-fork event-handler queue rather than a globally serialized signal handler.

Projection handlers validate both admission event types against current projection state, so a
stale worker observation cannot admit conflicting work. Provider-wide capacity belongs to provider
admission.

Compaction may overlap only the ordinary turn already active when Window first requires reduction.
The active result is after the frozen boundary. No later turn starts until Window reports `fits` and
Compaction is `idle`.

## Rewrite commit

Summary and lossy degradation produce one rewrite type:

```text
PreparedRewrite
  step identity
  frozen boundary
  replacement entries and cost
  strategy: summary | lossyElision
```

```text
[permanent] [frozen prefix] [post-boundary entries]
                  |
                  v
[permanent] [replacement]   [post-boundary entries]
```

Window stores the prepared rewrite from its durable event and applies it synchronously:

- immediately if preparation occurs at an idle turn boundary; or
- in the same projection transaction as the active turn's terminal event.

Window then recalculates fit and emits `rewriteApplied`. Compaction consumes that signal and returns
the matching step to `idle`. Workers observe only the settled projections. There is no asynchronous
injection decision.

## Robust execution and degradation

The shared model runner returns canonical output, typed terminal outcome, usage, and tool outcomes.
Compaction classifies the terminal outcome before interpreting output.

| Outcome | Policy |
| --- | --- |
| Context rejection | Replan with less input |
| Retryable transport failure | Backoff and retry |
| Valid reducing result | Prepare rewrite |
| Invalid, missing, or non-reducing result | Bounded retry, then lossy elision |
| Interruption | Return to idle with reduction still required |
| Model unavailable | Wait without admitting a step |
| Minimum request rejected | Window becomes incompatible |

Lossy elision is the deterministic liveness fallback after reasonable summary attempts. It replaces
an old prefix with a bounded omission marker or nothing. It is not a separate purge workflow: it
uses the same boundary, commit, preservation, accounting, and progress checks.

Compaction retains ordinary model-facing tool definitions for cache compatibility, but execution
permits exactly one successful `compact` result call. Other tools and repeated result calls fail
before execution.

## Model changes

A model change is an event-core ambient transaction. Window recomputes cost, thresholds, and fit
before projection readers or workers proceed.

```text
switch model
    |
    +-- Window fits ----------------> ordinary work may continue
    +-- Window needs reduction -----> sequential compaction
    `-- precondition fails ---------> typed incompatibility
```

An active step uses its captured configuration. A configuration change may supersede it; stale
output is ignored by identity. The next step uses the new model. A smaller compatible model does not
select lossy degradation unless normal summary attempts fail.

## Hydration

Hydration reconstructs projections without worker effects. After hydration, the runtime publishes a
transient reconciliation event that makes workers reread current truth.

```text
Window fits + Compaction idle        --> no compaction work
Window needs reduction + idle        --> CompactionWorker admits a step
Compaction compacting                 --> restart the identified side-effect-safe step
Compaction prepared                   --> Window applies at the replayed idle boundary
Window incompatible                   --> wait for configuration change
```

Restarted work retains its durable identity; stale outcomes are ignored. Historical signals are
unnecessary.

## Convergence

For a compatible model and finite accepted context:

1. Context rejection monotonically reduces the next step's variable input.
2. A summary or lossy rewrite reduces a finite prefix by a positive cost margin.
3. Post-boundary entries survive and cannot invalidate committed progress.
4. Window re-evaluates after every step using the current model.
5. In the worst case, lossy elision removes all non-permanent history.
6. The compatibility precondition leaves room for a bounded summary, useful new input, and output.

Therefore estimate errors, context rejection, model swaps, invalid output, interruption, and finite
concurrent appends cannot permanently fail compaction while a compatible model is available.
