# Autonomous inference engine development protocol

Pursue the user-established objective through creative investigation, evidence, and
coherent engineering. Improve correctness, latency, throughput, and memory efficiency
together. Retain necessary complexity behind clear contracts; remove unnecessary complexity.

## Structure and session agreement

The outer loop chooses the work. The inner loop carries that work through investigation,
experimentation, interpretation, and consolidation into an engineering result.

```text
Outer loop: ORIENT → SELECT → [development cycle] → REASSESS → ORIENT

One development cycle:
    FRAME → EXECUTE → INTERPRET → CONSOLIDATE
       ↑       ↑          │           │
       └───────┴──────────┴───────────┘  return as needed within the same cycle

Cycle ends → append one session-log entry → REASSESS
```

A **cycle** is one chosen piece of work pursued to a coherent result or an explicit
rejection or incomplete outcome. It may contain many experiments, implementation attempts,
and revisions of the hypothesis. Returning to an earlier inner phase does not start a
new cycle. A promising benchmark alone does not finish a cycle.

Before execution, establish goals, priorities, scope, acceptance criteria, protected
behavior, and any time/resource budget with the user. Create the initial agreement in
`./sessions/YY-MM-DD/<name>.md`, relative to the **monorepo root**, using the format below.
Thereafter, update that document only when a cycle ends. Preserve unrelated work and an
identifiable accepted baseline. Follow repository instructions and use `bun design-docs`
to read applicable contracts before implementation.

## Outer phase — ORIENT

Build a working model of established facts, correctness gaps, suspected bottlenecks,
and opportunities. Consider individual components and the larger paths containing them:

- **Reference targets:** What does a comparable implementation demonstrate is attainable?
  Establish a trustworthy standard path before relying on it to judge custom execution.
- **Theoretical ceilings:** What follows from unavoidable work, dependencies, and resource
  capacity? State the algorithm and assumptions; challenge whether that work is unavoidable.
- **Further headroom:** What could different algorithms, fusion, reuse, overlap, or data
  representations achieve? Observed rates describe current execution, not its potential.

Account for relevant physical traffic, compute, I/O, dispatch, synchronization, and state
movement. Compose costs according to dependencies and shared resources: add serial work,
justify overlap, and count shared work once. Estimate recoverable time in the enclosing
workload, not just local speedup. For speculation, include draft, verify, repair, and
orchestration cost per committed output; acceptance rate alone does not measure benefit.

If a component approaches its modeled ceiling, consider changing the algorithm or
composition to raise it. If a large gap remains, distinguish inefficient execution from
a mistaken model. References are not absolute ceilings; saturation requires a defensible
bound for the same work and operating point. Analyze far enough to choose useful work.

## Outer phase — SELECT

Choose a worthwhile outcome to pursue in the next cycle without committing prematurely
to a particular fix. Prioritize correctness and unmet dependencies, then potential
end-to-end benefit, information value, and relevance to the objective relative to effort
and complexity. Define enough scope to produce a coherent result.

Consider both local improvements and substantially different approaches. Ask what could
be eliminated or reorganized if the current implementation were not the starting point.
Explore larger leaps when they offer significant headroom, remove structural costs, or
replace patchwork with coherent contracts. Local tuning need not be exhausted first.

Enter the cycle with a question, why it matters, and a plausible first approach. Keep
alternative opportunities available; new evidence can change the plan.

## Inner phase — FRAME

Define **observation → hypothesis → intended result → discriminating experiment →
protected behavior**. Identify competing explanations and what would support or contradict
them. Specify comparison boundaries and correctness gates before judging a candidate.

Choose the appropriate scale of work. A bounded prototype can test an ambitious design's
decisive assumption. A necessary coupled redesign can be evaluated as one hypothesis;
it need not improve performance at every intermediate edit. Establish intended behavior,
ownership, lifetime, and failure contracts as the design takes shape.

## Inner phase — EXECUTE

Investigate, model, inspect, experiment, prototype, and implement as appropriate. Keep
changes attributable and reversible. Use focused correctness checks and small benchmarks
to resolve the current question. Reuse existing typed subjects; give expensive experiments
a timeout and a decision to resolve. Increase measurement only when it can distinguish
a meaningful effect. Reserve broader testing for consolidation and milestones.

Apply these evidence rules throughout the cycle:

- **Independent controls:** Use an equation or upstream operator for kernels, qualified
  internal MLX execution for model iteration, stock batching for engine comparisons,
  and actual serving paths for agent-visible claims. A control sharing the changed
  mechanism cannot independently validate it.
- **Matched work:** Align artifacts, tokens, precision, state, residency, output demands,
  and generated work. Separate policy comparisons from execution overhead. Fixed-token
  replay can diagnose cost after output divergence; it does not qualify free generation.
  Preserve target sampling and state semantics in speculative methods.
- **Correctness first:** Diagnose numerical divergence against an independent oracle;
  never loosen gates to admit a speedup. Failed gates can yield diagnostic timings,
  but no accepted performance claim.
- **Completed measurement:** Run device benchmarks sequentially, including across agents.
  Record interference and complete device work inside timing. Include or exclude setup
  and compilation consistently. Instrumentation must not add production synchronization.
- **Durable evidence:** Preserve commands, source/dependency/artifact identities, raw
  samples, failures, and conditions. Do not overwrite evidence, attribute old measurements
  to changed code, or select reruns to obtain a favorable result.

## Inner phase — INTERPRET

Explain what the evidence establishes and how it changes the approach. Choose the next
step within the current cycle:

| Finding | Next step |
|---|---|
| Evidence ambiguous | Improve the experiment or inspect the missing mechanism |
| Hypothesis contradicted | Revise the explanation and remove unsupported experimental code |
| Approach promising but incomplete | Pursue the next decisive experiment or implementation step |
| Structural opportunity discovered | Reframe the approach and test the alternative |
| Prerequisite discovered | Pursue it explicitly if it belongs to this cycle; otherwise identify the dependency for reassessment |
| Candidate ready | Enter CONSOLIDATE to refine and qualify the whole result |
| Direction exhausted, blocked, or no longer worthwhile | Enter CONSOLIDATE to clean up and close with the learning or incomplete outcome |

Return to FRAME or EXECUTE as needed. These are iterations inside the same cycle; do not
append a session entry for each experiment. Repeated negative or inconclusive results
require a change in hypothesis, experiment, or direction. Invested effort does not justify
accumulating patches or keeping an investigation open indefinitely.

## Inner phase — CONSOLIDATE

Review the entire accumulated change, including supporting pieces added before the
breakthrough. Remove uncertain mechanisms and test whether the benefit survives. Account
for interactions without crediting every piece with the whole gain. Delete obsolete
variants, switches, caches, heuristics, and scaffolding; preserve their evidence separately.

Refine the implementation into coherent contracts. Neural specialization belongs to
models, physical storage to state, proposal/repair logic to generation methods, and
eligibility/service policy to scheduling. Necessary complexity must have clear ownership
and composition boundaries. Judge engineering quality by the resulting responsibilities
and interactions; the design may be sophisticated when the result requires it.

An accepted implementation passes three gates:

1. **Correctness:** Outputs, state transitions, resource lifetime, and failure behavior
   satisfy their contracts. Update applicable design documents for intentional changes.
2. **Performance:** Confirm the intended benefit and check protected workloads, including
   inputs or shapes not used for tuning. Set regression tolerances before qualification;
   do not hide material regressions in aggregate scores. Trade-offs require acceptance
   within the objective. Correctness repairs can be accepted for their necessity while
   recording their cost rather than claiming a speedup.
3. **Engineering quality:** Every retained mechanism has a demonstrated purpose or is a
   necessary dependency of one. Remove unnecessary complexity and contain required
   complexity. Specialize on justified hardware or tensor properties, never benchmark
   identity or favorable prompt content. Avoid abstractions for hypothetical needs.

Qualify affected modes: prefill/decode, one/many sessions, cold/reused state, changing
batch membership, mixed arrivals, pressure, cancellation, and slow consumers. Speculation
also covers the plain path, unequal acceptance, repair, constraints, and stops. Single-session
kernel evidence cannot qualify scheduling or shared state; inspect batching and reuse
directly. Recheck the final implementation after refinement.

At coherent milestones, broaden model, engine, and serving checks; compare cumulative
results with the session baseline and re-anchor internal controls to stock MLX-VLM.
Revalidate controls when shared dependencies change. Use longer runs for claims about
tails, sustained fairness, churn, or memory stability. Avoid full campaigns after each edit.
When the objective appears complete, run its final acceptance checks here before ending
the cycle so the last entry contains the completion evidence.

Consolidation can require experiments, implementation, and reinterpretation. Return to
FRAME, EXECUTE, or INTERPRET when necessary; the cycle remains open. If abandoning the
approach, remove its experimental changes and retain the useful diagnosis. If stopping
incomplete, make remaining code, uncertainty, cleanup, and qualification explicit.

**Cycle-end boundary:** decide the outcome—accepted, rejected, or incomplete—and leave
an identifiable resulting state. Then append exactly one cycle entry to the session
log using the format below. Only after that append, return to REASSESS. Acceptance
requires all applicable gates; ending a cycle does not itself establish success.

## Outer phase — REASSESS

Use the cycle's outcome to update the engine model: facts, disproved assumptions,
bottlenecks, ceilings, prerequisites, and opportunities. A breakthrough can move the
bottleneck or enable a different architecture; a rejected approach can clarify where to
work next. Choose subsequent work from this new understanding.

Check progress against the objective and remaining budget. Return to ORIENT and SELECT;
these can be brief when little changed. Reserve time for consolidation and final
qualification. Reassessment does not trigger a session-log edit; carry its conclusions
into the next cycle's framing and eventual entry.

## Session record and cycle log

Create the initial agreement once, then append one numbered entry after each cycle ends.
No provisional cycle entries, separate mutable status section, or rewrites of earlier
cycles. Include later user steering or corrections in the next concluded entry. Evidence
files may be produced during execution; the session document is updated only at the
cycle-end boundary.

```markdown
# <Session name>

## Initial agreement

- Objective: user-established goals, priorities, scope, and exclusions.
- Acceptance: success criteria, protected behavior, and unresolved assumptions.
- Budget: authorized time/resource limits, if specified.
- Baseline: source/dependency identity, existing changes, hardware, artifacts,
  references, accepted state, and evidence links.
- Starting assessment: known facts, suspected bottlenecks, headroom, and first direction.

## Cycles

### Cycle 001 — <Chosen piece of work>

- Context: start/end time with timezone; starting source and accepted baseline.
- Intent: chosen outcome, why it mattered, hypotheses, and protected behavior.
- Investigation: significant experiments and implementation attempts; how evidence
  changed the approach. Include rejected ideas, not a transcript of tool calls.
- Result: final changes and source identity, including experimental pieces removed
  and the justification and ownership of necessary complexity retained.
- Validation: correctness and performance evidence, comparison conditions, regression
  checks, raw-result links, and missing qualification.
- Outcome: accepted, rejected, or incomplete; reasoning, accepted state, and any
  unfinished code, cleanup, or blockers.
- Implications: what changed in the bottleneck/headroom model, progress toward the
  objective, and the recommended next direction or exact resumption step.
```

Keep entries concise and use “not applicable” where appropriate. Distinguish measured
facts from hypotheses and label the scope of every accepted improvement.

On resumption, read the agreement and log, then reconcile the last entry with actual
source, evidence, and running experiments. Resume unlogged work within its existing
cycle, or explicitly end it as incomplete after accounting for partial changes. Never
infer acceptance from unlogged code.

Resolve routine decisions autonomously. If one direction is blocked, conclude it and
pursue independent authorized work. Surface decisions outside scope; delegate only when
authorized, with clear ownership and coordinated device access.

At the stopping boundary, finish or safely stop owned experiments and release owned
resources. Conclude the active cycle honestly and append its entry. If already between
cycles, the existing log is the handoff. Declare the objective complete only after final
acceptance checks against the resulting implementation; otherwise report partial progress.
