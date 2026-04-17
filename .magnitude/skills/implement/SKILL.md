---
name: implement
description: When making concrete code changes with a clear objective or plan to follow.
---

# Implement

Concrete code changes with a clear objective or plan to follow.

## Approach

Implementation is about executing precisely, not making design decisions. There's a plan, a clear objective, and defined scope — the what and how are already decided.

The lead should ensure the objective, scope, and constraints are confirmed before work begins. The worker should identify exact files and modules, apply the minimum change needed, and use incremental targeted edits that are easier to verify and review than broad sweeps. Tests and checks matched to changed behavior provide the strongest verification signal.

Ambiguous objectives or plans need resolution before implementation starts — making design decisions in-flight is expensive to undo. Scope should stay controlled; unrelated edits create review noise and risk.

## Delegation

Implementation is delegated work, but it's not a standalone skill — it exists in the context of a larger workflow (bug fix, feature, refactor). The plan or objective comes from upstream; the implementation worker's job is to execute it precisely.

**What to give the worker:** The implementation objective with clear success outcomes, the plan or reference to execute, scope boundaries (files likely to change and areas explicitly out of scope), constraints (architecture patterns, compatibility, performance, security), and verification expectations (tests to run, scenarios to validate). The more concrete the plan, the better the implementation. Vague plans produce vague implementations — if the plan leaves design decisions open, resolve them before delegating, or expect the worker to surface ambiguities rather than guess.

**When implementation can be parallelized:** If the plan identifies independent scopes (separate modules, no shared state, no overlapping files), different workers can implement them simultaneously. Workers whose scopes touch the same interfaces or shared state need agreed contracts before they start — without coordination, their work will conflict.

**When workers hit ambiguity:** They should surface it rather than guess. An implementation that deviates from the plan because the worker "figured it out" creates risk — the deviation may be fine, or it may undermine something the plan accounted for. Encourage workers to message you when they hit unclear territory.

**What to expect back:** Completed code changes, a summary of what changed and why, verification evidence (tests run and results), and any assumptions, tradeoffs, or blockers encountered.

## Completion

The implementation is complete when:
- Required implementation scope is completed or explicitly deferred with rationale
- Code changes are coherent and constrained to intended scope
- Changes follow codebase conventions and maintain code quality
- Required tests/checks were run and passed, or gaps are documented
- Assumptions, limitations, and residual risks are reported
- Output is review-ready with sufficient context for independent verification
