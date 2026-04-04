---
id: implement
label: Implement
description: Concrete code changes with a clear objective or plan to follow.
allowedAssignees: [builder]
---

<!-- @lead -->

## Scope and inputs

Implementation objective with clear success outcomes. The plan or reference to execute. Scope boundaries: files likely to change and areas explicitly out of scope. Constraints: architecture patterns, compatibility, performance, security. Verification expectations: tests to run, scenarios to validate, environment constraints.

## Output and coordination

Completed code changes matching objective and plan. Change summary: what changed, where, why. Verification evidence: tests/checks run and their results. Assumptions, tradeoffs, limitations, and blockers.

Ambiguous objectives or plans need resolution before implementation starts — builders making design decisions in-flight is expensive to undo. Scope should stay controlled; unrelated edits create review noise and risk. Route output to review; iterate on findings until resolved.

<!-- @worker -->

## Approach

Objective-plan alignment keeps edits relevant — confirm the objective, scope, and constraints before starting. Identify exact files and modules, then apply the minimum change needed. Incremental targeted edits are easier to verify and review than broad sweeps.

Tests and checks matched to changed behavior provide the strongest verification signal. Adding or updating tests for new behavior is part of the implementation, not a follow-up. Incidental drift, convention violations, and unhandled error paths create review churn and maintenance cost.

Report what changed, what was verified, what was assumed, and what's unresolved. If blocked, state the exact blocker, what was attempted, and what decision is needed.

<!-- @criteria -->

## Completion criteria

- [ ] Required implementation scope is completed or explicitly deferred with rationale.
- [ ] Code changes are coherent and constrained to intended scope.
- [ ] Changes follow codebase conventions and maintain code quality.
- [ ] Required tests/checks were run and passed, or gaps are documented.
- [ ] Assumptions, limitations, and residual risks are reported.
- [ ] Output is review-ready with sufficient context for independent verification.
