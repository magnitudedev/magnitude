---
name: implement
description: When making concrete code changes with a clear objective or plan to follow.
---

# Implement

Concrete code changes with a clear objective or plan to follow.

## Approach

Use this skill when the what and how are already decided — there's a plan, a clear objective, and defined scope. Implementation is about executing precisely, not making design decisions.

Confirm the objective, scope, and constraints before starting. Identify the exact files and modules. Apply the minimum change needed. Incremental targeted edits are easier to verify and review than broad sweeps.

Tests and checks matched to changed behavior provide the strongest verification signal. Adding or updating tests for new behavior is part of the implementation, not a follow-up. Incidental drift, convention violations, and unhandled error paths create review churn and maintenance cost.

Ambiguous objectives or plans need resolution before implementation starts — making design decisions in-flight is expensive to undo. Scope should stay controlled; unrelated edits create review noise and risk.

## Delegation

When assigning a worker to implement, share in your spawn message:

- The implementation objective with clear success outcomes
- The plan or reference to execute
- Scope boundaries: files likely to change and areas explicitly out of scope
- Constraints: architecture patterns, compatibility, performance, security
- Verification expectations: tests to run, scenarios to validate

Expect back: completed code changes, a summary of what changed and why, verification evidence (tests run and results), and any assumptions, tradeoffs, or blockers.

## Quality Bar

- Required implementation scope is completed or explicitly deferred with rationale.
- Code changes are coherent and constrained to intended scope.
- Changes follow codebase conventions and maintain code quality.
- Required tests/checks were run and passed, or gaps are documented.
- Assumptions, limitations, and residual risks are reported.
- Output is review-ready with sufficient context for independent verification.

## Skill Evolution

Update this skill when:
- The user expresses preferences about implementation style or scope discipline.
- Specific conventions or patterns emerge that all implementations should follow.
- A verification step is repeatedly missed — make it explicit.
