---
name: refactor
description: When restructuring, reorganizing, or cleaning up code without changing its external behavior.
---

# Refactor

Restructure, reorganize, or clean up code without changing its external behavior.

## Approach

Refactor work changes structure without changing behavior. The whole process is anchored to proving that behavior stayed the same.

**Baseline** — Before changing anything, capture the current state of tests, typechecks, lint, and builds. This is your reference point. Without a baseline, "no behavior change" is just a claim with no way to check it.

**Context** — Structural changes ripple. Map the dependency surface first: who calls what, who imports what, what tests cover what. The most common source of hidden breakage in refactors is dependencies you didn't know about.

**Design** — A refactor plan describes the current structure, the target structure, and what defines behavior parity between them. Break the work into small structural moves with verification between each one — that keeps every step reversible. If the refactor involves interface changes or tradeoffs that affect others, surface those to the user before starting.

**Implementation** — Small moves, verified between each one. If tests fail after a structural change, that's a behavior change — treat it as one. Don't update tests to make them pass unless you can explain why the new behavior is correct and the user agrees.

**Verification** — Final state should match the baseline. Same pre-existing failures, no new ones. Any behavior or interface change that crept in during the refactor needs to be called out explicitly and approved separately.

## Delegation

- **scan/explore-codebase** — Map the dependency surface before planning. What calls what, what imports what, what tests cover the affected area.
- **plan** — Design the structural moves and verification strategy. Share the baseline state and exploration findings.
- **implement** — Execute the plan in small, verifiable steps. Share the plan and baseline explicitly.
- **review** — Verify behavior parity against the baseline. Share the baseline state and the refactor plan.

## Quality Bar

- Target structural improvements are implemented as planned.
- Baseline validation parity is preserved — no new failures.
- No public interface or behavior changes occurred unless explicitly approved.
- Code quality is maintained or improved.
- User requirements for the refactor are satisfied.

## Skill Evolution

Update this skill when:
- A refactor pattern recurs in this project — capture the approach.
- The user has preferences about what counts as acceptable behavior change.
- Specific areas of the codebase have known fragility that warrants extra care.
