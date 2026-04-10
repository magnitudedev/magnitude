---
id: refactor
label: Refactor
description: Restructuring, reorganizing, or cleaning up code without changing its external behavior.
allowedAssignees: []
---

<!-- @lead -->

## Workflow

Refactor work changes structure without changing behavior. The whole process is anchored to proving that behavior stayed the same.

**Baseline** — Before changing anything, capture the current state of tests, typechecks, lint, and builds. This is your reference point. Without a baseline, "no behavior change" is just a claim with no way to check it.

**Context** — Structural changes ripple. Have explorers map the dependency surface: who calls what, who imports what, what tests cover what. The most common source of hidden breakage in refactors is dependencies you didn't know about.

**Design** — A refactor plan describes the current structure, the target structure, and what defines behavior parity between them. Break the work into small structural moves with verification between each one — that keeps every step reversible. If the refactor involves interface changes or tradeoffs that affect others, surface those to the user before starting.

**Implementation** — Small moves, verified between each one. If tests fail after a structural change, that's a behavior change — treat it as one. Don't update tests to make them pass unless you can explain why the new behavior is correct and the user agrees.

**Verification** — Final state should match the baseline. Same pre-existing failures, no new ones. Any behavior or interface change that crept in during the refactor needs to be called out explicitly and approved separately.

## Completion

The refactor is done when the target structure is in place and baseline parity is confirmed. Code quality should be better than before — that's the point. Any behavior changes that surfaced need to be explicitly approved, not silently shipped. User requirements for the refactor are satisfied.

<!-- @criteria -->

## Completion criteria

- [ ] Target structural improvements are implemented as planned.
- [ ] Baseline validation parity is preserved — no new failures.
- [ ] No public interface or behavior changes occurred unless explicitly approved.
- [ ] Code quality is maintained or improved.
- [ ] User requirements for the refactor are satisfied.
