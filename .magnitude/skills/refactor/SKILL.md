---
name: refactor
description: When restructuring, reorganizing, or cleaning up code without changing its external behavior.
---

# Refactor

Restructure, reorganize, or clean up code without changing its external behavior.

## Approach

The whole process is anchored to proving that behavior stayed the same using **TDD discipline: green/green**.

**Baseline** — Before changing anything, capture the current state of tests. Run the full test suite and record which tests pass and which fail. This is your reference point. Pre-existing failures are fine and expected; what matters is that no NEW failures appear after the refactor. Without a baseline, "no behavior change" is just a claim with no way to check it.

**Context** — Structural changes ripple. Map the dependency surface first: who calls what, who imports what, what tests cover what. The most common source of hidden breakage in refactors is dependencies you didn't know about.

**Design** — A refactor plan describes the current structure, the target structure, and what defines behavior parity between them. Break the work into small structural moves with verification between each one — that keeps every step reversible. If the refactor involves interface changes or tradeoffs that affect others, surface those to the user before starting.

**Implementation** — Small moves, verified between each one. The test suite is your specification of behavior. If tests fail after a structural change, that's a behavior change — treat it as one. Don't update tests to make them pass unless you can explain why the new behavior is correct and the user agrees. Follow TDD green/green: all tests that were green before must still be green after.

**Verification** — Final state should match the baseline. Same pre-existing failures, no new ones. Every test that passed before the refactor still passes after. This is the green/green guarantee — behavior is preserved. Any behavior or interface change that crept in during the refactor needs to be called out explicitly and approved separately.

## Delegation

Refactor follows a similar phased pattern to feature work (context, plan, implement, review), but the emphasis is different — baseline capture and green/green verification are critical, and the implementation must be broken into small verified moves rather than a single push.

**Phase 1: Context** — Use **scan** or **explore-codebase** to map the dependency surface before touching anything. What calls what, what imports what, what tests cover the affected area. The most common source of hidden breakage is dependencies you didn't know about — this phase prevents that.

**Phase 2: Design** — Use **plan**. Share the baseline test state and exploration findings. The plan must describe: current structure, target structure, behavior parity criteria (what "same behavior" means concretely), and the work broken into small structural moves with verification between each. Each move should be reversible on its own.

**Phase 3: Implementation** — Use **implement** or **tdd**. Share the plan and the baseline explicitly. Instruct the worker to follow green/green discipline: after each structural move, run the test suite and confirm every test that passed before still passes. If a test fails, that's a behavior change — stop and surface it, don't silently update the test to pass. Small moves with verification between each keeps every step reversible.

**Phase 4: Review** — Use **review** with a different worker than the implementer. Share the baseline state and the refactor plan. The reviewer's primary question: does every test that passed before still pass after? Any behavior or interface changes need to be called out explicitly.

## Completion

The refactor is complete when:
- Target structural improvements are implemented as planned
- **Green/green is confirmed** — every test that passed before the refactor still passes after
- No public interface or behavior changes occurred unless explicitly approved
- Code quality is maintained or improved
- User requirements for the refactor are satisfied
