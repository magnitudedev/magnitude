---
name: tdd
description: When using test-driven development — write a failing test first, then implement to make it pass.
---

# TDD

Test-driven development: write a failing test first, then implement to make it pass.

## Approach

TDD flips the usual implementation order: write a failing test that specifies the desired behavior, confirm it fails for the right reason, then implement the minimum change to make it pass. Red, Green, Refactor.

**Why red first matters.** A test written after the code can pass for the wrong reason — it might be testing something incidental, or the implementation might satisfy the assertion without actually solving the problem. A test written before and confirmed to fail locks in the exact specification. When it flips green, you know the implementation addressed that specific behavior. This is the core value: the failing test is a precise contract between "what's broken" and "what fixed it."

**When TDD is the right tool.** TDD is highly applicable for bug fixes — the reproduction test is the red test, and the fix makes it green. It's also valuable for features or changes where the expected behavior can be clearly specified in advance: pure functions, data transformations, protocol handling, API contracts, state machines. The more testable the behavior — deterministic inputs and outputs, clear boundaries, minimal external dependencies — the more TDD pays off. For exploratory work, UI-heavy features, or situations where the desired behavior isn't well understood yet, TDD adds overhead without proportional benefit; use it selectively on the testable pieces.

**What makes a good red test.** Not every failing test is useful. The test should fail because the desired behavior is absent, not because of a setup error, type mismatch, or wrong import. A good red test fails narrowly — it isolates the specific behavior gap. A test that fails broadly (entire module broken) doesn't tell you what to fix. If the test fails for the wrong reason, the green phase gives false confidence.

**Handling hard-to-test code.** When code depends on external systems, async timing, or shared state, the test itself may need design work — mocks, fixtures, test harnesses, dependency injection. This is part of the TDD cycle, not a reason to skip it. If the code is genuinely untestable without major restructuring, that's a signal: the code's design may need to change before TDD can apply. In that case, consider a refactor-first approach using TDD to lock down existing behavior, then refactor for testability, then TDD the new behavior.

## Delegation

TDD is not a standalone delegated task — it's a discipline you instruct workers to follow within other delegations (bug fixes, feature implementation, refactoring). You don't create a "TDD task"; you tell an implementation worker to use TDD as their approach.

**How to instruct workers to apply TDD:** Tell them the behavior to implement or the bug to fix, the existing test patterns or frameworks to follow, scope boundaries, and what "done" means (the specific assertion that must pass). Then instruct them to: write a failing test that isolates the specific behavior gap (not a broad smoke test), confirm the test fails for the right reason (desired behavior absent, not a setup error), implement only the minimum to make it pass, and verify adjacent tests still pass.

**Critical verification step:** Check that the worker actually confirmed the test fails before implementing. Workers sometimes skip this step — they write the test and the fix together, which defeats the purpose. Ask for confirmation: "Show me the test failing before your fix."

**For bug fixes:** The reproduction test IS the red test — confirm it fails, then the fix is the green phase. This is the most natural application of TDD.

**For features:** TDD the most testable boundaries first (pure functions, data transforms, API contracts). Less testable pieces can follow a standard implementation approach — don't force TDD where it adds friction without value.

**For refactoring:** TDD provides the green/green guarantee — the test suite that passed before the refactor must still pass after. The tests are the specification of behavior; if they still pass, behavior is preserved.

## Completion

The TDD cycle is complete when:
- Failing test is confirmed red before implementation begins, and it fails for the right reason (desired behavior absent, not setup error)
- Implementation is confirmed green — the test passes because the desired behavior is now present
- No regressions in existing tests
- The test isolates the specific behavior gap (narrow failure, not broad)
- For bug fixes: the reproduction test defines exactly what was broken and proves the fix addressed it
