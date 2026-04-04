---
id: bug
label: Bug
description: Unexpected behavior, errors, test failures, or regressions from previously working functionality.
allowedAssignees: []
---

<!-- @lead -->

## Workflow

Bug work goes from symptom to cause to fix to verification.

**Reproduction** — Get the bug to happen reliably first. A reproducible failure is the foundation for everything else — without it, diagnosis is guesswork and you can't verify the fix actually works. If reproduction is inconsistent, that tells you something: it's probably timing, environment, data shape, or shared state.

**Diagnosis** — Collect concrete evidence: logs, traces, return values, state at each step. Each observation rules out possible causes. Test hypotheses against the code, don't just reason about them. Falsified hypotheses are valuable — they stop you from going in circles. Diagnosis is done when you can trace from the trigger to the defect mechanism with evidence at each step.

**Fix** — The smallest fix that addresses the proven cause carries the least risk. Bigger "while we're here" changes make it hard to tell what actually fixed the bug. If there are multiple valid approaches with different tradeoffs, surface those to the user before picking one.

**Verification** — Before writing the fix, write a test that reproduces the failure and confirm it fails. Then apply the fix and confirm the test passes. This red-green sequence proves the fix actually addresses the defect — a test written after the fix can accidentally pass for the wrong reasons. Then check that nearby behavior still works and existing tests still pass.

## Decomposition

Separating diagnosis from implementation works well for bugs. A debugger establishing reproduction and root cause operates differently from a builder writing the fix — combining them tends to shortcut the evidence chain. Review adds value because independent verification breaks the confirmation bias from whoever diagnosed and fixed the issue.

## Completion

The bug is done when the user's requirements around the failure are satisfied. The reproduction case flipping to success is necessary but not sufficient — adjacent behavior should still work, code quality should hold up, and the fix should follow codebase conventions.

<!-- @criteria -->

## Completion criteria

- [ ] Root cause is identified with explicit supporting evidence.
- [ ] A minimal fix addressing the root cause is applied.
- [ ] The reproduction case flips from failure to success.
- [ ] Baseline validation confirms no regressions were introduced.
- [ ] The fix aligns with codebase conventions and does not degrade code quality.
- [ ] User requirements related to the bug are fully satisfied.
