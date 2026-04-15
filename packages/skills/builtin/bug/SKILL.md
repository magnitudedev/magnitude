---
name: bug
description: When fixing unexpected behavior, errors, test failures, or regressions from previously working functionality.
---

# Bug

Fix unexpected behavior, errors, test failures, or regressions from previously working functionality.

## Approach

Bug work goes from symptom to cause to fix to verification. The sequence matters — skipping diagnosis leads to fixing the wrong thing, and skipping the reproduction test means you can't prove the fix worked.

**Reproduction** — Get the bug to happen reliably first. A reproducible failure is the foundation for everything else — without it, diagnosis is guesswork and you can't verify the fix actually works. If reproduction is inconsistent, that tells you something: it's probably timing, environment, data shape, or shared state.

**Diagnosis** — Collect concrete evidence: logs, traces, return values, state at each step. Each observation rules out possible causes. Test hypotheses against the code, don't just reason about them. Falsified hypotheses are valuable — they stop you from going in circles. Diagnosis is done when you can trace from the trigger to the defect mechanism with evidence at each step.

**Reproduction test** — Once the cause is identified, write a test that captures the failure and confirm it goes red. This happens *before* any fix. A failing test locks in the exact defect — it removes ambiguity about what "fixed" means, prevents the fix from being declared done when it isn't, and catches regressions permanently. A test written after the fix can accidentally pass for the wrong reasons; a test written before cannot.

**Fix** — The smallest fix that addresses the proven cause carries the least risk. Bigger "while we're here" changes make it hard to tell what actually fixed the bug. If there are multiple valid approaches with different tradeoffs, surface those to the user before picking one.

**Verification** — Apply the fix and confirm the reproduction test flips green. Then check that nearby behavior still works and existing tests still pass.

## Delegation

Separating diagnosis from implementation works well for bugs:

- **debug** — Establish reproduction and root cause. Share the symptom, reproduction steps, error outputs, relevant code areas, and any prior hypotheses.
- **implement** — Apply the fix once root cause is confirmed. Share the diagnosis report, the proposed fix approach, and the reproduction test.
- **review** — Verify the fix independently. Share the original bug report and the reproduction test as the baseline.

For simple bugs, a single worker can handle diagnosis and fix together. Reserve the separation for bugs where the cause is genuinely unknown or the fix is non-trivial.

## Quality Bar

- Root cause is identified with explicit supporting evidence.
- A minimal fix addressing the root cause is applied.
- The reproduction case flips from failure to success.
- Baseline validation confirms no regressions were introduced.
- The fix aligns with codebase conventions and does not degrade code quality.
- User requirements related to the bug are fully satisfied.

## Skill Evolution

Update this skill when:
- A class of bugs keeps appearing — add a note about what to look for.
- The user has preferences about fix scope or approach.
- A debugging technique proves particularly effective for this codebase.
