---
name: bug
description: When fixing unexpected behavior, errors, test failures, or regressions from previously working functionality.
---

# Bug

Fix unexpected behavior, errors, test failures, or regressions from previously working functionality.

## Approach

Bug work follows a sequence from symptom to cause to fix to verification — skipping diagnosis leads to fixing the wrong thing, and skipping reproduction means you cannot prove the fix worked.

The lead should orchestrate this work in phases, ensuring each phase completes before the next begins. Reproduction must be stable before diagnosis proceeds. Diagnosis must identify the root cause with evidence before implementation begins. The fix must be minimal and verified before the work is considered complete.

Bug fixing follows **TDD discipline**: write a failing test first (the reproduction), confirm it fails, then implement the fix to make it green. The reproduction test defines exactly what "done" means — the bug is fixed when that test passes.

## Delegation

Bug work flows through phases that depend on each other — each phase's output is the next phase's input. The lead's primary role is managing handoffs and ensuring each phase is genuinely complete before moving on.

**Simple bugs** (cause is obvious or easy to isolate) can be handled by a single worker end-to-end: reproduce, diagnose, fix, verify. Give them the symptom and let them run.

**Complex bugs** (cause unknown, multiple possible explanations, subtle behavior) benefit from separating diagnosis from implementation. When the same worker diagnoses and fixes, there's a natural bias toward explanations that are convenient to fix — separating the roles gives you more honest diagnosis and more honest verification. Use separate workers for each phase:

1. **Diagnose** — worker's only job is finding the root cause. They should not start fixing. Give them: symptoms, error outputs, reproduction steps if known, relevant code areas. They return: confirmed reproduction steps, root cause with evidence chain (what they checked, what they ruled out, what they found), and a recommended fix direction. Write to `$M/reports/`.

2. **Implement with TDD** — worker gets the diagnosis report and applies the fix following TDD: the reproduction test is the red test, confirm it fails, then implement the fix to make it green. They return: the test (confirmed red then green), a summary of what changed, verification that no regressions were introduced.

3. **Review** (for non-trivial fixes) — a different worker than the implementer independently verifies the fix. Give them: original bug report, reproduction test, the applied fix. They return: verdict (pass/fail) with evidence.

The phases are sequential — diagnosis must complete before implementation starts, implementation must complete before review starts. Parallelism doesn't help here because each phase depends on the prior phase's output.

## Completion

The bug is fixed when:
- Root cause is identified with explicit supporting evidence (not speculation)
- A minimal fix addressing the root cause is applied following TDD discipline (red test first, then green)
- The reproduction case flips from failure to success
- Baseline validation confirms no regressions were introduced
- The fix aligns with codebase conventions and does not degrade code quality
- User requirements related to the bug are fully satisfied
