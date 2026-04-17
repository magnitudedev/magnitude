---
name: review
description: When verifying completed work for correctness, quality, and requirement coverage.
---

# Review

Verify completed work for correctness, quality, and requirement coverage.

## Approach

Review is independent verification. The reviewer's job is to find problems, not confirm success. Assume nothing works until there is proof. When in doubt, report the issue.

The lead should ensure the reviewer reads the requirement baseline (plan, scope, constraints) before evaluating code. The reviewer should check each expected behavior for full, partial, or missing implementation; inspect logic paths, state transitions, failure handling, and edge cases; run or inspect relevant tests; and confirm adjacent behavior remains intact. Evidence-backed findings are actionable; vague concerns are not. Severity reflects impact, not emphasis. Pass means no unresolved blocking issues remain.

## Delegation

Review is delegated, and the reviewer should be someone other than the person who built the thing being reviewed. The builder and their bugs share the same blind spots — a fresh pair of eyes catches what the builder structurally cannot.

**Why independence matters:** If the reviewer is the same worker who built the thing, the review is theater. They'll see what they intended, not what's actually there. Use a different worker.

**What to give the reviewer:** The canonical requirements source — approved plan, user request, or acceptance criteria. Without this, the reviewer can only judge code quality, not whether the right thing was built. Also share: scope (changed files, diff range, areas out of scope), risk focus areas (correctness edge cases, regressions, security, performance, integration boundaries), required validation (tests to run, what to check), and decision context for any known tradeoffs so the reviewer can distinguish intentional decisions from defects.

**What to expect back:** An explicit verdict (pass or fail) — not hedging. Findings with severity and evidence (file + line refs, test output, specific commands run). Coverage summary (what was reviewed, what was not). Residual risks called out separately.

**Handling findings:** Vague or unsupported findings need evidence before you accept them. Concrete findings get routed back to implementation for fixes. After fixes, re-review until issues are resolved or explicitly accepted with rationale. A pass verdict is only meaningful when the reviewer actually demonstrated requirement coverage and validation — a pass without evidence is just an opinion.

## Completion

The review is complete when:
- Verdict is explicit (pass or fail) and consistent with reported findings
- Every reported finding is concrete and includes verifiable evidence
- Requirement coverage is assessed, including callouts for anything not reviewed
- Correctness, regression, and quality checks were performed and summarized
- Unresolved issues are either fixed or explicitly accepted with rationale
