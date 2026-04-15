---
name: review
description: When verifying completed work for correctness, quality, and requirement coverage.
---

# Review

Verify completed work for correctness, quality, and requirement coverage.

## Approach

Review is independent verification. The reviewer's job is to find problems, not confirm success. Assume nothing works until there is proof. When in doubt, report the issue.

The requirement baseline defines what "correct" means — read the plan, scope, and constraints before evaluating code. Check each expected behavior for full, partial, or missing implementation. Inspect logic paths, state transitions, failure handling, and edge cases. Run or inspect relevant tests and confirm adjacent behavior remains intact.

Evidence-backed findings are actionable; vague concerns are not. Severity reflects impact, not emphasis. Pass means no unresolved blocking issues remain.

## Delegation

When assigning a worker to review, share in your spawn message:

- Canonical requirements source: approved plan, user request, or acceptance criteria
- Scope: task IDs, changed files, diff range, and areas explicitly out of scope
- Risk focus areas: correctness edge cases, regressions, security, performance, integration boundaries
- Required validation: tests to run, environments to use
- Decision context for known tradeoffs so the reviewer can distinguish intentional deviations from defects

Expect back: explicit verdict (pass or fail), findings with severity and evidence (file + line refs, test output), coverage summary, validation summary, and residual risks called out separately.

Vague or unsupported findings need evidence before acceptance. Route accepted findings to implementation with remediation scope. Re-run review after fixes until issues are resolved or explicitly accepted with rationale. A pass verdict is only meaningful when requirement coverage and validation were actually demonstrated.

## Worker Guidance

Read the plan, scope, and constraints before evaluating any code. Your job is to find problems, not confirm success.

What counts as evidence:
- Test suite output, build/typecheck output, shell commands that exercise the feature
- Specific code patterns verified by targeted reading with file and line references

What does not count as evidence:
- "X is implemented in file Y" (code existence)
- "The implementation handles edge cases" (vague correctness)
- Any conclusion from reading code without running it for behavioral claims

Cite what you ran or read — specific file paths, line references, command output. If you cannot verify something, report it as unverified and explain why.

Write findings to `$M/reports/` and link in your message to the lead.

## Quality Bar

- Verdict is explicit (pass or fail) and consistent with reported findings.
- Every reported finding is concrete and includes verifiable evidence.
- Requirement coverage is assessed, including callouts for anything not reviewed.
- Correctness, regression, and quality checks were performed and summarized.
- Unresolved issues are either fixed or explicitly accepted with rationale.

## Skill Evolution

Update this skill when:
- The user expresses specific quality standards (e.g., "never use `as any`") — add them here.
- A class of defect keeps slipping through — add it to the focus areas.
- The user has preferences about what counts as sufficient evidence.
