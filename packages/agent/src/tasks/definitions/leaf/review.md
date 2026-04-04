---
id: review
label: Review
description: Completed work that needs verification for correctness, quality, and requirement coverage.
allowedAssignees: [reviewer]
---

<!-- @lead -->

## Scope and inputs

Canonical requirements source: approved plan, user request, or acceptance criteria. Scope: task IDs, changed files, diff range, and areas explicitly out of scope. Risk focus areas: correctness edge cases, regressions, security, performance, integration boundaries. Required validation: tests to run, environments to use. Decision context for known tradeoffs so the reviewer can distinguish intentional deviations from defects.

## Output and coordination

Explicit verdict: pass or fail. Findings with severity, concrete issue statement, evidence (file + line refs, test output, runtime observation), and remediation direction. Coverage summary: what was reviewed and what wasn't. Validation summary: checks run and outcomes. Residual risks called out separately from confirmed defects.

Vague or unsupported findings need evidence before acceptance. Route accepted findings to implementation with remediation scope. Re-run review after fixes until issues are resolved or explicitly accepted with rationale. A pass verdict is only meaningful when requirement coverage and validation were actually demonstrated.

<!-- @worker -->

## Approach

The requirement baseline defines what "correct" means — read the plan, scope, and constraints before evaluating code. Check each expected behavior for full, partial, or missing implementation. Inspect logic paths, state transitions, failure handling, and edge cases. Run or inspect relevant tests and confirm adjacent behavior remains intact. Evaluate code quality against local patterns for structure, naming, maintainability, and consistency.

Evidence-backed findings are actionable; vague concerns are not. Severity reflects impact, not emphasis. Mark uncertain items explicitly as unverified rather than presenting them as confirmed defects. Pass means no unresolved blocking issues remain.

<!-- @criteria -->

## Completion criteria

- [ ] Verdict is explicit (pass or fail) and consistent with reported findings.
- [ ] Every reported finding is concrete and includes verifiable evidence.
- [ ] Requirement coverage is assessed, including callouts for anything not reviewed.
- [ ] Correctness, regression, and quality checks were performed and summarized.
- [ ] Unresolved issues are either fixed or explicitly accepted with rationale.
