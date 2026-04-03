---
id: review
label: Review
description: Completed work that needs verification for correctness, quality, and requirement coverage.
allowedAssignees: [reviewer]
---

<!-- @lead -->

## Inputs to provide the worker
- Canonical requirements source: approved plan, user request, acceptance criteria, or equivalent reference.
- Scope boundaries for review: task IDs, changed files, diff range, and explicitly out-of-scope areas.
- Risk focus areas: correctness edge cases, regressions, security/performance concerns, integration boundaries.
- Required validation expectations: tests/checks to run, environments to use, and any reproducibility notes.
- Decision context for known tradeoffs so the reviewer can distinguish intentional deviations from defects.

## Output to expect from the worker
- Explicit verdict: `pass` or `fail`.
- Findings list where each finding includes:
  - Severity/impact framing.
  - Concrete issue statement.
  - Verifiable evidence (file path + line refs, command/test output, or runtime observation).
  - Specific remediation direction.
- Coverage summary stating what was reviewed and what was not reviewed.
- Validation summary listing checks executed and outcomes.
- Residual risks/unknowns called out separately from confirmed defects.

## Coordination loop
1. Confirm the worker had sufficient context; if context was missing, provide it before accepting any verdict.
2. Validate finding quality; reject vague or unsupported claims and require evidence-backed revisions.
3. Route accepted findings to implementation with explicit remediation instructions and scope.
4. Re-run review after fixes until all required issues are resolved or explicitly accepted with rationale.
5. Do not close on a nominal `pass` unless requirement coverage and required validation were actually demonstrated.

<!-- @worker -->

## Objective
- Determine whether delivered work satisfies stated requirements and quality expectations.
- Surface concrete, evidence-backed defects, risks, and unverified claims.
- Provide a reliable go/no-go signal for lead decision-making.

## Procedure
1. Establish the review baseline by reading requirements/plan, scope boundaries, and stated constraints.
2. Validate requirement coverage by checking each expected behavior for full, partial, or missing implementation.
3. Validate correctness and robustness by inspecting logic paths, state transitions, failure handling, and edge cases.
4. Validate non-regression by running or inspecting relevant tests/checks and confirming adjacent behavior remains intact.
5. Evaluate code quality against local patterns for structure, naming, maintainability, and consistency.
6. Record only evidence-backed findings, and mark uncertain items explicitly as unverified rather than inferred.

## Output contract
- Return a top-level verdict: `pass` only when no unresolved blocking issues remain; otherwise `fail`.
- Return findings as concrete items, each with severity, issue, evidence, and remediation direction.
- Include coverage summary, validation summary, and residual risks/unknowns.
- Explicitly list unverified claims and why they could not be verified.
- Avoid generic advice without a specific defect/risk and evidence trail.

<!-- @criteria -->

## Completion criteria
- [ ] Verdict is explicit (`pass` or `fail`) and consistent with reported findings.
- [ ] Every reported finding is concrete and includes verifiable evidence.
- [ ] Requirement coverage is assessed, including explicit callouts for anything not reviewed.
- [ ] Correctness, regression, and quality checks were performed and summarized.
- [ ] Unresolved issues are either fixed or explicitly accepted by lead/user with rationale.
