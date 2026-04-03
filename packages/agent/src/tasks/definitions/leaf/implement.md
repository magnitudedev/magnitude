---
id: implement
label: Implement
description: Concrete code changes with a clear objective or plan to follow.
allowedAssignees: [builder]
---

<!-- @lead -->

## Inputs to provide the worker
- Implementation objective with clear success outcomes and required behavior changes.
- Authoritative plan/reference to execute (task output, design doc, or explicit ordered steps).
- Scope boundaries: files/components likely to change and areas explicitly out of scope.
- Constraints and standards: architecture patterns, compatibility requirements, performance/security expectations.
- Verification expectations: tests/checks to run, scenarios to validate, and any environment constraints.
- Priority/risk guidance for tradeoff decisions when constraints conflict.

## Output to expect from the worker
- Completed code changes aligned to objective and plan intent.
- Structured change summary describing what changed, where, and why.
- Verification evidence: commands/tests/checks run with pass/fail outcomes.
- Explicit assumptions, tradeoffs, limitations, and follow-up items.
- Clear blocker/escalation notes for anything preventing full completion.

## Coordination loop
1. Validate assignment readiness; if objective/plan is ambiguous, resolve planning gaps before implementation proceeds.
2. Keep scope controlled; reject unrelated edits unless explicitly approved.
3. Review worker output for requirement alignment, plan fidelity, and sufficient verification evidence.
4. Route output to independent review; return implementation follow-ups when review findings surface.
5. Iterate until completion criteria are met or remaining gaps are explicitly accepted with rationale.

<!-- @worker -->

## Objective
- Produce scoped code changes that satisfy the provided objective/plan.
- Preserve system consistency by following established local patterns and constraints.
- Provide verification evidence sufficient for independent review without guesswork.

## Procedure
1. Confirm assignment contract: objective, scope, constraints, and verification expectations; request missing context before broad changes.
2. Map implementation path: identify exact files/modules, dependencies, and minimum viable change sequence.
3. Apply changes incrementally with targeted edits tied directly to required outcomes.
4. Validate behavior by running relevant tests/checks and adding/updating tests when needed for changed behavior.
5. Audit for scope and quality: remove incidental drift, ensure consistency with project conventions, and confirm error-path handling.
6. Prepare delivery summary with verification results, assumptions, and any unresolved risks.

## Output contract
- Return scope recap showing objective coverage and boundary adherence.
- Return change summary grouped by file/area with rationale for each material change.
- Return verification section including commands run, outcomes, and any checks not run with reason.
- Return assumptions/tradeoffs/risks and explicit follow-up recommendations where applicable.
- If blocked, return exact blocker details, attempted mitigations, and decision needed from lead.

<!-- @criteria -->

## Completion criteria
- [ ] Required implementation scope is completed or explicitly deferred/accepted by lead/user.
- [ ] Code changes are coherent and constrained to intended scope.
- [ ] Required tests/checks were run and passed, or unrun checks are explicitly documented and accepted.
- [ ] Assumptions, limitations, and residual risks are explicitly reported.
- [ ] Output is review-ready without missing context needed for independent verification.
