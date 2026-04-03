---
id: refactor
label: Refactor
description: Restructuring, reorganizing, or cleaning up code without changing its external behavior.
allowedAssignees: []
---

<!-- @lead -->

## Suggested decomposition

```
- refactor: {id}
  - research: {id}-research (explorer)
  - plan: {id}-plan (planner)
    - approve: {id}-plan-approve (user)
  - implement: {id}-impl (builder)
    OR
    group: {id}-impl
      - implement: {id}-impl-{scope} (builder) +
  - review: {id}-review (reviewer)
```

## Orchestration procedure

1. Establish complete structural understanding and verification baseline.
   - Deploy explorers to map the full refactor scope: structure, dependencies, callers, consumers, and relevant tests.
   - Clarify with the user which concrete structural problem is being solved.
   - Create a task to record baseline test/typecheck/lint/build outcomes.
2. Produce a behavior-preserving refactor plan.
   - Document current structure, concrete pain points, and target structure.
   - Document invariants that must not change.
   - Break work into ordered, incremental structural steps.
   - Keep behavior changes out of refactor steps.
3. Execute incrementally with verification at every step.
   - Apply one structural change at a time.
   - Re-run baseline validation after each step or batch.
   - If new failures appear, treat as behavior change, revert, and reassess.
4. Validate final parity and close.
   - Confirm final validation matches baseline expectations (same pre-existing failures, no new failures).
   - Confirm no unintended interface or behavior changes were introduced.
   - If any behavior or interface change is needed, escalate as a separate, explicitly approved effort.

## Oversight responsibilities

- Enforce the non-negotiable invariant: no external behavior change without explicit approval.
- Ensure baseline evidence exists before implementation starts.
- Ensure dependency mapping is complete enough to avoid hidden breakage.
- Ensure each implementation step remains small, reversible, and verified.
- Ensure any modified tests are justified as structural accommodation, not behavior masking.
- Surface risk when verification coverage is weak and align with the user before proceeding.

<!-- @criteria -->

## Completion criteria

- [ ] Target structural improvements are implemented as planned.
- [ ] Baseline validation parity is preserved with no new failures.
- [ ] No public interface or behavior changes occurred unless explicitly approved by the user.
- [ ] Refactor outcome and verification evidence are documented clearly.
