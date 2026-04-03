---
id: feature
label: Feature
description: New functionality, behavior, or capabilities that don't exist yet.
allowedAssignees: []
---

<!-- @lead -->

## Suggested decomposition

```
- feature: {id}
  - research: {id}-research (explorer)
    OR
    group: {id}-research
      - research: {id}-research-{area} (explorer) +
  - plan: {id}-plan (planner)
    - approve: {id}-plan-approve (user)
  - implement: {id}-impl (builder)
    OR
    group: {id}-impl
      - implement: {id}-impl-{scope} (builder) +
  - review: {id}-review (reviewer)
```

## Orchestration procedure

1. Establish requirements and context before design.
   - Deploy explorers to map relevant code: entry points, related modules, types, patterns, conventions, and integration points.
   - Clarify requirements with the user, including explicit in-scope and out-of-scope boundaries.
   - If external APIs or unfamiliar technologies are involved, create research tasks rather than assuming API surfaces.
2. Produce and align on an execution plan.
   - Decompose work into child tasks with clear ownership and sequencing.
   - Document functional behavior, integration design, file-level changes, and verification strategy.
   - Iterate with the user until plan direction is approved.
3. Coordinate implementation across scoped child tasks.
   - Parallelize where safe, and sequence where dependencies require it.
   - Ensure implementation follows discovered codebase conventions so the feature integrates naturally.
4. Drive review and closure.
   - Run independent review for requirement coverage, correctness, edge cases, regressions, and code quality.
   - Route findings back to implementation and repeat review until resolved or explicitly accepted.

## Oversight responsibilities

- Maintain scope control: prevent accidental expansion beyond agreed feature boundaries.
- Maintain pattern consistency: enforce alignment with existing architecture and conventions.
- Maintain integration quality: verify all touched interfaces and dependent flows behave correctly.
- Maintain iteration discipline: ensure reviewer findings are concrete and fully closed before completion.
- Maintain stakeholder alignment: surface tradeoffs and decisions, and keep the user informed at key gates.

<!-- @criteria -->

## Completion criteria

- [ ] All planned child tasks are completed and reviewed.
- [ ] The integrated result satisfies the user's stated requirements and scope boundaries.
- [ ] Correctness, edge cases, and regression checks meet project quality expectations.
- [ ] User confirms the feature works as intended.
