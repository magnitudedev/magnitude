---
id: group
label: Group
description: Related tasks that should be organized together under a common objective.
allowedAssignees: []
---

<!-- @lead -->

## Suggested decomposition

```
- group: {id}
  - <task-type>: {id}-<scope-a> (<role or user>)
  - <task-type>: {id}-<scope-b> (<role or user>)
  - <task-type>: {id}-integration (<role>) [optional]
  - review: {id}-review (reviewer) [recommended]
```

## Orchestration procedure

1. Define the group objective and boundaries.
   - State the concrete outcome the group must deliver.
   - Define scope boundaries and exclusions so child tasks can be evaluated consistently.
2. Create a complete child-task set.
   - Decompose the objective into children that collectively cover the full scope.
   - Assign clear ownership and expected deliverables per child.
   - Sequence dependencies and identify candidates for parallel execution.
3. Coordinate execution and integration.
   - Track status across children and resolve blockers quickly.
   - Rebalance scope or sequencing when upstream findings change downstream needs.
   - Add integration and review tasks when cross-child coupling introduces risk.
4. Close the group intentionally.
   - Verify all required children are complete.
   - Verify outputs compose into the intended objective without gaps.
   - Confirm any deferred work or accepted risk is explicitly documented.

## Oversight responsibilities

- Maintain decomposition quality so no critical scope is missing or duplicated.
- Maintain dependency clarity so downstream work is not blocked unexpectedly.
- Maintain progress visibility across all children and escalation paths.
- Maintain integration quality by validating that child outputs work together, not just independently.
- Maintain decision traceability when scope, sequencing, or acceptance decisions change.

<!-- @criteria -->

## Completion criteria

- [ ] Child tasks collectively cover the full group objective and required scope.
- [ ] All required child tasks are completed or explicitly dispositioned.
- [ ] Integrated outputs satisfy the group objective without unresolved blockers.
- [ ] Outstanding risks, tradeoffs, and follow-ups are documented clearly.
