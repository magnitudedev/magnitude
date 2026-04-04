---
id: group
label: Group
description: Related tasks that should be organized together under a common objective.
allowedAssignees: []
---

<!-- @lead -->

## Purpose

A group task is a container for related work that shares an objective. Its value is coordination — making sure the pieces add up to a coherent result, not just a collection of individually completed tasks.

## Decomposition

Break the objective into child tasks that collectively cover the full scope. Each child should have clear ownership and a concrete deliverable. Think about dependencies: which tasks can run in parallel, which ones need to wait on others. Missing a child task means missing part of the scope — gaps tend to surface late when they're expensive to fill.

## Coordination

Track progress across children and unblock things quickly. When one child's output changes what another child needs, adjust. The biggest risk in group work is at the boundaries between children — where their outputs meet. Integration problems show up there, not inside individual tasks. If children touch shared interfaces or produce artifacts that need to compose, verify that they actually fit together.

## Completion

The group is done when the objective is met, not when all children are individually done. Check that the integrated result actually works as a whole. Any deferred work or accepted risk should be documented so it doesn't get lost. User requirements for the group objective are satisfied.

<!-- @criteria -->

## Completion criteria

- [ ] Child tasks collectively cover the full group objective.
- [ ] All required child tasks are completed or explicitly dispositioned.
- [ ] Integrated outputs satisfy the group objective — not just individually correct, but working together.
- [ ] Code quality and codebase conventions are maintained.
- [ ] Outstanding risks and follow-ups are documented.
- [ ] User requirements are satisfied.
