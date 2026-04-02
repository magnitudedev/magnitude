---
id: feature
label: Feature
description: New functionality, behavior, or capabilities that don't exist yet.
allowedAssignees: []
---

# Feature

You are implementing a new feature — introducing novel behavior into the codebase.

## Suggested Task Decomposition

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

## Procedure

### Research

Understand the codebase and the problem space before designing anything.

1. Read files directly that are clearly relevant — entry points, related modules, types, config.
2. Deploy workers to explore broader context: existing patterns, conventions, similar features, integration points.
3. While workers run, ask the user to clarify requirements and scope — what exactly should this do, and what should it NOT do?

If the feature involves unfamiliar libraries, external APIs, or technology choices — research them. Do not guess at API surfaces or design around assumptions about how a library works.

Research is complete when you can answer:
- What does the feature do, precisely?
- Where does it integrate with existing code?
- What existing patterns and conventions must it follow?
- What is explicitly out of scope?

### Plan

Decompose into child tasks. Create a plan covering:
- Functional behavior and integration design
- Scope boundaries (what it does NOT do)
- File-level changes
- Verification strategy

Iterate with the user until they agree with the approach.

### Implement

Deploy workers for parallelizable work. Follow the patterns and conventions identified during research — the feature should look like it belongs in this codebase.

### Review

Review the integrated result for correctness, requirement coverage, edge cases, regressions, and code quality. Iterate with implementer until resolved.

## Completion criteria

- All child tasks completed and reviewed.
- Integrated result meets user's stated requirements.
- User has confirmed the feature works as intended.
