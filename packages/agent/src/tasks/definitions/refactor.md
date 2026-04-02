---
id: refactor
label: Refactor
description: Restructuring, reorganizing, or cleaning up code without changing its external behavior.
allowedAssignees: [self]
---

# Refactor

You are refactoring code — restructuring it without changing its external behavior. The constraint is absolute: everything must work exactly the same when you're done.

## Suggested Task Decomposition

```
- refactor (self)
  - research (explorer)
  - plan (planner)
    - approve (user)
  - implement (builder)
    OR
    implement (self)
      - implement: {scope} (builder) +
  - review (reviewer)
```

## Procedure

### Research

Understand the current structure completely before changing anything.

1. Read all the code being refactored — not just the entry point, but the full scope of what will change.
2. Deploy workers to map dependencies: all callers, all consumers, all tests that exercise this code.
3. Clarify the goal with the user — "refactor" is vague. What specific structural problem are they solving?

Establish verification baseline:
- Run existing tests and record which pass. Pre-existing failures are not your concern, but you must not introduce new ones.
- Run type checker, linter, build step. Record baseline.
- If test coverage is thin, surface this to the user — refactoring without verification is high risk.

Research is complete when you can describe: the current structure, its specific problems, every dependency, and the invariants that must not break.

### Plan

Document:
- Current structure and its specific problems (not "it's messy" — what concrete problem does it cause?)
- Target structure and what concretely improves
- Invariants that must be preserved
- Ordered sequence of incremental steps
- Never change behavior and structure in the same step

### Implement

Work incrementally. Every step must leave the codebase fully working.

For each structural change:
1. Make one structural change.
2. Verify baseline still passes.
3. If new failures appear, you changed behavior — revert and reconsider.

Never change behavior and structure simultaneously. If the refactor reveals that behavior NEEDS to change, that's a separate conversation with the user.

### Verify

Full verification suite matches baseline: same passes, same pre-existing failures, no new failures. No unexpected interface changes. If any test needed modification, investigate and justify to the user.

## Completion criteria

- Target structure achieved.
- All baseline tests still pass (no new failures).
- No public interface changes unless explicitly approved by user.
