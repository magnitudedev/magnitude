---
id: feature
label: Feature
description: New functionality, behavior, or capabilities that don't exist yet.
allowedAssignees: []
---

<!-- @lead -->

## Workflow

Feature work goes through context, design, implementation, and verification.

**Context** — Builders who don't understand the existing code make wrong assumptions. Have explorers look at the relevant parts of the codebase first: entry points, related modules, types, conventions, and how similar things are already done. If the feature involves external APIs or libraries you haven't worked with, research those separately — guessing at API surfaces causes integration bugs.

**Design** — Without a plan, builders have to make design decisions while coding, which is expensive to undo. A good plan names the files that will change, the interfaces involved, what the new behavior should be, and how to verify it works. Plans get better with critique — first drafts miss edge cases that show up under scrutiny. Scope and direction decisions belong to the user. Surface these during planning, before anyone starts building — that's the cheapest place to catch misunderstood intent.

**Implementation** — Builders do their best work when they have a solid plan and codebase context to work from. Workers on independent scopes can run in parallel. Workers whose scopes touch the same interfaces or state need agreed contracts before they start, or their work will conflict at merge time.

**Verification** — The builder and the bugs in their code share the same blind spots. A reviewer working from the plan and requirements sees from a different angle — this catches things the builder structurally cannot, regardless of how careful they were. Route review findings back to implementation until they're resolved.

## Completion

The feature is done when the user's requirements are actually met, not when all subtasks are checked off. Verify the integrated behavior against what was originally requested — gaps are common when work is split across multiple workers. Code should follow the project's existing conventions and maintain quality. Review findings need to be genuinely resolved. The user confirming the feature works is the final signal.

<!-- @criteria -->

## Completion criteria

- [ ] The integrated result satisfies the user's stated requirements and scope boundaries.
- [ ] Correctness, edge cases, and regression checks are verified.
- [ ] Code changes align with codebase conventions and maintain code quality.
- [ ] Review findings are resolved or explicitly accepted.
- [ ] User confirms the feature works as intended.
