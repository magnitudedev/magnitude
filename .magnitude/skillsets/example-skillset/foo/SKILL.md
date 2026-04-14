---
name: foo
description: When building new functionality, behavior, or capabilities that don't exist yet.
---

# Feature

Build new functionality, behavior, or capabilities that don't exist yet.

## Approach

Feature work goes through four phases: context, design, implementation, and verification. Skipping phases creates compounding problems — skipping context leads to wrong assumptions, skipping design leads to expensive in-flight decisions, skipping verification leads to gaps between what was built and what was asked for.

**Context** — Builders who don't understand the existing code make wrong assumptions. Explore the relevant parts of the codebase first: entry points, related modules, types, conventions, and how similar things are already done. If the feature involves external APIs or libraries you haven't worked with, research those separately — guessing at API surfaces causes integration bugs.

**Design** — Without a plan, builders make design decisions while coding, which is expensive to undo. A good plan names the files that will change, the interfaces involved, what the new behavior should be, and how to verify it works. Plans get better with critique — first drafts miss edge cases that show up under scrutiny. Scope and direction decisions belong to the user. Surface these during planning, before anyone starts building.

**Implementation** — Builders do their best work when they have a solid plan and codebase context to work from. Workers on independent scopes can run in parallel. Workers whose scopes touch the same interfaces or state need agreed contracts before they start, or their work will conflict at merge time.

**Verification** — The builder and the bugs in their code share the same blind spots. A reviewer working from the plan and requirements sees from a different angle — this catches things the builder structurally cannot, regardless of how careful they were. Route review findings back to implementation until they're resolved.

## Delegation

Typical worker breakdown for a feature:

- **scan/explore-codebase** — Map relevant code before design begins. Share specific questions about entry points, related modules, and existing patterns.
- **explore-docs** — Research external APIs or libraries if the feature depends on them.
- **plan** — Produce an implementation-ready design. Share the user's requirements, constraints, and any exploration findings. The plan names files, interfaces, and verification strategy.
- **implement** — Execute the plan. Share the approved plan and relevant codebase context. Independent scopes can run in parallel.
- **review** — Verify the integrated result. Share the original requirements and plan as the baseline.

For simple features, collapse phases — a combined scan+plan, or a single implement+review pass, may be sufficient. Match depth to complexity.

## Quality Bar

The feature is done when the user's requirements are actually met, not when all subtasks are checked off. Verify the integrated behavior against what was originally requested — gaps are common when work is split across multiple workers.

- The integrated result satisfies the user's stated requirements and scope boundaries.
- Correctness, edge cases, and regression checks are verified.
- Code changes align with codebase conventions and maintain code quality.
- Review findings are resolved or explicitly accepted.
- User confirms the feature works as intended.

## Skill Evolution

Update this skill when:
- A phase is repeatedly skipped and causes problems — make it explicit.
- A delegation pattern works particularly well for this project — capture it.
- The user expresses preferences about how features should be built or verified.
- Project-specific conventions emerge that should always be followed.
