---
name: feature
description: When building new functionality, behavior, or capabilities that don't exist yet.
---

# Feature

Build new functionality, behavior, or capabilities that don't exist yet.

## Approach

Feature work goes through four phases: context, design, implementation, and verification. Skipping phases creates compounding problems — skipping context leads to wrong assumptions, skipping design leads to expensive in-flight decisions, skipping verification leads to gaps between what was built and what was asked for.

The lead should orchestrate each phase, ensuring it completes before the next begins. Context must map the relevant codebase before design starts. Design must produce a concrete, implementation-ready plan before building begins. Implementation must follow the approved plan. Verification must confirm the integrated result meets the original requirements.

## Delegation

Feature work is the most involved delegation pattern — it has the most phases, the most handoffs, and the most room for parallelism. The lead's job is to keep the phases flowing and ensure the final result actually matches what the user asked for.

**Phase 1: Context** — Map relevant code before design begins. Use **scan** or **explore-codebase**. This is fast, focused work — give the worker specific questions about entry points, related modules, existing patterns, and how similar features are already built. The output feeds directly into planning, so emphasize what matters for design decisions.

**Phase 2: Design** — Produce an implementation-ready plan. Use **plan** (or **web-research** if external APIs are involved). Share the user's requirements, constraints, and the context findings. This is where the lead is most involved — review the plan for gaps, push back on unclear parts, and shape it into something the user can evaluate. The plan names files, interfaces, ordered steps, and verification strategy.

**Phase 3: Implementation** — Execute the plan. This is where parallelism can help. If the plan identifies independent scopes (e.g., backend API + frontend component that don't share state), different workers can implement them in parallel. Workers touching shared interfaces or state need agreed contracts before they start, or their work will conflict at merge time. Share the approved plan and relevant context with each worker.

**Phase 4: Verification** — Use **review**. The reviewer should be a different worker than the implementer — they bring fresh eyes. Share the original requirements and the plan as the baseline. The reviewer verifies the integrated result, not just individual pieces — gaps between "each piece works" and "the whole thing does what was asked" are common.

**For simple features**, collapse phases — a combined scan+plan, or a single implement+review pass, may be sufficient. Match depth to complexity.

The biggest risk in feature work is building something that technically checks all the subtask boxes but doesn't actually satisfy the user's original request. The lead should verify the integrated behavior against what was originally asked for, not just mark subtasks complete.

## Completion

The feature is done when:
- The integrated result satisfies the user's stated requirements and scope boundaries
- Correctness, edge cases, and regression checks are verified
- Code changes align with codebase conventions and maintain code quality
- Review findings are resolved or explicitly accepted
- User confirms the feature works as intended
