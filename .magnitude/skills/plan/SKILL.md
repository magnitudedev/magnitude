---
name: plan
description: When a concrete, implementation-ready design or strategy is needed before building begins.
---

# Plan

Produce a concrete, implementation-ready design or strategy before building begins.

## Approach

Use this skill when work is complex enough that building without a plan risks expensive rework. The plan is the contract that builders will follow — it must be specific enough to implement without ambiguity.

A good plan names the files that will change, the interfaces involved, the ordered steps, and how to verify the result. Plans that stay abstract — not naming files or interfaces — aren't ready for implementation.

Open questions need resolution before builder handoff. Critical unknowns go to research or scan tasks first, then the planner revises. The approved plan becomes the baseline for both implementation and review.

## Delegation

When assigning a worker to plan, share in your spawn message:

- Problem statement and desired outcomes
- Requirements and constraints: functional, technical, compatibility, performance
- Relevant codebase context — files, modules, existing patterns, prior decisions, known limitations
- Prior research or scan findings
- Decision priorities and acceptable tradeoffs

Expect back: a concrete plan with scope boundaries, recommended approach with rationale, ordered steps with specific files and modules, verification strategy tied to requirements, and open questions requiring decisions.

Plans get better with critique — review the plan against the user's actual requirements. If the plan misunderstands the requirements, this is where it gets caught, before any building starts.

## Worker Guidance

Requirements interpretation anchors all downstream decisions — restate goals, success criteria, and constraints before designing. Concrete file and module touchpoints reduce implementation variance. Verification design is part of implementation design — untestable requirements are unstable.

Analyze approaches and select one with explicit tradeoff reasoning. Define ordered steps with concrete impact. Document assumptions, risks, and unresolved questions — distinguish blocking decisions from in-flight ones.

Write the plan to `$M/plans/` and link it in your message to the lead.

## Quality Bar

- Plan provides a clear, ordered, implementation-ready path.
- Scope, approach rationale, and verification strategy are concrete and internally consistent.
- Affected files/components and sequencing dependencies are identified.
- Risks, assumptions, and open questions are explicitly documented.
- Lead confirms the plan is actionable for implementation without additional discovery.

## Skill Evolution

Update this skill when:
- The user has preferences about plan format or depth.
- Plans consistently miss a certain type of consideration — add it explicitly.
