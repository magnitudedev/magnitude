---
id: plan
label: Plan
description: Work that needs a concrete design or implementation strategy before building.
allowedAssignees: [planner]
---

<!-- @lead -->

## Scope and inputs

Problem statement and desired outcomes. Requirements and constraints: functional, technical, compatibility, performance, timeline. Relevant codebase context — files, modules, existing patterns, prior decisions, known limitations. Prior research or scan findings. Decision priorities and acceptable tradeoffs.

## Output and coordination

A concrete, implementation-ready plan: scope boundaries, recommended approach with rationale, ordered steps with specific files and modules, verification strategy tied to requirements, open questions requiring decisions.

Plans that stay abstract — not naming files or interfaces — aren't ready for implementation. Open questions need resolution before builder handoff. Critical unknowns go to research or scan tasks first, then the planner revises. The approved plan version becomes the baseline for both implementation and review.

<!-- @worker -->

## Approach

Requirements interpretation anchors all downstream decisions — restate goals, success criteria, and constraints before designing. Concrete file and module touchpoints reduce implementation variance. Verification design is part of implementation design — untestable requirements are unstable.

Analyze approaches and select one with explicit tradeoff reasoning. Define ordered steps with concrete impact. Define how each requirement will be verified. Document assumptions, risks, and unresolved questions — distinguish blocking decisions from in-flight ones.

Internal consistency matters: scope should match approach, sequence should respect dependencies, verification should cover requirements.

<!-- @criteria -->

## Completion criteria

- [ ] Plan provides a clear, ordered, implementation-ready path.
- [ ] Scope, approach rationale, and verification strategy are concrete and internally consistent.
- [ ] Affected files/components and sequencing dependencies are identified.
- [ ] Risks, assumptions, and open questions are explicitly documented.
- [ ] Lead confirms the plan is actionable for implementation without additional discovery.
