---
id: ideate
label: Ideate
description: Explore option space, surface tradeoffs, and produce recommendations — not an executable plan.
allowedAssignees: [planner]
---

<!-- @lead -->

## Scope and inputs

The decision question to explore, with scope boundaries. Constraints and preferences: technical, product, UX, timeline, risk tolerance, maintainability. Relevant context: prior decisions, research findings, architecture constraints, known failure modes. How the output will be used — decision meeting, planning input, architecture choice. Non-negotiables and evaluation criteria.

## Output and coordination

Multiple viable options described concretely. Structured tradeoff analysis: pros, cons, risks, uncertainty, operational implications. Clear recommendation with decision logic grounded in constraints, not preference. Explanation of why alternatives were not selected. Workspace document suitable for decision-making.

Shallow comparisons need deeper analysis before they're decision-ready. Recommendations without constraint-grounded rationale are opinions. Close ideation when outputs support a concrete decision, then route the selected direction into planning or implementation.

<!-- @worker -->

## Approach

Frame the decision before comparing solutions — restate the question, constraints, success criteria, and key tradeoff dimensions. Generate multiple distinct approaches before converging, including non-obvious or hybrid options.

Analyze each option concretely: how it works, value, complexity, risks, dependencies, failure modes. Compare on shared criteria to expose meaningful tradeoffs. Identify which choices are reversible vs hard-to-reverse, and where uncertainty materially affects selection.

Recommend a preferred option with explicit rationale. Explain why alternatives were not selected. Document the full analysis in a workspace artifact for decision-making.

<!-- @criteria -->

## Completion criteria

- [ ] Multiple distinct options are explored with concrete descriptions.
- [ ] Tradeoffs are assessed honestly, including downsides, risks, and uncertainty.
- [ ] A clear recommendation is provided with explicit rationale.
- [ ] Analysis depth is sufficient to support an actual decision.
- [ ] A workspace document is produced and shared for downstream use.
