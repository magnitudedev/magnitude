---
id: ideate
label: Ideate
description: Explore option space, surface tradeoffs, and produce recommendations — not an executable plan.
allowedAssignees: [planner]
---

<!-- @lead -->

## Inputs to provide the worker
- Decision question to explore, with explicit scope boundaries.
- Constraints and preferences (technical, product, UX, timeline, risk tolerance, maintainability).
- Relevant context: prior decisions, research findings, architecture constraints, known failure modes.
- Intended downstream use of the output (decision meeting, planning input, architecture choice, sequencing).
- Any non-negotiables and evaluation criteria that must shape option comparison.

## Output to expect from the worker
- Multiple viable options described concretely, not abstractly.
- Structured tradeoff analysis covering pros, cons, risks, uncertainty, and operational implications.
- Clear recommendation with explicit decision logic grounded in constraints.
- Explanation of why alternatives were not selected.
- Workspace document suitable for stakeholder decision-making and downstream planning.

## Coordination loop
1. Assign with a concrete decision question, constraints, and success criteria for the ideation output.
2. If context is missing or conflicting, resolve critical ambiguities before deep option analysis.
3. Review option breadth and tradeoff rigor; request deeper analysis where comparison is shallow.
4. Require an explicit recommendation and rationale tied to constraints rather than preference.
5. Close ideation when outputs are decision-ready, then route selected direction into planning/implementation tasks.

<!-- @worker -->

## Objective
- Explore the solution space, compare meaningful alternatives, and produce a recommendation that enables a concrete decision.

## Procedure
1. Frame the decision by restating the question, constraints, success criteria, and key tradeoff dimensions.
2. Generate multiple distinct approaches before converging, including non-obvious or hybrid options when relevant.
3. Analyze each option concretely: how it would work, value, complexity/cost, risks, dependencies, and failure modes.
4. Compare options on common criteria to expose meaningful tradeoffs and key decision pivots.
5. Identify reversible versus hard-to-reverse choices and uncertainty that materially affects selection.
6. Recommend a preferred option (or staged combination), justify the choice, and explain why alternatives were not selected.
7. Document full analysis in a workspace artifact that can be used directly for decision-making.

## Output contract
- Return a decision framing section (question, scope, constraints, success criteria).
- Return an option set with concrete descriptions for each viable approach.
- Return a structured tradeoff comparison (matrix or equivalent).
- Return a recommendation with explicit rationale and non-selected option reasoning.
- Return risks/unknowns plus suggested de-risking steps and confidence notes.
- Return a clear statement of what decision this analysis enables.

<!-- @criteria -->

## Completion criteria
- [ ] Multiple distinct options are explored with concrete descriptions.
- [ ] Tradeoffs are assessed honestly, including downsides, risks, and uncertainty.
- [ ] A clear recommendation is provided with explicit rationale.
- [ ] Analysis depth is sufficient to support an actual decision.
- [ ] A workspace document is produced and shared for downstream use.