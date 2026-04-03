---
id: plan
label: Plan
description: Work that needs a concrete design or implementation strategy before building.
allowedAssignees: [planner]
---

<!-- @lead -->

## Inputs to provide the worker
- Problem statement and desired outcomes, including what success means operationally.
- Requirements and constraints: functional, technical, compatibility, security, performance, and timeline boundaries.
- Relevant codebase and system context: files/modules, existing patterns, prior decisions, and known limitations.
- Existing evidence: prior research/scan/diagnose outputs, failed attempts, and open questions.
- Decision priorities and risk posture: what to optimize for and what tradeoffs are acceptable.

## Output to expect from the worker
- Concrete, execution-ready plan with:
  - Requirements interpretation and scope boundaries.
  - Recommended approach and tradeoff rationale.
  - Ordered implementation steps with file/module touchpoints.
  - Verification strategy tied to requirements.
  - Dependencies, assumptions, risks, and rollback/mitigation notes where relevant.
  - Explicit open questions that require lead/user decisions.
- Plan version that is actionable by implementers without inventing missing steps.

## Coordination loop
1. Evaluate plan concreteness; reject broad strategy that lacks ordered, file-aware execution detail.
2. Validate alignment with requirements and constraints; resolve inconsistencies before handoff.
3. Resolve or escalate open decisions (lead/user) prior to implementation start.
4. If critical unknowns remain, commission prerequisite research/scan/diagnose and request plan revision.
5. Lock approved plan scope/version for implementation and use it as review baseline.

<!-- @worker -->

## Objective
- Convert requirements and constraints into a concrete, executable implementation strategy.
- Reduce ambiguity and decision risk before implementation begins.
- Produce a plan that supports predictable build and review cycles.

## Procedure
1. Clarify objective by restating goals, success criteria, and hard constraints from provided context.
2. Analyze solution approaches and select a recommended path with explicit tradeoff reasoning.
3. Define ordered implementation steps with concrete file/module impact and sequencing dependencies.
4. Define verification strategy that maps each major requirement to tests/checks and acceptance signals.
5. Document assumptions, risks, and unresolved questions; separate blocking decisions from in-flight decisions.
6. Validate internal consistency across scope, approach, sequence, and verification before delivering.

## Output contract
- Return requirements interpretation and scope definition.
- Return recommended approach with rationale and key tradeoffs.
- Return ordered implementation steps with concrete touchpoints and dependency notes.
- Return requirement-linked verification strategy.
- Return risks, assumptions, dependencies, and open decision items with recommended resolution paths.
- If context is insufficient, state assumptions explicitly and identify required follow-up discovery.

<!-- @criteria -->

## Completion criteria
- [ ] Plan provides a clear, ordered, implementation-ready path.
- [ ] Scope, approach rationale, and verification strategy are concrete and internally consistent.
- [ ] Affected files/components and sequencing dependencies are identified with sufficient specificity.
- [ ] Risks, assumptions, and open questions are explicitly documented and prioritized.
- [ ] Lead confirms the plan is actionable for implementation without additional discovery.
