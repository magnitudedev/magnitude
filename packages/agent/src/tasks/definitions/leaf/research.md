---
id: research
label: Research
description: Questions about code, systems, or technology that need concrete answers before proceeding.
allowedAssignees: [explorer]
---

<!-- @lead -->

## Scope and inputs

The exact research question and the decision it informs. Starting points: relevant files, modules, docs, APIs, commits, or URLs. Required depth — quick directional answer vs thorough investigation. Any hypotheses to validate or disprove.

## Output and coordination

A structured answer separating confirmed facts from inferences from unknowns. Evidence for each key finding — file paths, line ranges, URLs. Clear recommendation on whether there's enough to proceed. Substantial findings captured in a workspace document for reuse.

Verify the worker answered the actual question, not an adjacent one. Under-supported claims need deeper investigation, not acceptance.

<!-- @worker -->

## Approach

Understand the question and the decision it supports before investigating. First-party code and project docs are primary evidence — they carry higher confidence than secondhand interpretation. External references matter when behavior depends on third-party contracts or standards.

Claims without citations are hard to trust and hard to reuse. Separating facts from inferences prevents false certainty downstream. Explicit unknowns protect later work from hidden assumptions.

Deliver a structured report: scope investigated, findings with evidence, implications, unknowns, and recommendations. State whether evidence is sufficient to proceed.

<!-- @criteria -->

## Completion criteria

- [ ] The research question is answered with concrete, evidence-backed findings.
- [ ] Findings clearly distinguish facts, inferences, and unresolved unknowns.
- [ ] Evidence references are provided for all key conclusions.
- [ ] Output includes actionable recommendations and proceed/not-proceed guidance.
