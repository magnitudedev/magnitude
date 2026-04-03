---
id: research
label: Research
description: Questions about code, systems, or technology that need concrete answers before proceeding.
allowedAssignees: [explorer]
---

<!-- @lead -->

## Inputs to provide the worker
- The exact research question(s) to answer and the decision they inform.
- Starting artifacts: relevant files, modules, docs, APIs, commits, or URLs.
- Required depth (quick directional answer vs deep investigation).
- Constraints: timeline, risk tolerance, and acceptable uncertainty level.
- Any hypotheses to validate or disprove.

## Output to expect from the worker
- A structured research report covering investigated scope, findings, and implications.
- Evidence-backed conclusions with file/line references and URLs where applicable.
- Clear separation of confirmed facts, inferences, and unresolved unknowns.
- Recommended next actions and whether evidence is sufficient to proceed.
- A workspace document when findings are substantial or reused across tasks.

## Coordination loop
1. Check that the worker answered the stated question, not an adjacent one.
2. Validate evidence quality; return for deeper investigation if claims are under-supported.
3. Use findings to drive planning/implementation decisions and cite the research output.
4. If uncertainty remains high, commission focused follow-up research on unresolved unknowns.

<!-- @worker -->

## Objective
- Produce decision-ready, evidence-backed answers to the research question, with explicit uncertainty handling.

## Procedure
1. Parse the question and success condition before starting investigation.
2. Inspect first-party code and project docs directly for primary evidence.
3. Use external references when needed for third-party behavior, APIs, or standards.
4. Record findings with source citations as you go.
5. Separate facts, conclusions, and open questions explicitly.
6. Synthesize implications for the current decision and propose next actions.

## Output contract
- Provide a structured report with: scope investigated, findings, implications, unknowns, and recommendations.
- Cite evidence for each key finding (file path + line range, and/or URL).
- State confidence level and whether evidence is sufficient to proceed now.
- Include a workspace document link when output is substantial or cross-cutting.

<!-- @criteria -->

## Completion criteria
- [ ] The research question is answered with concrete, evidence-backed findings.
- [ ] Findings clearly distinguish facts, inferences, and unresolved unknowns.
- [ ] Evidence references are provided for all key conclusions.
- [ ] Output includes actionable recommendations and proceed/not-proceed guidance.
