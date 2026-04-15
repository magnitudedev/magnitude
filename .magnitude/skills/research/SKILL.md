---
name: research
description: When answering specific questions about code, systems, or technology with evidence-backed findings.
---

# Research

Answer specific questions about code, systems, or technology with concrete, evidence-backed findings.

## Approach

Use this skill when a question needs a thorough answer before work can proceed — not just a quick scan, but an investigation that produces findings you can rely on.

Understand the question and the decision it supports before investigating. First-party code and project docs are primary evidence — they carry higher confidence than secondhand interpretation. External references matter when behavior depends on third-party contracts or standards.

Claims without citations are hard to trust and hard to reuse. Separate facts from inferences to prevent false certainty downstream. Explicit unknowns protect later work from hidden assumptions.

## Delegation

When assigning a worker to research, share in your spawn message:

- The exact research question and the decision it informs
- Starting points: relevant files, modules, docs, APIs, commits, or URLs
- Required depth — quick directional answer vs thorough investigation
- Any hypotheses to validate or disprove

Expect back: a structured report with findings separated into confirmed facts, inferences, and unknowns; evidence for each key finding (file paths, line ranges, URLs); a clear recommendation on whether there's enough to proceed; and substantial findings in a workspace document.

Verify the worker answered the actual question, not an adjacent one. Under-supported claims need deeper investigation, not acceptance.

## Worker Guidance

Understand the question and the decision it supports before investigating. Work iteratively — read files, search for patterns, follow references. Let what you find guide what you look at next.

Deliver a structured report: scope investigated, findings with evidence, implications, unknowns, and recommendations. State whether evidence is sufficient to proceed. Write the report to `$M/reports/` and link it in your message to the lead.

## Quality Bar

- The research question is answered with concrete, evidence-backed findings.
- Findings clearly distinguish facts, inferences, and unresolved unknowns.
- Evidence references are provided for all key conclusions.
- Output includes actionable recommendations and proceed/not-proceed guidance.
- Substantial findings are in a workspace document for reuse.

## Skill Evolution

Update this skill when:
- Certain types of research questions recur — add guidance on where to look first.
- The user has preferences about research depth or report format.
