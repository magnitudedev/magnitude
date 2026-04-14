---
name: explore-docs
description: When researching external documentation, APIs, libraries, or standards.
---

# Explore Docs

Research external documentation, APIs, libraries, or standards.

## Approach

Use this skill when the work depends on understanding something outside the codebase — a library's API, a protocol's spec, a framework's behavior, or a third-party service's contracts.

Guessing at external API surfaces causes integration bugs. When the feature or fix depends on third-party behavior, research it from primary sources before building.

Primary sources are authoritative: official docs, specs, source code of the dependency. Secondary sources (blog posts, Stack Overflow) are useful for context but not for contract details.

## Delegation

When assigning a worker to explore docs, share in your spawn message:

- The specific question to answer (what API, what behavior, what contract)
- Known starting points: library name, version, relevant doc URLs, or package paths
- What decision the research informs
- Any hypotheses to validate or disprove

Expect back: a structured report with findings separated into confirmed facts, inferences, and unknowns; evidence with URLs or source references for key conclusions; and a recommendation on whether there's enough to proceed.

## Worker Guidance

Use web search and fetch as your primary tools for external docs. Check the project's installed package version first — API surfaces differ across versions.

Separate what you confirmed from primary sources from what you inferred. Explicit unknowns protect downstream work from hidden assumptions.

Deliver a structured report: question investigated, findings with source references, implications for the work, unknowns, and recommendations. Write to `$M/reports/` and link in your message.

## Quality Bar

- The question is answered from primary sources where possible.
- Findings distinguish confirmed facts from inferences.
- Source references (URLs, doc sections) are provided for key conclusions.
- Output includes actionable recommendations.

## Skill Evolution

Update this skill when:
- Certain libraries or APIs are used repeatedly — add notes on where their docs live.
- The user has preferences about research depth or source priority.
