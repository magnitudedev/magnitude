---
name: web-research
description: When researching external documentation, APIs, libraries, protocols, frameworks, or services.
---

# Web Research

Research external documentation, APIs, libraries, protocols, frameworks, or services.

## Approach

Primary sources are authoritative: official docs, specs, source code of dependencies. Guessing at external API surfaces causes integration bugs — when the feature or fix depends on third-party behavior, research it from primary sources before building.

**Tactical guidance for the lead:**
- Use web fetch and web search tools as primary tools for external research
- When available, look for and fetch `llms.txt` on documentation sites — it provides a condensed, LLM-friendly overview of the site's documentation
- Check the project's installed package version first — API surfaces differ across versions
- Separate confirmed facts from inferences; cite sources (URLs, doc sections)

Secondary sources (blog posts, Stack Overflow) are useful for context but not for contract details.

## Delegation

Web research is delegated — the worker uses fetch and search tools independently while the lead continues other work. The key factors are framing the question precisely and ensuring the worker knows to prioritize primary sources.

**What to give the worker:** The specific question to answer (what API, what behavior, what contract), known starting points (library name, version, relevant doc URLs, or package paths), what decision the research informs, and any hypotheses to validate or disprove. The more specific the question, the more targeted the research.

**Instruct the worker on tactical approach:** Use web fetch and web search tools as primary tools. Check for and fetch `llms.txt` when available on documentation sites — it's a condensed, LLM-friendly overview that can save significant time. Check the project's installed package version first — API surfaces differ across versions and you need the right one. Separate what they confirmed from primary sources from what they inferred — reports that present inferences as facts create false confidence downstream.

**What to expect back:** A structured report (written to `$M/reports/`) with findings separated into confirmed facts, inferences, and unknowns; evidence with URLs or source references for key conclusions; and a recommendation on whether there's enough information to proceed or whether more research is needed.

**Evaluating the output:** If the report mixes facts and inferences without distinguishing them, send it back — that's a sign the worker isn't being rigorous about source quality. If critical claims lack source references, ask for them. The whole point of research is to replace assumptions with evidence; if the output doesn't do that, it hasn't done its job.

## Completion

The web research is complete when:
- The question is answered from primary sources where possible
- Findings distinguish confirmed facts from inferences
- Source references (URLs, doc sections) are provided for key conclusions
- Output includes actionable recommendations
