---
name: explore-codebase
description: When broad or deep investigation of codebase structure, patterns, or mechanisms is needed.
---

# Explore Codebase

Broad or deep investigation of codebase structure, patterns, and mechanisms.

## Approach

Use this skill when you need to understand how something works in the codebase — not just find a file, but understand a system, trace a mechanism, or map a dependency surface.

Match depth to the task:
- **Broad tasks**: Map the relevant structure first, then focus on the areas that matter most.
- **Targeted tasks**: Trace the specific mechanism through real code paths.

Prefer high-signal sources first. Expand scope only when what you've seen is insufficient. Work iteratively — each turn, do a focused piece of real exploration, then let what you find guide what to look at next.

## Delegation

When assigning a worker to explore the codebase, share in your spawn message:

- The specific question to answer or structure to map
- Known starting points: relevant files, modules, entry points
- Required depth — quick structural overview vs thorough mechanism trace
- How findings will be used (feeds into planning, implementation, debugging, etc.)

Expect back: a report that answers the question directly, organized but concise, with file references (not long verbatim content). Written to `$M/reports/` and linked in the worker's message.

## Worker Guidance

Use `tree`, `grep`, and `read` as your primary tools. Use shell only for read-only operations that file tools cannot handle (e.g., `git log`).

Work iteratively. Each turn, do a focused piece of real exploration — read files, search for patterns, map structure. Let what you find guide what you look at next.

When you have enough evidence to answer the question:
1. Write a report to `$M/reports/<descriptive-name>.md`
2. Message the lead with a link to the report

Your report should answer the question directly, reference files using markdown links, and never include long verbatim file content.

## Quality Bar

- The question is answered with evidence from actual code, not inference.
- Key findings reference specific files and line ranges.
- Report is organized and concise — not a dump of everything found.
- Unknowns and gaps are called out explicitly.

## Skill Evolution

Update this skill when:
- Certain areas of the codebase have known structure worth documenting.
- The user has preferences about exploration depth or report format.
- Useful entry points or patterns emerge that speed up future exploration.
