---
name: explore-codebase
description: When broad or deep investigation of codebase structure, patterns, or mechanisms is needed.
---

# Explore Codebase

Broad or deep investigation of codebase structure, patterns, or mechanisms.

## Approach

The lead should match depth to the task: broad tasks need structural mapping first, then focused deep dives; targeted tasks should trace specific mechanisms through real code paths. Prefer high-signal sources first. Work iteratively — each turn, the worker should do a focused piece of real exploration, then let what they find guide what to look at next.

## Delegation

The lead delegates exploration because it requires reading through the codebase — that's worker work. The critical factor is how precisely you frame the question. Vague questions produce vague reports. Specific questions produce specific answers.

**Framing the question:** Tell the worker exactly what you need to understand. "How does the auth flow work?" is too broad — you'll get a tour. "How does the session token get validated on each request after login?" will get you a precise trace through the relevant code. Share known starting points (relevant files, modules, entry points) so the worker doesn't spend their first turn just finding where to start.

**Depth:** Tell the worker whether you need a structural overview (map the modules, name the key types, identify the boundaries) or a mechanism trace (follow the actual code paths step by step). Over-investigating wastes time; under-investigating leaves gaps that show up later when you try to build on the findings.

**How findings feed downstream:** Exploration is usually early-stage work — its output feeds into planning, implementation, or debugging. If you know how the findings will be used, share that context so the worker can emphasize what matters for the next step and skip what doesn't.

**What to expect back:** A report written to `$M/reports/` that answers the question directly, with file references (not long verbatim content). Organized and concise — not a dump of everything found. Unknowns and gaps called out explicitly.

## Completion

The exploration is complete when:
- The question is answered with evidence from actual code, not inference
- Key findings reference specific files and line ranges
- Report is organized and concise — not a dump of everything found
- Unknowns and gaps are called out explicitly
